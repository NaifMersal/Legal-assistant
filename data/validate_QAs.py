# === QA Validation and Correction Pipeline ===
# Add these components to your existing code
from typing import List, Dict, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field
from create_eval import PromptTemplate, LawData, LawInfo,QADataset,GeneratedQA,DatasetGenerator,\
      ProgressTracker,MAX_CONTEXT_SIZE, MAX_QUESTIONS_PER_CALL, setup_logging, count_tokens,\
        initialize_llm, load_and_prepare_laws, get_openai_callback

import datetime
logger = setup_logging()

class ValidationStatus(str, Enum):
    VALID = "valid"
    CORRECTED = "corrected"
    REJECTED = "rejected"

class LLMValidationOutput(BaseModel):
    """Ultra-compact LLM validation output to minimize tokens"""
    is_answerable: bool = Field(..., description="False=reject, True=proceed")
    suggested_refs: List[int] = Field(default_factory=list, description="Valid supporting references only")

class QAValidationResult(BaseModel):
    """Compact validation result"""
    status: ValidationStatus
    original_references: List[int]
    corrected_references: Optional[List[int]] = None
    rejection_reason: Optional[str] = None

# === Compact Validation Pipeline ===
class CompactQAValidationPipeline:
    def __init__(self, llm):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(LLMValidationOutput)
        self.validation_prompt = self._create_compact_prompt()
        self.validation_chain = self.validation_prompt | self.structured_llm
        
    def _create_compact_prompt(self) -> PromptTemplate:
        template = """
You are a legal expert validating QA pairs against provided articles.

ARTICLES:\n {selected_articles_context}
QUESTION:\n {question}
ANSWER:\n {answer}
CLAIMED_REFS: {claimed_references}

Validate:
1. Are all claimed refs in the articles? 
2. Do they support the answer?
3. If refs are wrong/missing, suggest correct ones from articles only.
4. If unanswerable, set is_answerable=false.

OUTPUT FORMAT:
{{"is_answerable": true, "suggested_refs": [12,30]}}

Analyze:
"""
        return PromptTemplate(
            input_variables=["selected_articles_context", "question", "answer", "claimed_references"],
            template=template
        )
    
    def _build_articles_context(self, law_data: LawData, selected_article_ids) -> str:
        """Build a structured and LLM-friendly context string from selected articles."""
        context_parts = []
        for part_title, part in law_data.parts.items():
            for art_id, art_info in part.articles.items():
                if art_id in selected_article_ids:
                    article_block = (
                        f"========== ARTICLE ==========\n"
                        f"Article ID: {art_id}\n"
                        f"Text:\n{art_info.text.strip()}\n"
                        f"=============================\n"
                    )
                    context_parts.append(article_block)
        return "\n".join(context_parts).strip()

    
    def validate_qa_pair(
        self, 
        question: str, 
        answer: str, 
        claimed_references: List[int],
        selected_article_ids: List[int],
        law_data: LawData
    ) -> QAValidationResult:
        """Compact validation logic"""
        # Quick checks first
        if not claimed_references:
            return QAValidationResult(
                status=ValidationStatus.REJECTED,
                original_references=[],
                rejection_reason="No refs provided"
            )
        
        articles_context = self._build_articles_context(law_data, selected_article_ids)
        if not articles_context.strip():
            return QAValidationResult(
                status=ValidationStatus.REJECTED,
                original_references=claimed_references,
                rejection_reason="No articles available"
            )
        
        # LLM validation
        try:
            result = self.validation_chain.invoke({
                "selected_articles_context": articles_context,
                "question": question,
                "answer": answer,
                "claimed_references": str(claimed_references)
            })
            
            # Apply your logic: is_answerable=False => reject
            if not result.is_answerable:
                return QAValidationResult(
                    status=ValidationStatus.REJECTED,
                    original_references=claimed_references,
                    rejection_reason="Unanswerable from provided articles"
                )
            
            # suggested_refs == claimed_refs => valid
            if set(result.suggested_refs) == set(claimed_references):
                return QAValidationResult(
                    status=ValidationStatus.VALID,
                    original_references=claimed_references
                )
            
            # suggested_refs != claimed_refs => corrected
            if result.suggested_refs and set(result.suggested_refs).issubset(set(selected_article_ids)):  # Has valid suggestions
                return QAValidationResult(
                    status=ValidationStatus.CORRECTED,
                    original_references=claimed_references,
                    corrected_references=result.suggested_refs
                )
            else:
                return QAValidationResult(
                    status=ValidationStatus.REJECTED,
                    original_references=claimed_references,
                    rejection_reason="No valid references found"
                )
                
        except Exception as e:
            return QAValidationResult(
                status=ValidationStatus.REJECTED,
                original_references=claimed_references,
                rejection_reason=f"Validation error: {str(e)}"
            )


# === Integration with Main Generator ===
class ValidatedDatasetGenerator(DatasetGenerator):
    """Extended generator with validation pipeline"""
    
    def __init__(self, llm):
        super().__init__(llm)
        self.validator = CompactQAValidationPipeline(llm)
        self.validation_stats = {
            "total_checked": 0,
            "valid": 0,
            "corrected": 0,
            "rejected": 0
        }
    
    def generate_qa_dataset(self, laws_data: List[LawInfo], output_file: str) -> QADataset:
        """Override with validation enabled"""
        tracker = ProgressTracker(output_file)
        dataset, qa_id = tracker.load_existing_dataset()
        processed_laws = tracker.get_processed_laws(dataset)
        
        logger.info(f"Starting QA generation with validation. Already processed: {len(processed_laws)} laws")
        
        model_name = self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown"
        dataset.metadata.model = model_name
        
        for idx, law in enumerate(laws_data, 1):
            if law.law_name in processed_laws:
                logger.info(f"[{idx}/{len(laws_data)}] Skipping: {law.law_name}")
                continue
            
            logger.info(f"[{idx}/{len(laws_data)}] Processing: {law.law_name}")
            
            phase = self.phase_classifier.classify_law_phase(law)
            total_q_needed = self.phase_classifier.calculate_qa_quantity(law)
            
            logger.info(f"  Phase: {phase}, Target QAs: {total_q_needed}")

            q_generated_for_law = 0
            chunk_num = 1
            
            all_articles_list = [art for part in law.law_data.parts.values() for art in part.articles.values()]
            all_articles_list.sort(key=lambda a: a.id)
            total_articles_in_law = len(all_articles_list)
            p3_article_start_index = 0
            
            while q_generated_for_law < total_q_needed:
                num_q_this_chunk = min(MAX_QUESTIONS_PER_CALL, total_q_needed - q_generated_for_law)
                if num_q_this_chunk <= 0:
                    break
                
                logger.info(f"  Chunk {chunk_num}: Target {num_q_this_chunk} QAs")
                
                # Get context and selected articles (same as before)
                context, selected_articles = "", []
                
                if phase == "phase1":
                    if chunk_num > 1: break
                    context, selected_articles = self.context_preparer.prepare_context_phase1(law.law_data)
                
                elif phase == "phase2":
                    context, selected_articles = self.context_preparer.prepare_context_phase2(law.law_data)
                    if count_tokens(context) > MAX_CONTEXT_SIZE:
                        logger.warning(f"  Switching to phase3")
                        phase = "phase3"
                
                if phase == "phase3":
                    if p3_article_start_index >= total_articles_in_law:
                        break
                    context, selected_articles, p3_article_start_index = self.context_preparer.prepare_context_phase3(
                        law.law_data, MAX_CONTEXT_SIZE, p3_article_start_index, all_articles_list
                    )
                
                if not context.strip() or not selected_articles:
                    logger.warning(f"  Empty context, skipping chunk")
                    chunk_num += 1
                    continue
                
                try:
                    # Generate QAs
                    with get_openai_callback() as cb:
                        qa_output = self.qa_generator.generate_qa(context, num_q_this_chunk)
                        logger.info(f"    Generated {len(qa_output.qa_pairs)} QAs")
                    
                    # === VALIDATION STEP ===
                    validated_count = 0
                    for qa in qa_output.qa_pairs:
                        self.validation_stats["total_checked"] += 1
                        
                        # Validate the QA pair
                        validation_result = self.validator.validate_qa_pair(
                            question=qa.question,
                            answer=qa.answer,
                            claimed_references=qa.references_ids,
                            selected_article_ids=selected_articles,
                            law_data=law.law_data
                        )
                        
                        # Handle validation result
                        if validation_result.status == ValidationStatus.VALID:
                            self.validation_stats["valid"] += 1
                            final_refs = qa.references_ids
                            validated_count += 1
                            
                        elif validation_result.status == ValidationStatus.CORRECTED:
                            self.validation_stats["corrected"] += 1
                            final_refs = validation_result.corrected_references
                            selected_articles
                            validated_count += 1
                            logger.info(f"    ✓ Corrected refs: {qa.references_ids} → {final_refs}")
                            
                        elif validation_result.status == ValidationStatus.REJECTED:
                            self.validation_stats["rejected"] += 1
                            logger.warning(f"    ✗ Rejected: {validation_result.rejection_reason}")
                            continue  # Skip this QA pair
                        
                        # Add validated QA to dataset
                        generated_qa = GeneratedQA(
                            id=f"qa_{qa_id:04d}",
                            law_name=law.law_name,
                            phase=phase,
                            category=law.category,
                            question=qa.question,
                            answer=qa.answer,
                            references_ids=final_refs,
                            type="factual",
                            selected_articles=selected_articles
                        )
                        dataset.qa_pairs.append(generated_qa)
                        qa_id += 1
                    
                    logger.info(f"    Validation: {validated_count}/{len(qa_output.qa_pairs)} passed")
                    q_generated_for_law += validated_count
                    chunk_num += 1
                    
                    # Save progress
                    dataset.metadata.total_qa_pairs = len(dataset.qa_pairs)
                    dataset.metadata.last_processed_law = law.law_name
                    tracker.save_dataset(dataset)
                    
                except Exception as e:
                    logger.error(f"    Error in chunk {chunk_num}: {str(e)}", exc_info=True)
                    chunk_num += 1
                    continue
            
            processed_laws.add(law.law_name)
            dataset.metadata.laws_processed = len(processed_laws)
        
        # Log validation statistics
        logger.info("="*60)
        logger.info("VALIDATION STATISTICS:")
        logger.info(f"  Total checked: {self.validation_stats['total_checked']}")
        logger.info(f"  Valid: {self.validation_stats['valid']}")
        logger.info(f"  Corrected: {self.validation_stats['corrected']}")
        logger.info(f"  Rejected: {self.validation_stats['rejected']}")
        logger.info("="*60)
        
        dataset.metadata.generation_end = datetime.now().isoformat()
        tracker.save_dataset(dataset)
        
        return dataset


# === Updated Main Function ===
def main():
    logger.info("="*60)
    logger.info("Starting QA Dataset Generation with Validation")
    logger.info("="*60)

    output_file = "law_qa_dataset_validated.json"
    
    try:
        llm = initialize_llm()
        laws_data = load_and_prepare_laws("saudi_laws_scraped.json")
        
        # Use validated generator instead of regular one
        generator = ValidatedDatasetGenerator(llm)
        dataset = generator.generate_qa_dataset(laws_data, output_file)
        
        logger.info("="*60)
        logger.info(f"SUCCESS! Generated {len(dataset.qa_pairs)} validated QA pairs")
        logger.info(f"Output file: {output_file}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ =='__main__':
    main()