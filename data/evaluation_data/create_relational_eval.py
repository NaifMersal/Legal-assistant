import uuid
import random
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Type
from pydantic import BaseModel, Field
from create_eval import QAPair, Article,ContextPreparer, PromptTemplate, LawData, LawInfo,QADataset,GeneratedQA,DatasetGenerator,\
      ProgressTracker, MAX_QUESTIONS_PER_CALL, setup_logging, count_tokens,\
        initialize_llm, load_and_prepare_laws, get_openai_callback

MAX_CONTEXT_SIZE = 12000  # Max tokens for context in relational generation
# --- Setup Logging ---
logger = setup_logging("relational_generation.log")


# ===================================================
# === Relational Generation Models
# ===================================================

class RelationalStep1(BaseModel):
    """Output model for Step 1: Core Concept Discovery"""
    primary_concept: str
    secondary_concept: str
    interaction_type: str
    evidence_snippet: str

class RelationalStep2(BaseModel):
    """Output model for Step 2: Concept Expansion"""
    expanded_concept: str
    supporting_concepts: List[str] = Field(default_factory=list)
    interpretation_gaps: List[str] = Field(default_factory=list)
    key_evidence: str

class RelationalStep3(BaseModel):
    """Output model for Step 3: Scenario Generation"""
    scenario: str
    key_concepts: List[str] = Field(default_factory=list)
    expected_answer: str

# ===================================================
# === Relational Prompt Templates
# ===================================================
class RelationalPrompts:
    """Relational evaluation prompts using the Intelligence-First approach.
    This implementation separates conceptual question design from article ID mapping
    to force genuine legal reasoning and search behavior.
    
    Key features:
    - No article IDs in Steps 1-3 (prevents ID-centric questions)
    - Scenario-based questions requiring concept discovery
    - Real-world problem-solving mirroring legal practice
    - Final ID mapping as separate post-processing step
    """
    
    STEP1_TEMPLATE = """You are a legal concept analyzer. Your task is to identify key legal concepts that interact in the given law context.

Rules:
1. Focus on finding meaningful relationships (e.g., principle/exception, procedure/penalty)
2. Concepts must be clearly defined in the text
3. Avoid generic or overly broad concepts
4. Use specific legal terminology when possible
5. Concepts must be central to the legal text's purpose, not trivial or isolated details.

Context:
{context}

Instructions:
Before outputting, double-check: Is the interaction logical? Is the evidence a direct quote supporting this specific interaction?

Return ONLY a JSON object with EXACTLY this structure:
{{
    "primary_concept": "<specific name of main concept>",
    "secondary_concept": "<specific name of related concept>",
    "interaction_type": "<exactly one of: modifies, enables, overrides, defines, restricts>",
    "evidence_snippet": "<direct quote from context showing interaction>"
}}

Do not include any explanation or other text. Return ONLY the JSON object."""

    STEP2_TEMPLATE = """You are a legal concept analyzer examining concept relationships. Your task is to expand on the previously identified concepts:

Primary: {primary_concept}
Secondary: {secondary_concept}
Type: {interaction_type}

Review the context and identify:
1. Supporting concepts that strengthen this relationship
2. Any conditions or exceptions that modify or limit it
3. Key evidence showing these interactions

Context:
{context}

Instructions:
Return ONLY a JSON object with EXACTLY this structure:
{{
    "expanded_concept": "<refined name combining main concepts>",
    "supporting_concepts": [
        "<concept1>",
        "<concept2>"
    ],
    "limiting_conditions_or_exceptions": [
        "<condition or exception 1>",
        "<condition or exception 2>"
    ],
    "key_evidence": "<quote from context showing support>"
}}

Do not include any explanation or other text. Return ONLY the JSON object."""

    STEP3_TEMPLATE = """
You are a legal examiner. Create a high-stakes scenario where a practitioner must:  
1. Identify which legal concepts from the context resolve the problem,  
2. Explain how they interact,  
3. Justify their application using the law.  

Target Language: Arabic

Requirements:  
- The scenario MUST NOT mention article numbers, section names, or "the law says...".  
- It MUST force the user to search for concepts like:  
  * "{primary_concept}"  
  * "{secondary_concept}"  
  * "{supporting_concepts[0]}"
- The scenario must include:
  • At least 1 ambiguous detail requiring legal interpretation of the concept interaction.
  • 1 "red herring" (a plausible but incorrect legal concept from the same context).
  • Requirement to connect 3+ concepts to solve the problem.
- The scenario MUST be a plausible, realistic situation directly inspired by the situations or purpose described in the source text. Do not invent fantastical situations.
- The solution must require understanding the interaction between these concepts.  

Context:  
{source_text}  
Expanded concepts: {expanded_concept}, {supporting_concepts}

Output ONLY in JSON. All JSON values (scenario, expected_answer) MUST be written mostly in Arabic.
"""

    STEP4_TEMPLATE = """
You are an expert legal analyst. Your task is to link the given scenario and expected answer to the correct article IDs from the provided context.

**Target Language:** Arabic

**Context (with article IDs):**
{context}

**Scenario and Concepts to Map:**
Scenario: {scenario}
Expected Answer: {answer}
Key Concepts: {concepts}

**Instructions:**
1. Carefully read the context and identify all articles directly related to the scenario and answer.
2. Use these article IDs to justify the reasoning.
3. Ensure that every relevant article ID is explicitly listed under "references_ids".
4. Do not invent or guess any IDs not present in the context.

**Output Format (JSON only):**
{{
  "question": "The scenario text exactly as provided, mostly written in Arabic",
  "answer": "The expected_answer rewritten or enhanced with proper references, in Arabic",
  "references_ids": [<all relevant article IDs from the context>]
}}
"""


# ===================================================
# === Relational Generation Strategy with LangChain
# ===================================================

class RelationalQAStrategy:
    """
    Implements the 3-step plan to generate a single relational QA pair using LangChain.
    """
    def __init__(
        self,
        all_articles_map: Dict[int, Article],
    ):
        """
        Initializes the strategy with LangChain models.
        
        Args:
            all_articles_map: A map of {article_id: Article} for the entire law.
        """
        self.step1_2_llm = initialize_llm()
        
        self.step3_llm = initialize_llm(temperature=0.1)

        
        # Create structured output chains
        self.step1_chain = self.step1_2_llm.with_structured_output(RelationalStep1)
        self.step2_chain = self.step1_2_llm.with_structured_output(RelationalStep2)
        self.step3_chain = self.step3_llm.with_structured_output(RelationalStep3)
        self.step4_chain = self.step1_2_llm.with_structured_output(QAPair)
        
        self.all_articles_map = all_articles_map
        self.prompts = RelationalPrompts()
        
        

    def generate_scenario(self, context: str) -> Optional[Tuple[RelationalStep3, dict]]:
        """
        Generates a legal scenario through steps 1-3 without article ID mapping.
        
        Args:
            context: A string containing a chunk of the law.
            
        Returns:
            Tuple of (scenario result, metadata) if successful, or None
        """
        
        # --- Step 1: Core Concept Discovery ---
        logger.info("Starting Step 1: Core Concept Discovery")
        prompt1 = self.prompts.STEP1_TEMPLATE.format(context=context)
        
        try:
            step1_result = self.step1_chain.invoke(prompt1)
      
        except Exception as e:
            logger.error(f"Step 1: LLM call failed. Error: {str(e)}")
            return None

        if not step1_result:
            logger.info("Step 1: No core concepts found in this chunk.")
            return None
        
        logger.info(f"Step 1: Found interaction between {step1_result.primary_concept} and {step1_result.secondary_concept}")

        # --- Step 2: Concept Expansion ---
        logger.info("Starting Step 2: Concept Expansion")
        prompt2 = self.prompts.STEP2_TEMPLATE.format(
            primary_concept=step1_result.primary_concept,
            secondary_concept=step1_result.secondary_concept,
            interaction_type=step1_result.interaction_type,
            context=context
        )
        
        try:
            step2_result = self.step2_chain.invoke(prompt2)
            
            
            if step2_result and not step2_result.expanded_concept:
                logger.error("Step 2: Missing expanded concept in LLM output")
                step2_result = None
                
        except Exception as e:
            logger.warning(f"Step 2: LLM call failed. Error: {str(e)}. Falling back to core concepts.")
            step2_result = None

        if not step2_result:
            logger.warning("Step 2: Expansion failed. Using core concepts only.")
            expanded_concept = f"{step1_result.primary_concept} - {step1_result.secondary_concept}"
            supporting_concepts = []
        else:
            expanded_concept = step2_result.expanded_concept.strip()
            supporting_concepts = [c.strip() for c in step2_result.supporting_concepts if c.strip()]
            logger.info(f"Step 2: Expanded to concept: {expanded_concept} with support: {supporting_concepts}")

        # --- Step 3: Scenario Generation ---
        logger.info("Starting Step 3: Scenario Generation")
        source_text = context  # Using full context since we're not ID-bound yet
            
        prompt3 = self.prompts.STEP3_TEMPLATE.format(
            source_text=source_text,
            primary_concept=step1_result.primary_concept,
            secondary_concept=step1_result.secondary_concept,
            supporting_concepts=supporting_concepts,
            expanded_concept=expanded_concept
        )
        
        try:
            step3_result = self.step3_chain.invoke(prompt3)
        except Exception as e:
            logger.error(f"Step 3: LLM call failed. {e}")
            return None
            
        if not step3_result:
            logger.error("Step 3: Failed to generate valid scenario.")
            return None

        logger.info(f"Step 3: Successfully generated scenario exploring {len(step3_result.key_concepts)} concepts")
        

        return step3_result



def main_generation_loop(output_file: str = "relational_qa_dataset.json"):
    """
    Main loop showing how to use the RelationalQAStrategy with ProgressTracker.
    
    Args:
        output_file: Path to save the dataset
    """
    logger.info("--- Starting Relational Generation Example ---")
    
    # Initialize Progress Tracker and load data
    tracker = ProgressTracker(output_file)
    dataset, next_id = tracker.load_existing_dataset()
    
    laws_data = load_and_prepare_laws("saudi_laws_scraped.json")
    all_articles_map = {
        art.id: art 
        for law in laws_data
        for part in law.law_data.parts.values() 
        for art in part.articles.values()
    }
    
    processed_laws = tracker.get_processed_laws(dataset)
    logger.info(f"Starting QA generation. Already processed: {len(processed_laws)} laws")
    # Initialize the Relational Strategy
    relational_strategy = RelationalQAStrategy(
            all_articles_map=all_articles_map,
        )
    
    for idx, law_info in enumerate(laws_data, 1):
        if law_info.law_name in processed_laws:
            logger.info(f"[{idx}/{len(laws_data)}] Skipping already processed: {law_info.law_name}")
            continue
            
        logger.info(f"[{idx}/{len(laws_data)}] Processing: {law_info.law_name}")
        

        
        # Prepare article list for phase 3 (if needed)
        all_articles_list = [art for part in law_info.law_data.parts.values() for art in part.articles.values()]
        all_articles_list.sort(key=lambda a: a.id)
        
        # Try Phase 1 first (full law) - without IDs for concept discovery
        concept_context, _ = ContextPreparer.prepare_context_phase1(law_info.law_data, include_id=False)
        current_phase = "phase1"
        
        if not concept_context or count_tokens(concept_context) > MAX_CONTEXT_SIZE:
            # If Phase 1 fails, try Phase 2 (random parts)
            concept_context, _ = ContextPreparer.prepare_context_phase2(law_info.law_data, include_id=False)
            current_phase = "phase2"
            
            if count_tokens(concept_context) > MAX_CONTEXT_SIZE:
                # If Phase 2 fails, try Phase 3 (controlled chunks)
                logger.info(f"Context too large for Phase 2, falling back to Phase 3 for {law_info.law_name}")
                concept_context, _, _ = ContextPreparer.prepare_context_phase3(
                    law_info.law_data,
                    MAX_CONTEXT_SIZE,
                    0,  # Start from first article
                    all_articles_list,
                    include_id=False
                )
                current_phase = "phase3"
        
        if concept_context:
            logger.info(f"Processing concepts using {current_phase}...")
            
            try:
                # Generate scenario without IDs
                step3_result = relational_strategy.generate_scenario(concept_context)
                
                if step3_result:

                    # Now get context with IDs for mapping step
                    id_context, selected_article_ids = ContextPreparer.prepare_context_phase1(law_info.law_data, include_id=True)
                    
                    # Step 4: Map to article IDs
                    logger.info("Starting Step 4: Article ID Mapping")
                    prompt4 = RelationalPrompts.STEP4_TEMPLATE.format(
                        scenario=step3_result.scenario,
                        answer=step3_result.expected_answer,
                        concepts=step3_result.key_concepts,
                        context=id_context
                    )
                    
                    try:
                        qa_pair = relational_strategy.step4_chain.invoke(prompt4)  
                        intersection_ids = set(qa_pair.references_ids).intersection(set(selected_article_ids))
                        if qa_pair and intersection_ids:
                            logger.info(f"Step 4: Successfully mapped to {len(qa_pair.references_ids)} articles")
                            
                            # Create final GeneratedQA
                            generated_qa = GeneratedQA(
                                id=str(uuid.uuid4()),
                                law_name=law_info.law_name,
                                phase=current_phase,
                                category=law_info.category,
                                question=qa_pair.question,
                                answer=qa_pair.answer,
                                references_ids=list(intersection_ids),
                                selected_articles=selected_article_ids,
                                type="relational",
                            )
                        else:
                            logger.error("Step 4: Failed to map concepts to article IDs")
                            continue
                    except Exception as e:
                        logger.error(f"Step 4: Article mapping failed: {str(e)}")
                        continue
                    
                    # Add to dataset and save
                    dataset.qa_pairs.append(generated_qa)
                    dataset.metadata.total_qa_pairs = len(dataset.qa_pairs)
                    dataset.metadata.laws_processed += 1
                    dataset.metadata.last_processed_law = law_info.law_name
                    
                    if current_phase not in dataset.metadata.phases_used:
                        dataset.metadata.phases_used.append(current_phase)
                    if "relational" not in dataset.metadata.phases_used:
                        dataset.metadata.phases_used.append("relational")
                    
                    tracker.save_dataset(dataset)
                    logger.info(f"Successfully added relational QA pair (using {current_phase}) to dataset and saved.")
                    
            except Exception as e:
                logger.error(f"Error processing {law_info.law_name}: {str(e)}", exc_info=True)
                continue
                
        processed_laws.add(law_info.law_name)
    
    # Final metadata update
    dataset.metadata.generation_end = datetime.now().isoformat()
    tracker.save_dataset(dataset)
    logger.info(f"Generation complete! Total QAs: {len(dataset.qa_pairs)}")
        


if __name__ == "__main__":
    main_generation_loop()