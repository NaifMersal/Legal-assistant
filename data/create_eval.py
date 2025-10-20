# === Step 1: Setup and Imports ===
import json
import math
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from transformers import AutoTokenizer

# === Step 1.1: Logging Setup ===
def setup_logging(log_file: str = "qa_generation.log"):
    """Setup logging configuration with both file and console handlers"""
    logger = logging.getLogger("QAGenerator")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# Initialize tokenizer for Arabic
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

# === Step 1.2: Global Constants ===
MAX_CONTEXT_SIZE = 3600  # Max tokens for the context window
MAX_QUESTIONS_PER_CALL = 5  # Max QAs to request in a single LLM call
PHASE3_ARTICLE_OVERLAP = 1   # Number of articles to overlap between phase 3 chunks


# === Step 2: Pydantic Models for Structured Output ===
class Article(BaseModel):
    id: int
    title: str
    text: str

class LawPart(BaseModel):
    name: str
    articles: Dict[int, Article] = Field(default_factory=dict)

class LawData(BaseModel):
    brief: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parts: Dict[str, LawPart] = Field(default_factory=dict)

class LawInfo(BaseModel):
    law_name: str
    law_data: LawData
    total_tokens: int
    num_parts: int
    category: str

class QAPair(BaseModel):
    """QA pair without type field - type is added when saving"""
    question: str
    answer: str
    references_ids: List[int]  # COMMENT: Should be List[int] but LLM might return strings

class QAOutput(BaseModel):
    qa_pairs: List[QAPair] = Field(default_factory=list)

class GeneratedQA(BaseModel):
    """
    MODIFIED: Now only stores selected_articles (List[int])
    Since parts consist of articles, we only need to track which articles were selected
    """
    id: str
    law_name: str
    phase: str
    category: str
    question: str
    answer: str
    references_ids: List[int]
    type: str = "factual"  # Hard-coded when saving
    selected_articles: Optional[List[int]] = None  # CHANGED: Only this field now

class DatasetMetadata(BaseModel):
    total_qa_pairs: int = 0
    laws_processed: int = 0
    phases_used: List[str] = Field(default_factory=list)
    model: str = "unknown"
    generation_start: Optional[str] = None
    generation_end: Optional[str] = None
    last_processed_law: Optional[str] = None

class QADataset(BaseModel):
    metadata: DatasetMetadata = Field(default_factory=DatasetMetadata)
    qa_pairs: List[GeneratedQA] = Field(default_factory=list)

# === Step 3: LLM Initialization ===
def initialize_llm():
    from langchain_openai.chat_models.base import ChatOpenAI
    logger.info("Initializing LLM")
    return ChatOpenAI(
        openai_api_base ="http://localhost:11434/v1",
        api_key ="good",
        model="gemma3:27b-it-qat",
        temperature=0.0,
        max_tokens=2000
    )

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

# === Step 4: Data Loading and Preparation ===
def load_and_prepare_laws(file_path: str) -> List[LawInfo]:
    logger.info(f"Loading laws data from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file: {str(e)}")
        raise
    
    laws_data = []
    for main_cat, sub_categories in data.items():
        for sub_cat, laws in sub_categories.items():
            for law_title, law_obj in laws.items():
                # Reconstruct structured parts with proper naming
                parts_dict = {}
                total_text = ""

                # Handle 'brief' and metadata
                brief = law_obj.get("brief", "")
                total_text += brief + "\n"

                # Process 'parts': keys are actual Arabic titles or 'main'
                raw_parts = law_obj.get("parts", {})
                for part_key, articles_list in raw_parts.items():
                    part_name = part_key
                    part_articles = {}
                    for art in articles_list:
                        art_id = art.get("id", "")
                        art_text = art.get("Article_Text", "")
                        art_title = art.get("Article_Title", "")
                        # COMMENT: art_id should be int, but JSON might have it as string
                        try:
                            art_id_int = int(art_id)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid article ID: {art_id}, skipping")
                            continue
                        
                        part_articles[art_id_int] = Article(
                            id=art_id_int,
                            title=art_title,
                            text=art_text
                        )
                        total_text += art_text + "\n"
                    
                    parts_dict[part_key] = LawPart(
                        name=part_name,
                        articles=part_articles
                    )

                total_tokens = count_tokens(total_text)

                laws_data.append(LawInfo(
                    law_name=law_title,
                    law_data=LawData(
                        brief=brief,
                        metadata=law_obj.get('metadata', {}),
                        parts=parts_dict
                    ),
                    total_tokens=total_tokens,
                    num_parts=len(parts_dict),
                    category=f"{main_cat} > {sub_cat}"
                ))
    
    logger.info(f"Loaded {len(laws_data)} laws successfully")
    return laws_data

# === Step 5: Phase Classification ===
class PhaseClassifier:
    @staticmethod
    def classify_law_phase(law_info: LawInfo) -> str:
        tokens = law_info.total_tokens
        parts = law_info.num_parts
        
        if tokens < MAX_CONTEXT_SIZE:
            return "phase1"
        elif tokens >= MAX_CONTEXT_SIZE and parts > 1:
            return "phase2"
        else:
            return "phase3"

    @staticmethod
    def calculate_qa_quantity(law_info: LawInfo) -> int:
        tokens = law_info.total_tokens
        num_qa = max(1, math.ceil(tokens / 500))
        return num_qa


# === Step 6: Context Preparation ===
class ContextPreparer:
    @staticmethod
    def prepare_context_phase1(law_data: LawData) -> Tuple[str, List[int]]:
        """
        MODIFIED: Returns List[int] of all article IDs instead of part names
        """
        parts = law_data.parts
        law_name = law_data.metadata.get("الاسم", "قانون غير محدد")
        context_lines = [f"اسم القانون: {law_name}"]
        
        selected_articles = []  # CHANGED: Collect article IDs

        if law_data.brief:
            context_lines.append(f"نبذة عن القانون: {law_data.brief}")
        
        for part_key, part_info in parts.items():
            context_lines.append(f"\n--- {part_info.name} ---")
            for art_info in part_info.articles.values():
                context_lines.append(f'Article id: {art_info.id}\n')
                context_lines.append(f"The Article:\n {art_info.text}")
                selected_articles.append(art_info.id)  # CHANGED: Add article ID
        
        return "\n".join(context_lines), selected_articles

    @staticmethod
    def prepare_context_phase2(law_data: LawData) -> Tuple[str, List[int]]:
        """
        MODIFIED: Returns List[int] of selected article IDs instead of part names
        """
        parts = law_data.parts
        part_keys = list(parts.keys())
        if not part_keys:
            return "", []
        
        selected_keys = []
        
        if 'main' in part_keys:
            selected_keys.append('main')
        else:
            selected_keys.append(part_keys[0])
            if len(part_keys) > 1:
                largest = max(part_keys, key=lambda k: len(parts[k].articles))
                if largest not in selected_keys:
                    selected_keys.append(largest)
            remaining = [k for k in part_keys if k not in selected_keys]
            if remaining:
                selected_keys.append(random.choice(remaining))
        
        law_name = law_data.metadata.get("الاسم", "قانون غير محدد")
        context_lines = [f"اسم القانون: {law_name}"]
        
        selected_articles = []  # CHANGED: Collect article IDs

        if law_data.brief:
            context_lines.append(f"نبذة عن القانون: {law_data.brief}")
        
        for key in selected_keys:
            part = parts[key]
            context_lines.append(f"\n--- {part.name} ---")
            for art_info in part.articles.values():
                context_lines.append(f'Article id: {art_info.id}\n')
                context_lines.append(f"The Article:\n {art_info.text}")
                selected_articles.append(art_info.id)  # CHANGED: Add article ID
        
        return "\n".join(context_lines), selected_articles

    @staticmethod
    def prepare_context_phase3(
        law_data: LawData, 
        max_tokens: int,
        start_article_index: int,
        all_articles_list: List[Article]
    ) -> Tuple[str, List[int], int]:
        """
        MODIFIED: This is the "controlled" chunking phase.
        It builds context from a list of all articles, starting at 'start_article_index',
        respecting 'max_tokens'.
        It returns the context, selected article IDs, and the 'next_start_index'
        for the next chunk, including overlap.
        """
        law_name = law_data.metadata.get("الاسم", "قانون غير محدد")
        context_lines = [f"اسم القانون: {law_name}"]
        selected_article_ids = []

        if law_data.brief:
            context_lines.append(f"نبذة عن القانون: {law_data.brief}")
        
        base_context = "\n".join(context_lines)
        current_tokens = count_tokens(base_context)
        
        if current_tokens > max_tokens:
            logger.warning(f"  [Phase 3] Base context for {law_name} ({current_tokens} tokens) already exceeds limit {max_tokens}. Returning empty context.")
            # Return 'start_article_index' to avoid infinite loop
            return "", [], start_article_index

        if not all_articles_list or start_article_index >= len(all_articles_list):
            logger.warning(f"  [Phase 3] No articles to process or start index is out of bounds.")
            return base_context, [], len(all_articles_list)

        last_part_name_added = None
        articles_added_count = 0
        current_article_index = start_article_index

        while current_article_index < len(all_articles_list):
            art = all_articles_list[current_article_index]
            
            # Find which part this article belongs to
            part_name = ""
            for p in law_data.parts.values():
                if art.id in p.articles:
                    part_name = p.name
                    break
            
            article_lines = []
            if part_name and part_name != last_part_name_added:
                article_lines.append(f"\n--- {part_name} ---")
                last_part_name_added = part_name
            
            article_lines.append(f'Article id: {art.id}\n')
            article_lines.append(f"The Article:\n {art.text}")

            article_text_to_add = "\n".join(article_lines)
            article_tokens = count_tokens(article_text_to_add)
            
            # Check if adding this article would exceed the limit
            if current_tokens + article_tokens > max_tokens and articles_added_count > 0:
                # We can't add this article, so stop.
                # Only stop if we've added at least one article, otherwise we'd get stuck.
                logger.info(f"  [Phase 3] Token limit {max_tokens} reached. Stopping article inclusion for this chunk.")
                break
            
            # If it fits (or it's the first article and must be included)
            context_lines.extend(article_lines)
            selected_article_ids.append(art.id)
            current_tokens += article_tokens
            articles_added_count += 1
            current_article_index += 1
            
            # Special case: If even the *first* article was too big, break
            if articles_added_count == 1 and current_tokens > max_tokens:
                 logger.warning(f"  [Phase 3] First article ({art.id}) in chunk is larger ({current_tokens}) than max_tokens ({max_tokens}).")
                 break

        if not selected_article_ids:
             logger.warning(f"  [Phase 3] No articles could be added for {law_name} in this chunk (start_index: {start_article_index})")

        # Calculate next start index with overlap
        # We want to overlap, but *always* advance by at least 1
        # 'current_article_index' is the index of the *next* article we would have processed
        next_start_index = max(start_article_index + 1, current_article_index - PHASE3_ARTICLE_OVERLAP)
        
        # Ensure next_start_index doesn't go backwards if context was empty
        if articles_added_count == 0 and start_article_index < len(all_articles_list):
             next_start_index = start_article_index + 1

        return "\n".join(context_lines), selected_article_ids, next_start_index

# === Step 7: QA Generation with Structured Output ===
class QAGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(QAOutput)
        self.prompt_template = self._create_qa_generation_prompt()
        self.chain = self.prompt_template | self.structured_llm

    def _create_qa_generation_prompt(self) -> PromptTemplate:
        template = """
    أنت خبير قانوني سعودي. النص التالي يخص أحد الأنظمة السعودية.

    **النص القانوني:**
    {context}

    **تعليمات إنشاء الأسئلة:**
    - أنشئ أسئلة قانونية دقيقة ذات صلة مباشرة بالقانون المذكور في النص.
    - يجب أن تكون الأسئلة مستندة إلى النص فقط.
    - لا تكتب الإجابات.
    - أدرج قائمة تحتوي فقط على **المعرفات (id)** للمواد التي يمكن أن يُستمد منها الجواب.
    - مثال للإخراج المتوقع:

    [
    {{ "question": "ما هي العقوبة على ترك العامل عمله بدون إشعار؟", "reference_ids": [12] }},
    {{ "question": "ما الشروط اللازمة لإنهاء عقد العمل؟", "reference_ids": [40, 41] }}
    ]

    **عدد الأسئلة المطلوب:** {num_questions}
    """
        return PromptTemplate(
            input_variables=["context", "num_questions"],
            template=template
        )


    def generate_qa(self, context: str, num_questions: int) -> QAOutput:
        """Generate QA pairs using structured output"""
        try:
            result = self.chain.invoke({
                "context": context, 
                "num_questions": num_questions
            })
            for qa_pair in result.qa_pairs:
                qa_pair.references_ids = [
                    int(ref_id) if isinstance(ref_id, str) else ref_id 
                    for ref_id in qa_pair.references_ids
                ]
            return result
        except Exception as e:
            logger.error(f"Error generating QA: {str(e)}")
            return QAOutput()

# === Step 8: Progress Tracking and Resumption ===
class ProgressTracker:
    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.checkpoint_file = self.output_file.with_suffix('.checkpoint.json')
        
    def load_existing_dataset(self) -> Tuple[QADataset, int]:
        """Load existing dataset if available"""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dataset = QADataset(**data)
                next_id = len(dataset.qa_pairs) + 1
                logger.info(f"Resuming from existing dataset with {len(dataset.qa_pairs)} QA pairs")
                return dataset, next_id
            except Exception as e:
                logger.warning(f"Could not load existing dataset: {str(e)}")
        
        return QADataset(
            metadata=DatasetMetadata(generation_start=datetime.now().isoformat())
        ), 1
    
    def get_processed_laws(self, dataset: QADataset) -> set:
        """Get set of already processed law names"""
        return {qa.law_name for qa in dataset.qa_pairs}
    
    def save_dataset(self, dataset: QADataset):
        """Save dataset to file atomically"""
        try:
            # Save to temporary file first
            temp_file = self.output_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(dataset.model_dump_json(indent=2))
            
            # Atomic rename
            temp_file.replace(self.output_file)
            logger.debug(f"Dataset saved successfully ({len(dataset.qa_pairs)} QA pairs)")
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            raise

# === Step 9: Main Generation Function ===
class DatasetGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.qa_generator = QAGenerator(llm)
        self.phase_classifier = PhaseClassifier()
        self.context_preparer = ContextPreparer()

    def generate_qa_dataset(self, laws_data: List[LawInfo], output_file: str) -> QADataset:
        tracker = ProgressTracker(output_file)
        dataset, qa_id = tracker.load_existing_dataset()
        processed_laws = tracker.get_processed_laws(dataset)
        
        logger.info(f"Starting QA generation. Already processed: {len(processed_laws)} laws")
        
        model_name = self.llm.model_name if hasattr(self.llm, 'model_name') else "unknown"
        dataset.metadata.model = model_name
        
        for idx, law in enumerate(laws_data, 1):
            # Skip already processed laws
            if law.law_name in processed_laws:
                logger.info(f"[{idx}/{len(laws_data)}] Skipping already processed: {law.law_name}")
                continue
            
            logger.info(f"[{idx}/{len(laws_data)}] Processing: {law.law_name}")
            
            phase = self.phase_classifier.classify_law_phase(law)
            total_q_needed = self.phase_classifier.calculate_qa_quantity(law)
            
            logger.info(f"  Initial Phase: {phase}, Total Target QAs: {total_q_needed}, Tokens: {law.total_tokens}")

            q_generated_for_law = 0
            chunk_num = 1
            
            # Pre-calculate article list for phase 3 chunking
            all_articles_list = [art for part in law.law_data.parts.values() for art in part.articles.values()]
            all_articles_list.sort(key=lambda a: a.id)
            total_articles_in_law = len(all_articles_list)
            p3_article_start_index = 0
            
            # === NEW CHUNK LOOP ===
            # Keep generating chunks until we meet the total QAs needed
            while q_generated_for_law < total_q_needed:
                
                # Determine QAs for this specific chunk
                num_q_this_chunk = min(MAX_QUESTIONS_PER_CALL, total_q_needed - q_generated_for_law)
                if num_q_this_chunk <= 0:
                    break  # Should not happen, but as a safeguard
                
                num_chunks_total = math.ceil(total_q_needed / MAX_QUESTIONS_PER_CALL)
                logger.info(f"  Generating chunk {chunk_num}/{num_chunks_total}. Target QAs: {num_q_this_chunk} (Total: {q_generated_for_law}/{total_q_needed})")
                
                context = ""
                selected_articles = []
                
                # --- Context Preparation ---
                if phase == "phase1":
                    # Phase 1 is for small laws, only run once.
                    if chunk_num > 1: 
                        logger.info("  [Phase 1] Already processed, stopping chunks.")
                        break 
                    context, selected_articles = self.context_preparer.prepare_context_phase1(law.law_data)
                
                elif phase == "phase2":
                    # Phase 2's randomness is our "chunking".
                    # We just call it again to get a new random selection of parts.
                    context, selected_articles = self.context_preparer.prepare_context_phase2(law.law_data)
                    context_tokens = count_tokens(context)
                    
                    if context_tokens > MAX_CONTEXT_SIZE:
                        logger.warning(f"  [Phase 2] Context too large ({context_tokens} > {MAX_CONTEXT_SIZE}). Switching to controlled Phase 3 for *all* remaining chunks.")
                        phase = "phase3" # Permanently switch for this law
                        # Fall-through to the 'phase3' block below
                    else:
                        # Phase 2 context is valid, proceed to generation
                        pass
                
                # Note: 'if' not 'elif', as phase2 can fall-through
                if phase == "phase3": 
                    if p3_article_start_index >= total_articles_in_law:
                        logger.info(f"  [Phase 3] No more articles to process. Stopping chunk generation.")
                        break # We've exhausted all articles
                        
                    context, selected_articles, p3_article_start_index = self.context_preparer.prepare_context_phase3(
                        law.law_data, 
                        MAX_CONTEXT_SIZE, 
                        p3_article_start_index,
                        all_articles_list
                    )
                # --- End Context Preparation ---

                if not context.strip() or not selected_articles:
                    logger.warning(f"  Empty context or no selected articles for {law.law_name} (Phase {phase}, Chunk {chunk_num}), skipping chunk.")
                    if phase == "phase1": break # P1 failed
                    if phase == "phase3" and not selected_articles: break # P3 exhausted
                    chunk_num += 1
                    continue
                
                # Final safety check
                final_context_tokens = count_tokens(context)
                if final_context_tokens > MAX_CONTEXT_SIZE:
                     logger.error(f"  FATAL: Context for {law.law_name} ({phase}) still too large ({final_context_tokens} > {MAX_CONTEXT_SIZE}) after preparation. Skipping chunk.")
                     chunk_num += 1
                     continue
                
                try:
                    # --- Generation for this chunk ---
                    with get_openai_callback() as cb:
                        qa_output = self.qa_generator.generate_qa(context, num_q_this_chunk)
                        logger.info(f"    Generated {len(qa_output.qa_pairs)} QAs | API Tokens: {cb.total_tokens}")
                    
                    if not qa_output.qa_pairs:
                        logger.warning(f"    LLM returned 0 QAs for this chunk.")
                        if phase == "phase3" and p3_article_start_index >= total_articles_in_law:
                             logger.info("    [Phase 3] Stopping as 0 QAs returned and all articles processed.")
                             break # Exhausted articles and got 0
                        # Otherwise, just continue to the next chunk
                    
                    # --- Save results for this chunk ---
                    for qa in qa_output.qa_pairs:
                        generated_qa = GeneratedQA(
                            id=f"qa_{qa_id:04d}",
                            law_name=law.law_name,
                            phase=phase, # This will correctly log "phase3" if it fell back
                            category=law.category,
                            question=qa.question,
                            answer=qa.answer,
                            references_ids=qa.references_ids,
                            type="factual",
                            selected_articles=selected_articles
                        )
                        dataset.qa_pairs.append(generated_qa)
                        qa_id += 1
                    
                    q_generated_for_law += len(qa_output.qa_pairs)
                    chunk_num += 1
                    
                    # Update metadata and save after *each chunk*
                    dataset.metadata.total_qa_pairs = len(dataset.qa_pairs)
                    dataset.metadata.last_processed_law = law.law_name
                    phases_set = set(dataset.metadata.phases_used)
                    phases_set.add(phase)
                    dataset.metadata.phases_used = sorted(list(phases_set))
                    
                    tracker.save_dataset(dataset)
                    
                except Exception as e:
                    logger.error(f"    Error processing chunk {chunk_num} for {law.law_name}: {str(e)}", exc_info=True)
                    # Skip this chunk and try the next one
                    chunk_num += 1
                    if phase == "phase1": break # If P1 fails, stop.
                    continue
            
            # --- End of NEW CHUNK LOOP ---
            
            # Add to processed list *after* all chunks for a law are done
            processed_laws.add(law.law_name)
            dataset.metadata.laws_processed = len(processed_laws)

        # Final save with end timestamp
        dataset.metadata.generation_end = datetime.now().isoformat()
        tracker.save_dataset(dataset)
        
        logger.info(f"Generation complete! Total QAs: {len(dataset.qa_pairs)}")
        return dataset
# === Step 10: Main Execution ===
def main():
    logger.info("="*60)
    logger.info("Starting QA Dataset Generation")
    logger.info("="*60)
    
    try:
        # Initialize components
        llm = initialize_llm()
        
        # Load laws data
        laws_data = load_and_prepare_laws("saudi_laws_scraped.json")
        
        # Generate dataset
        generator = DatasetGenerator(llm)
        dataset = generator.generate_qa_dataset(laws_data, "law_qa_dataset.json")
        
        logger.info("="*60)
        logger.info(f"SUCCESS! Generated {len(dataset.qa_pairs)} QA pairs")
        logger.info(f"Output file: law_qa_dataset.json")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()