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
        max_token_size = 3000
        
        if tokens < max_token_size:
            return "phase1"
        elif tokens >= max_token_size and parts > 1:
            return "phase2"
        else:
            return "phase3"

    @staticmethod
    def calculate_qa_quantity(law_info: LawInfo, phase: str) -> int:
        tokens = law_info.total_tokens
        if phase == "phase1":
            return max(1, math.ceil(tokens / 500))
        elif phase == "phase2":
            return max(1, math.ceil(tokens / 400))
        else:
            return max(1, math.ceil(tokens / 500))

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
                context_lines.append(f'id: {art_info.id}')
                context_lines.append(f"{art_info.title}: {art_info.text}")
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
                context_lines.append(f'id: {art_info.id}')
                context_lines.append(f"{art_info.title}: {art_info.text}")
                selected_articles.append(art_info.id)  # CHANGED: Add article ID
        
        return "\n".join(context_lines), selected_articles

    @staticmethod
    def prepare_context_phase3(law_data: LawData) -> Tuple[str, List[int]]:
        """
        Already returns List[int], no changes needed in return type
        """
        parts = law_data.parts
        if not parts:
            return "", []
        
        part_key = next(iter(parts.keys()))
        part = parts[part_key]
        articles = part.articles
        art_keys = list(articles.keys())
        if not art_keys:
            return "", []
        
        selected_arts = set(art_keys[:2])
        if len(art_keys) > 2:
            mid = len(art_keys) // 2
            selected_arts.add(art_keys[mid])
            selected_arts.add(art_keys[-1])
        
        law_name = law_data.metadata.get("الاسم", "قانون غير محدد")
        context_lines = [f"اسم القانون: {law_name}"]

        if law_data.brief:
            context_lines.append(f"نبذة عن القانون: {law_data.brief}")
        
        context_lines.append(f"\n--- {part.name} ---")
        selected_article_ids = []  # CHANGED: Use consistent variable name
        for art_key in selected_arts:
            art = articles[art_key]
            context_lines.append(f'id: {art.id}')
            context_lines.append(f"{art.title}: {art.text}")
            selected_article_ids.append(art.id)
        
        return "\n".join(context_lines), selected_article_ids

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
    {{ "question": "ما هي العقوبة على ترك العامل عمله بدون إشعار؟", "reference_ids": ["12"] }},
    {{ "question": "ما الشروط اللازمة لإنهاء عقد العمل؟", "reference_ids": ["40", "41"] }}
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
            # COMMENT: LLM might return references_ids as strings, need to convert to int
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
            num_q = self.phase_classifier.calculate_qa_quantity(law, phase)
            
            logger.info(f"  Phase: {phase}, Target QAs: {num_q}, Tokens: {law.total_tokens}")
            
            # MODIFIED: All phases now return List[int] of article IDs
            if phase == "phase1":
                context, selected_articles = self.context_preparer.prepare_context_phase1(law.law_data)
            elif phase == "phase2":
                context, selected_articles = self.context_preparer.prepare_context_phase2(law.law_data)
            else:  # phase3
                context, selected_articles = self.context_preparer.prepare_context_phase3(law.law_data)
            
            if not context.strip():
                logger.warning(f"  Empty context for {law.law_name}, skipping")
                continue
            
            try:
                with get_openai_callback() as cb:
                    qa_output = self.qa_generator.generate_qa(context, num_q)
                    logger.info(f"  Generated {len(qa_output.qa_pairs)} QAs | API Tokens: {cb.total_tokens}")
                
                # MODIFIED: Only pass selected_articles (no selected_parts)
                for qa in qa_output.qa_pairs:
                    generated_qa = GeneratedQA(
                        id=f"qa_{qa_id:04d}",
                        law_name=law.law_name,
                        phase=phase,
                        category=law.category,
                        question=qa.question,
                        answer=qa.answer,
                        references_ids=qa.references_ids,
                        type="factual",
                        selected_articles=selected_articles  # CHANGED: Only this field
                    )
                    dataset.qa_pairs.append(generated_qa)
                    qa_id += 1
                
                # Update metadata
                dataset.metadata.total_qa_pairs = len(dataset.qa_pairs)
                dataset.metadata.laws_processed = len(processed_laws) + 1
                dataset.metadata.last_processed_law = law.law_name
                phases_set = set(dataset.metadata.phases_used)
                phases_set.add(phase)
                dataset.metadata.phases_used = sorted(list(phases_set))
                
                # Save after each law
                tracker.save_dataset(dataset)
                processed_laws.add(law.law_name)
                
            except Exception as e:
                logger.error(f"  Error processing {law.law_name}: {str(e)}", exc_info=True)
                continue
        
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