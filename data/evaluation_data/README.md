# Legal QA Dataset Generation (Saudi Laws)

Generates high-quality question-answer pairs from Saudi legal texts using a structured, multi-phase approach that adapts to document size and complexity.

## Question Types

- **Simple-Factual**: Direct, single-concept questions with clear answers from individual articles.  
- **Relational**: Complex, scenario-based questions requiring legal reasoning, built via a 4-step process:
  1. **Core Concept Discovery** – Identify key legal concepts and relationships.  
  2. **Concept Expansion** – Add conditions, exceptions, and hierarchies.  
  3. **Scenario Generation** – Create realistic, ambiguous Arabic-language scenarios (no article numbers).  
  4. **Article Mapping** – Link scenarios to relevant, validated articles.

## Processing Phases

Based on document size (token-limited at 3900 for simple QA, 12000 for relational):

- **Phase 1 (Small)**: Full document processed in one pass.  
- **Phase 2 (Medium)**: Prioritizes main section + largest + one random part; falls back if over limit.  
- **Phase 3 (Large)**: Sequential chunking with overlap to preserve context and structure.

## Data & Generation

- **Structure**: Pydantic models (`LawData`, `Article`, `QAPair`, etc.) ensure consistency.  
- **Context Prep**: Includes law name, summary, and hierarchical structure with clear markers.  
- **QA Generation**:
  - *Simple*: LLM prompt → concise Q&A (1–2 sentences), no article numbers in questions.  
  - *Relational*: Intelligence-first design → multi-article, reasoning-intensive Q&A.

## Quality Control

- **LLM Validation Pipeline** checks:
  - Reference existence and accuracy  
  - Answer support from cited articles  
  - Completeness and correctness  
- Auto-corrects references, logs stats (valid/corrected/rejected), and provides rejection reasons.  
- Enforces token limits, structural integrity, and context coherence.

## Output

JSON format with:
- QA pairs (categorized by type)  
- Validated article references  
- Phase metadata   


