# Legal QA Dataset Generation Methodology

This document describes the methodology used for generating question-answer pairs from Saudi legal documents.

## Overview

The system generates high-quality question-answer pairs from Saudi legal texts using a phased approach that handles documents of varying sizes and complexities. The process is designed to maintain context coherence while respecting token limits and ensuring comprehensive coverage of legal content.

## Question Types

The system generates two distinct types of questions to comprehensively evaluate legal understanding:

### 1. Simple-Factual Questions
- Direct questions about specific legal facts, procedures, or requirements
- Single-concept focused queries
- Clear, unambiguous answers derived from individual articles
- Suitable for testing basic legal knowledge and fact retrieval

### 2. Relational Questions
Generated through a sophisticated 4-step process:
1. **Core Concept Discovery**
   - Identifies key legal concepts and their interactions
   - Maps relationships (e.g., principle/exception, procedure/penalty)
   - Focuses on meaningful concept pairs central to the law

2. **Concept Expansion**
   - Expands primary concepts with supporting elements
   - Identifies conditions and exceptions
   - Maps concept hierarchies and dependencies

3. **Scenario Generation**
   - Creates realistic, high-stakes legal scenarios
   - Includes ambiguous details requiring legal interpretation
   - Incorporates multiple interrelated concepts
   - Written primarily in Arabic
   - Avoids direct references to article numbers

4. **Article ID Mapping**
   - Links scenarios to relevant legal articles
   - Ensures comprehensive coverage of related provisions
   - Validates reference accuracy

## Components

### 1. Data Structure
- Uses Pydantic models for structured data handling
- Key models:
  - `LawData`: Represents a law with its metadata, brief, and parts
  - `LawPart`: Contains named sections of the law and their articles
  - `Article`: Individual legal articles with ID, title, and text
  - `QAPair`: Question-answer pairs with reference article IDs
  - `GeneratedQA`: Final format with additional metadata

### 2. Phase Classification

The system employs three phases based on document characteristics:

#### Phase 1 (Small Documents)
- For documents with total tokens < MAX_CONTEXT_SIZE (3900 tokens)
- Processes entire document in one pass
- Includes all articles and sections

#### Phase 2 (Medium-sized, Multi-part Documents)
- For documents exceeding MAX_CONTEXT_SIZE with multiple parts
- Strategically selects parts:
  1. Main section (if exists)
  2. Largest part (by article count)
  3. Random additional part
- Falls back to Phase 3 if selected content exceeds token limit

#### Phase 3 (Large Documents)
- For documents exceeding token limits or Phase 2 fallbacks
- Uses controlled chunking approach:
  - Processes articles sequentially
  - Respects token limits per chunk
  - Maintains article order and part structure
  - Includes overlap between chunks for context continuity

### 3. Context Preparation

For each phase:
1. Includes law name and brief summary
2. Maintains hierarchical structure (parts → articles)
3. Preserves article IDs for reference tracking
4. Formats content with clear section markers

### 4. QA Generation

The QA generation process varies by question type:

#### Simple-Factual QA Generation:
1. Uses a specialized LLM prompt template
2. Generates structured QA pairs with:
   - Clear, specific questions
   - Concise answers (1-2 sentences)
   - Article reference IDs
3. Enforces guidelines:
   - No direct article number references in questions
   - Questions based on content meaning
   - Answers strictly from referenced articles
   - No external information or interpretation

#### Relational QA Generation:
1. Implements the 4-step intelligence-first approach
2. Ensures complex concept interactions
3. Creates realistic scenarios requiring legal reasoning
4. Maps to multiple relevant articles
5. Generates answers that demonstrate concept relationships

### 5. Progress Tracking

Features robust progress tracking:
- Maintains checkpoints
- Supports resumption of interrupted processes
- Tracks processed laws and generated QA pairs
- Records metadata including:
  - Total QA pairs
  - Laws processed
  - Phases used
  - Generation timestamps

## Quality Control

1. **LLM-based Validation Pipeline**
   - Implements an automated validation system using a specialized LLM
   - Validates each QA pair against referenced articles
   - Performs three key validations:
     - Verifies all claimed references exist in context
     - Confirms referenced articles fully support the answer
     - Validates answer accuracy and completeness
   - Can correct reference IDs when necessary
   - Maintains validation statistics (valid, corrected, rejected)
   - Provides detailed rejection reasons for failed validations

2. **Reference Validation**
   - Ensures all referenced articles exist in the context
   - Validates article IDs against selected content
   - Tracks and corrects mismatched references
   - Reports validation metrics and statistics

3. **Content Constraints**
   - Enforces token limits
   - Maintains context coherence
   - Preserves legal document structure



## Output Format

The final dataset is stored in JSON format with:
- Comprehensive metadata
- Structured QA pairs
- Article references (validated and corrected)
- Phase information
- Category labels
- Validation status for each QA pair

## Technical Details

- Uses transformers tokenizer for accurate token counting
- Implements atomic saves for data safety
- Supports parallel processing
- MAX_CONTEXT_SIZE: 3900 tokens for simple questions, 12000 for relational
- MAX_QUESTIONS_PER_CALL: 5 questions per LLM call
- Validation pipeline with structured output
