import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .retriever import Retriever
from langchain_core.language_models import BaseChatModel


class QueryRewriterSubcategoryOutput(BaseModel):
    """Structured output for query rewriting."""
    query: str = Field(description="The rewritten query")
    subcategory_filters: List[str] = Field(default_factory=list, description="List of subcategory filters to apply")
    relevant_terms: List[str] = Field(default_factory=list, description="List of relevant terms extracted from the query")


class QueryRewriterRetrieverSubcategory(Retriever):
    """
    A class that rewrites queries and extracts relevant terms and subcategory filters
    using LangChain with structured output.
    """

    def __init__(self, llm:BaseChatModel, faiss_index_path: str, documents_path: str,
                 embeddings_model: Any, metric_type: str = 'ip'):
        """Initialize the system.

        Args:
            retriever: An object with `hybrid` and `get_article_metadata` methods.
            llm: A LangChain-compatible LLM instance (e.g., ChatOpenAI, ChatGoogleGenerativeAI)
                 that supports tool calling.
            config: Configuration object (uses defaults if None)
        """
        super().__init__(faiss_index_path, documents_path, embeddings_model, metric_type)
        self.llm=llm.with_structured_output(QueryRewriterSubcategoryOutput)
        self._setup_logging()

        self._initialize_chain()
        self.logger.info("✓ Query Rewriter Retriever initialized")
        

    def _setup_logging(self) -> None:
        """Configure logging for the Query Rewriter Retriever."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel("INFO")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


    def _initialize_chain(self) -> None:
        """Initialize the QueryRewriter chain with LangChain components."""
        subcategories = ", ".join(self.subcategories_article_ids.keys())
        prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert in **analyzing and reformulating legal queries** in Arabic, specializing in **Saudi Arabian regulations**.

## Your Task:
Analyze the user's query and reformulate it to improve search results in Saudi legal documents.

## Available Subcategories:
[{subcategories}]

## Core Guidelines:

### 1. Query Reformulation (query):
- Reformulate the query using formal, precise legal language
- Maintain the **user's original intent** completely without adding or removing information
- Use common Saudi legal terminology where appropriate
- Make the formulation clear, direct, and suitable for semantic search
- Avoid overly general or complex formulations

### 2. Subcategory Filters (subcategory_filters):
**⚠️ Critical Warning:** Selecting any subcategory will restrict the search **exclusively** to articles within that subcategory/subcategories only, and will exclude **all** other categories.

**Given there are only 20 subcategories:**
- **If you select any category, try to include all partially or indirectly related categories**
- Think comprehensively: Could the query relate to more than one category?
- Add related categories even if the connection is indirect or potential

**When to leave the list empty []:**
- The query is too general and cannot be linked to specific categories
- The query relates to general concepts that span most or all categories
- **When in significant doubt:** It's better to leave the list empty to allow comprehensive search

### 3. Relevant Terms (relevant_terms):
- Extract key legal terms or phrases from the original query that are crucial for search

## Required Output:
**IMPORTANT: You must respond in Arabic only.**

Return a JSON-formatted response containing:
1. **query**: The reformulated query text (Arabic text)
2. **subcategory_filters**: List of subcategories from the list above - **Remember: If you add a category, try to add all related categories** (may be empty [])
3. **relevant_terms**: List of key legal terms or phrases extracted from the original query that are crucial for search (may be empty [])
        """),
        ("human", "Original query: {query}\n\nAnalyze the query and reformulate it with the required structure. Remember to respond in Arabic.")
    ])
        self.query_rewriter_chain = prompt | self.llm


    def rewrite_query(self, query: str) -> QueryRewriterSubcategoryOutput:
        """Rewrite the query and extract relevant terms and subcategory filters.

        Args:
            query: The original user query

        Returns:
            QueryRewriterOutput: Contains the rewritten query, relevant terms, and subcategory filters
        """
        try:
            result = self.query_rewriter_chain.invoke({"query": query})
            return result
        except Exception as e:
            self.logger.error(f"Error rewriting query: {str(e)}")
            # Return a default structure in case of error
            return QueryRewriterSubcategoryOutput(
                query=query,
                relevant_terms=[],
                subcategory_filters=[]
            )

    def retrieve(self, query: str, k:int) -> Tuple[np.ndarray, List[int]]:
        """Process the query and return a dictionary with all components.

        Args:
            query: The original user query

        Returns:
           Tuple of (normalized scores, ordinal doc_ids)
        """
        rewriter_output = self.rewrite_query(query)
        self.logger.info(f"Rewritten Query: {rewriter_output.query}")
        self.logger.info(f"Relevant Terms: {rewriter_output.relevant_terms}")
        self.logger.info(f"Subcategory Filters: {rewriter_output.subcategory_filters}")
        results = self.re_ranked_search(rewriter_output.query,
                                        relevant_terms = rewriter_output.relevant_terms,
                                        subcategory_filters =rewriter_output.subcategory_filters,
                                        k=k
                                        )
        
        return results
    
    def __call__(self, *args, **kwds):
        return self.retrieve(*args, **kwds)
    

class Law2StepRetriever(Retriever):
    """
    A class that retrieves legal documents using a two-step process:
    first retrieving relevant laws, then retrieving articles within those laws.
    """
    def __init__(self, laws_faiss_index_path: str, articles_faiss_index_path: str, documents_path: str,
                 embeddings_model: Any, metric_type: str = 'ip'):
        super().__init__(articles_faiss_index_path, documents_path, embeddings_model, metric_type)
        self.laws_faiss_index_path = laws_faiss_index_path
        self.laws_faiss_index = self._load_faiss_index(self.laws_faiss_index_path)

    def retrieve(self, query: str, k:int, top_laws:int=30) -> Tuple[np.ndarray, List[int]]:
        """Process the query and return a dictionary with all components.

        Args:
            query: The original user query

        Returns:
           Tuple of (normalized scores, ordinal doc_ids)
        """
        # Step 1: Retrieve top relevant laws
        law_scores, law_ids = self._hybrid(self.laws_faiss_index, query, k=top_laws)
        law_names = [self.law_id_to_name[law_id] for law_id in law_ids]

        # Step 2: Retrieve articles within the top laws
        results = self.re_ranked_search(query,
                                        relevant_terms=[query],
                                        laws_filters=law_names,
                                        k=k)
        
        return results