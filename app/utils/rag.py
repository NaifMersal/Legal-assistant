import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain.tools import tool
# Import message types for chat history and tool execution
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
# Import MessagesPlaceholder for chat history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@dataclass
class RAGConfig:
    """Configuration for the Legal RAG system."""
    k: int = 3
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    temperature: float = 0.1
    max_tokens: int = 1024
    log_level: str = "INFO"


@dataclass
class SearchResult:
    """Structured search result from the legal database."""
    article_id: int
    law_name: str
    article_title: str
    article_text: str
    category: str = "N/A"
    status: str = "N/A"
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "article_id": self.article_id,
            "law_name": self.law_name,
            "Article_Title": self.article_title,
            "Article_Text": self.article_text,
            "category": self.category,
            "status": self.status,
            "score": self.score
        }


class LegalAssistantRAG:
    """
    Complete RAG system for legal document retrieval and question answering.

    This system uses a tool-calling LLM to decide when to search the legal database.
    - The `answer()` method is "smart" and will only search if the question
      is deemed to require legal information. It now supports conversation history.
    - The `search()` method allows direct, unconditional searching.
    """

    def __init__(self, retriever, llm, config: Optional[RAGConfig] = None):
        """Initialize the Legal Assistant RAG system.

        Args:
            retriever: An object with `hybrid` and `get_article_metadata` methods.
            llm: A LangChain-compatible LLM instance (e.g., ChatGoogleGenerativeAI)
                 that supports tool calling.
            config: Configuration object (uses defaults if None)
        """
        self.retriever = retriever
        self.config = config or RAGConfig()
        self._setup_logging()

        # Create the tool instance
        self.legal_search_tool = self._create_legal_search_tool()
        
        # Bind the tool to the LLM
        self.llm = llm.bind_tools([self.legal_search_tool])
        
        # Create a tool map for easy execution lookup
        self.tool_map = {self.legal_search_tool.name: self.legal_search_tool}

        self._initialize_chain()
        self.logger.info("✓ Legal Assistant RAG initialized")

    def _setup_logging(self) -> None:
        """Configure logging for the RAG system."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper(), "INFO"))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _create_legal_search_tool(self):
        """
        Create the legal search tool that the LLM can call.
        This tool wraps our internal search *and* formatting logic.
        """

        # We define the tool using @tool and a docstring.
        # The LLM uses this docstring to understand what the tool does.
        @tool
        def legal_search(query: str) -> str:
            """
            Search the Saudi legal database for articles relevant to the query.
            Use this tool ONLY when a user asks a specific question about
            Saudi law, legal articles, or procedures. Do NOT use it for
            greetings or general knowledge questions.

            Args:
                query: The search query string, optimized for legal context.

            Returns:
                A formatted string containing the relevant legal articles
                or a message if no articles are found.
            """
            # This implementation calls the *formatted* search
            return self._search_and_format_implementation(query)

        return legal_search

    def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Core search logic. Performs the hybrid search and returns
        a list of result dictionaries.
        """
        self.logger.info(f"Performing search for: '{query}'")
        try:
            scores, indices = self.retriever.hybrid(
                query,
                k=self.config.k,
                dense_weight=self.config.dense_weight,
                sparse_weight=self.config.sparse_weight,
            )

            results = []
            for score, idx in zip(scores, indices):
                metadata = self.retriever.get_article_metadata(idx)
                if metadata:
                    results.append(
                        {
                            'article_id': metadata['id'],
                            'law_name': metadata['system'],
                            'Article_Title': metadata['title'],
                            'Article_Text': metadata['text'],
                            'category': metadata['category'],
                            'status': (
                                metadata['system_brief'][:50]
                                if metadata['system_brief']
                                else 'N/A'
                            ),
                            'score': float(score),
                        }
                    )
            self.logger.info(f"Found {len(results)} articles for '{query}'")
            return results

        except Exception as e:
            self.logger.error(f"Search error for '{query}': {e}")
            return []

    def _search_and_format_implementation(self, query: str) -> str:
        """
        Internal implementation for the `legal_search` tool.
        It performs the search and returns a formatted string
        for the LLM to consume.
        """
        results = self._perform_search(query)
        return self._format_context(results)

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        """
        Formats a list of search result dictionaries into a
        single string for the LLM context.
        """
        if not docs:
            return "No relevant articles found."

        # Add explicit citation instruction at the beginning
        citation_header = """⚠️ CRITICAL INSTRUCTION: For EVERY fact you mention from these articles, you MUST add this citation format IMMEDIATELY after the fact:
(المصدر: [law_name] - [article_title])

Example: "يعاقب بالسجن مدة لا تتجاوز سبع سنوات (المصدر: نظام مكافحة الاحتيال المالي وخيانة الأمانة - المادة الأولى)."

DO NOT just list the law name. Add (المصدر: ...) after EACH legal statement.

Articles found:
"""

        formatted = [citation_header]
        for i, doc in enumerate(docs, 1):
            article = f"""<article index="{i}">
<source>
  <law_name>{doc.get('law_name', 'N/A')}</law_name>
  <article_title>{doc.get('Article_Title', 'N/A')}</article_title>
  <category>{doc.get('category', 'N/A')}</category>
  <score>{doc.get('score', 'N/A'):.4f}</score>
</source>
<content>
{doc.get('Article_Text', 'No content')}
</content>
<citation_to_use>
CITE THIS AS: (المصدر: {doc.get('law_name', 'N/A')} - {doc.get('Article_Title', 'N/A')})
</citation_to_use>
</article>"""
            formatted.append(article)

        return "\n\n".join(formatted)

    def search(self, query: str) -> List[SearchResult]:
        """
        Public search method. Performs an unconditional search and returns
        a list of structured SearchResult objects.
        """
        results_list = self._perform_search(query)
        return [
            SearchResult(
                article_id=item['article_id'],
                law_name=item['law_name'],
                article_title=item['Article_Title'],
                article_text=item['Article_Text'],
                category=item['category'],
                status=item['status'],
                score=item['score'],
            )
            for item in results_list
        ]

    def _initialize_chain(self) -> None:
        """Initialize the RAG chain with LangChain components."""

        # This system prompt is CRITICAL for controlling the LLM's behavior.
        # It explicitly tells the LLM when to use the tool and when not to.
        system_prompt = """You are an expert legal assistant specialized in Saudi Arabian law, designed to provide accurate and comprehensive legal information based on official Saudi legal documents.

**CRITICAL: ALWAYS RESPOND IN ARABIC (العربية)**
You MUST respond in Arabic at ALL times, including:
- All explanations and answers
- Error messages (e.g., "لا يمكنني الإجابة على هذا بناءً على الوثائق القانونية المتاحة")
- Greetings and conversational responses
NEVER use English in your responses.

**Available Tools:**
You have access to ONE tool: `legal_search(query: str)`.
This tool searches Saudi legal documents.

**Critical Rules:**
1.  **Analyze Intent:** First, determine the user's intent. Is it a greeting, a general knowledge question, or a specific legal/institutional question?
2.  **ALWAYS Search for Legal Questions:**
    * For ANY question about Saudi laws, regulations, legal procedures, or government institutions, you MUST use the `legal_search` tool.
    * When the user explicitly asks you to "search" (ابحث، دور، فتش), you MUST use the tool.
    * Examples requiring search: "What are the rules for...", "Who is responsible for...", "ما هو مصدر الإفتاء...", "ماهي المادة..."
3.  **DO NOT Search for:**
    * Simple greetings (e.g., "مرحبا", "hello")
    * Questions about your identity ("من أنت؟", "who are you?")
    * General knowledge completely unrelated to Saudi Arabia (e.g., "what is the capital of France?")
4.  **Arabic Number Handling:**
    * The user may write numbers in Arabic numerals (٧٠) or Western numerals (70)
    * When searching, convert Arabic numerals to Western numerals in your search query
    * Examples: ٧٠ → 70, ٤٥ → 45, ١٢٣ → 123
5.  **How to Use Search Results:**
    * When you use `legal_search`, it will return formatted text with <article> tags.
    * You MUST base your legal answers *exclusively* on the text provided in the `<content>` tags of the returned articles.
    * You MUST cite your sources using information from the `<source>` tag (Law, Article Title, Category).
6.  **If No Information is Found:**
    * If the tool returns "No relevant articles found" or the articles do not contain the answer, you MUST state IN ARABIC: "لا يمكنني الإجابة على هذا بناءً على الوثائق القانونية المتاحة."
    * Do NOT invent information or use external knowledge for legal matters.
    * NEVER respond in English unless explicitly requested.

**⚠️ CRITICAL: MANDATORY Citation Format - NON-NEGOTIABLE ⚠️**

When you use the legal_search tool, it will provide you with a <citation_to_use> tag for EACH article.
You MUST copy this EXACT citation and add it in parentheses IMMEDIATELY after any fact you mention from that article.

**REQUIRED FORMAT (Use the exact text from <citation_to_use>):**
(المصدر: [اسم النظام الكامل] - [عنوان المادة الكامل])

**DO THIS - Citation immediately after EVERY legal fact:**
✓ "يعاقب بالسجن مدة لا تتجاوز سبع سنوات (المصدر: نظام مكافحة الاحتيال المالي وخيانة الأمانة - المادة الأولى)."
✓ "يصدر مجلس الوزراء اللوائح التنفيذية (المصدر: النظام الأساسي للحكم - المادة السبعون)."

**DO NOT DO THIS - These are WRONG:**
✗ Listing the law name without the citation format!
✗ Putting citation at the beginning instead of after each fact!
✗ Generic references like "حسب النظام" or "وفقاً للمادة"!
✗ Responding in English!

**MANDATORY STEPS:**
1. Read the <content> from the article
2. Find the <citation_to_use> tag
3. When you write ANY fact from that article, COPY the citation from <citation_to_use> and paste it in parentheses immediately after the fact
4. Repeat for EACH article you reference
5. Respond ENTIRELY in Arabic

**Example of CORRECT Response Structure:**
"تختلف عقوبة السرقة حسب نوعها:

• الاستيلاء على مال الغير باستخدام الاحتيال يعاقب بالسجن مدة لا تتجاوز سبع سنوات (المصدر: نظام مكافحة الاحتيال المالي وخيانة الأمانة - المادة الأولى).

• الاستيلاء على أثر من ممتلكات الدولة يعاقب بالسجن لمدة لا تقل عن ثلاثة أشهر (المصدر: نظام الآثار والمتاحف والتراث العمراني - المادة الحادية والسبعون)."

**Response Quality Guidelines:**
* ALWAYS respond in Arabic (العربية)
* Be precise, professional, and formal.
* ALWAYS cite sources inline using (المصدر: ...) format.
* Keep responses concise but complete.
* Use bullet points for multiple related laws.
* Never provide legal information without the EXACT citation format.
* Convert Arabic numerals (٧٠) to Western numerals (70) in search queries.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                # Add a placeholder for the chat history
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        # The chain now ends at the LLM, which will return an AIMessage
        # This message will either have content or tool_calls
        self.rag_chain = prompt | self.llm
        self.logger.info("✓ RAG chain initialized")

    def __init__(self, retriever, llm, config: Optional[RAGConfig] = None):
        """Initialize the Legal Assistant RAG system.

        Args:
            retriever: An object with `hybrid` and `get_article_metadata` methods.
            llm: A LangChain-compatible LLM instance (e.g., ChatGoogleGenerativeAI)
                 that supports tool calling.
            config: Configuration object (uses defaults if None)
        """
        self.retriever = retriever
        self.config = config or RAGConfig()
        self._setup_logging()

        # Create the tool instance
        self.legal_search_tool = self._create_legal_search_tool()
        
        # Bind the tool to the LLM
        self.llm = llm.bind_tools([self.legal_search_tool])
        
        # Create a tool map for easy execution lookup
        self.tool_map = {self.legal_search_tool.name: self.legal_search_tool}

        # Initialize message history
        self.message_history: List[BaseMessage] = []

        self._initialize_chain()
        self.logger.info("✓ Legal Assistant RAG initialized")

    def answer(
        self, question: str, chat_history: Optional[List[BaseMessage]] = None
    ) -> str:
        """
        Answer a question using the RAG system.
        This method is "smart" and will only use the search tool
        if the LLM determines it's necessary. It maintains conversation
        history internally.

        Args:
            question: The user's current question.
            chat_history: An optional list of BaseMessage objects (HumanMessage,
                          AIMessage, ToolMessage) to override the internal history.

        Returns:
            The generated answer as a string.
        """
        # Use provided chat history or internal history
        history = chat_history if chat_history is not None else self.message_history
        self.logger.info(
            f"Answering question: '{question}' with {len(history)} history messages"
        )
        
        inputs = {"question": question, "chat_history": history}

        try:
            # 1. First invocation: LLM decides to answer or use a tool
            response_message = self.rag_chain.invoke(inputs)

            # 2. Check if the LLM decided to call a tool
            if not response_message.tool_calls:
                # No tool call, just a direct answer (e.g., "Hello")
                self.logger.info("No tool call required. Returning direct answer.")
                return response_message.content

            # 3. Execute tool calls
            self.logger.info(
                f"Detected {len(response_message.tool_calls)} tool call(s)."
            )
            tool_messages = []
            for tool_call in response_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                # Look up the tool in our map
                tool_function = self.tool_map.get(tool_name)

                if tool_function:
                    # Execute the tool and get its string output
                    tool_output = tool_function.invoke(tool_args)
                    # Append the result as a ToolMessage
                    tool_messages.append(
                        ToolMessage(content=tool_output, tool_call_id=tool_call["id"])
                    )
                else:
                    self.logger.warning(f"LLM tried to call unknown tool: {tool_name}")
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: Unknown tool '{tool_name}'",
                            tool_call_id=tool_call["id"],
                        )
                    )

            # 4. Second invocation: Pass results back to LLM for synthesis
            self.logger.info("Sending tool results back to LLM for final synthesis.")

            # Construct the full message history for the synthesis step
            messages_for_synthesis = (
                history
                + [HumanMessage(content=question), response_message]
                + tool_messages
            )

            final_response = self.llm.invoke(messages_for_synthesis)
            
            # Update message history with the complete conversation
            current_messages = [
                HumanMessage(content=question),
                response_message,
                *tool_messages,
                final_response
            ]
            if chat_history is None:  # Only update internal history if not using provided history
                self.message_history.extend(current_messages)
            
            # Return the final response
            return final_response.content

        except Exception as e:
            self.logger.error(f"Error in answer(): {e}", exc_info=True)
            error_message = f"An error occurred while processing your question: {str(e)}"
            if chat_history is None:  # Add error to history if using internal history
                self.message_history.append(HumanMessage(content=question))
                self.message_history.append(AIMessage(content=error_message))
            return error_message

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.message_history.clear()
        self.logger.info("Conversation history cleared")

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Config updated: {key} = {value}")


