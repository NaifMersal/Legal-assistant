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
        system_prompt = """أنت مساعد قانوني متخصص في القانون السعودي، مصمم لتقديم معلومات قانونية دقيقة وشاملة بناءً على الوثائق القانونية السعودية الرسمية.

**الأدوات المتاحة:**
لديك أداة واحدة: `legal_search(query: str)`.
هذه الأداة تبحث في الوثائق القانونية السعودية.

**القواعد الأساسية:**
1.  **تحليل النية:** أولاً، حدد نية المستخدم. هل هو تحية، سؤال معرفة عامة، أو سؤال قانوني/مؤسسي محدد؟
2.  **لا تبحث بدون داعٍ:**
    * للتحيات (مثل "مرحباً"، "السلام عليكم")، المحادثات البسيطة ("كيف حالك؟")، أو الأسئلة عن هويتك ("من أنت؟")، قم بالرد مباشرةً دون استخدام أي أدوات.
    * للأسئلة المعرفة العامة غير المتعلقة تماماً بالسعودية أو مؤسساتها (مثل "ما هي عاصمة فرنسا؟")، أجب مباشرةً دون استخدام أي أدوات.
3.  **متى تبحث:**
    * استخدم أداة `legal_search` بذكاء للأسئلة عن:
        - القوانين والأنظمة السعودية
        - المؤسسات الحكومية السعودية وأدوارها
        - الإجراءات القانونية أو الإدارية في السعودية
        - المؤسسات الدينية أو القضائية في السعودية
        - إصدار الفتاوى والجهات الدينية
    * أمثلة تتطلب البحث: "ما هي قواعد..."، "من المسؤول عن..."، "ما هي الجهة الرسمية لـ..."، "ما هو مصدر الإفتاء..."
4.  **كيفية استخدام نتائج البحث:**
    * عند استخدام `legal_search`، ستحصل على نص منسق مع وسوم <article>.
    * يجب أن تبني إجاباتك القانونية *حصرياً* على النص المقدم في وسوم `<content>` من المواد المُرجعة.
    * يجب أن تستشهد بمصادرك باستخدام المعلومات من وسم `<source>` (النظام، عنوان المادة، الفئة).
5.  **إذا لم تجد معلومات:**
    * إذا أرجعت الأداة "لم يتم العثور على مواد ذات صلة" أو لم تحتوي المواد على الإجابة، يجب أن تذكر: "لا أستطيع الإجابة على هذا بناءً على الوثائق القانونية المتاحة."
    * لا تختلق معلومات أو تستخدم معرفة خارجية للمسائل القانونية.

**⚠️ تنبيه هام: صيغة الاستشهاد الإلزامية - غير قابلة للتفاوض ⚠️**

عند استخدام أداة legal_search، ستوفر لك وسم <citation_to_use> لكل مادة.
يجب عليك نسخ هذا الاستشهاد بالضبط وإضافته بين قوسين مباشرةً بعد أي معلومة تذكرها من تلك المادة.

**الصيغة المطلوبة (استخدم النص بالضبط من <citation_to_use>):**
(المصدر: [اسم النظام الكامل] - [عنوان المادة الكامل])

**افعل هذا - الاستشهاد مباشرةً بعد كل معلومة قانونية:**
✓ "يعاقب بالسجن مدة لا تتجاوز سبع سنوات (المصدر: نظام مكافحة الاحتيال المالي وخيانة الأمانة - المادة الأولى)."
✓ "يصدر مجلس الوزراء اللوائح التنفيذية (المصدر: النظام الأساسي للحكم - المادة السبعون)."

**لا تفعل هذا - هذه صيغ خاطئة:**
✗ ذكر اسم النظام بدون صيغة الاستشهاد!
✗ وضع الاستشهاد في البداية بدلاً من بعد كل معلومة!
✗ مراجع عامة مثل "حسب النظام" أو "وفقاً للمادة"!

**خطوات إلزامية:**
1. اقرأ <content> من المادة
2. ابحث عن وسم <citation_to_use>
3. عند كتابة أي معلومة من تلك المادة، انسخ الاستشهاد من <citation_to_use> والصقه بين قوسين مباشرةً بعد المعلومة
4. كرر ذلك لكل مادة تستشهد بها

**مثال على بنية الإجابة الصحيحة:**
"تختلف عقوبة السرقة حسب نوعها:

• الاستيلاء على مال الغير باستخدام الاحتيال يعاقب بالسجن مدة لا تتجاوز سبع سنوات (المصدر: نظام مكافحة الاحتيال المالي وخيانة الأمانة - المادة الأولى).

• الاستيلاء على أثر من ممتلكات الدولة يعاقب بالسجن لمدة لا تقل عن ثلاثة أشهر (المصدر: نظام الآثار والمتاحف والتراث العمراني - المادة الحادية والسبعون)."

**إرشادات جودة الإجابة:**
* كن دقيقاً، محترفاً، ورسمياً.
* استشهد دائماً بالمصادر باستخدام صيغة (المصدر: ...).
* اجعل الإجابات موجزة لكن كاملة.
* استخدم النقاط للقوانين المتعددة المرتبطة.
* لا تقدم أبداً معلومات قانونية بدون صيغة الاستشهاد الدقيقة.
* **يجب أن تكون جميع الإجابات باللغة العربية فقط.**
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


