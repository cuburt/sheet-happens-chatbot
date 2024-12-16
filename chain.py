import uuid
import time
import logging
from typing import List, Dict, Tuple, Any, Sequence
from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from llm import GeminiLLM, CodeyLLM, Gemini15LLM, GeminiFlashLLM, Llama3LLM
from prompt_template import IQPromptTemplate, DocumentParserPromptTemplate
from retriever import RetrieverFactory
from session import Session
from routers import SemanticRouter
from vector_index import FAISSstore


class ConversationBufferWindowMemory(InMemoryChatMessageHistory):
    """
    A chat message history that only keeps the last K messages.
    """
    buffer_size: int
    
    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if self.buffer_size > 0:
            self.messages.extend(messages)
            self.messages = self.messages[-(self.buffer_size * 2):]
        else:
            self.messages = []


class Chain:
    def __init__(self):
        """
        Initialize LLM, Code Interpreter's prompt template, vector store, and retriever.
        """
        # try:
        self.gemini = GeminiFlashLLM()
        self.llama = Llama3LLM()
        self.session_manager = Session()
        self.memory_window = 3
        self.chat_history = {}
        self.default_session_id = str(uuid.uuid4())
        self.vectorstore = FAISSstore(db_dir="vector_dbs", db_file="index")

        retriever = RetrieverFactory(vectorstore=self.vectorstore.vectorstore,
                                      k=3,
                                      skip_longcontext_reorder=False)
        # retriever.router = SemanticRouter()
        self.gemini_custom_chain = self.build_qa_chain(self.gemini, IQPromptTemplate().prompt, retriever)
        self.llama_custom_chain = self.build_qa_chain(self.llama, IQPromptTemplate().prompt, retriever)
        self.attached_text = ""
        # except Exception as e:
        #     raise Exception(str(e))

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves history from BigQuery
        """
        if session_id not in self.chat_history:
            history = []
            try:
                start = time.time()
                history = self.session_manager.get_conversations(session_id, self.memory_window)
                logging.info(f"BQSelect RT: {(time.time() - start):.4f}")
            except Exception as e:
                logging.warning(str(e))
            self.chat_history[session_id] = ConversationBufferWindowMemory(buffer_size=self.memory_window)
            for entry in eval(history.to_json(orient='records')):
                self.chat_history[session_id].add_user_message(entry["message"])
                self.chat_history[session_id].add_ai_message(entry["response"])
        return self.chat_history[session_id]

    def build_qa_chain(self, llm, prompt_template, retriever) -> ConversationalRetrievalChain:
        """
        QA chain's constructor
        """
        doc_chain = create_stuff_documents_chain(llm,
                                                 prompt=prompt_template,
                                                 document_prompt=DocumentParserPromptTemplate().prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        return RunnableWithMessageHistory(rag_chain,
                                          get_session_history=self.get_session_history,
                                          input_messages_key="input",
                                          history_messages_key="chat_history",
                                          output_messages_key="answer")

    @staticmethod
    def run_qa_chain(qa_chain, query: str, session_id: str):
        """
        Retriever's function. can be used in function calling or as agent tool
        """
        try:
            return qa_chain.invoke({"input": query}, config={"configurable": {"session_id": session_id}})
        except Exception as e:
            raise Exception(str(e))

    def __call__(self, query: str, llm: str = 'gemini', task: str = 'docs', enable_rag=True, session_id: str = None) -> Tuple[
        str, str, List[Tuple[Any, Any]]]:
        """
        Invoked for RAG response. For non-RAG response, invoke LLM directly.
        """
        if not enable_rag:
            response, source_docs = {"answer": eval(f"self.experimental_chain.{llm}")}, []
        else:
            response = self.run_qa_chain(
                qa_chain=eval(f"self.{llm}_custom_chain"),
                query=f"{query} \n File: \n {self.attached_text}",
                session_id=session_id if session_id else self.default_session_id)
            print(response)
            source_docs = [(x.page_content, x.metadata['source']) for x in response['context']]

        assert response["answer"], "'answer' key not found in model response."
        message_id = str(uuid.uuid4())
        if session_id:
            start = time.time()
            self.session_manager.add_to_conversations(message_id=message_id, session_id=session_id, message=str(query), response=str(response["answer"]))
            logging.info(f"BQUpdate RT: {(time.time() - start):.4f}")

        return message_id, response["answer"].encode("utf-8"), source_docs
