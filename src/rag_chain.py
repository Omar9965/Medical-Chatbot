"""
RAG Chain Module
Handles RAG chain setup, retrieval, and document processing
"""
from src.helper import (
    download_embeddings,
    get_vector_store,
    get_retriever,
    get_chat_model,
    get_prompt_template,
    get_question_answer_chain,
    format_docs
)
from utils import get_settings


class RAGChain:
    """Manages the RAG (Retrieval-Augmented Generation) pipeline"""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_chain()
    
    def _setup_chain(self):
        """Initialize all RAG components using helper functions"""
        # Initialize embeddings
        self.embeddings_model = download_embeddings()
        
        # Connect to existing index and create retriever
        self.docsearch = get_vector_store(
            index_name="medical-chatbot",
            embeddings=self.embeddings_model
        )
        self.retriever = get_retriever(
            vector_store=self.docsearch,
            search_type="similarity",
            k=3
        )
        
        # Initialize chat model and prompt
        self.chat_model = get_chat_model()
        self.prompt = get_prompt_template()
        
        # Create question-answer chain
        self.question_answer_chain = get_question_answer_chain(
            chat_model=self.chat_model,
            prompt=self.prompt
        )
    
    def retrieve(self, query: str):
        """Retrieve relevant documents for a query"""
        return self.retriever.invoke(query)
    
    def generate_response(self, query: str) -> dict:
        """
        Generate a response for a query using RAG
        Returns dict with 'content'
        """
        # Retrieve documents
        retrieved_docs = self.retrieve(query)
        
        # Format context and generate response
        context = format_docs(retrieved_docs)
        response = self.question_answer_chain.invoke({
            "context": context, 
            "input": query
        })
        
        return {
            "content": response.content,
        }


# Global RAG chain instance (lazy initialization)
_rag_chain = None


def get_rag_chain() -> RAGChain:
    """Get or create the RAG chain instance"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
