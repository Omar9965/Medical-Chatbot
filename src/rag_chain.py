"""
RAG Chain Module
Handles RAG chain setup, retrieval, and document processing
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_embeddings
from src.prompt import system_prompt
from utils import get_settings


class RAGChain:
    """Manages the RAG (Retrieval-Augmented Generation) pipeline"""
    
    def __init__(self):
        self.settings = get_settings()
        self._setup_chain()
    
    def _setup_chain(self):
        """Initialize all RAG components"""
        # Initialize embeddings
        self.embeddings_model = download_embeddings()
        
        # Initialize Pinecone
        pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
        index_name = "medical-chatbot"
        
        # Connect to existing index
        self.docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings_model
        )
        self.retriever = self.docsearch.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # Initialize chat model
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.settings.GEMINI_API_KEY
        )
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create question-answer chain
        self.question_answer_chain = self.prompt | self.chat_model
    
    @staticmethod
    def format_docs(docs) -> str:
        """Format retrieved documents into a single context string"""
        return "\n\n".join([d.page_content for d in docs])
    
    @staticmethod
    def get_sources(docs) -> list:
        """Extract unique source filenames from documents"""
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            # Extract just the filename from the path
            if source and source != "Unknown":
                source = os.path.basename(source)
            if source not in sources:
                sources.append(source)
        return sources
    
    def retrieve(self, query: str):
        """Retrieve relevant documents for a query"""
        return self.retriever.invoke(query)
    
    def generate_response(self, query: str) -> dict:
        """
        Generate a response for a query using RAG
        Returns dict with 'content' and 'sources'
        """
        # Retrieve documents
        retrieved_docs = self.retrieve(query)
        sources = self.get_sources(retrieved_docs)
        
        # Format context and generate response
        context = self.format_docs(retrieved_docs)
        response = self.question_answer_chain.invoke({
            "context": context, 
            "input": query
        })
        
        return {
            "content": response.content,
            "sources": sources
        }


# Global RAG chain instance (lazy initialization)
_rag_chain = None


def get_rag_chain() -> RAGChain:
    """Get or create the RAG chain instance"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
