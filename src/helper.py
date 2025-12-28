from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from langchain_core.documents import Document

from src.prompt import system_prompt
from utils import get_settings


# ============ PDF Loading & Document Processing ============

#Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs




# Chunking the documents into smaller pieces
def split_documents(docs: List[Document], chunk_size: int =500, chunk_overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    return text_splitter.split_documents(docs)


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a single context string"""
    return "\n\n".join([d.page_content for d in docs])


# ============ Embeddings ============

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings


# ============ Vector Store & Retriever ============

def get_vector_store(index_name: str = "medical-chatbot", embeddings=None):
    """
    Connect to an existing Pinecone vector store.
    """
    if embeddings is None:
        embeddings = download_embeddings()
    
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )


def get_retriever(vector_store=None, search_type: str = "similarity", k: int = 3):
    """
    Create a retriever from a vector store.
    """
    if vector_store is None:
        vector_store = get_vector_store()
    
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )


# ============ LLM & Prompt ============

def get_chat_model(model: str = "gemini-2.5-flash", temperature: float = 0.2):
    """
    Initialize and return the ChatGoogleGenerativeAI model.
    """
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=settings.GEMINI_API_KEY
    )


def get_prompt_template():
    """
    Create and return the chat prompt template.
    """
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])


def get_question_answer_chain(chat_model=None, prompt=None):
    """
    Create the question-answer chain by composing prompt and model.
    """
    if chat_model is None:
        chat_model = get_chat_model()
    if prompt is None:
        prompt = get_prompt_template()
    
    return prompt | chat_model


# Global embeddings model instance
embeddings_model = download_embeddings()