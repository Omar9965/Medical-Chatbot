from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from src.helper import download_embeddings
from src.prompt import system_prompt
from utils import get_settings
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load settings
settings = get_settings()

# Initialize embeddings
embeddings_model = download_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = "medical-chatbot"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# Helper function to format documents
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Create RAG chain
question_answer_chain = prompt | chat_model

rag_chain = (
    RunnableParallel({
        "context": lambda x: format_docs(retriever.invoke(x["input"])),
        "input": lambda x: x["input"]
    })
    | question_answer_chain
)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form["msg"]
        
        response = rag_chain.invoke({"input": msg})
        
        return jsonify({"response": response.content})
    
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
