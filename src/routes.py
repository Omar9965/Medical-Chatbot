"""
Routes Module
Flask Blueprint containing all API route handlers
"""
from flask import Blueprint, render_template, jsonify, request

from src.chat_store import chat_store
from src.rag_chain import get_rag_chain


# Create Blueprint
api = Blueprint('api', __name__)


@api.route("/")
def index():
    """Render the chat interface"""
    return render_template('chat.html')


@api.route("/get", methods=["POST"])
def chat():
    """Handle chat messages and generate AI responses"""
    try:
        msg = request.form["msg"]
        
        # Get RAG chain and generate response
        rag_chain = get_rag_chain()
        result = rag_chain.generate_response(msg)
        
        # Store in chat history
        chat_entry = chat_store.add_chat(
            question=msg,
            answer=result["content"],
            sources=result["sources"]
        )
        
        return jsonify({
            "response": result["content"],
            "sources": result["sources"],
            "chat_id": chat_entry["id"]
        })
    
    except Exception as e:
        return jsonify({
            "response": f"Error: {str(e)}", 
            "sources": [], 
            "chat_id": -1
        })


@api.route("/feedback", methods=["POST"])
def feedback():
    """Store user feedback on responses"""
    try:
        data = request.get_json()
        chat_id = data.get("chat_id")
        feedback_type = data.get("feedback")  # "up" or "down"
        
        chat_store.add_feedback(chat_id, feedback_type)
        
        return jsonify({
            "status": "success", 
            "message": f"Feedback '{feedback_type}' recorded"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@api.route("/history", methods=["GET"])
def get_history():
    """Get chat history"""
    return jsonify({"history": chat_store.get_all_history()})


@api.route("/clear", methods=["POST"])
def clear_history():
    """Clear chat history"""
    chat_store.clear_history()
    return jsonify({"status": "success", "message": "Chat history cleared"})


@api.route("/export", methods=["GET"])
def export_chat():
    """Export chat history as text"""
    return jsonify({"export": chat_store.export_history()})


@api.route("/stats", methods=["GET"])
def get_stats():
    """Get usage statistics"""
    return jsonify(chat_store.get_stats())
