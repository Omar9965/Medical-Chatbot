"""
Chat Store Module
Handles chat history and feedback storage management
"""
from datetime import datetime


class ChatStore:
    """Manages chat history and feedback in memory"""
    
    def __init__(self):
        self.chat_history = []
        self.feedback_store = []
    
    def add_chat(self, question: str, answer: str, sources: list) -> dict:
        """Add a new chat entry and return it"""
        chat_entry = {
            "id": len(self.chat_history),
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources
        }
        self.chat_history.append(chat_entry)
        return chat_entry
    
    def get_chat(self, chat_id: int) -> dict | None:
        """Get a specific chat entry by ID"""
        if 0 <= chat_id < len(self.chat_history):
            return self.chat_history[chat_id]
        return None
    
    def get_all_history(self) -> list:
        """Get all chat history"""
        return self.chat_history
    
    def clear_history(self) -> None:
        """Clear all chat history"""
        self.chat_history = []
    
    def add_feedback(self, chat_id: int, feedback_type: str) -> dict:
        """Add feedback for a chat entry"""
        feedback_entry = {
            "chat_id": chat_id,
            "feedback": feedback_type,
            "timestamp": datetime.now().isoformat()
        }
        
        chat = self.get_chat(chat_id)
        if chat:
            feedback_entry["question"] = chat["question"]
            feedback_entry["answer"] = chat["answer"]
        
        self.feedback_store.append(feedback_entry)
        return feedback_entry
    
    def get_stats(self) -> dict:
        """Get usage statistics"""
        return {
            "total_chats": len(self.chat_history),
            "positive_feedback": sum(1 for f in self.feedback_store if f["feedback"] == "up"),
            "negative_feedback": sum(1 for f in self.feedback_store if f["feedback"] == "down")
        }
    
    def export_history(self) -> str:
        """Export chat history as formatted text"""
        export_text = "Medical Chatbot - Chat Export\n"
        export_text += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        export_text += "=" * 50 + "\n\n"
        
        for entry in self.chat_history:
            export_text += f"Q: {entry['question']}\n"
            export_text += f"A: {entry['answer']}\n"
            if entry['sources']:
                export_text += f"Sources: {', '.join(entry['sources'])}\n"
            export_text += "-" * 30 + "\n\n"
        
        return export_text


# Global store instance
chat_store = ChatStore()
