"""
Medical Chatbot Application
Flask entry point - imports and registers routes from modules
"""
from flask import Flask
import os
import warnings

from src.routes import api

warnings.filterwarnings("ignore")

# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Register routes blueprint
app.register_blueprint(api)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

