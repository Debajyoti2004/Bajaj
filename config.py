import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")

EMBEDDING_MODEL = "models/embedding-001"
ANSWER_LLM_MODEL = "gemini-2.0-flash"
QUERY_LLM_MODEL = "gemini-2.0-flash-lite"
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")