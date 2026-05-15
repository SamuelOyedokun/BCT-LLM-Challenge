import os
from dotenv import load_dotenv

load_dotenv("../../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

SHARED_DATA_PATH = "../../shared"
