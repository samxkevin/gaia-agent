import os

from dotenv import load_dotenv

load_dotenv()

GAIA_API_URL = os.getenv(
    "GAIA_API_URL",
    "https://agents-course-unit4-scoring.hf.space",
)
HF_TOKEN = os.getenv("HF_TOKEN", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_USERNAME = os.getenv("HF_USERNAME", "")
AGENT_CODE_URL = os.getenv("AGENT_CODE_URL", "")
