import os

from dotenv import load_dotenv

load_dotenv()

GAIA_API_URL = os.getenv(
    "GAIA_API_URL",
    "https://agents-course-unit4-scoring.hf.space",
)

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "")
AGENT_CODE_URL = os.getenv("AGENT_CODE_URL", "")

COHERE_MODEL = os.getenv(
    "COHERE_MODEL",
    "command-a-plus-05-2026",
)

COHERE_BASE_URL = os.getenv(
    "COHERE_BASE_URL",
    "https://api.cohere.ai/compatibility/v1",
)

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "20"))
