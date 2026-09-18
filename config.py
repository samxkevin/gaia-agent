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

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "18"))
PLANNING_INTERVAL = int(os.getenv("PLANNING_INTERVAL", "5"))
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "8"))
WEB_RATE_LIMIT = float(os.getenv("WEB_RATE_LIMIT", "1.0"))

MAX_TOOL_TEXT = int(os.getenv("MAX_TOOL_TEXT", "100000"))
MAX_WEBPAGE_TEXT = int(os.getenv("MAX_WEBPAGE_TEXT", "30000"))

TRANSCRIPTION_MODEL = os.getenv(
    "TRANSCRIPTION_MODEL",
    "cohere-transcribe-03-2026",
)
