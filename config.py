import os

from dotenv import load_dotenv

load_dotenv()

GAIA_API_URL = os.getenv(
    "GAIA_API_URL",
    "https://agents-course-unit4-scoring.hf.space",
)

COHERE_PRIMARY_API_KEY = os.getenv("COHERE_PRIMARY_API_KEY", "") or os.getenv("COHERE_API_KEY", "")
COHERE_FALLBACK_API_KEY = os.getenv("COHERE_FALLBACK_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "") or COHERE_PRIMARY_API_KEY

COHERE_PRIMARY_MODEL = os.getenv(
    "COHERE_PRIMARY_MODEL",
    "command-a-plus-05-2026",
)
COHERE_FALLBACK_MODEL = os.getenv(
    "COHERE_FALLBACK_MODEL",
    "command-a-reasoning-08-2025",
)
COHERE_VISION_MODEL = os.getenv(
    "COHERE_VISION_MODEL",
    "command-a-vision-07-2025",
)

COHERE_BASE_URL = os.getenv(
    "COHERE_BASE_URL",
    "https://api.cohere.ai/compatibility/v1",
)

COHERE_PRIMARY_TRANSCRIPTION_MODEL = os.getenv(
    "COHERE_PRIMARY_TRANSCRIPTION_MODEL",
    "cohere-transcribe-03-2026",
)
COHERE_FALLBACK_TRANSCRIPTION_MODEL = os.getenv(
    "COHERE_FALLBACK_TRANSCRIPTION_MODEL",
    "cohere-transcribe-03-2026",
)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_USERNAME = os.getenv("HF_USERNAME", "")
AGENT_CODE_URL = os.getenv("AGENT_CODE_URL", "")

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
PLANNING_INTERVAL = int(os.getenv("PLANNING_INTERVAL", "0"))
WEB_MAX_RESULTS = int(os.getenv("WEB_MAX_RESULTS", "8"))
WEB_RATE_LIMIT = float(os.getenv("WEB_RATE_LIMIT", "1.0"))

MAX_TOOL_TEXT = int(os.getenv("MAX_TOOL_TEXT", "100000"))
MAX_WEBPAGE_TEXT = int(os.getenv("MAX_WEBPAGE_TEXT", "15000"))

FAILOVER_ATTEMPTS = int(os.getenv("FAILOVER_ATTEMPTS", "4"))
FAILOVER_COOLDOWN_SECONDS = float(os.getenv("FAILOVER_COOLDOWN_SECONDS", "45"))
MODEL_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "0"))
MODEL_TIMEOUT_SECONDS = int(os.getenv("MODEL_TIMEOUT_SECONDS", "120"))
