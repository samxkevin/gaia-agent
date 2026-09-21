# GAIA Agent

A general purpose agent for the GAIA benchmark, built around Cohere Command A+ and smolagents.

## Architecture

GAIA question
    |
Question parser
    |
ToolCallingAgent with Command A+
    |-- DuckDuckGo web search
    |-- Wikipedia retrieval
    |-- webpage retrieval
    |-- Python execution
    |-- local text, PDF, DOCX, XLSX reading
    |-- image analysis with Command A+
    |
Verification and exact answer cleanup
    |
GAIA answer payload

## Development rule

evaluation/debug.py runs exactly one task and never calls /submit.

The full submission runner refuses to submit partial results.

## First question

The live GAIA endpoint currently exposes the first Level 1 task as a Mercedes Sosa studio album question. The debugger defaults to index 0.

## Setup

Create the environment and install dependencies:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

Copy .env.example to .env and set the Cohere credentials.

For numerical YouTube maximum tasks, the agent can use Adversal as a free tier visual ingestion layer. It checks the live remaining quota before submission and falls back to the native FFmpeg and Cohere pipeline when Adversal is unavailable, unauthenticated, or under quota. No paid Adversal path is used.

The Adversal MCP client requires a working Python 3.13 environment through `adversal-cli`. The agent can launch it with `uvx` when available. Authenticate once through an MCP client before the first live run. The agent never stores Adversal credentials in this repository.

## Run one question

    python -m evaluation.debug

Run another task:

    python -m evaluation.debug --index 3
    python -m evaluation.debug --task-id YOUR_TASK_ID

The debugger does not submit anything.

## Full submission

Only after local validation:

    python -m evaluation.run

The submission runner requires HF_USERNAME and AGENT_CODE_URL and will submit only after every question produces an answer.

Never commit API keys or runtime artifacts.
