# GAIA Agent

A tool using general purpose AI agent for the GAIA benchmark.

## Goal

Build a reliable agent for the GAIA Level 1 validation subset used by the Agents Course Unit 4 evaluation.

The agent is designed around:

- strong reasoning and planning
- web search and webpage retrieval
- local file handling
- Python execution for calculations and data processing
- answer verification
- exact answer extraction for GAIA submission

## Architecture

```
GAIA question
    ↓
Agent planner
    ↓
Tool use
    ├── web search
    ├── webpage retrieval
    ├── file analysis
    └── Python execution
    ↓
Verification
    ↓
Final answer extraction
    ↓
GAIA submission
```

## Status

Initial repository scaffold. Model and tool implementations will be added incrementally and evaluated against the official course evaluation API.

## Configuration

Copy `.env.example` to `.env` and configure the required API credentials.

Never commit API keys or other secrets.
