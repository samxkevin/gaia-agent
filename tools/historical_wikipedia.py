from __future__ import annotations

from datetime import date as date_type
import requests
from smolagents import Tool

from config import MAX_TOOL_TEXT


class WikipediaPageAsOfTool(Tool):
    name = "wikipedia_page_as_of"
    description = (
        "Fetch the latest English Wikipedia revision of a page on or before a "
        "specified date. Use this when a GAIA task refers to a historical "
        "Wikipedia version or a year such as 'latest 2022 version'. "
        "Returns the page wikitext, revision timestamp, and revision ID."
    )
    inputs = {
        "title": {
            "type": "string",
            "description": "Exact or near exact Wikipedia page title.",
        },
        "date": {
            "type": "string",
            "description": "Cutoff date in YYYY-MM-DD format, usually December 31 of the requested year.",
        },
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self._cache: dict[tuple[str, str], str] = {}

    def forward(self, title: str, date: str) -> str:
        try:
            cutoff = date_type.fromisoformat(date)
        except ValueError:
            return f"Invalid date: {date}. Use YYYY-MM-DD."

        cache_key = (title.strip().lower(), cutoff.isoformat())
        cached = self._cache.get(cache_key)
        if cached is not None:
            return (
                f"Historical Wikipedia page already fetched for '{title}' on "
                f"{cutoff.isoformat()}. Reuse the earlier observation instead of "
                "fetching the same revision again.\n\n"
                f"{cached}"
            )

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "titles": title,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "rvlimit": "1",
            "rvstart": f"{cutoff.isoformat()}T23:59:59Z",
            "rvdir": "older",
        }

        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers={"User-Agent": "gaia-agent/1.0"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return f"Could not fetch historical Wikipedia revision: {exc}"

        pages = payload.get("query", {}).get("pages", [])
        if not pages:
            return f"No Wikipedia page found for '{title}'."

        page = pages[0]
        if "missing" in page:
            return f"No Wikipedia page found for '{title}'."

        revisions = page.get("revisions", [])
        if not revisions:
            return (
                f"No revision for '{title}' was found on or before {cutoff.isoformat()}."
            )

        revision = revisions[0]
        slots = revision.get("slots", {})
        main = slots.get("main", {})
        content = main.get("content", "")

        if not content:
            return f"Historical revision for '{title}' contains no page content."

        result = (
            f"Wikipedia page: {page.get('title', title)}\n"
            f"Revision ID: {revision.get('revid', '')}\n"
            f"Revision timestamp: {revision.get('timestamp', '')}\n"
            f"Cutoff date: {cutoff.isoformat()}\n\n"
            f"{content[:MAX_TOOL_TEXT]}"
        )
        self._cache[cache_key] = result
        return result
