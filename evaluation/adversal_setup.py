from __future__ import annotations

import asyncio
import shutil

from config import ADVERSAL_CLI_PACKAGE
from tools.adversal import (
    _result_text,
    parse_remaining_minutes,
    resolve_adversal_command,
)


async def main() -> None:
    command = resolve_adversal_command()
    if command is None:
        raise SystemExit(
            "Adversal client unavailable. Install uv, or set ADVERSAL_CLI to adversal-cli."
        )

    executable, args = command
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        raise SystemExit(
            f"MCP client unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    print(f"Adversal client: {executable}")
    print(f"Package: {ADVERSAL_CLI_PACKAGE}")
    print("Starting browser authentication...")
    print("Complete the Adversal sign in in your browser, then return here.")

    transport = StdioServerParameters(command=executable, args=args)
    async with stdio_client(transport) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            auth = await session.call_tool("authenticate", {})
            auth_text = _result_text(auth)
            print(auth_text)

            quota = await session.call_tool("check_remaining_quota", {})
            quota_text = _result_text(quota)
            remaining = parse_remaining_minutes(quota_text)
            print(quota_text)
            if remaining is not None:
                print(f"Parsed remaining minutes: {remaining:g}")


if __name__ == "__main__":
    asyncio.run(main())
