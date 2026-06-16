"""CLI: list Discord text channels (MCP list_channels equivalent)."""
from __future__ import annotations

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
_GUILD_ID = os.getenv("DISCORD_GUILD_ID", "")
_BASE = "https://discord.com/api/v10"
_TEXT = 0  # GUILD_TEXT


def main() -> None:
    if not _TOKEN:
        print("DISCORD_BOT_TOKEN missing in .env", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bot {_TOKEN}"}
    lines: list[str] = []

    with httpx.Client(timeout=30.0, headers=headers) as client:
        me = client.get(f"{_BASE}/users/@me").raise_for_status().json()
        lines.append(f"Bot: {me['username']} (id: {me['id']})")

        guilds = client.get(f"{_BASE}/users/@me/guilds").raise_for_status().json()
        if _GUILD_ID and not any(g["id"] == _GUILD_ID for g in guilds):
            probe = client.get(f"{_BASE}/guilds/{_GUILD_ID}/channels")
            lines.append(
                f"Note: DISCORD_GUILD_ID={_GUILD_ID} not in bot guilds "
                f"(probe status {probe.status_code})."
            )

        target = [g for g in guilds if not _GUILD_ID or g["id"] == _GUILD_ID]
        if not target and guilds:
            target = guilds

        for g in target:
            guild = client.get(f"{_BASE}/guilds/{g['id']}").raise_for_status().json()
            lines.append(f"\nServer: {guild['name']} (id: {guild['id']})")
            channels = client.get(f"{_BASE}/guilds/{g['id']}/channels").raise_for_status().json()
            for ch in sorted(
                (c for c in channels if c.get("type") == _TEXT),
                key=lambda c: c["name"],
            ):
                lines.append(f"  #{ch['name']} (id: {ch['id']})")

    if len(lines) <= 1:
        lines.append("\nNo text channels — invite the bot to a server first.")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
