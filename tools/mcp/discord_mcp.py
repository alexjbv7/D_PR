"""
Discord MCP Server — Cursor integration
Allows Cursor to interact with Discord via MCP protocol.

Tools available:
- send_message: Send a message to a Discord channel
- read_messages: Read recent messages from a channel
- list_channels: List all channels in the server
- run_daily_briefing: Trigger daily briefing manually
- run_weekly_briefing: Trigger weekly briefing manually
- bot_status: Check if bot is connected and running

Usage (in .cursor/mcp.json):
{
  "mcpServers": {
    "discord": {
      "command": "python",
      "args": ["-m", "tools.mcp.discord_mcp"],
      "env": {
        "DISCORD_BOT_TOKEN": "<your_token>",
        "DISCORD_GUILD_ID": "<your_server_id>"
      }
    }
  }
}
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import discord
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

load_dotenv()

_ET = ZoneInfo("America/New_York")

BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
GUILD_ID = int(os.getenv("DISCORD_GUILD_ID", "0"))

# Discord client (read/write)
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
discord_client = discord.Client(intents=intents)

# MCP server
mcp_server = Server("discord-quant-bot")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="send_message",
            description="Send a message to a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_name": {"type": "string", "description": "Channel name (e.g. 'alpaca-bot')"},
                    "message": {"type": "string", "description": "Message to send"},
                },
                "required": ["channel_name", "message"],
            },
        ),
        Tool(
            name="read_messages",
            description="Read recent messages from a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_name": {"type": "string", "description": "Channel name"},
                    "limit": {"type": "integer", "description": "Number of messages to read (default 10)", "default": 10},
                },
                "required": ["channel_name"],
            },
        ),
        Tool(
            name="list_channels",
            description="List all text channels in the Discord server",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="run_daily_briefing",
            description="Trigger the daily briefing manually for a specific date",
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format (default: today)"},
                },
            },
        ),
        Tool(
            name="run_weekly_briefing",
            description="Trigger the weekly briefing manually for a specific week",
            inputSchema={
                "type": "object",
                "properties": {
                    "week": {"type": "string", "description": "ISO week e.g. 2026-W21 (default: current week)"},
                },
            },
        ),
        Tool(
            name="bot_status",
            description="Check Discord bot connection status and list available channels",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


def _get_channel(name: str) -> discord.TextChannel | None:
    for guild in discord_client.guilds:
        for channel in guild.text_channels:
            if channel.name == name:
                return channel
    return None


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "send_message":
        channel = _get_channel(arguments["channel_name"])
        if not channel:
            return [TextContent(type="text", text=f"Channel '{arguments['channel_name']}' not found.")]
        await channel.send(arguments["message"])
        return [TextContent(type="text", text=f"Message sent to #{arguments['channel_name']}")]

    elif name == "read_messages":
        channel = _get_channel(arguments["channel_name"])
        if not channel:
            return [TextContent(type="text", text=f"Channel '{arguments['channel_name']}' not found.")]
        limit = arguments.get("limit", 10)
        messages = []
        async for msg in channel.history(limit=limit):
            messages.append(f"[{msg.created_at.strftime('%Y-%m-%d %H:%M')}] {msg.author.name}: {msg.content}")
        messages.reverse()
        return [TextContent(type="text", text="\n".join(messages) if messages else "No messages found.")]

    elif name == "list_channels":
        channels = []
        for guild in discord_client.guilds:
            for channel in guild.text_channels:
                channels.append(f"#{channel.name} (id: {channel.id})")
        return [TextContent(type="text", text="\n".join(channels) if channels else "No channels found.")]

    elif name == "run_daily_briefing":
        date_str = arguments.get("date", datetime.now(tz=_ET).strftime("%Y-%m-%d"))
        result = subprocess.run(
            [sys.executable, "-m", "tools.briefing.daily", "--date", date_str],
            capture_output=True, text=True,
        )
        output = result.stdout + (f"\nERROR: {result.stderr}" if result.returncode != 0 else "")
        return [TextContent(type="text", text=output.strip())]

    elif name == "run_weekly_briefing":
        now = datetime.now(tz=_ET)
        default_week = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
        week = arguments.get("week", default_week)
        result = subprocess.run(
            [sys.executable, "-m", "tools.briefing.weekly", "--week", week],
            capture_output=True, text=True,
        )
        output = result.stdout + (f"\nERROR: {result.stderr}" if result.returncode != 0 else "")
        return [TextContent(type="text", text=output.strip())]

    elif name == "bot_status":
        if not discord_client.is_ready():
            return [TextContent(type="text", text="Bot is not connected.")]
        guilds = [g.name for g in discord_client.guilds]
        channels = [f"#{c.name}" for g in discord_client.guilds for c in g.text_channels]
        status = f"Bot connected as: {discord_client.user}\nServers: {', '.join(guilds)}\nChannels: {', '.join(channels)}"
        return [TextContent(type="text", text=status)]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main() -> None:
    # Start Discord client in background
    asyncio.create_task(discord_client.start(BOT_TOKEN))

    # Wait for Discord to connect
    await discord_client.wait_until_ready()
    print(f"Discord bot connected as {discord_client.user}", file=sys.stderr)

    # Start MCP server
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())