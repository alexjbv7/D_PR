"""Convert markdown briefing output to Slack mrkdwn."""
from __future__ import annotations

import re

_SLACK_MAX_CHARS = 38_000


def md_to_slack_mrkdwn(md: str) -> str:
    """Translate a limited markdown subset to Slack mrkdwn.

    Slack differences:
    - ``**bold**`` → ``*bold*``
    - ``[text](url)`` → ``<url|text>``
    - Tables → monospace code block (Slack has no table support)
    """
    out = md
    out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out)
    out = re.sub(r"\[(.+?)\]\((.+?)\)", r"<\2|\1>", out)
    out = _wrap_tables_as_codeblock(out)
    if len(out) > _SLACK_MAX_CHARS:
        out = out[:_SLACK_MAX_CHARS] + "\n\n*[Truncated. See full briefing in repo.]*"
    return out


def _wrap_tables_as_codeblock(md: str) -> str:
    """Wrap pipe-table lines in triple-backtick fences."""
    lines = md.split("\n")
    out: list[str] = []
    in_table = False
    for line in lines:
        is_table_line = line.lstrip().startswith("|")
        if is_table_line and not in_table:
            out.append("```")
            in_table = True
        elif not is_table_line and in_table:
            out.append("```")
            in_table = False
        out.append(line)
    if in_table:
        out.append("```")
    return "\n".join(out)
