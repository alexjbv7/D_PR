"""
Briefing Scheduler — APScheduler
Runs daily briefing Mon-Fri at 6pm ET
Runs weekly briefing every Friday at 7pm ET

Usage:
    python -m tools.briefing.scheduler
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv()

_ET = ZoneInfo("America/New_York")


def run_daily() -> None:
    date_str = datetime.now(tz=_ET).strftime("%Y-%m-%d")
    print(f"[scheduler] Running daily briefing for {date_str}", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.briefing.daily", "--date", date_str],
        capture_output=True,
        text=True,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"[scheduler] ERROR: {result.stderr}", flush=True)


def run_weekly() -> None:
    now = datetime.now(tz=_ET)
    week_iso = f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"
    print(f"[scheduler] Running weekly briefing for {week_iso}", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "tools.briefing.weekly", "--week", week_iso],
        capture_output=True,
        text=True,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"[scheduler] ERROR: {result.stderr}", flush=True)


async def main() -> None:
    scheduler = AsyncIOScheduler(timezone=_ET)

    # Daily briefing: Mon-Fri at 6pm ET
    scheduler.add_job(
        run_daily,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=0, timezone=_ET),
        id="daily_briefing",
        name="Daily Briefing",
        replace_existing=True,
    )

    # Weekly briefing: Friday at 7pm ET
    scheduler.add_job(
        run_weekly,
        CronTrigger(day_of_week="fri", hour=19, minute=0, timezone=_ET),
        id="weekly_briefing",
        name="Weekly Briefing",
        replace_existing=True,
    )

    scheduler.start()
    print("[scheduler] Started. Daily=Mon-Fri 18:00 ET | Weekly=Fri 19:00 ET", flush=True)

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("[scheduler] Stopped.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())