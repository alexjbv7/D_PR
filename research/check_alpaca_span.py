"""
Diagnostic: how much SPY daily history does each Alpaca feed return?

Run:  python check_alpaca_span.py
Needs ALPACA_API_KEY / ALPACA_API_SECRET exported in the environment.

Writes to a throwaway cache dir so it does NOT pollute the training data.
"""
from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data.alpaca_bars import AlpacaBarsIngestor

START = datetime(2008, 1, 1, tzinfo=timezone.utc)
END = datetime.now(timezone.utc)


def probe(feed: str) -> None:
    print(f"\n=== feed={feed} ===")
    try:
        out = Path(tempfile.gettempdir()) / f"alpaca_probe_{feed}"
        ing = AlpacaBarsIngestor.from_env(output_dir=out, feed=feed)
        with ing:
            ing.ingest(["SPY"], timeframe="1d", start=START, end=END)
            df = ing._load_df("SPY", "1d")
        if df is None or df.empty:
            print("  no bars returned")
            return
        print(f"  bars={len(df)}  span={df.index.min().date()} -> {df.index.max().date()}")
        yrs = (df.index.max() - df.index.min()).days / 365.25
        print(f"  ~{yrs:.1f} years of daily data")
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    print(f"Requesting SPY 1d from {START.date()} to {END.date()} on each feed...")
    for f in ("iex", "sip"):
        probe(f)
    print("\nIf sip errors with a subscription/403 message, your account lacks the")
    print("paid SIP data plan. iex span is what you actually have for free.")
