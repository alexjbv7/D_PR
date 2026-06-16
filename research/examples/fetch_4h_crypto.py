"""
fetch_4h_crypto.py — Descarga barras 4h de un universo cripto a la caché Parquet.

Pobla ``research/data/alpaca_bars/bars/4h/<SYM>.parquet`` para que el screening de
pares (``alpha.statarb.screen``) pueda buscar cointegración en 4h (Rama B, E5).

Prerrequisitos (tu venv):
  - ``pip install alpaca-py``
  - credenciales en el entorno:
      ALPACA_API_KEY, ALPACA_API_SECRET
  - ejecutar con PYTHONPATH que incluya ``research`` (o desde la raíz; el script
    añade research/ a sys.path).

Uso:
  python research/examples/fetch_4h_crypto.py                 # universo por defecto
  python research/examples/fetch_4h_crypto.py --symbols XRP/USD BTC/USD ETH/USD
  python research/examples/fetch_4h_crypto.py --start 2019-01-01

El anchor del screening es XRP/USD; el resto son las cripto-mayores con las que se
empareja. Cripto opera 24/7 → 4h ≈ 6 barras/día (mucha más muestra que diario).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_RESEARCH = Path(__file__).resolve().parents[1]
if str(_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_RESEARCH))

# Universo por defecto: XRP (anchor) + majors líquidas en Alpaca crypto.
DEFAULT_UNIVERSE = [
    "XRP/USD", "BTC/USD", "ETH/USD", "SOL/USD",
    "LTC/USD", "LINK/USD", "DOGE/USD", "AVAX/USD",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Descarga 4h cripto a la caché Parquet")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_UNIVERSE)
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--start", default="2018-01-01", help="YYYY-MM-DD (UTC)")
    ap.add_argument("--output-dir", default=str(_RESEARCH / "data" / "alpaca_bars"))
    args = ap.parse_args()

    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or not secret:
        print("ERROR: define ALPACA_API_KEY y ALPACA_API_SECRET en el entorno.",
              file=sys.stderr)
        return 2

    from data.alpaca_bars import AlpacaBarsIngestor

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    ingestor = AlpacaBarsIngestor(api_key=key, api_secret=secret,
                                  output_dir=args.output_dir)
    ingestor.connect()
    report = ingestor.ingest(args.symbols, timeframe=args.timeframe,
                             start=start, end=datetime.now(tz=timezone.utc))

    print(f"\n=== Descarga {args.timeframe} — {report.succeeded}/{report.total_symbols} OK ===")
    for r in report.reports:
        status = "OK  " if r.success else "FAIL"
        print(f"  {status} {r.symbol:10s} bars={r.bar_count:6d} last={r.last_ts} {r.error or ''}")
    print(f"\nCaché: {args.output_dir}/bars/{args.timeframe}/")
    print("Siguiente: correr el screening de pares (alpha.statarb.screen.screen_universe).")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
