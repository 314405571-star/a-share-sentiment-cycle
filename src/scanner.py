"""baostock adapter: turn a stock pool into one day of classified signals.

This is the only module that touches the network. ``baostock`` is imported
lazily so the rest of the package stays importable without it.

Notes on the data choices, which matter for the indicator semantics:

* ``adjustflag="3"`` (unadjusted) is deliberate. A price limit is defined on
  the raw price, so forward-adjusted bars would move the +10% reference.
* ``turn`` is turnover in percent, as returned by baostock.
* The pool CSV needs at least a ``code`` column; a ``name`` column is optional.

Usage::

    python -m src.scanner --pool stock_pool.csv --date 2026-11-10
    python -m src.scanner --build-pool stock_pool.csv     # all listed A-shares
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .indicators import Bar, StockSignal, to_signal
from .sentiment import Config, EmotionEngine, State

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn"


class ScannerError(RuntimeError):
    pass


def normalise_code(raw: str) -> str:
    raw = str(raw).strip()
    if raw.startswith(("sh.", "sz.", "bj.")):
        return raw
    if raw.startswith("6"):
        return "sh." + raw
    if raw.startswith(("0", "3")):
        return "sz." + raw
    if raw.startswith(("4", "8", "9")):
        return "bj." + raw
    return "sh." + raw


class BaoStockScanner:
    def __init__(self, pool_csv: Optional[str] = None, state_file: str = "emotion_state.json",
                 industry_cache: str = "industry_cache.json", max_workers: int = 1,
                 max_concurrent_requests: int = 2, request_delay: float = 0.5,
                 max_retries: int = 3, verbose: bool = True):
        self.pool_csv = pool_csv
        self.state_file = state_file
        self.industry_cache = industry_cache
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.verbose = verbose

        self.codes: List[str] = []
        self.names: Dict[str, str] = {}
        self.industry: Dict[str, str] = {}
        self._lock = threading.Semaphore(max_concurrent_requests)  # rate limit
        self._logged_in = False

        if pool_csv:
            self._load_pool(pool_csv)

    # -- logging utilities ------------------------------------------------
    def log(self, *args):
        if self.verbose:
            print(*args)

    # -- pool -------------------------------------------------------------
    def _load_pool(self, path: str):
        import pandas as pd
        df = pd.read_csv(path, encoding="utf-8-sig")
        col = "code" if "code" in df.columns else df.columns[0]
        self.codes = [normalise_code(c) for c in df[col].tolist()]
        if "name" in df.columns:
            self.names = dict(zip(self.codes, df["name"].astype(str)))
        self.log(f"pool: {len(self.codes)} codes from {path}")

    # -- session ----------------------------------------------------------
    def __enter__(self):
        import baostock as bs
        for attempt in range(self.max_retries):
            try:
                lg = bs.login()
                if lg.error_code == "0":
                    self._logged_in = True
                    return self
            except Exception:
                pass
            time.sleep(2 ** attempt)
        raise ScannerError("baostock login failed")

    def __exit__(self, *exc):
        if self._logged_in:
            import baostock as bs
            bs.logout()
            self._logged_in = False

    # -- raw data ---------------------------------------------------------
    def fetch_bars(self, code: str, end_date: str, days: int = 70) -> Optional[List[Bar]]:
        import baostock as bs
        start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days * 2)).strftime("%Y-%m-%d")
        for attempt in range(self.max_retries + 1):
            with self._lock:
                try:
                    time.sleep(self.request_delay + (3 ** attempt) * 0.2 + np.random.uniform(0, 0.5))
                    rs = bs.query_history_k_data_plus(
                        code, FIELDS, start_date=start, end_date=end_date,
                        frequency="d", adjustflag="3")
                    rows = []
                    while rs.error_code == "0" and rs.next():
                        rows.append(rs.get_row_data())
                except (ConnectionResetError, ConnectionAbortedError, TimeoutError, OSError) as exc:
                    if self.verbose:
                        print(f"  [{code}] network error: {exc}")
                    continue
            if not rows:
                return None
            bars = []
            for r in rows:
                try:
                    pre = float(r[6])
                    if pre <= 0:
                        continue
                    bars.append(Bar(code=r[1], preclose=pre, open=float(r[2]),
                                    high=float(r[3]), low=float(r[4]), close=float(r[5]),
                                    turnover=float(r[9]) if r[9] else 0.0,
                                    industry=self.industry.get(code),
                                    name=self.names.get(code, ""), date=r[0]))
                except (ValueError, IndexError):
                    continue
            return bars[-days:] if bars else None
        return None

    def hs300_return(self, end_date: str) -> Optional[float]:
        bars = self.fetch_bars("sh.000300", end_date, days=5)
        if not bars or len(bars) < 2:
            return None
        return bars[-1].close / bars[-2].close - 1

    # -- industry ---------------------------------------------------------
    def load_industry(self, refresh: bool = False):
        if not refresh and os.path.exists(self.industry_cache):
            with open(self.industry_cache, encoding="utf-8") as f:
                data = json.load(f)
            self.industry = data.get("industry", {})
            self.log(f"industry cache: {len(self.industry)} codes")
            return
        import baostock as bs
        self.log("building industry cache (one request per code)...")
        for i, code in enumerate(self.codes):
            try:
                rs = bs.query_stock_industry(code)
                best = None
                while rs.error_code == "0" and rs.next():
                    row = rs.get_row_data()
                    name, level = row[1], (row[2] if len(row) > 2 else "")
                    if "二级" in level:
                        best = name
                        break
                    best = best or name
                if best:
                    self.industry[code] = best
            except Exception:
                continue
            if (i + 1) % 200 == 0:
                self.log(f"  {i + 1}/{len(self.codes)}")
            time.sleep(0.05)
        with open(self.industry_cache, "w", encoding="utf-8") as f:
            json.dump({"industry": self.industry}, f, ensure_ascii=False)
        self.log(f"industry cache saved: {len(self.industry)} codes")

    # -- one trading day --------------------------------------------------
    def scan(self, date: str = None) -> Tuple[List[StockSignal], Optional[float]]:
        """Classify every stock in the pool for ``date``.

        Bars whose last date is not ``date`` are dropped rather than scored:
        on a non-trading day baostock happily returns the previous session, and
        silently re-scoring yesterday's tape is the easiest way to look-ahead
        your own state machine.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")
        hs_ret = self.hs300_return(date)

        def work(code):
            bars = self.fetch_bars(code, date, days=70)
            if not bars or bars[-1].close <= 0:
                return None
            if bars[-1].date and bars[-1].date != date:
                return None
            if not bars[-1].industry:
                bars[-1].industry = self.industry.get(code)
            return to_signal(bars[-1], bars[:-1])

        signals: List[StockSignal] = []
        if self.max_workers <= 1:
            for i, code in enumerate(self.codes):
                s = work(code)
                if s:
                    signals.append(s)
                if (i + 1) % 500 == 0:
                    self.log(f"  scanned {i + 1}/{len(self.codes)}")
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(work, c) for c in self.codes]
                for i, fut in enumerate(as_completed(futures)):
                    s = fut.result()
                    if s:
                        signals.append(s)
                    if (i + 1) % 500 == 0:
                        self.log(f"  scanned {i + 1}/{len(self.codes)}")
        if not signals:
            self.log(f"no bars dated {date} -- non-trading day, or the pool is empty")
        self.log(f"scanned {len(signals)}/{len(self.codes)} codes, hs300 {hs_ret}")
        return signals, hs_ret


def build_pool(csv_path: str, date: Optional[str] = None):
    """Write every listed A-share to a pool CSV (code,name)."""
    import baostock as bs
    import pandas as pd

    date = date or datetime.now().strftime("%Y-%m-%d")
    lg = bs.login()
    if lg.error_code != "0":
        raise ScannerError("login failed")
    try:
        rs = bs.query_all_stock(day=date)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=["code", "tradeStatus", "code_name"])
        df = df[df["code"].str.startswith(("sh.6", "sz.0", "sz.3", "bj."))]
        df[["code", "code_name"]].rename(columns={"code_name": "name"}).to_csv(
            csv_path, index=False, encoding="utf-8-sig")
        print(f"wrote {len(df)} codes to {csv_path}")
    finally:
        bs.logout()


def run_day(pool_csv: str, date: str = None, state_file: str = "emotion_state.json",
            refresh_industry: bool = False, max_workers: int = 1):
    scanner = BaoStockScanner(pool_csv, state_file=state_file, max_workers=max_workers)
    with scanner:
        scanner.load_industry(refresh=refresh_industry)
        signals, hs_ret = scanner.scan(date)
        engine = EmotionEngine(Config(), State.load(state_file))
        snap = engine.update(date or datetime.now().strftime("%Y-%m-%d"), signals, hs_ret)
        engine.state.save(state_file)
        print(snap.report())
        log_path = "sentiment_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(snap.as_dict(), ensure_ascii=False) + "\n")
        return snap


def main(argv: Sequence[str] = None):
    ap = argparse.ArgumentParser(description="A-share sentiment cycle - one day scan")
    ap.add_argument("--pool", help="stock pool CSV (needs a 'code' column)")
    ap.add_argument("--date", help="trading day, YYYY-MM-DD (default: today)")
    ap.add_argument("--state", default="emotion_state.json")
    ap.add_argument("--refresh-industry", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--build-pool", metavar="CSV", help="write all A-shares to CSV and exit")
    args = ap.parse_args(argv)

    if args.build_pool:
        build_pool(args.build_pool, args.date)
        return
    if not args.pool:
        ap.error("--pool is required unless --build-pool is given")
    run_day(args.pool, args.date, args.state, args.refresh_industry, args.workers)


if __name__ == "__main__":
    main()
