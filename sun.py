"""
SUN scanner — stocks trading near their N-period MONTHLY high (breakout watch).

For every stock in the swing/positional universe (all NSE EQ + BE + BZ, same as scanner.py):
    * download monthly candles (yfinance 1mo, up to ~8 years)
    * for each lookback window (6M, 1Y, 2Y, 3Y, 4Y, 5Y, 6Y, 7Y) take the highest HIGH
      of the last N COMPLETED monthly candles (current running month is excluded,
      so the high is the level the stock has to break)
    * LTP = latest close
    * dist% = (LTP - high) / high * 100   (negative = below the high, positive = already above)
The dashboard (sun.html) filters by window and by +/- tolerance (1 / 2 / 5 %).
A stock without enough history for a window gets null for that window (never a fake high).

Output: sun.json  (read by sun.html). Run once a day near market close (Mon-Fri).
"""
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

IST = timezone(timedelta(hours=5, minutes=30))
OUT_FILE = "sun.json"
CHUNK = 100
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
WINDOWS = {"6M": 6, "1Y": 12, "2Y": 24, "3Y": 36, "4Y": 48, "5Y": 60, "6Y": 72, "7Y": 84}
MIN_COVER = 0.9          # window counts if >= 90% of its months exist (allows a missing yf month)


def get_csv(urls, check=None):
    for u in urls:
        try:
            r = requests.get(u, headers=HEADERS, timeout=30)
            if r.ok and (check is None or check(r.text)):
                return r.text
        except Exception:
            continue
    return None


# ---------------- universe: same as scanner.py ----------------
txt = get_csv(("https://archives.nseindia.com/content/equities/EQUITY_L.csv",
               "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"))
if not txt:
    raise SystemExit("EQUITY_L.csv download failed")
eq = pd.read_csv(pd.io.common.StringIO(txt))
eq.columns = eq.columns.str.strip()
eq["SERIES"] = eq["SERIES"].astype(str).str.strip()
eq = eq[eq["SERIES"].isin(("EQ", "BE", "BZ"))]
base_symbols = eq["SYMBOL"].astype(str).str.strip().tolist()
series_map = dict(zip(base_symbols, eq["SERIES"]))

fno_set = set()
t = get_csv(("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
             "https://archives.nseindia.com/content/fo/fo_mktlots.csv"), lambda s: len(s) > 200)
if t:
    for line in t.splitlines()[1:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 2 and p[1] and p[1].upper() == p[1] and " " not in p[1]:
            fno_set.add(p[1])

sector = {}
t = get_csv(("https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
             "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"))
if t:
    n5 = pd.read_csv(pd.io.common.StringIO(t))
    n5.columns = n5.columns.str.strip()
    ic = next((c for c in n5.columns if c.lower() == "industry"), None)
    if ic:
        sector = dict(zip(n5["Symbol"].astype(str).str.strip(), n5[ic].astype(str).str.strip()))

print(f"Universe: {len(base_symbols)} | F&O: {len(fno_set)} | Nifty500 sectors: {len(sector)}")

now = datetime.now(IST)
cur_month = pd.Period(now.date(), "M")


def r2(v):
    return None if v is None or pd.isna(v) else round(float(v), 2)


def process(base, df):
    df = df[["High", "Close"]].dropna()
    df = df[(df["High"] > 0) & (df["Close"] > 0)]
    if df.empty:
        return None
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df.copy()
    df["m"] = idx.to_period("M")
    # yfinance sometimes returns two rows for the running month -> merge per month
    g = df.groupby("m").agg(High=("High", "max"), Close=("Close", "last")).sort_index()
    ltp = g["Close"].iloc[-1]
    done = g[g.index < cur_month]                      # completed months only
    w = {}
    for k, n in WINDOWS.items():
        start = cur_month - n
        part = done[done.index >= start]
        if len(part) >= int(n * MIN_COVER) and done.index.min() <= start + 1:
            hi = part["High"].max()
            hm = part["High"].idxmax()
            w[k] = [r2(hi), r2((ltp - hi) / hi * 100), str(hm)]   # [high, dist%, month of high]
        else:
            w[k] = None
    return {"s": base, "ltp": r2(ltp), "fno": 1 if base in fno_set else 0,
            "ser": series_map.get(base, "EQ"), "sec": sector.get(base, ""),
            "hist": len(done), "w": w}


rows, failed = [], 0
syms = [s + ".NS" for s in base_symbols]
for i in range(0, len(syms), CHUNK):
    chunk = syms[i:i + CHUNK]
    try:
        data = yf.download(chunk, period="8y", interval="1mo", group_by="ticker",
                           threads=True, progress=False, auto_adjust=False)
    except Exception:
        failed += len(chunk)
        continue
    for sym in chunk:
        base = sym[:-3]
        try:
            df = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            row = process(base, df.dropna(how="all"))
            if row:
                rows.append(row)
            else:
                failed += 1
        except Exception:
            failed += 1
    print(f"{min(i + CHUNK, len(syms))}/{len(syms)} | ok={len(rows)} fail={failed}")
    time.sleep(1.0)

out = {"updated": now.strftime("%Y-%m-%d %H:%M IST"),
       "updatedUtc": datetime.now(timezone.utc).isoformat(),
       "universe": f"NSE EQ/BE/BZ ({len(rows)} ok)",
       "windows": list(WINDOWS.keys()), "rows": rows}
with open(OUT_FILE, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(f"Wrote {OUT_FILE}: {len(rows)} rows, {failed} failed")
