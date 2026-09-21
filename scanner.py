"""
SWING / POSITIONAL SCANNER — all NSE EQ + trade-to-trade (BE/BZ) stocks
Timeframes: Monthly, Weekly (+ Daily context)

Ships raw features to data.json. Strategy logic (doji breakout,
rectangle breakout, range-width buckets) lives in the dashboard JS,
so thresholds can be tuned without re-running this scanner.

Also ships:
  * "ld"  — NSE listing date, so the dashboard can compute a rolling
            "listed in the last N years" IPO filter at render time.
  * "pat" — chart-pattern intel per timeframe (V / U recovery, flag,
            ascending & descending triangle) with EARLY / DONE stage
            and a 0-100 quality score. See patterns.py.

One yf.download batch call per chunk (daily 3y), then Weekly/Monthly
are resampled locally — ~6x fewer network calls than per-ticker history.
"""

import json
import math
import time
import logging
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import patterns as PAT

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

IST = timezone(timedelta(hours=5, minutes=30))
CHUNK = 50
OUT_FILE = "data.json"

MIN_BARS = 60          # normal minimum history
MIN_BARS_NEW = 25      # relaxed minimum for recently listed IPOs
NEW_LISTING_YEARS = 3  # keep IPOs this recent even with thin history

# ===============================
# UNIVERSE: all NSE EQ + F&O flag
# ===============================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

resp = requests.get(
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
    headers=HEADERS, timeout=30,
)
resp.raise_for_status()
eq = pd.read_csv(pd.io.common.StringIO(resp.text))
eq.columns = eq.columns.str.strip()
eq["SERIES"] = eq["SERIES"].astype(str).str.strip()
# EQ = normal rolling settlement; BE / BZ = trade-to-trade (no intraday, delivery only).
# T2T names are where most circuit action happens, so they are included and tagged.
UNIVERSE_SERIES = ("EQ", "BE", "BZ")
eq = eq[eq["SERIES"].isin(UNIVERSE_SERIES)]
base_symbols = eq["SYMBOL"].astype(str).str.strip().tolist()
series_map = dict(zip(base_symbols, eq["SERIES"]))

# ---- listing dates (drives the rolling "last N years IPO" filter) ----
listing = {}
_ld_col = next((c for c in eq.columns if "LISTING" in c.upper()), None)
if _ld_col:
    _raw = eq[_ld_col].astype(str).str.strip()
    _ld = pd.to_datetime(_raw, format="%d-%b-%Y", errors="coerce")
    if _ld.notna().sum() < len(_raw) * 0.5:      # NSE changed the format
        _ld = pd.to_datetime(_raw, errors="coerce", dayfirst=True)
    for _s, _d in zip(base_symbols, _ld):
        if pd.notna(_d):
            listing[_s] = _d.strftime("%Y-%m-%d")
print(f"Listing dates parsed: {len(listing)}")

_today = datetime.now(IST).date()

def listed_days(base):
    d = listing.get(base)
    if not d:
        return None
    return (_today - datetime.strptime(d, "%Y-%m-%d").date()).days

fno_set = set()
for _url in ("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
             "https://archives.nseindia.com/content/fo/fo_mktlots.csv"):
    try:
        _r = requests.get(_url, headers=HEADERS, timeout=30)
        if _r.ok and len(_r.text) > 200:
            for _line in _r.text.splitlines()[1:]:
                _p = [x.strip() for x in _line.split(",")]
                if len(_p) >= 2 and _p[1] and _p[1].upper() == _p[1] and " " not in _p[1]:
                    fno_set.add(_p[1])
            if fno_set:
                break
    except Exception:
        continue
if not fno_set:
    print("WARNING: F&O list fetch failed from all sources — fno flags will be 0")

# ---- price bands + surveillance remarks (NSE sec_list.csv) ----
# Band is 2 / 5 / 10 / 20, or "No Band" (F&O names — dynamic flexing, never hard-locked).
band_map, surv_map = {}, {}
for _url in ("https://nsearchives.nseindia.com/content/equities/sec_list.csv",
             "https://archives.nseindia.com/content/equities/sec_list.csv"):
    try:
        _r = requests.get(_url, headers=HEADERS, timeout=30)
        if not (_r.ok and "Symbol" in _r.text[:300] and "Band" in _r.text[:300]):
            continue
        _sl = pd.read_csv(pd.io.common.StringIO(_r.text))
        _sl.columns = _sl.columns.str.strip()
        _sl["Symbol"] = _sl["Symbol"].astype(str).str.strip()
        _sl["Series"] = _sl["Series"].astype(str).str.strip()
        _want = _sl["Symbol"].map(series_map)
        _sl = _sl.assign(_miss=_sl["Series"].ne(_want)).sort_values("_miss", kind="stable")  # matching series wins
        for _, _rr in _sl.drop_duplicates("Symbol").iterrows():
            try:
                band_map[_rr["Symbol"]] = float(str(_rr.get("Band", "")).strip())
            except ValueError:
                band_map[_rr["Symbol"]] = None
            _rm = str(_rr.get("Remarks", "")).strip().strip('"')
            if _rm and _rm not in ("-", "nan"):
                surv_map[_rr["Symbol"]] = _rm
        if band_map:
            break
    except Exception:
        continue
if not band_map:
    print("WARNING: price-band list fetch failed — circuit fields will be empty")
print(f"Price bands: {sum(1 for v in band_map.values() if v)} banded | surveillance remarks: {len(surv_map)}")

symbols = [s + ".NS" for s in base_symbols]
print(f"Universe: {len(symbols)} symbols | F&O flags: {len(fno_set)}")


# ===============================
# HELPERS
# ===============================
def r2(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(f) or np.isinf(f):
        return None
    return round(f, 2)


def tick_for(price):
    """NSE CM tick size by price slab (revised 15-Apr-2025)."""
    if price is None or price < 250:
        return 0.01
    if price <= 1000:
        return 0.05
    if price <= 5000:
        return 0.10
    if price <= 10000:
        return 0.50
    if price <= 20000:
        return 1.0
    return 5.0


def band_limits(pc, band, tk):
    """Upper / lower circuit price: band applied to prev close, rounded INSIDE the band
    to a valid tick. This rounding is why a 5% stock locks at 4.9x%."""
    uc = math.floor(round(pc * (1 + band / 100) / tk, 6)) * tk
    lc = math.ceil(round(pc * (1 - band / 100) / tk, 6)) * tk
    # sub-rupee stocks: band can round to zero ticks — assume at least one tick of room
    uc = max(uc, pc + tk)
    lc = max(tk, min(lc, pc - tk))
    return round(uc, 2), round(lc, 2)


def candle_pattern(o, h, l, c):
    body = abs(c - o)
    rng = h - l
    if rng == 0 or pd.isna(rng):
        return "Flat"
    upper = h - max(o, c)
    lower = min(o, c) - l
    b, u, lo = body / rng, upper / rng, lower / rng
    if b < 0.10:
        return "Doji"
    if b > 0.80:
        return "Bullish Marubozu" if c > o else "Bearish Marubozu"
    if lo > 0.50 and b < 0.30 and u < 0.20:
        return "Hammer"
    if u > 0.50 and b < 0.30 and lo < 0.20:
        return "Shooting Star"
    return "Bullish" if c > o else "Bearish"


def wilder_rsi(close, period=14):
    close = pd.Series(close).dropna()
    if len(close) < period + 1:
        return None
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss))
    return r2(rsi.iloc[-1])


def ema_last(close, span):
    close = pd.Series(close).dropna()
    if len(close) < span:
        return None
    return r2(close.ewm(span=span, adjust=False).mean().iloc[-1])


def calc_vwap(df, window=20):
    d = df.tail(window).dropna(subset=["High", "Low", "Close", "Volume"])
    if d.empty or d["Volume"].sum() == 0:
        return None
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    return r2((tp * d["Volume"]).sum() / d["Volume"].sum())


def hhmm(ts, bar_minutes=5):
    t = ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)
    return (t + timedelta(minutes=bar_minutes)).strftime("%H:%M")


def cross_times(today5, up_levels, dn_levels):
    out = {k: None for k in list(up_levels) + list(dn_levels)}
    if today5 is None or today5.empty:
        return out
    for ts, bar in today5.iterrows():
        c = bar.get("Close")
        if c is None or pd.isna(c):
            continue
        for k, lv in up_levels.items():
            if out[k] is None and lv is not None and c > lv:
                out[k] = hhmm(ts)
        for k, lv in dn_levels.items():
            if out[k] is None and lv is not None and c < lv:
                out[k] = hhmm(ts)
    return out


def candle_block(bar):
    """Prev completed candle OHLC + pattern + fib golden zone."""
    o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
    return {
        "o": r2(o), "h": r2(h), "l": r2(l), "c": r2(c),
        "pat": candle_pattern(o, h, l, c),
        "f50": r2(l + 0.5 * (h - l)),
        "f618": r2(l + 0.618 * (h - l)),
    }


def prev_completed(res_df, current_period_start):
    """Last fully completed bar of a resampled frame."""
    if res_df.empty:
        return None
    if res_df.index[-1] >= current_period_start:
        return res_df.iloc[-2] if len(res_df) >= 2 else None
    return res_df.iloc[-1]


def rect(closes_hist):
    """Rectangle bounds + width%% from a series of completed closes."""
    cs = pd.Series(closes_hist).dropna()
    if len(cs) < 5:
        return None, None, None
    mx, mn = float(cs.max()), float(cs.min())
    if mn <= 0:
        return None, None, None
    return r2(mx), r2(mn), r2((mx - mn) / mn * 100)


AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def ema_block(closes):
    return {"e9": ema_last(closes, 9), "e21": ema_last(closes, 21),
            "e50": ema_last(closes, 50), "e200": ema_last(closes, 200)}


def process_symbol(base, df):
    df = df.dropna(subset=["Close"])
    ld_days = listed_days(base)
    is_new = ld_days is not None and ld_days <= NEW_LISTING_YEARS * 365
    if len(df) < (MIN_BARS_NEW if is_new else MIN_BARS):
        return None

    now = datetime.now(IST)
    last_bar_date = df.index[-1].date()
    daily_is_live = last_bar_date == now.date() and now.hour < 16

    ltp = float(df["Close"].iloc[-1])

    weekly = df.resample("W-FRI").agg(AGG).dropna(subset=["Close"])
    monthly = df.resample("MS").agg(AGG).dropna(subset=["Close"])

    # ---- completed-close histories for rectangles ----
    d_closes = df["Close"]
    d_hist = d_closes.iloc[:-1].tail(21) if daily_is_live else d_closes.tail(21)

    week_start = pd.Timestamp(now.date() - timedelta(days=now.weekday()))
    month_start = pd.Timestamp(now.year, now.month, 1)
    if weekly.index.tz is not None:
        week_start = week_start.tz_localize(weekly.index.tz)
    if monthly.index.tz is not None:
        month_start = month_start.tz_localize(monthly.index.tz)

    # weekly index = week END (Fri); the last bar is the current week if its end >= this week's Monday
    w_hist = (weekly["Close"].iloc[:-1] if weekly.index[-1] >= week_start else weekly["Close"]).tail(21)
    m_hist = (monthly["Close"].iloc[:-1] if monthly.index[-1] >= month_start else monthly["Close"]).tail(23)

    m_max, m_min, m_w = rect(m_hist)
    w_max, w_min, w_w = rect(w_hist)
    d_max, d_min, d_w = rect(d_hist)

    # ---- previous completed candles ----
    pm_bar = prev_completed(monthly, month_start)
    pw_bar = prev_completed(weekly, week_start)
    pd_bar = df.iloc[-2] if (daily_is_live and len(df) >= 2) else df.iloc[-1]

    row = {
        "s": base,
        "ltp": r2(ltp),
        "fno": 1 if base in fno_set else 0,
        "ld": listing.get(base),
        "sr": series_map.get(base, "EQ"),
        "mMax": m_max, "mMin": m_min, "mW": m_w,
        "wMax": w_max, "wMin": w_min, "wW": w_w,
        "dMax": d_max, "dMin": d_min, "dW": d_w,
        "pm": candle_block(pm_bar) if pm_bar is not None else None,
        "pw": candle_block(pw_bar) if pw_bar is not None else None,
        "pd": candle_block(pd_bar) if pd_bar is not None else None,
        "rsiD": wilder_rsi(df["Close"]),
        "rsiW": wilder_rsi(weekly["Close"]),
        "rsiM": wilder_rsi(monthly["Close"]),
        "e9": ema_last(df["Close"], 9),
        "e21": ema_last(df["Close"], 21),
        "e50": ema_last(df["Close"], 50),
        "e200": ema_last(df["Close"], 200),
        "ema": {
            "d": ema_block(df["Close"]),
            "w": ema_block(weekly["Close"]),
            "m": ema_block(monthly["Close"]),
        },
        "vwap": calc_vwap(df),
    }

    vol = df["Volume"].dropna()
    comp_vol = vol.iloc[:-1] if daily_is_live else vol
    row["v7"] = r2(comp_vol.tail(7).mean()) if len(comp_vol) else None

    # ---- price band / circuit ----
    # base price = close of the session BEFORE the latest one (live or finished)
    cl = df["Close"].to_numpy(dtype=float)
    hi = df["High"].to_numpy(dtype=float)
    lo = df["Low"].to_numpy(dtype=float)
    if len(cl) >= 2 and cl[-2] > 0:
        pc = float(cl[-2])
        row["pc"] = r2(pc)
        row["chg"] = r2((ltp - pc) / pc * 100)
        row["dh"] = r2(hi[-1])
        row["dl"] = r2(lo[-1])
        band = None if base in fno_set else band_map.get(base)
        if band:
            # tick is fixed monthly from the previous month-end close
            ref = float(pm_bar["Close"]) if pm_bar is not None else pc
            tk = tick_for(ref)
            uc, lc = band_limits(pc, band, tk)
            row.update({"band": band, "tk": tk, "uc": uc, "lc": lc})

            # approx history with TODAY's band: circuit closes in last 20 sessions + current streak
            cb, streak = 0, 0
            start = max(1, len(cl) - 20)
            flags = []
            for i in range(start, len(cl)):
                t_i = tick_for(cl[i - 1])
                u_i, l_i = band_limits(cl[i - 1], band, t_i)
                f = 1 if cl[i] >= u_i - t_i else (-1 if cl[i] <= l_i + t_i else 0)
                flags.append(f)
                cb += 1 if f else 0
            for f in reversed(flags):
                if f == 0 or (streak and (f > 0) != (streak > 0)):
                    break
                streak += f
            if cb:
                row["cb20"] = cb
            if streak:
                row["cs"] = streak
    if base in surv_map:
        row["sv"] = surv_map[base]

    # ---- chart-pattern intel on completed bars of each timeframe ----
    d_comp = df.iloc[:-1] if daily_is_live else df
    w_comp = weekly.iloc[:-1] if weekly.index[-1] >= week_start else weekly
    m_comp = monthly.iloc[:-1] if monthly.index[-1] >= month_start else monthly

    pat = {}
    for tf, frame in (("d", d_comp), ("w", w_comp), ("m", m_comp)):
        try:
            hits = PAT.detect(frame, tf, ltp)
        except Exception:
            hits = []
        if hits:
            pat[tf] = hits
    if pat:
        row["pat"] = pat

    return row


# ===============================
# BATCH DOWNLOAD + PROCESS
# ===============================
rows = []
failed = 0

for i in range(0, len(symbols), CHUNK):
    chunk = symbols[i:i + CHUNK]
    try:
        data = yf.download(
            chunk, period="5y", interval="1d",
            group_by="ticker", threads=True,
            progress=False, auto_adjust=False,
        )
    except Exception:
        failed += len(chunk)
        continue

    for sym in chunk:
        base = sym.replace(".NS", "")
        try:
            df = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(how="all")
            if df.empty:
                failed += 1
                continue
            row = process_symbol(base, df)
            if row:
                rows.append(row)
            else:
                failed += 1
        except Exception:
            failed += 1

    done = min(i + CHUNK, len(symbols))
    print(f"{done}/{len(symbols)} processed | ok={len(rows)} fail={failed}")
    time.sleep(1.0)

# ===============================
# BREAKOUT-TIME PASS (today's 5m, only symbols beyond a level)
# ===============================
by_sym = {r["s"]: r for r in rows}
cands = []
for r in rows:
    ltp = r.get("ltp")
    if ltp is None:
        continue
    hit = False
    for mxk, mnk, cbk in (("mMax", "mMin", "pm"), ("wMax", "wMin", "pw"), ("dMax", "dMin", "pd")):
        mx, mn, cb = r.get(mxk), r.get(mnk), r.get(cbk)
        if (mx is not None and ltp > mx) or (mn is not None and ltp < mn):
            hit = True
        if cb and cb.get("pat") == "Doji" and (ltp > cb["c"] or ltp < cb["l"]):
            hit = True
    if hit:
        cands.append(r["s"])
cands = cands[:400]
print(f"Breakout-time pass: {len(cands)} candidates")

now_ist = datetime.now(IST)
for i in range(0, len(cands), CHUNK):
    chunk = [s + ".NS" for s in cands[i:i + CHUNK]]
    try:
        d5 = yf.download(chunk, period="2d", interval="5m", group_by="ticker",
                         threads=True, progress=False, auto_adjust=False)
    except Exception:
        continue
    for sym in chunk:
        base = sym.replace(".NS", "")
        r = by_sym.get(base)
        if r is None:
            continue
        try:
            df5 = d5[sym] if isinstance(d5.columns, pd.MultiIndex) else d5
            df5 = df5.dropna(how="all")
            if df5.empty:
                continue
            idx = df5.index
            idx_ist = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
            last_day = max(idx_ist.date)      # last trading day (works on weekends/holidays)
            today5 = df5[idx_ist.date == last_day]
            bt = {}
            for tf, mxk, mnk, cbk in (("m", "mMax", "mMin", "pm"), ("w", "wMax", "wMin", "pw"), ("d", "dMax", "dMin", "pd")):
                cb = r.get(cbk)
                up = {"u": r.get(mxk)}
                dn = {"l": r.get(mnk)}
                if cb:
                    up["dh"] = cb.get("h")
                    up["dc"] = cb.get("c")
                    dn["dl"] = cb.get("l")
                res = cross_times(today5, up, dn)
                if any(v for v in res.values()):
                    bt[tf] = {k: v for k, v in res.items() if v}
            if bt:
                r["bt"] = bt
        except Exception:
            continue
    time.sleep(0.6)

# ===============================
# SAVE
# ===============================
payload = {
    "updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
    "updatedUtc": datetime.now(timezone.utc).isoformat(),
    "count": len(rows),
    "rows": rows,
}
with open(OUT_FILE, "w") as f:
    json.dump(payload, f, separators=(",", ":"))

print(f"Saved {OUT_FILE}: {len(rows)} rows, {failed} failed")
