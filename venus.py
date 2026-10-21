"""
VENUS scanner — Elliott-wave style 3-candle box breakout, multi-timeframe.

Combos (higher TF reference  ->  trigger TF):
    M30 : previous MONTH high/low  -> 30-min candles of the current month
    W15 : previous WEEK  high/low  -> 15-min candles of the current week
    D5  : previous DAY   high/low  -> 5-min  candles of the current day

Upside (mirror for downside):
    C1 = first trigger candle that CLOSES above prev-period high
    C2 = first later candle that CLOSES above C1 high   (closes below C1 low are ignored, no reset)
    C3 = first later candle that CLOSES above C2 high
    BOX: bottom = C1 high, top = C3 high
    Status of latest closed candle:
        close > top    -> BREAKOUT
        close < bottom -> PATTERN FAIL
        else           -> INSIDE
Downside: C1 closes below prev low, C2 below C1 low, C3 below C2 low,
          box top = C1 low, bottom = C3 low; close < bottom = BREAKOUT, close > top = PATTERN FAIL.
Up and down patterns are tracked independently.

Data: yfinance. One 5-min download per chunk (from the earlier of month-start / week-start),
resampled to 15m / 30m on a 09:15 anchor. Prev month/week/day high-low from daily bars.
Universe: Nifty 500 (NSE index constituents CSV).
Output: venus.json (read by venus.html)
"""
import argparse
import json
import time
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

IST = timezone(timedelta(hours=5, minutes=30))
OUT_FILE = "venus.json"
CHUNK = 50
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SESSION_OPEN = (9, 15)
SESSION_CLOSE = (15, 30)

COMBOS = {               # key: (reference period, trigger minutes)
    "M30": ("M", 30),
    "W15": ("W", 15),
    "D5":  ("D", 5),
    "D3":  ("D", 3),     # built from 1-min data (yfinance has no 3m interval)
}


# ===============================
# UNIVERSE: Nifty 500
# ===============================
def load_universe():
    urls = (
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    )
    for u in urls:
        try:
            r = requests.get(u, headers=HEADERS, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(r.text))
            df.columns = df.columns.str.strip()
            syms = df["Symbol"].astype(str).str.strip().tolist()
            ind_col = next((c for c in df.columns if c.lower() == "industry"), None)
            sectors = dict(zip(syms, df[ind_col].astype(str).str.strip())) if ind_col else {}
            if len(syms) > 400:
                print(f"Universe: {len(syms)} symbols from {u} | sectors: {len(set(sectors.values()))}")
                return syms, sectors
        except Exception as e:
            print(f"Universe source failed ({u}): {e}")
    raise SystemExit("Could not load Nifty 500 list")


# ===============================
# HELPERS
# ===============================
def to_ist(df):
    idx = df.index
    idx = idx.tz_localize("UTC").tz_convert(IST) if idx.tz is None else idx.tz_convert(IST)
    out = df.copy()
    out.index = idx
    return out


def clean_ohlc(df):
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df[(df["High"] > 0) & (df["Low"] > 0)]


def in_session(df):
    t = df.index.hour * 60 + df.index.minute
    lo = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    hi = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    return df[(t >= lo) & (t < hi)]


def resample_closed(df5, minutes, now_ist, base_min=5):
    """Build `minutes` bars from base (5m or 1m) bars on a 09:15 anchor, keep only fully closed bars."""
    if df5.empty:
        return df5
    # keep only base bars that have finished
    df5 = df5[df5.index + pd.Timedelta(minutes=base_min) <= now_ist]
    if df5.empty:
        return df5
    last_5m_end = df5.index[-1] + pd.Timedelta(minutes=base_min)

    if minutes == base_min:
        bars = df5.copy()
    else:
        bars = df5.resample(f"{minutes}min", origin="start_day", offset="15min",
                            label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()

    # a bar is closed when its end (capped at session close) is covered by data
    ends = []
    for ts in bars.index:
        cap = ts.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1], second=0)
        ends.append(min(ts + pd.Timedelta(minutes=minutes), cap))
    bars = bars[[e <= last_5m_end for e in ends]]
    return bars


def prev_period_hl(daily, period, cur_day):
    """High/low of the previous completed month / week / day relative to cur_day."""
    d = daily[daily.index.date < cur_day] if period == "D" else daily
    if period == "M":
        key = d.index.to_period("M")
        cur = pd.Period(cur_day, "M")
    elif period == "W":
        key = d.index.to_period("W-SUN")
        cur = pd.Period(cur_day, "W-SUN")
    else:
        key = pd.PeriodIndex(d.index.date, freq="D")
        cur = pd.Period(cur_day, "D")
    prev = d[key < cur]
    if prev.empty:
        return None
    if period == "D":
        last = prev.index[-1].date()
        prev = prev[prev.index.date == last]
    else:
        pkey = key[key < cur]
        prev = prev[pkey == pkey.max()]
    return float(prev["High"].max()), float(prev["Low"].min())


def period_start(period, cur_day):
    if period == "M":
        return cur_day.replace(day=1)
    if period == "W":
        return cur_day - timedelta(days=cur_day.weekday())
    return cur_day


def fmt(ts, period):
    return ts.strftime("%d-%b %H:%M") if period != "D" else ts.strftime("%H:%M")


def r2(v):
    return None if v is None else round(float(v), 2)


# ===============================
# PATTERN ENGINE
# ===============================
def bar_end_ms(ts, minutes):
    cap = ts.replace(hour=SESSION_CLOSE[0], minute=SESSION_CLOSE[1], second=0)
    return int(min(ts + pd.Timedelta(minutes=minutes), cap).timestamp() * 1000)


def detect(bars, level, side, period, minutes):
    """Run the C1 -> C2 -> C3 -> box state machine on closed trigger bars."""
    up = side == "U"
    c = [None, None, None]
    stage = 0
    for ts, row in bars.iterrows():
        cl, hi, lo = row["Close"], row["High"], row["Low"]
        if stage == 0:
            ok = cl > level if up else cl < level
        else:
            prev = c[stage - 1]
            ok = cl > prev["h"] if up else cl < prev["l"]
        if ok and stage < 3:
            c[stage] = {"t": fmt(ts, period), "ts": ts, "h": r2(hi), "l": r2(lo)}
            stage += 1
            if stage == 3:
                break
    if stage == 0:
        return None

    out = {"side": side, "stage": stage, "lvl": r2(level),
           "c1": None, "c2": None, "c3": None,
           "top": None, "bot": None, "st": "FORM", "bt": None, "ft": None, "since": None,
           "bte": None, "fte": None, "sincee": None}
    for i in range(stage):
        out[f"c{i+1}"] = {k: c[i][k] for k in ("t", "h", "l")}

    last_close = float(bars["Close"].iloc[-1])
    out["ltp"] = r2(last_close)
    if stage < 3:
        return out

    if up:
        top, bot = c[2]["h"], c[0]["h"]
    else:
        top, bot = c[0]["l"], c[2]["l"]
    out["top"], out["bot"] = top, bot

    after = bars[bars.index > c[2]["ts"]]

    def status(x):
        if up:
            return "BO" if x > top else ("FAIL" if x < bot else "IN")
        return "BO" if x < bot else ("FAIL" if x > top else "IN")

    st_prev = None
    since = since_e = None
    for ts, row in after.iterrows():
        s = status(row["Close"])
        if s == "BO" and out["bt"] is None:
            out["bt"] = fmt(ts, period)
            out["bte"] = bar_end_ms(ts, minutes)
        if s == "FAIL" and out["ft"] is None:
            out["ft"] = fmt(ts, period)
            out["fte"] = bar_end_ms(ts, minutes)
        if s != st_prev:
            since = fmt(ts, period)
            since_e = bar_end_ms(ts, minutes)
            st_prev = s
    if after.empty:
        out["st"] = "IN"
    else:
        out["st"] = st_prev
        out["since"] = since
        out["sincee"] = since_e   # candle-close time (epoch ms) when price moved outside the box
    return out


def process(base, daily, df5, now_ist, combos=COMBOS, base_min=5):
    daily = clean_ohlc(daily)
    df5 = in_session(clean_ohlc(to_ist(df5)))
    if daily.empty or df5.empty:
        return None
    cur_day = df5.index[-1].date()           # last trading day in data (weekend/holiday safe)
    today = df5[df5.index.date == cur_day]
    done5 = today[today.index + pd.Timedelta(minutes=base_min) <= now_ist]
    prev_d = daily[daily.index.date < cur_day]
    pc = float(prev_d["Close"].iloc[-1]) if not prev_d.empty else None
    ltp = float(today["Close"].iloc[-1])
    quote = {"ltp": r2(ltp), "dh": r2(today["High"].max()), "dl": r2(today["Low"].min()),
             "pc": r2(pc), "chg": r2((ltp / pc - 1) * 100) if pc else None,
             "asof": bar_end_ms(done5.index[-1], base_min) if not done5.empty else None}
    res = {}
    for key, (period, mins) in combos.items():
        hl = prev_period_hl(daily, period, cur_day)
        if hl is None:
            continue
        start = period_start(period, cur_day)
        seg = df5[df5.index.date >= start]
        bars = resample_closed(seg, mins, now_ist, base_min)
        if bars.empty:
            continue
        found = []
        for side, lvl in (("U", hl[0]), ("D", hl[1])):
            p = detect(bars, lvl, side, period, mins)
            if p:
                p["bar"] = fmt(bars.index[-1], period)
                found.append(p)
        if found:
            res[key] = found
    return res, quote


# ===============================
# MAIN
# ===============================
def download(chunk, tries=3, **kw):
    """yf.download with retry (Yahoo sometimes rate-limits frequent runs)."""
    for t in range(tries):
        try:
            df = yf.download(chunk, group_by="ticker", threads=True, progress=False,
                             auto_adjust=False, **kw)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            print(f"download error (try {t + 1}): {e}")
        time.sleep(3 * (t + 1))
    raise RuntimeError("download failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default="M30,W15,D5", help="comma list, e.g. D5")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()
    run = {k: COMBOS[k] for k in args.combos.split(",") if k in COMBOS}
    fast = set(run) <= {"D5", "D3"}     # day-only mode: tiny downloads, runs every 3-5 min
    base_iv, base_min = ("1m", 1) if "D3" in run else ("5m", 5)

    symbols, sectors = load_universe()
    now_ist = datetime.now(IST)
    today = now_ist.date()
    if fast:
        start_5m = today - timedelta(days=5)      # covers weekends / a holiday
        daily_period = "10d"
    else:
        start_5m = min(today.replace(day=1), today - timedelta(days=today.weekday())) - timedelta(days=4)
        daily_period = "4mo"
    combos = {k: [] for k in run}
    quotes = {}
    ok = fail = 0

    for i in range(0, len(symbols), CHUNK):
        chunk = [s + ".NS" for s in symbols[i:i + CHUNK]]
        try:
            dd = download(chunk, period=daily_period, interval="1d")
            if base_min == 1:      # 1-min only needed for the latest session
                d5 = download(chunk, period="1d", interval="1m")
            else:
                d5 = download(chunk, start=start_5m.isoformat(), interval="5m")
        except Exception as e:
            print(f"chunk {i} download failed: {e}")
            fail += len(chunk)
            continue
        for sym in chunk:
            base = sym.replace(".NS", "")
            try:
                a = dd[sym] if isinstance(dd.columns, pd.MultiIndex) else dd
                b = d5[sym] if isinstance(d5.columns, pd.MultiIndex) else d5
                out = process(base, a.dropna(how="all"), b.dropna(how="all"), now_ist, run, base_min)
                if out is None:
                    fail += 1
                    continue
                r, quote = out
                ok += 1
                quotes[base] = quote
                for key, pats in r.items():
                    for p in pats:
                        p["s"] = base
                        combos[key].append(p)
            except Exception as e:
                fail += 1
        print(f"{min(i + CHUNK, len(symbols))}/{len(symbols)} | ok={ok} fail={fail}")
        time.sleep(0.3 if fast else 0.8)

    payload = {
        "updated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        "updatedUtc": datetime.now(timezone.utc).isoformat(),
        "universe": "Nifty 500",
        "sector": sectors,        # symbol -> NSE industry (Nifty 500 list)
        "scanned": ok,
        "asof": max([q["asof"] for q in quotes.values() if q.get("asof")], default=None),
        "quotes": quotes,         # today's ltp / day high / day low per symbol (for circuit filters)
        "combos": combos,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Saved {args.out}: " + ", ".join(f"{k}={len(v)}" for k, v in combos.items()))


if __name__ == "__main__":
    main()
