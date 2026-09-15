"""
patterns.py — geometric chart-pattern detection for the Swing / Positional scanner.

Detects, per timeframe (day / week / month):
    V    — V-shape recovery        (sharp fall, sharp snap-back, narrow bottom)
    U    — U-shape recovery        (fall, broad rounded base, grind back up)
    FLAG — bull / bear flag        (impulse pole + tight counter-drift consolidation)
    ASCT — ascending triangle      (flat resistance + rising lows)
    DESCT— descending triangle     (flat support + falling highs)

Every detection carries a STAGE:
    "E" = EARLY  — pattern is still forming / recovery in progress (early detection)
    "D" = DONE   — pattern completed or broken out (confirmed)

Pure numpy. Input = completed OHLC bars of ONE timeframe + live LTP.
Output = compact list of dicts, JSON-ready, so data.json stays small.
"""

import numpy as np

# ---------------------------------------------------------------
# per-timeframe thresholds — tuned for NSE swing / positional use
# ---------------------------------------------------------------
TF_CFG = {
    "d": dict(look=60, minBars=25, declMin=14.0, poleMin=11.0, flatTol=2.2,
              flagMax=13, poleLens=(5, 7, 10, 14)),
    "w": dict(look=52, minBars=22, declMin=22.0, poleMin=18.0, flatTol=3.2,
              flagMax=11, poleLens=(4, 6, 9, 12)),
    "m": dict(look=30, minBars=16, declMin=30.0, poleMin=25.0, flatTol=4.5,
              flagMax=7,  poleLens=(3, 4, 6, 8)),
}

MIN_SCORE = 40        # anything weaker is not shipped
DONE_REC  = 0.88      # recovery ratio that counts as "recovery done"
EARLY_REC = 0.18      # below this the recovery has not really started


# ===============================================================
# small numeric helpers
# ===============================================================
def _r(v, n=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, n)


def _linfit(y):
    """Closed-form least squares. Returns (slope, intercept, r2)."""
    n = len(y)
    if n < 3:
        return 0.0, float(y[-1]) if n else 0.0, 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0:
        return 0.0, ym, 0.0
    slope = ((x - xm) * (y - ym)).sum() / sxx
    icpt = ym - slope * xm
    ss_tot = ((y - ym) ** 2).sum()
    if ss_tot <= 0:
        return slope, icpt, 0.0
    ss_res = ((y - (slope * x + icpt)) ** 2).sum()
    return slope, icpt, max(0.0, 1.0 - ss_res / ss_tot)


def _quadfit(y):
    """Quadratic least squares. Returns (a, b, c, r2) for a*x^2+b*x+c."""
    n = len(y)
    if n < 6:
        return 0.0, 0.0, 0.0, 0.0
    x = np.arange(n, dtype=float)
    try:
        a, b, c = np.polyfit(x, y, 2)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot <= 0:
        return a, b, c, 0.0
    ss_res = ((y - (a * x * x + b * x + c)) ** 2).sum()
    return float(a), float(b), float(c), max(0.0, 1.0 - ss_res / ss_tot)


def _pw_r2(y, pki, t):
    """R^2 of a two-segment (down-leg / up-leg) linear fit hinged at the trough.
    High for a V, low for a U — the flat base is badly fitted by a sharp hinge."""
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    res = 0.0
    for seg in (y[pki:t + 1], y[t:]):
        if len(seg) < 2:
            continue
        x = np.arange(len(seg), dtype=float)
        s, i, _ = _linfit(seg)
        res += ((seg - (s * x + i)) ** 2).sum()
    if pki > 0:                       # bars before the peak are unmodelled
        head = y[:pki + 1]
        res += ((head - head.mean()) ** 2).sum()
    return max(0.0, 1.0 - res / ss_tot)


def _pivots(h, l, k=2):
    """Fractal swing highs / lows. Returns (idx_highs, idx_lows)."""
    ph, pl = [], []
    n = len(h)
    for i in range(k, n - k):
        w_h = h[i - k:i + k + 1]
        w_l = l[i - k:i + k + 1]
        if h[i] >= w_h.max() and h[i] > h[i - 1]:
            ph.append(i)
        if l[i] <= w_l.min() and l[i] < l[i - 1]:
            pl.append(i)
    return ph, pl


def _clip01(v):
    return float(min(1.0, max(0.0, v)))


# ===============================================================
# 1 + 2.  V-shape and U-shape recovery
# ===============================================================
def _detect_vu(o, h, l, c, ltp, cf):
    n = len(c)
    t = int(np.argmin(l))
    if t < 3 or t > n - 3:
        return []

    pki = int(np.argmax(h[:t + 1]))
    pk = float(h[pki])
    low = float(l[t])
    if pk <= 0 or low <= 0 or pk <= low:
        return []

    fall = pk - low
    dd = fall / pk * 100.0                       # drawdown %
    if dd < cf["declMin"]:
        return []

    rec = (ltp - low) / fall                     # 0..1+  retracement of the fall
    if rec < EARLY_REC:
        return []

    leftBars = t - pki
    rightBars = (n - 1) - t
    if leftBars < 2 or rightBars < 2:
        return []

    sym = min(leftBars, rightBars) / max(leftBars, rightBars)

    # base width measured relative to the SIZE OF THE FALL, not to price —
    # this is what separates a spike bottom (V) from a rounded base (U).
    tightBand = low + 0.10 * fall
    wideBand = low + 0.22 * fall
    baseTight = int((l <= tightBand).sum())
    baseWide = int((l <= wideBand).sum())
    tightRatio = baseTight / n
    wideRatio = baseWide / n

    _, _, r2_left = _linfit(c[pki:t + 1])        # clean down-leg?
    _, _, r2_right = _linfit(c[t:])              # clean up-leg?
    a, b, _, r2q = _quadfit(c)                   # rounded-bottom test
    vertex = (-b / (2 * a)) if a != 0 else -1.0
    r2v = _pw_r2(c, pki, t)                      # spike-bottom test

    rising = ltp > float(c[t]) and ltp > low
    stage = "D" if rec >= DONE_REC else ("E" if rising else None)
    if stage is None:
        return []

    # the two shapes compete: whichever geometry explains the path better wins
    v_ok = (r2v >= 0.78 and tightRatio <= 0.18 and sym >= 0.28
            and r2_left >= 0.70 and r2_right >= 0.70)
    u_ok = (a > 0 and r2q >= 0.70 and 0.22 * n <= vertex <= 0.82 * n
            and wideRatio >= 0.25 and r2_left >= 0.55)
    if v_ok and u_ok:
        if r2v >= r2q:
            u_ok = False
        else:
            v_ok = False

    common = {"st": stage, "lo": _r(low), "hi": _r(pk),
              "key": _r(low + 0.5 * fall), "dd": _r(dd, 1), "rc": _r(rec * 100, 1)}

    if v_ok:
        sc = (18 * _clip01(dd / (2 * cf["declMin"]))
              + 18 * _clip01(sym)
              + 24 * _clip01((r2v - 0.75) / 0.25)
              + 18 * _clip01(rec / DONE_REC)
              + 22 * (1 - _clip01(tightRatio / 0.18)))
        if sc >= MIN_SCORE:
            return [dict(common, p="V", sc=int(round(sc)), bb=baseTight)]

    if u_ok:
        base_l = l[l <= wideBand]
        flat = 1 - _clip01(float(base_l.std()) / max(0.10 * fall, 1e-9))
        vsym = 1 - _clip01(abs(vertex - n / 2) / (n / 2))
        sc = (18 * _clip01(dd / (2 * cf["declMin"]))
              + 30 * _clip01((r2q - 0.68) / 0.28)
              + 18 * _clip01(rec / DONE_REC)
              + 22 * flat
              + 12 * vsym)
        if sc >= MIN_SCORE:
            return [dict(common, p="U", sc=int(round(sc)), bb=baseWide)]

    return []


# ===============================================================
# 3.  Flag  (bull / bear)
# ===============================================================
def _detect_flag(o, h, l, c, ltp, cf):
    n = len(c)
    best = None
    maxFlag = min(cf["flagMax"], max(3, n // 3))

    for flagLen in range(3, maxFlag + 1):
        fs = n - flagLen                          # flag starts here
        if fs < 5:
            break
        fh = float(h[fs:].max())
        fl = float(l[fs:].min())
        if fl <= 0:
            continue
        fRange = (fh - fl) / fl * 100.0
        slope, _, r2f = _linfit(c[fs:])
        fMean = float(c[fs:].mean())
        slopePct = slope / fMean * 100.0 if fMean else 0.0

        for poleLen in cf["poleLens"]:
            ps = fs - poleLen
            if ps < 0:
                continue
            pole_o = float(c[ps])
            pole_x = float(c[fs - 1])
            if pole_o <= 0:
                continue
            move = (pole_x - pole_o) / pole_o * 100.0
            poleHi = float(h[ps:fs].max())
            poleLo = float(l[ps:fs].min())
            height = poleHi - poleLo
            if height <= 0:
                continue

            # ---------- BULL flag ----------
            if move >= cf["poleMin"]:
                retr = (poleHi - fl) / height
                if (0.04 <= retr <= 0.62 and fRange <= 0.50 * abs(move)
                        and slopePct <= 0.20 and fh <= poleHi * 1.03):
                    stage = "D" if ltp > fh else ("E" if ltp >= fl * 0.98 else None)
                    if stage:
                        sc = (26 * _clip01(move / (2 * cf["poleMin"]))
                              + 24 * (1 - _clip01(fRange / (0.50 * abs(move))))
                              + 18 * (1 - _clip01(abs(retr - 0.35) / 0.35))
                              + 16 * _clip01(r2f)
                              + 16 * _clip01((0.20 - slopePct) / 0.60))
                        if sc >= MIN_SCORE and (best is None or sc > best["sc"]):
                            best = {"p": "FLAG", "d": "U", "st": stage,
                                    "sc": int(round(sc)), "lo": _r(fl), "hi": _r(fh),
                                    "key": _r(fh), "pl": _r(move, 1),
                                    "rt": _r(retr * 100, 1), "fb": flagLen}

            # ---------- BEAR flag ----------
            if move <= -cf["poleMin"]:
                retr = (fh - poleLo) / height
                if (0.04 <= retr <= 0.62 and fRange <= 0.50 * abs(move)
                        and slopePct >= -0.20 and fl >= poleLo * 0.97):
                    stage = "D" if ltp < fl else ("E" if ltp <= fh * 1.02 else None)
                    if stage:
                        sc = (26 * _clip01(abs(move) / (2 * cf["poleMin"]))
                              + 24 * (1 - _clip01(fRange / (0.50 * abs(move))))
                              + 18 * (1 - _clip01(abs(retr - 0.35) / 0.35))
                              + 16 * _clip01(r2f)
                              + 16 * _clip01((slopePct + 0.20) / 0.60))
                        if sc >= MIN_SCORE and (best is None or sc > best["sc"]):
                            best = {"p": "FLAG", "d": "D", "st": stage,
                                    "sc": int(round(sc)), "lo": _r(fl), "hi": _r(fh),
                                    "key": _r(fl), "pl": _r(move, 1),
                                    "rt": _r(retr * 100, 1), "fb": flagLen}

    return [best] if best else []


# ===============================================================
# 4 + 5.  Ascending / Descending triangle
# ===============================================================
def _flatness(vals, tol):
    """1.0 = perfectly flat, 0.0 = at tolerance. None = fails tolerance."""
    m = float(np.mean(vals))
    if m <= 0:
        return None
    spread = (float(np.max(vals)) - float(np.min(vals))) / m * 100.0
    if spread > tol:
        return None
    return 1 - _clip01(spread / tol)


def _detect_triangles(o, h, l, c, ltp, cf):
    n = len(c)
    if n < 15:
        return []
    ph, pl = _pivots(h, l, k=2)
    out = []
    tol = cf["flatTol"]

    # ---------- ASCENDING : flat resistance + rising lows ----------
    if len(ph) >= 2 and len(pl) >= 2:
        hi_idx = ph[-3:] if len(ph) >= 3 else ph[-2:]
        lo_idx = pl[-3:] if len(pl) >= 3 else pl[-2:]
        hv = np.array([h[i] for i in hi_idx], dtype=float)
        lv = np.array([l[i] for i in lo_idx], dtype=float)
        flat = _flatness(hv, tol)
        if flat is not None and lv[0] > 0:
            slope, icpt, r2l = _linfit(lv)
            rise = (lv[-1] - lv[0]) / lv[0] * 100.0
            res = float(hv.mean())
            if slope > 0 and rise >= tol * 0.8 and lv[-1] < res:
                gap0 = (res - lv[0]) / res
                gap1 = (res - lv[-1]) / res
                if gap1 < gap0:
                    stage = "D" if ltp > res * 1.002 else ("E" if ltp >= lv[-1] * 0.97 else None)
                    if stage:
                        sc = (28 * flat
                              + 24 * _clip01(r2l)
                              + 18 * _clip01(1 - gap1 / max(gap0, 1e-9))
                              + 15 * _clip01((len(hi_idx) + len(lo_idx) - 4) / 2)
                              + 15 * _clip01(rise / (3 * tol)))
                        if sc >= MIN_SCORE:
                            out.append({"p": "ASCT", "st": stage, "sc": int(round(sc)),
                                        "lo": _r(float(lv.min())), "hi": _r(res),
                                        "key": _r(res), "lv": _r(res),
                                        "sp": _r(rise, 1), "tc": len(hi_idx) + len(lo_idx)})

    # ---------- DESCENDING : flat support + falling highs ----------
    if len(ph) >= 2 and len(pl) >= 2:
        hi_idx = ph[-3:] if len(ph) >= 3 else ph[-2:]
        lo_idx = pl[-3:] if len(pl) >= 3 else pl[-2:]
        hv = np.array([h[i] for i in hi_idx], dtype=float)
        lv = np.array([l[i] for i in lo_idx], dtype=float)
        flat = _flatness(lv, tol)
        if flat is not None and hv[0] > 0:
            slope, icpt, r2h = _linfit(hv)
            drop = (hv[-1] - hv[0]) / hv[0] * 100.0
            sup = float(lv.mean())
            if slope < 0 and drop <= -tol * 0.8 and hv[-1] > sup:
                gap0 = (hv[0] - sup) / sup
                gap1 = (hv[-1] - sup) / sup
                if gap1 < gap0:
                    stage = "D" if ltp < sup * 0.998 else ("E" if ltp <= hv[-1] * 1.03 else None)
                    if stage:
                        sc = (28 * flat
                              + 24 * _clip01(r2h)
                              + 18 * _clip01(1 - gap1 / max(gap0, 1e-9))
                              + 15 * _clip01((len(hi_idx) + len(lo_idx) - 4) / 2)
                              + 15 * _clip01(abs(drop) / (3 * tol)))
                        if sc >= MIN_SCORE:
                            out.append({"p": "DESCT", "st": stage, "sc": int(round(sc)),
                                        "lo": _r(sup), "hi": _r(float(hv.max())),
                                        "key": _r(sup), "lv": _r(sup),
                                        "sp": _r(drop, 1), "tc": len(hi_idx) + len(lo_idx)})

    return out


# ===============================================================
# public entry point
# ===============================================================
def detect(df, tf, ltp):
    """
    df  : completed-bar OHLC DataFrame for ONE timeframe (oldest -> newest)
    tf  : 'd' | 'w' | 'm'
    ltp : live last traded price (drives EARLY vs DONE)
    """
    cf = TF_CFG[tf]
    if df is None or len(df) < cf["minBars"] or ltp is None or ltp <= 0:
        return []

    d = df.tail(cf["look"])
    o = d["Open"].to_numpy(dtype=float)
    h = d["High"].to_numpy(dtype=float)
    l = d["Low"].to_numpy(dtype=float)
    c = d["Close"].to_numpy(dtype=float)
    if not np.isfinite(c).all() or not np.isfinite(h).all() or not np.isfinite(l).all():
        m = np.isfinite(c) & np.isfinite(h) & np.isfinite(l) & np.isfinite(o)
        o, h, l, c = o[m], h[m], l[m], c[m]
        if len(c) < cf["minBars"]:
            return []

    res = []
    try:
        res += _detect_vu(o, h, l, c, float(ltp), cf)
    except Exception:
        pass
    try:
        res += _detect_flag(o, h, l, c, float(ltp), cf)
    except Exception:
        pass
    try:
        res += _detect_triangles(o, h, l, c, float(ltp), cf)
    except Exception:
        pass

    res.sort(key=lambda x: -x["sc"])
    return res[:3]
