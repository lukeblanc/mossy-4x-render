from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class BacktestResult:
    trades: int
    net_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    profit_factor: float


def ema(values: list[float], length: int) -> list[float]:
    if not values:
        return []
    k = 2 / (length + 1)
    out = [values[0]]
    for p in values[1:]:
        out.append(p * k + out[-1] * (1 - k))
    return out


def rsi(values: list[float], length: int) -> float:
    if length <= 0 or len(values) < length + 2:
        return 50.0
    gains = []
    losses = []
    for i in range(1, length + 1):
        d = values[-i] - values[-(i + 1)]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(highs: list[float], lows: list[float], closes: list[float], length: int) -> float:
    if length <= 0 or len(highs) < length + 2:
        return 0.0
    trs = []
    for i in range(1, length + 1):
        h = highs[-i]
        l = lows[-i]
        prev = closes[-(i + 1)]
        trs.append(max(h - l, abs(h - prev), abs(l - prev)))
    return sum(trs) / length


def adx(highs: list[float], lows: list[float], closes: list[float], length: int) -> float:
    if length <= 0 or len(highs) < length + 1:
        return 0.0
    tr_list, plus_dm, minus_dm = [], [], []
    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus = up_move if up_move > down_move and up_move > 0 else 0.0
        minus = down_move if down_move > up_move and down_move > 0 else 0.0
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)
    if len(tr_list) < length:
        return 0.0
    tr_s = sum(tr_list[:length])
    p_s = sum(plus_dm[:length])
    m_s = sum(minus_dm[:length])

    def _dx(tv: float, pv: float, mv: float) -> float:
        if tv <= 0:
            return 0.0
        pdi = 100.0 * pv / tv
        mdi = 100.0 * mv / tv
        d = pdi + mdi
        return 100.0 * abs(pdi - mdi) / d if d > 0 else 0.0

    dxs = [_dx(tr_s, p_s, m_s)]
    for i in range(length, len(tr_list)):
        tr_s = tr_s - (tr_s / length) + tr_list[i]
        p_s = p_s - (p_s / length) + plus_dm[i]
        m_s = m_s - (m_s / length) + minus_dm[i]
        dxs.append(_dx(tr_s, p_s, m_s))
    if not dxs:
        return 0.0
    initial = min(length, len(dxs))
    val = sum(dxs[:initial]) / initial
    for d in dxs[initial:]:
        val = ((val * (length - 1)) + d) / length
    return val


def _summarize(pnls: list[float]) -> BacktestResult:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return BacktestResult(
        trades=len(pnls),
        net_pnl=sum(pnls),
        win_rate=(len(wins) / len(pnls) * 100.0) if pnls else 0.0,
        avg_win=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        max_drawdown=mdd,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
    )


def run_backtest(candles: Iterable[dict], *, enhanced: bool) -> BacktestResult:
    candles = list(candles)
    closes = [float(c["c"]) for c in candles]
    highs = [float(c["h"]) for c in candles]
    lows = [float(c["l"]) for c in candles]
    fast, slow, rsi_len, atr_len = 12, 26, 14, 14
    min_bars = max(fast, slow, rsi_len, atr_len) + 2
    pnls: list[float] = []
    for i in range(min_bars, len(candles) - 1):
        c = closes[: i + 1]
        h = highs[: i + 1]
        l = lows[: i + 1]
        ef = ema(c, fast)
        es = ema(c, slow)
        r = rsi(c, rsi_len)
        a = atr(h, l, c, atr_len)
        x = adx(h, l, c, atr_len)
        cross_up = ef[-2] <= es[-2] and ef[-1] > es[-1]
        cross_down = ef[-2] >= es[-2] and ef[-1] < es[-1]
        spread_ok = abs(ef[-1] - es[-1]) >= (a * 0.05)

        buy = cross_up and r > 52.0 and a >= 0.00005
        sell = cross_down and r < 48.0 and a >= 0.00005
        if enhanced:
            buy = buy and x >= 18.0 and spread_ok
            sell = sell and x >= 18.0 and spread_ok

        if not (buy or sell):
            continue

        entry = closes[i]
        exit_ = closes[i + 1]
        pnl = (exit_ - entry) if buy else (entry - exit_)
        pnls.append(pnl)

    return _summarize(pnls)
