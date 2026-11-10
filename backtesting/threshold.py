"""Backtest that only takes trades the model rates above a chosen probability."""

from typing import Any, Dict, Optional

import numpy as np


def threshold_backtest(pnl: np.ndarray, probabilities: np.ndarray, threshold: float,
                       initial_capital: float) -> Dict[str, Any]:
    """
    Equity curve and stats when taking trades (in order) whose probability is at least `threshold`.
    Once the account reaches zero it is wiped out: no further trades are taken.
    """
    pnl = np.asarray(pnl, dtype=float)
    taken = np.asarray(probabilities, dtype=float) >= threshold
    equity = initial_capital + np.concatenate([[0.0], np.cumsum(np.where(taken, pnl, 0.0))])

    ruined_after: Optional[int] = None
    broke = np.nonzero(equity[1:] <= 0)[0]
    if broke.size:
        last = int(broke[0])                # index of the trade that emptied the account
        taken[last + 1:] = False
        equity[last + 1:] = 0.0
        ruined_after = int(taken[:last + 1].sum())

    peak = np.maximum.accumulate(equity)
    drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
    trades = int(taken.sum())
    wins = int(np.sum(pnl[taken] > 0))
    return {
        "equity_curve": equity.tolist(),
        "final_capital": float(equity[-1]),
        "trades": trades,
        "win_rate": (wins / trades * 100.0) if trades else 0.0,
        "max_drawdown_pct": float(drawdown.max() * 100.0),
        "ruined_after_trades": ruined_after,
    }
