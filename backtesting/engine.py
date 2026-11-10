import pandas as pd
import numpy as np

from ml_engine.synthetic_data import generate_synthetic_options_data
from ml_engine.classifier import get_predictor

# Must differ from the classifier's TRAINING_SEED so the backtest runs on
# trades the model has never seen (out-of-sample).
BACKTEST_SEED = 7

class BacktestEngine:
    def __init__(self, n_trades: int = 1000, initial_capital: float = 100000.0,
                 data_seed: int = BACKTEST_SEED):
        self.n_trades = n_trades
        self.data_seed = data_seed
        self.initial_capital = initial_capital
        
    def run_backtest(self) -> dict:
        """
        Runs a backtest comparing three strategies over synthetic historical trades.
        1. Baseline: Take all trades.
        2. Rule-Based: Drop trades with Theta Burn > 10% or VIX > 25.
        3. ML-Based: Take trades approved by the Random Forest classifier.
        """
        predictor = get_predictor()
        if self.data_seed == predictor.training_seed:
            raise ValueError("Backtest data_seed must differ from the model's training seed "
                             "(otherwise the backtest re-scores training data).")

        # Out-of-sample synthetic trades: same generating process, independent draw.
        df = generate_synthetic_options_data(n_samples=self.n_trades, seed=self.data_seed)
        ml_predictions = predictor.predict_batch(df)
        
        # Win is +5000, Loss is -3000 (just hypothetical PnL per trade)
        profit_pnl = 5000.0
        loss_pnl = -3000.0
        
        baseline_equity = [self.initial_capital]
        rule_equity = [self.initial_capital]
        ml_equity = [self.initial_capital]
        
        baseline_trades = 0
        rule_trades = 0
        ml_trades = 0
        
        baseline_wins = 0
        rule_wins = 0
        ml_wins = 0
        
        for i in range(self.n_trades):
            row = df.iloc[i]
            is_success = (row["is_profitable"] == 1)
            trade_pnl = profit_pnl if is_success else loss_pnl
            
            # 1. Baseline
            baseline_equity.append(baseline_equity[-1] + trade_pnl)
            baseline_trades += 1
            if is_success: baseline_wins += 1
            
            # 2. Rule-Based
            # Drop if hazardous (e.g., VIX > 25 or Theta Burn > 10%)
            is_hazardous = (row["theta_burn_pct"] > 10.0) or (row["vix"] > 25.0)
            if not is_hazardous:
                rule_equity.append(rule_equity[-1] + trade_pnl)
                rule_trades += 1
                if is_success: rule_wins += 1
            else:
                rule_equity.append(rule_equity[-1])
                
            # 3. ML-Based
            if ml_predictions[i] == 1:
                ml_equity.append(ml_equity[-1] + trade_pnl)
                ml_trades += 1
                if is_success: ml_wins += 1
            else:
                ml_equity.append(ml_equity[-1])
                
        # Calculate Max Drawdowns
        def calc_mdd(equity_curve):
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            return abs(np.min(drawdown)) * 100.0
            
        return {
            "baseline": {
                "equity_curve": baseline_equity,
                "final_capital": baseline_equity[-1],
                "trades": baseline_trades,
                "success_rate": (baseline_wins / baseline_trades * 100) if baseline_trades > 0 else 0,
                "max_drawdown": calc_mdd(baseline_equity)
            },
            "rule_based": {
                "equity_curve": rule_equity,
                "final_capital": rule_equity[-1],
                "trades": rule_trades,
                "success_rate": (rule_wins / rule_trades * 100) if rule_trades > 0 else 0,
                "max_drawdown": calc_mdd(rule_equity)
            },
            "ml_based": {
                "equity_curve": ml_equity,
                "final_capital": ml_equity[-1],
                "trades": ml_trades,
                "success_rate": (ml_wins / ml_trades * 100) if ml_trades > 0 else 0,
                "max_drawdown": calc_mdd(ml_equity)
            },
            "feature_importances": predictor.feature_importances_
        }
