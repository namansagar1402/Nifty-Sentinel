import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

from .synthetic_data import generate_synthetic_options_data

TRAINING_SEED = 42
TRAINING_SAMPLES = 5000

class OptionOutcomePredictor:
    def __init__(self, training_seed: int = TRAINING_SEED):
        self.training_seed = training_seed
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42
        )
        self.is_trained = False
        self.feature_names = ["delta", "gamma", "theta_burn_pct", "vix", "days_to_expiry", "moneyness"]
        self.feature_importances_ = None
        
    def train(self):
        """Trains the Random Forest model on synthetic historical data."""
        df = generate_synthetic_options_data(n_samples=TRAINING_SAMPLES, seed=self.training_seed)
        
        X = df[self.feature_names]
        y = df["is_profitable"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        self.is_trained = True
        self.feature_importances_ = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return acc

    def predict_trade(self, delta: float, gamma: float, theta_burn_pct: float, vix: float, dte: int, moneyness: float) -> int:
        """
        Returns 1 if the model predicts the trade will be profitable, 0 otherwise.
        """
        if not self.is_trained:
            self.train()
            
        features = pd.DataFrame([{
            "delta": delta,
            "gamma": gamma,
            "theta_burn_pct": theta_burn_pct,
            "vix": vix,
            "days_to_expiry": dte,
            "moneyness": moneyness
        }])
        
        return self.model.predict(features)[0]

    def predict_batch(self, df_features: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            self.train()
            
        return self.model.predict(df_features[self.feature_names])

# Singleton instance
_predictor_instance = None

def get_predictor() -> OptionOutcomePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = OptionOutcomePredictor()
        _predictor_instance.train()
    return _predictor_instance
