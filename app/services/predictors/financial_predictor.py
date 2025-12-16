import joblib
import pandas as pd
import numpy as np
from app.core.config import settings


class FinancialRiskPredictor:
    _instance = None
    _model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            model_path = f"{settings.MODEL_STORAGE_PATH}/finance/credit_risk_xgboost_v1.0.0.pkl"
            print(f"--- [LOADING] Carregando modelo de: {model_path} ---")
            cls._model = joblib.load(model_path)
        return cls._instance

    def predict_risk_probability(self, input_data: dict) -> float:

        features = ['income', 'debt_ratio', 'loan_amount', 'late_payments_count']
        df_input = pd.DataFrame([input_data], columns=features)

        probability = self._model.predict_proba(df_input)[0][1]

        return float(probability)