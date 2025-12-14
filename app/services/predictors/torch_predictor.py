# app/services/predictors/torch_predictor.py
import torch
import numpy as np
from app.services.model_loader import loader


class HealthPredictor:

    @staticmethod
    def predict_iris(features: list) -> dict:
        model = loader.models.get("iris_neural")
        scaler = loader.scalers.get("iris_scaler")

        if not model or not scaler:
            raise RuntimeError("Modelo de saúde não está carregado.")

        # 1. Pré-processamento (StandardScaler)
        # O modelo espera 2D array: [[5.1, 3.5, 1.4, 0.2]]
        raw_data = np.array([features])
        scaled_data = scaler.transform(raw_data)

        # 2. Conversão para Tensor
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32)

        # 3. Inferência
        with torch.no_grad():
            logits = model(tensor_data)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        # Mapeamento de classes (opcional, bom ter num config)
        class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

        return {
            "class_id": int(predicted_class.item()),
            "class_name": class_names.get(int(predicted_class.item()), "Unknown"),
            "confidence": float(confidence.item()),
            "model_version": "v2.0-torch"
        }