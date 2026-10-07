"""
GeoDemand AI — Inference & TreeSHAP Explainability Engine

Member 3 — Machine Learning & MLOps

Provides production inference interface:
- Point prediction (expected_customer_count)
- Rigorous prediction interval (P10 <= P50 <= P90)
- TreeSHAP local explanations (top positive and negative drivers)
- Human-readable explanation generation for API (Member 4)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = ROOT / "models"
ARTIFACT_DIR = MODELS_DIR / "lgbm_v1"


class DemandModelPredictor:
    """
    Production-grade inference and explainability engine for next-hour demand forecasting.
    Loads models trained via train_demand_model.py.
    """

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, bundle_path: Optional[Union[str, Path]] = None):
        self.model_dir = Path(model_dir) if model_dir else ARTIFACT_DIR
        self.bundle_path = Path(bundle_path) if bundle_path else (MODELS_DIR / "demand_model.pkl")

        self.model = None
        self.p10_model = None
        self.p90_model = None
        self.feature_cols: List[str] = []
        self.label_encoders: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.feature_schema: Dict[str, Any] = {}

        self._load()

    def _load(self) -> None:
        """Load artifacts from directory or fallback to bundle."""
        if (self.model_dir / "model.joblib").exists():
            logger.info(f"Loading champion model from {self.model_dir / 'model.joblib'}")
            self.model = joblib.load(self.model_dir / "model.joblib")

            if (self.model_dir / "p10_model.joblib").exists():
                self.p10_model = joblib.load(self.model_dir / "p10_model.joblib")
            if (self.model_dir / "p90_model.joblib").exists():
                self.p90_model = joblib.load(self.model_dir / "p90_model.joblib")
            if (self.model_dir / "label_encoders.joblib").exists():
                self.label_encoders = joblib.load(self.model_dir / "label_encoders.joblib")

            meta_file = self.model_dir / "metadata.json"
            if not meta_file.exists():
                meta_file = self.model_dir / "model_metadata.json"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    self.metadata = json.load(f)
                    self.feature_cols = self.metadata.get("feature_columns", [])

            schema_file = self.model_dir / "feature_schema.json"
            if schema_file.exists():
                with open(schema_file, "r") as f:
                    self.feature_schema = json.load(f)
                    if not self.feature_cols:
                        self.feature_cols = self.feature_schema.get("feature_columns", [])
        elif self.bundle_path.exists():
            logger.info(f"Loading champion model from bundle {self.bundle_path}")
            bundle = joblib.load(self.bundle_path)
            self.model = bundle.get("model")
            self.p10_model = bundle.get("p10_model")
            self.p90_model = bundle.get("p90_model")
            self.feature_cols = bundle.get("feature_cols", [])
            self.label_encoders = bundle.get("label_encoders", {})
            self.metadata = bundle.get("meta", {})
        else:
            raise FileNotFoundError(
                f"No model artifacts found at {self.model_dir} or {self.bundle_path}. "
                "Run `python scripts/train_model.py` first."
            )

    def preprocess(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Align and preprocess input features to match the exact schema expected by the model.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Encode categoricals using saved encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                known_classes = set(le.classes_)
                df[col] = df[col].apply(lambda v: le.transform([v])[0] if v in known_classes else 0)

        # Convert booleans to int
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

        # Ensure all expected feature columns exist (fill missing with 0.0)
        missing_cols = [c for c in self.feature_cols if c not in df.columns]
        if missing_cols:
            logger.debug(f"Imputing missing feature columns with 0.0: {missing_cols}")
            for c in missing_cols:
                df[c] = 0.0

        # Select and order columns strictly
        X = df[self.feature_cols].astype(float)
        return X

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> np.ndarray:
        """
        Predict expected_customer_count.
        Non-negative clipped.
        """
        X = self.preprocess(data)
        raw = self.model.predict(X)
        return np.clip(raw, 0, None)

    def predict_with_interval(
        self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, float]]:
        """
        Predict expected customer count with prediction interval (P10 <= P50 <= P90).
        Enforces monotonicity and non-negativity.
        """
        X = self.preprocess(data)
        p50 = np.clip(self.model.predict(X), 0, None)

        if self.p10_model is not None and self.p90_model is not None:
            p10 = np.clip(self.p10_model.predict(X), 0, None)
            p90 = np.clip(self.p90_model.predict(X), 0, None)
            # Enforce monotonicity: P10 <= P50 <= P90
            p10 = np.minimum(p10, p50)
            p90 = np.maximum(p50, p90)
        else:
            # Fallback heuristic interval (+/- 25%) if quantile models missing
            p10 = np.clip(p50 * 0.75, 0, None)
            p90 = np.clip(p50 * 1.25, 0, None)

        results = []
        for i in range(len(p50)):
            p10_val = round(float(p10[i]), 2)
            p50_val = round(float(p50[i]), 2)
            p90_val = round(float(p90[i]), 2)
            results.append({
                "expected_customers": p50_val,
                "p10": p10_val,
                "p50": p50_val,
                "p90": p90_val,
                "prediction_range": round(p90_val - p10_val, 2),
            })
        return results

    def explain(
        self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        TreeSHAP local explanation for each sample.
        Returns top positive and top negative contributing features.
        """
        X = self.preprocess(data)
        preds = np.clip(self.model.predict(X), 0, None)

        # LightGBM native pred_contrib: shape (n_samples, n_features + 1)
        contribs = self.model.predict(X, pred_contrib=True)

        explanations = []
        for i in range(len(X)):
            sample_contribs = contribs[i][:-1]  # exclude bias
            base_bias = float(contribs[i][-1])
            pred_val = float(preds[i])

            features_and_contribs = list(zip(self.feature_cols, sample_contribs))

            # Positive factors (increase demand)
            positives = [(feat, val) for feat, val in features_and_contribs if val > 0]
            positives.sort(key=lambda x: x[1], reverse=True)

            # Negative factors (decrease demand)
            negatives = [(feat, val) for feat, val in features_and_contribs if val < 0]
            negatives.sort(key=lambda x: x[1])  # most negative first

            top_pos = [feat for feat, _ in positives[:top_k]]
            top_neg = [feat for feat, _ in negatives[:top_k]]

            formatted_text = self.format_shap_explanation(
                predicted_count=pred_val,
                top_positive=top_pos,
                top_negative=top_neg,
            )

            explanations.append({
                "predicted_customer_count": round(pred_val, 2),
                "base_bias": round(base_bias, 4),
                "top_positive_factors": top_pos,
                "top_negative_factors": top_neg,
                "formatted_explanation": formatted_text,
                "feature_contributions": {feat: round(float(val), 4) for feat, val in features_and_contribs},
            })

        return explanations

    @staticmethod
    def format_shap_explanation(
        predicted_count: float,
        top_positive: List[str],
        top_negative: List[str],
    ) -> str:
        """
        Format SHAP explanation strictly according to Member 3 contract:
        Example:
        Prediction: 34 customers/hour

        Top positive factors:
        + commercial_density
        + hour_of_day
        + nearby_transport

        Top negative factors:
        - heavy_rain
        - low_activity
        """
        lines = [f"Prediction: {int(round(predicted_count))} customers/hour", ""]

        if top_positive:
            lines.append("Top positive factors:")
            for feat in top_positive:
                lines.append(f"+ {feat}")
            lines.append("")

        if top_negative:
            lines.append("Top negative factors:")
            for feat in top_negative:
                lines.append(f"- {feat}")

        return "\n".join(lines).strip()
