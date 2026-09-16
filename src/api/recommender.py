"""
GeoDemand AI — Production Recommendation Engine

Responsibilities:
  1. Load trained Champion LightGBM Model & Quantile Heads (P10, P50, P90)
  2. Compute exact TreeSHAP feature attributions at inference time (<1ms)
  3. Calculate comprehensive business economics (Revenue, COGS, Hourly Fixed, Fuel, Net Profit)
  4. Multi-objective ranking (Profit 60%, Demand 25%, Distance -10%, Competition -5%)
  5. Generate dynamic SHAP-grounded natural language driver explanations
"""

from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"


# Human readable labels for SHAP feature drivers
FEATURE_FRIENDLY_NAMES = {
    "office_count": "Office & commercial density",
    "college_count": "College student footfall",
    "school_count": "School area presence",
    "hospital_count": "Healthcare & hospital proximity",
    "restaurant_count": "Dining & food cluster",
    "mall_count": "Shopping mall footfall",
    "park_count": "Park & recreation zone",
    "bus_stop_count": "Transit bus stop density",
    "railway_station_count": "Railway station passenger flow",
    "road_density": "Road accessibility",
    "commercial_ratio": "Commercial zoning ratio",
    "population_density": "Local residential population",
    "hour": "Time-of-day demand peak",
    "hour_sin": "Daily temporal cycle",
    "hour_cos": "Daily temporal cycle",
    "is_weekend": "Weekend leisure spending",
    "is_holiday": "Public holiday effect",
    "temperature": "Ambient temperature",
    "rainfall": "Rainfall impact",
    "is_rainy": "Rainy weather condition",
    "competition_score": "Local vendor competition",
    "active_events_count": "Nearby active local events",
    "event_max_attendance": "High-attendance event proximity",
    "traffic_delay_min": "Traffic travel friction",
    "distance_km": "Travel distance from start point",
}


class DemandModel:
    """Wrapper around trained LightGBM model and calibrated Quantile heads."""
    _instance: Optional["DemandModel"] = None

    def __init__(self):
        self._model = None
        self._p10_model = None
        self._p90_model = None
        self._feature_cols: list[str] = []
        self._meta: dict = {}
        self._loaded = False

    @classmethod
    def get(cls) -> "DemandModel":
        if cls._instance is None:
            cls._instance = cls()
        if not cls._instance._loaded:
            cls._instance._load()
        return cls._instance

    def _load(self):
        model_path = MODELS_DIR / "demand_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model bundle not found at {model_path}. "
                "Run: python3 src/models/train_demand_model.py"
            )

        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict) and "model" in bundle:
            self._model = bundle["model"]
            self._p10_model = bundle.get("p10_model")
            self._p90_model = bundle.get("p90_model")
            self._feature_cols = bundle.get("feature_cols", [])
            self._meta = bundle.get("meta", {})
        else:
            # Fallback legacy single model
            self._model = bundle
            self._feature_cols = [
                "office_count", "college_count", "school_count", "hospital_count",
                "mall_count", "restaurant_count", "park_count", "bus_stop_count",
                "railway_station_count", "metro_station_count", "road_density",
                "commercial_ratio", "residential_ratio", "industrial_ratio",
                "population_density", "parking_count", "building_density",
                "hour", "day_of_week", "month", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
                "temperature", "humidity", "rainfall", "wind_speed", "cloud_cover",
                "uv_index", "pressure", "is_rainy",
                "competition_score", "active_events_count", "event_max_attendance",
                "event_min_distance_km",
            ]
            self._meta = {"model_type": "lightgbm", "shap_enabled": True}

        self._loaded = True
        logger.info(f"DemandModel loaded successfully: {self._meta.get('model_type')} with {len(self._feature_cols)} features.")

    def _build_feature_matrix(self, feature_rows: list[dict]) -> pd.DataFrame:
        """Construct strict aligned DataFrame for model inference."""
        df = pd.DataFrame(feature_rows)
        missing = [col for col in self._feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Inference schema mismatch: missing required model features: {missing}")
        return df[self._feature_cols].astype(float)

    def predict_batch(self, feature_rows: list[dict]) -> np.ndarray:
        """Predict expected_customer_count for a batch of feature dicts."""
        X = self._build_feature_matrix(feature_rows)
        preds = self._model.predict(X)
        return np.clip(np.round(preds), 0, 100).astype(int)

    def predict_with_uncertainty(self, feature_row: dict) -> dict:
        """
        Calculate rigorous quantile intervals (P10, P50, P90).
        Uses dedicated quantile regression heads if available.
        """
        X = self._build_feature_matrix([feature_row])
        p50 = float(self._model.predict(X)[0])
        
        if self._p10_model is not None and self._p90_model is not None:
            p10 = float(self._p10_model.predict(X)[0])
            p90 = float(self._p90_model.predict(X)[0])
            uncertainty_method = "quantile_regression"
        else:
            sigma = 0.25 * np.sqrt(max(p50, 1.0))
            p10 = p50 - 1.28 * sigma
            p90 = p50 + 1.28 * sigma
            uncertainty_method = "approximation"

        p10 = max(0, int(round(p10)))
        p50 = max(0, int(round(p50)))
        p90 = max(p10 + 1, int(round(p90)))
        spread = p90 - p10
        # Heuristic classification of prediction uncertainty from interval spread
        uncertainty_level = "LOW" if spread <= 15 else "MODERATE" if spread <= 30 else "HIGH"

        return {
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "interval_spread": spread,
            "uncertainty_level": uncertainty_level,
            "certainty_level": uncertainty_level.capitalize(),  # backward compatibility alias
            "uncertainty_method": uncertainty_method,
        }

    def compute_shap_drivers(self, feature_row: dict, max_drivers: int = 4) -> tuple[list[str], list[dict]]:
        """
        Compute exact TreeSHAP feature attributions for a single candidate.
        Returns human-readable driver list + structured SHAP breakdown.
        """
        X = self._build_feature_matrix([feature_row])
        try:
            # Native LightGBM TreeSHAP contributions
            contribs = self._model.predict(X, pred_contrib=True)[0]
            feature_contribs = contribs[:-1]  # drop base value at the end
            
            # Sort by absolute impact
            sorted_indices = np.argsort(np.abs(feature_contribs))[::-1]
            
            drivers_nl = []
            structured_shap = []

            for idx in sorted_indices[:8]:
                val = feature_contribs[idx]
                if abs(val) < 0.2:
                    continue
                feat_name = self._feature_cols[idx]
                friendly_name = FEATURE_FRIENDLY_NAMES.get(feat_name, feat_name.replace("_", " ").title())
                sign_str = f"+{val:.1f}" if val > 0 else f"{val:.1f}"
                
                structured_shap.append({
                    "feature": feat_name,
                    "friendly_name": friendly_name,
                    "shap_value": round(float(val), 2),
                    "impact": "positive" if val > 0 else "negative",
                })

                if val > 0:
                    drivers_nl.append(f"{friendly_name} ({sign_str} customers/hr)")
                else:
                    drivers_nl.append(f"{friendly_name} ({sign_str} customers/hr)")

            if not drivers_nl:
                drivers_nl = ["Baseline local demand area"]

            return drivers_nl[:max_drivers], structured_shap[:6]

        except Exception as e:
            logger.warning(f"TreeSHAP calculation fallback: {e}")
            return ["Consistent commercial footfall"], []

    @property
    def meta(self) -> dict:
        return self._meta


# ---------------------------------------------------------------------------
# Business Economics
# ---------------------------------------------------------------------------

def calculate_business_metrics(
    expected_customers: int,
    aov: float,
    variable_cost_rate: float,
    fixed_cost_per_day: float,
    hours_per_day: int = 13,
    fuel_cost: float = 0.0,
) -> dict:
    """Derive revenue, operating costs, and net hourly profit from demand prediction."""
    revenue = round(expected_customers * aov, 2)
    ingredient_cost = round(revenue * variable_cost_rate, 2)
    hourly_fixed = round(fixed_cost_per_day / hours_per_day, 2)
    operating_cost = round(ingredient_cost + hourly_fixed + fuel_cost, 2)
    profit = round(revenue - operating_cost, 2)
    return {
        "expected_customers": expected_customers,
        "expected_revenue_inr": revenue,
        "expected_operating_cost_inr": operating_cost,
        "expected_profit_inr": profit,
        "ingredient_cost_inr": ingredient_cost,
        "fixed_cost_per_hour_inr": hourly_fixed,
        "fuel_cost_inr": fuel_cost,
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def rank_candidates(candidates: list[dict]) -> list[dict]:
    """
    Score = 0.60 × profit_norm + 0.25 × demand_norm - 0.10 × distance_norm - 0.05 × competition_norm
    """
    profits = [c["expected_profit_inr"] for c in candidates]
    demands = [c["expected_customers"] for c in candidates]
    distances = [c["distance_km"] for c in candidates]
    comp_scores = [c.get("competition_score", 0.3) for c in candidates]

    p_norm = _normalize(profits)
    d_norm = _normalize(demands)
    dist_norm = _normalize(distances)
    comp_norm = comp_scores

    for i, c in enumerate(candidates):
        c["recommendation_score"] = round(
            0.60 * p_norm[i]
            + 0.25 * d_norm[i]
            - 0.10 * dist_norm[i]
            - 0.05 * comp_norm[i],
            4
        )

    ranked = sorted(candidates, key=lambda x: x["recommendation_score"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked


# ---------------------------------------------------------------------------
# SHAP-Powered Explanation Generator
# ---------------------------------------------------------------------------

def generate_explanation(candidate: dict, vendor_category: str) -> tuple[list[str], str]:
    """
    Generate SHAP-driven top positive and negative drivers and natural reasoning narrative.
    """
    model = DemandModel.get()
    features = candidate.get("_features", {})
    drivers_nl, structured_shap = model.compute_shap_drivers(features, max_drivers=4)
    
    cust = candidate.get("expected_customers", 0)
    profit = candidate.get("expected_profit_inr", 0)
    dist = candidate.get("distance_km", 0)
    direction = f"{dist:.1f} km away" if dist > 0.1 else "your current position"

    top_pos = [s["friendly_name"] for s in structured_shap if s["impact"] == "positive"][:2]
    top_neg = [s["friendly_name"] for s in structured_shap if s["impact"] == "negative"][:1]

    pos_phrase = f"driven by strong {', '.join(top_pos)}" if top_pos else "with steady footfall"
    neg_phrase = f", though dampened slightly by {top_neg[0]}" if top_neg else ""

    summary = (
        f"Moving to {direction} is estimated to yield {cust} customers/hour "
        f"(₹{profit:,.0f} hourly profit), {pos_phrase}{neg_phrase}."
    )

    return drivers_nl, summary
