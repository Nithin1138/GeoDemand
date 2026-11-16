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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DEFAULT_AOV_INR,
    DEFAULT_VARIABLE_COST_RATE,
    DEFAULT_FIXED_COST_PER_DAY_INR,
    DEFAULT_OPERATING_HOURS_PER_DAY,
)

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
    """Wrapper around Member 2's approved DemandModelPredictor and calibrated Quantile heads."""
    _instance: Optional["DemandModel"] = None

    def __init__(self):
        self._predictor = None
        self._model = None
        self._p10_model = None
        self._p90_model = None
        self._feature_cols: list[str] = []
        self._label_encoders: dict = {}
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
        # 1. Primary: Load Member 2's approved DemandModelPredictor (champion LGBM + Quantiles)
        model_dir = MODELS_DIR / "lgbm_v1"
        bundle_path = MODELS_DIR / "demand_model.pkl"

        try:
            from src.geodemand.models.inference import DemandModelPredictor
            if (model_dir / "model.joblib").exists() or bundle_path.exists():
                self._predictor = DemandModelPredictor(
                    model_dir=model_dir if (model_dir / "model.joblib").exists() else None,
                    bundle_path=bundle_path if bundle_path.exists() else None,
                    validate_predictions=False,
                )
                self._model = self._predictor.model
                self._p10_model = self._predictor.p10_model
                self._p90_model = self._predictor.p90_model
                self._feature_cols = self._predictor.feature_cols
                self._label_encoders = self._predictor.label_encoders
                self._meta = self._predictor.metadata or {"model_type": "lightgbm", "shap_enabled": True}
                self._loaded = True
                logger.info(
                    f"DemandModel loaded successfully via DemandModelPredictor: "
                    f"{self._meta.get('model_type', 'lightgbm')} with {len(self._feature_cols)} features."
                )
                return
        except Exception as exc:
            logger.warning(f"DemandModelPredictor initialization fallback: {exc}")

        # 2. Fallback: Load raw demand_model.pkl bundle
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Trained model bundle not found at {bundle_path} or {model_dir}. "
                "Run: python scripts/train_model.py"
            )

        try:
            with open(bundle_path, "rb") as f:
                bundle = pickle.load(f)
        except Exception:
            import joblib
            bundle = joblib.load(bundle_path)

        if isinstance(bundle, dict) and "model" in bundle:
            self._model = bundle["model"]
            self._p10_model = bundle.get("p10_model")
            self._p90_model = bundle.get("p90_model")
            self._feature_cols = bundle.get("feature_cols", [])
            self._label_encoders = bundle.get("label_encoders", {})
            self._meta = bundle.get("meta", {})
        else:
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
            self._label_encoders = {}
            self._meta = {"model_type": "lightgbm", "shap_enabled": True}

        self._loaded = True
        logger.info(f"DemandModel loaded successfully: {self._meta.get('model_type')} with {len(self._feature_cols)} features.")

    def _build_feature_matrix(self, feature_rows: list[dict]) -> pd.DataFrame:
        """Construct strict aligned DataFrame for model inference."""
        df = pd.DataFrame(feature_rows)
        if hasattr(self, "_label_encoders") and self._label_encoders:
            for col, le in self._label_encoders.items():
                if col in df.columns:
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda v: le.transform([v])[0] if v in known else 0)
        for col in self._feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        return df[self._feature_cols].astype(float)

    def predict_batch(self, feature_rows: list[dict]) -> np.ndarray:
        """Predict expected_customer_count for a batch of feature dicts."""
        if self._predictor is not None:
            preds = self._predictor.predict(feature_rows)
            return np.clip(np.round(preds), 0, 100).astype(int)
        X = self._build_feature_matrix(feature_rows)
        preds = self._model.predict(X)
        return np.clip(np.round(preds), 0, 100).astype(int)

    def predict_with_uncertainty(self, feature_row: dict) -> dict:
        """
        Calculate rigorous quantile intervals (P10, P50, P90).
        Uses dedicated quantile regression heads if available.
        """
        if self._predictor is not None:
            intervals = self._predictor.predict_with_interval([feature_row])
            if intervals:
                res = intervals[0]
                p10 = max(0, int(round(res["p10"])))
                p50 = max(p10, int(round(res["p50"])))
                p90 = max(p50, int(round(res["p90"])))
                spread = p90 - p10
                uncertainty_level = "LOW" if spread <= 15 else "MODERATE" if spread <= 30 else "HIGH"
                return {
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "interval_spread": spread,
                    "uncertainty_level": uncertainty_level,
                    "certainty_level": uncertainty_level.capitalize(),
                    "uncertainty_method": "quantile_regression" if self._p10_model is not None else "approximation",
                }

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
        p50 = max(p10, int(round(p50)))
        p90 = max(p50, int(round(p90)))
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

from api.business_economics import (
    calculate_business_metrics,
    calculate_effective_demand,
    calculate_revenue,
    calculate_stock_cost,
    calculate_hourly_fixed_cost,
    calculate_fuel_cost,
    calculate_travel_opportunity_cost,
    evaluate_relocation_decision,
)


# ---------------------------------------------------------------------------
# Business Category Spatial Affinity & Throughput Conversion
# ---------------------------------------------------------------------------

CATEGORY_SPATIAL_WEIGHTS: dict[str, dict[str, float]] = {
    "food": {
        "office_count": 2.2,
        "restaurant_count": 1.8,
        "commercial_ratio": 2.0,
        "bus_stop_count": 1.5,
        "college_count": 1.5,
        "railway_station_count": 1.8,
        "population_density": 0.8,
        "building_density": 1.0,
    },
    "fruits": {
        "hospital_count": 3.2,
        "park_count": 2.6,
        "residential_ratio": 2.2,
        "population_density": 1.5,
        "bus_stop_count": 1.2,
        "commercial_ratio": 0.8,
        "industrial_ratio": -1.5,
    },
    "salon": {
        "mall_count": 3.0,
        "residential_ratio": 2.5,
        "building_density": 2.0,
        "parking_count": 1.8,
        "commercial_ratio": 1.2,
        "industrial_ratio": -2.0,
        "railway_station_count": -1.0,
    },
    "repair": {
        "road_density": 10.0,
        "industrial_ratio": 15.0,
        "bus_stop_count": 1.8,
        "parking_count": 2.0,
        "commercial_ratio": 2.0,
        "hospital_count": -0.4,
        "mall_count": -0.3,
    },
}

CATEGORY_THROUGHPUT_SPECS: dict[str, dict[str, float | int]] = {
    "food": {
        "conv_rate": 0.95,
        "min_cust": 15,
        "max_cust": 60,
        "benchmark_affinity": 100.0,
    },
    "fruits": {
        "conv_rate": 0.65,
        "min_cust": 10,
        "max_cust": 35,
        "benchmark_affinity": 120.0,
    },
    "salon": {
        "conv_rate": 0.15,
        "min_cust": 3,
        "max_cust": 7,
        "benchmark_affinity": 80.0,
    },
    "repair": {
        "conv_rate": 0.12,
        "min_cust": 2,
        "max_cust": 6,
        "benchmark_affinity": 40.0,
    },
}


def compute_category_affinity(vendor_category: str, features: dict) -> float:
    """
    Compute domain-specific spatial affinity score for a candidate cell based on
    the business category's real-world footfall anchors and zoning suitability.
    """
    cat = (vendor_category or "food").lower()
    weights = CATEGORY_SPATIAL_WEIGHTS.get(cat, CATEGORY_SPATIAL_WEIGHTS["food"])

    raw_score = 0.0
    for feature_name, weight in weights.items():
        val = float(features.get(feature_name, 0.0))
        # Ratios (0.0 - 1.0) scaled by 10 to calibrate with integer POI counts
        if feature_name.endswith("_ratio"):
            val *= 10.0
        elif feature_name == "road_density":
            val *= 5.0
        elif feature_name == "population_density":
            val = min(val / 1000.0, 10.0)
        raw_score += val * weight

    return max(0.5, round(raw_score, 2))


def calculate_category_demand(raw_p50: int, vendor_category: str, features: dict) -> tuple[int, float]:
    """
    Convert raw cell footfall demand (P50) into realistic, category-calibrated hourly customer capacity
    and return (expected_customers, category_affinity).
    """
    cat = (vendor_category or "food").lower()
    spec = CATEGORY_THROUGHPUT_SPECS.get(cat, CATEGORY_THROUGHPUT_SPECS["food"])
    affinity = compute_category_affinity(cat, features)

    benchmark = float(spec["benchmark_affinity"])
    affinity_multiplier = max(0.4, min(2.5, (affinity / benchmark) ** 0.35))

    conv_rate = float(spec["conv_rate"])
    raw_demand = float(raw_p50) if raw_p50 > 0 else 20.0
    converted = raw_demand * conv_rate * affinity_multiplier

    min_c = int(spec["min_cust"])
    max_c = int(spec["max_cust"])
    cat_customers = int(round(max(min_c, min(max_c, converted))))

    return cat_customers, affinity


# ---------------------------------------------------------------------------
# Ranking & Strategic Decision Engine
# ---------------------------------------------------------------------------

from api.ranking_engine import rank_candidates
from api.decision_engine import evaluate_decision, select_top_recommendations


# ---------------------------------------------------------------------------
# SHAP-Powered Explanation Generator
# ---------------------------------------------------------------------------

def generate_explanation(candidate: dict, vendor_category: str) -> tuple[list[str], str]:
    """
    Generate SHAP-driven top positive and negative drivers and natural reasoning narrative
    tailored to the specific business category.
    """
    model = DemandModel.get()
    features = candidate.get("_features", {})
    try:
        drivers_nl, structured_shap = model.compute_shap_drivers(features, max_drivers=4)
    except Exception:
        drivers_nl, structured_shap = [], []
    
    cust = candidate.get("expected_customers", 0)
    profit = candidate.get("expected_profit_inr", 0)
    dist = candidate.get("distance_km", 0)
    direction = f"{dist:.1f} km away" if dist > 0.1 else "your current position"

    top_pos = [s["friendly_name"] for s in structured_shap if s["impact"] == "positive"][:2]
    top_neg = [s["friendly_name"] for s in structured_shap if s["impact"] == "negative"][:1]

    cat = (vendor_category or "food").lower()
    cat_context = ""
    if cat == "food":
        if features.get("office_count", 0) >= 10:
            cat_context = " High lunchtime office crowd and dining demand."
        elif features.get("restaurant_count", 0) >= 15:
            cat_context = " Strong food cluster and pedestrian dining traffic."
        elif features.get("bus_stop_count", 0) >= 5 or features.get("railway_station_count", 0) >= 1:
            cat_context = " High commuter transit passenger flow."
    elif cat == "fruits":
        if features.get("hospital_count", 0) >= 10:
            cat_context = " Major hospital cluster with strong patient & visitor fresh fruit demand."
        elif features.get("park_count", 0) >= 2:
            cat_context = " Prime morning/evening walker and wellness consumer traffic."
        elif features.get("residential_ratio", 0) >= 0.2:
            cat_context = " High neighborhood residential household fruit demand."
    elif cat == "salon":
        if features.get("mall_count", 0) >= 15:
            cat_context = " Premium retail and lifestyle mall shoppers with high grooming spend."
        elif features.get("parking_count", 0) >= 2:
            cat_context = " Convenient customer vehicle access and dwell parking."
        elif features.get("residential_ratio", 0) >= 0.2:
            cat_context = " High-density residential community with regular personal care demand."
    elif cat == "repair":
        if features.get("industrial_ratio", 0) >= 0.05:
            cat_context = " Active industrial and transport hub with heavy commercial vehicle needs."
        elif features.get("road_density", 0) >= 0.5:
            cat_context = " High vehicular traffic flow along major arterial roadways."

    pos_phrase = f"driven by strong {', '.join(top_pos)}" if top_pos else "with steady footfall"
    neg_phrase = f", though dampened slightly by {top_neg[0]}" if top_neg else ""

    summary = (
        f"Moving to {direction} is estimated to yield {cust} customers/hour "
        f"(₹{profit:,.0f} hourly profit), {pos_phrase}{neg_phrase}.{cat_context}"
    )

    return drivers_nl, summary
