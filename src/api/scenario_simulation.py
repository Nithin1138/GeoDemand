"""
GeoDemand AI — Scenario Simulation & What-If Counterfactual Engine

Enables real-time sensitivity and what-if counterfactual analysis WITHOUT model retraining:
- Rainfall and weather condition variations (precipitation spikes, heavy monsoon rain, storm)
- Competitor density variations (arrival/departure of neighboring stalls, market dilution)
- Fuel price spikes (evaluating travel friction and relocation feasibility)
- Product and vendor commercial category shifts (food vs beverages vs fruits vs repair)
- Time-of-day operational shifts (morning rush, lunch, evening peak, night hours)
- Price and Average Order Value (AOV) sensitivity (measuring margin elasticity)
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from config.settings import (
    DEFAULT_AOV_INR,
    DEFAULT_VARIABLE_COST_RATE,
    DEFAULT_FIXED_COST_PER_DAY_INR,
    FUEL_PRICE_INR_PER_LITRE,
    FUEL_EFFICIENCY_KM_PER_L,
)
from config.product_pricing import PRODUCT_CATEGORIES, get_category_pricing
from api.location import get_location_context
from api.feature_assembler import FeatureAssembler
from api.recommender import (
    DemandModel,
    calculate_category_demand,
    calculate_business_metrics,
)
from api.providers.competition import CompetitionProviderFactory
from api.providers.events import EventProviderFactory
from api.providers.fuel import FuelPriceProvider
from features.weather_client import OpenMeteoClient

logger = logging.getLogger(__name__)


class ScenarioSimulator:
    """Production-grade what-if scenario simulation engine."""

    def __init__(
        self,
        model: Optional[DemandModel] = None,
        assembler: Optional[FeatureAssembler] = None,
    ):
        self.model = model or DemandModel.get()
        self.assembler = assembler or FeatureAssembler()

    def simulate(
        self,
        latitude: float,
        longitude: float,
        vendor_category: str = "food",
        baseline_aov: float = DEFAULT_AOV_INR,
        simulated_aov: Optional[float] = None,
        simulated_rainfall_mm: Optional[float] = None,
        simulated_is_rain: Optional[bool] = None,
        simulated_weather_condition: Optional[str] = None,
        simulated_competitor_count_delta: Optional[int] = None,
        simulated_competition_score: Optional[float] = None,
        simulated_fuel_price_inr: Optional[float] = None,
        simulated_vendor_category: Optional[str] = None,
        simulated_hour: Optional[int] = None,
        distance_km: float = 1.5,
        weather_client: Optional[OpenMeteoClient] = None,
        competition_provider=None,
        event_provider=None,
        fuel_provider: Optional[FuelPriceProvider] = None,
        reference_timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute full counterfactual simulation evaluating baseline vs simulated parameters.
        All calculations reuse existing ML model weights and feature schemas without retraining.
        """
        now = reference_timestamp or datetime.now(timezone.utc)
        loc_ctx = get_location_context(latitude, longitude)

        # Initialize providers if not injected
        wc = weather_client or OpenMeteoClient()
        comp_prov = competition_provider or CompetitionProviderFactory.get_provider()
        event_prov = event_provider or EventProviderFactory.get_provider()
        fuel_prov = fuel_provider or FuelPriceProvider()

        # -----------------------------------------------------------------------
        # 1. Fetch Baseline Telemetry & Context
        # -----------------------------------------------------------------------
        baseline_weather = wc.get_hourly_forecast(latitude, longitude, target_hour_offset=1)
        baseline_comp = comp_prov.get_competition_score(loc_ctx.h3_cell, vendor_category)
        baseline_event = event_prov.get_event_importance(loc_ctx.h3_cell, now)

        base_pricing = get_category_pricing(vendor_category)
        base_cogs_rate = base_pricing.default_cogs_rate if base_pricing else DEFAULT_VARIABLE_COST_RATE
        base_fuel_price = FUEL_PRICE_INR_PER_LITRE

        # Baseline Fuel Cost
        base_fuel_cost = round((distance_km / max(FUEL_EFFICIENCY_KM_PER_L, 1.0)) * base_fuel_price, 2)

        baseline_profile = {
            "vendor_id": "sim_user",
            "vendor_category": vendor_category,
            "vendor_name": "Mobile Vendor",
            "average_order_value": baseline_aov,
            "variable_cost_rate": base_cogs_rate,
            "fixed_cost_per_day": DEFAULT_FIXED_COST_PER_DAY_INR,
            "inventory_capacity": base_pricing.default_inventory_capacity if base_pricing else 200,
            "preparation_time": 5.0,
        }

        # Baseline Features & Inference
        base_features, _ = self.assembler.assemble(
            h3_cell=loc_ctx.h3_cell,
            vendor_profile=baseline_profile,
            weather=baseline_weather,
            competition=baseline_comp,
            event=baseline_event,
            decision_timestamp=now,
        )

        base_uncertainty = self.model.predict_with_uncertainty(base_features)
        base_raw_p50 = base_uncertainty["p50"]

        base_customers, _ = calculate_category_demand(
            raw_p50=base_raw_p50,
            vendor_category=vendor_category,
            features=base_features,
        )

        base_scale = base_customers / max(1, base_raw_p50)
        base_quantiles = {
            "p10": max(1, int(round(base_uncertainty.get("p10", base_raw_p50 * 0.8) * base_scale))),
            "p50": base_customers,
            "p90": max(base_customers, int(round(base_uncertainty.get("p90", base_raw_p50 * 1.2) * base_scale))),
        }

        base_biz = calculate_business_metrics(
            expected_customers=base_customers,
            aov=baseline_aov,
            variable_cost_rate=base_cogs_rate,
            fixed_cost_per_day=DEFAULT_FIXED_COST_PER_DAY_INR,
            fuel_cost=base_fuel_cost,
            travel_time_minutes=int(round((distance_km / 15.0) * 60.0)),
            vendor_category=vendor_category,
        )

        # -----------------------------------------------------------------------
        # 2. Formulate Counterfactual Simulated Context
        # -----------------------------------------------------------------------
        sim_category = simulated_vendor_category or vendor_category
        sim_pricing = get_category_pricing(sim_category)
        sim_cogs_rate = sim_pricing.default_cogs_rate if sim_pricing else DEFAULT_VARIABLE_COST_RATE
        sim_aov = simulated_aov if simulated_aov is not None else (
            sim_pricing.default_aov_inr if simulated_vendor_category else baseline_aov
        )

        # Weather Counterfactuals
        sim_weather = copy.deepcopy(baseline_weather)
        if simulated_rainfall_mm is not None:
            sim_weather["precipitation_mm"] = float(simulated_rainfall_mm)
            sim_weather["rainfall"] = float(simulated_rainfall_mm)
            sim_weather["is_rain"] = bool(simulated_rainfall_mm >= 0.5)
            if simulated_rainfall_mm >= 10.0:
                sim_weather["weather_condition"] = "heavy_rain"
            elif simulated_rainfall_mm > 0.5:
                sim_weather["weather_condition"] = "rainy"
            else:
                sim_weather["weather_condition"] = "clear"

        if simulated_is_rain is not None:
            sim_weather["is_rain"] = simulated_is_rain
            if simulated_is_rain and sim_weather.get("precipitation_mm", 0.0) == 0.0:
                sim_weather["precipitation_mm"] = 5.0
                sim_weather["rainfall"] = 5.0
                sim_weather["weather_condition"] = "rainy"

        if simulated_weather_condition:
            sim_weather["weather_condition"] = simulated_weather_condition
            if "rain" in simulated_weather_condition.lower() and sim_weather.get("precipitation_mm", 0.0) == 0.0:
                sim_weather["precipitation_mm"] = 6.0
                sim_weather["rainfall"] = 6.0
                sim_weather["is_rain"] = True

        # Competition Counterfactuals
        sim_comp = copy.deepcopy(baseline_comp)
        if simulated_competition_score is not None:
            sim_comp["competition_score"] = min(1.0, max(0.0, float(simulated_competition_score)))
        elif simulated_competitor_count_delta is not None:
            orig_cnt = sim_comp.get("raw_competitor_count", 5)
            new_cnt = max(0, orig_cnt + simulated_competitor_count_delta)
            sim_comp["raw_competitor_count"] = new_cnt
            # Calibrated sensitivity: 1 competitor stall shift approx shifts score by ~0.08
            score_delta = simulated_competitor_count_delta * 0.08
            sim_comp["competition_score"] = min(1.0, max(0.02, sim_comp["competition_score"] + score_delta))

        # Fuel Price Counterfactuals
        sim_fuel_price = simulated_fuel_price_inr if simulated_fuel_price_inr is not None else base_fuel_price
        sim_fuel_cost = round((distance_km / max(FUEL_EFFICIENCY_KM_PER_L, 1.0)) * sim_fuel_price, 2)

        # Operational Hour Counterfactuals
        if simulated_hour is not None:
            # Replace target hour in simulation timestamp
            sim_timestamp = now.replace(hour=simulated_hour, minute=0, second=0)
        else:
            sim_timestamp = now

        sim_profile = {
            "vendor_id": "sim_user",
            "vendor_category": sim_category,
            "vendor_name": "Mobile Vendor",
            "average_order_value": sim_aov,
            "variable_cost_rate": sim_cogs_rate,
            "fixed_cost_per_day": DEFAULT_FIXED_COST_PER_DAY_INR,
            "inventory_capacity": sim_pricing.default_inventory_capacity if sim_pricing else 200,
            "preparation_time": 5.0,
        }

        # Simulated Features & Inference
        sim_features, _ = self.assembler.assemble(
            h3_cell=loc_ctx.h3_cell,
            vendor_profile=sim_profile,
            weather=sim_weather,
            competition=sim_comp,
            event=baseline_event,
            decision_timestamp=sim_timestamp,
        )

        sim_uncertainty = self.model.predict_with_uncertainty(sim_features)
        sim_raw_p50 = sim_uncertainty["p50"]

        sim_customers, _ = calculate_category_demand(
            raw_p50=sim_raw_p50,
            vendor_category=sim_category,
            features=sim_features,
        )

        # Domain physics adjustments for weather impact on footfall
        if sim_weather.get("is_rain") or sim_weather.get("precipitation_mm", 0.0) >= 1.0:
            rain_mm = sim_weather.get("precipitation_mm", 5.0)
            # Food and fruit stalls experience 20-40% drop during heavy rain; beverages/snacks drop less
            if sim_category in ("food", "fruits"):
                rain_penalty = min(0.50, 0.15 + (rain_mm / 30.0) * 0.35)
                sim_customers = max(2, int(round(sim_customers * (1.0 - rain_penalty))))
            elif sim_category in ("beverages", "snacks"):
                rain_penalty = min(0.25, 0.08 + (rain_mm / 40.0) * 0.17)
                sim_customers = max(2, int(round(sim_customers * (1.0 - rain_penalty))))

        sim_scale = sim_customers / max(1, sim_raw_p50)
        sim_quantiles = {
            "p10": max(1, int(round(sim_uncertainty.get("p10", sim_raw_p50 * 0.8) * sim_scale))),
            "p50": sim_customers,
            "p90": max(sim_customers, int(round(sim_uncertainty.get("p90", sim_raw_p50 * 1.2) * sim_scale))),
        }

        sim_biz = calculate_business_metrics(
            expected_customers=sim_customers,
            aov=sim_aov,
            variable_cost_rate=sim_cogs_rate,
            fixed_cost_per_day=DEFAULT_FIXED_COST_PER_DAY_INR,
            fuel_cost=sim_fuel_cost,
            travel_time_minutes=int(round((distance_km / 15.0) * 60.0)),
            vendor_category=sim_category,
        )

        # -----------------------------------------------------------------------
        # 3. Comparative Delta & Sensitivity Metrics
        # -----------------------------------------------------------------------
        base_profit = base_biz["relocation_adjusted_realized_profit_inr"]
        sim_profit = sim_biz["relocation_adjusted_realized_profit_inr"]
        profit_delta = round(sim_profit - base_profit, 2)
        profit_pct_change = round((profit_delta / max(abs(base_profit), 1.0)) * 100, 1)

        customer_delta = sim_customers - base_customers
        customer_pct_change = round((customer_delta / max(base_customers, 1)) * 100, 1)

        revenue_delta = round(sim_biz["expected_revenue_inr"] - base_biz["expected_revenue_inr"], 2)
        revenue_pct_change = round((revenue_delta / max(base_biz["expected_revenue_inr"], 1.0)) * 100, 1)

        fuel_delta = round(sim_fuel_cost - base_fuel_cost, 2)

        # -----------------------------------------------------------------------
        # 4. Identify Scenario Classification & Primary Drivers
        # -----------------------------------------------------------------------
        drivers = []
        scenario_types = []

        if simulated_aov is not None and simulated_aov != baseline_aov:
            drivers.append(f"AOV Shift (₹{baseline_aov:.0f} → ₹{sim_aov:.0f})")
            scenario_types.append("pricing")

        if simulated_rainfall_mm is not None or simulated_is_rain is not None or simulated_weather_condition:
            drivers.append(f"Rainfall/Weather ({sim_weather.get('weather_condition', 'rainy')}, {sim_weather.get('precipitation_mm', 0):.1f}mm)")
            scenario_types.append("weather")

        if simulated_competitor_count_delta is not None or simulated_competition_score is not None:
            comp_diff = sim_comp['competition_score'] - baseline_comp['competition_score']
            drivers.append(f"Competition Shift ({'+' if comp_diff >= 0 else ''}{comp_diff:.2f} index)")
            scenario_types.append("competition")

        if simulated_fuel_price_inr is not None and simulated_fuel_price_inr != base_fuel_price:
            drivers.append(f"Fuel Price (₹{base_fuel_price:.1f} → ₹{sim_fuel_price:.1f}/L)")
            scenario_types.append("fuel")

        if simulated_vendor_category is not None and simulated_vendor_category != vendor_category:
            drivers.append(f"Category Pivot ({vendor_category} → {sim_category})")
            scenario_types.append("category")

        if simulated_hour is not None and simulated_hour != now.hour:
            drivers.append(f"Starting Hour Shift ({now.hour}:00 → {simulated_hour}:00)")
            scenario_types.append("schedule")

        scenario_type = "_".join(scenario_types) if scenario_types else "multi_factor"
        primary_driver = drivers[0] if drivers else "Baseline Comparison"

        # -----------------------------------------------------------------------
        # 5. Strategic Advice Formulation
        # -----------------------------------------------------------------------
        advice_parts = []
        if sim_weather.get("is_rain") or sim_weather.get("precipitation_mm", 0.0) >= 2.0:
            advice_parts.append(
                "Rain significantly impacts pedestrian footfall. Deploy waterproof overhead shelter, "
                "offer warm/hot items (tea, pakoras, soups), and consider relocating near transit bus shelters."
            )
        if sim_comp["competition_score"] > 0.65:
            advice_parts.append(
                "High competitive density compresses margins. Offer unique value combos or relocate to "
                "underserved commercial clusters with lower vendor density."
            )
        if sim_fuel_cost > base_fuel_cost + 15.0:
            advice_parts.append(
                "Elevated fuel costs erode relocation gains. Strictly adhere to moves within a 1.5 km radius "
                "or operate via pushcart/bicycle where feasible."
            )
        if sim_profit > base_profit:
            advice_parts.append(
                f"Favorable scenario! Expected hourly profit increases by +₹{profit_delta:.2f} ({profit_pct_change:+.1f}%). "
                f"Prepare sufficient inventory capacity to avoid stockout revenue losses."
            )
        elif sim_profit < base_profit:
            advice_parts.append(
                f"Adverse scenario: Profit drops by ₹{abs(profit_delta):.2f}/hr ({profit_pct_change:.1f}%). "
                f"Tighten operating expenses, minimize perishable overstock, and adjust operating hours."
            )

        strategic_advice = " ".join(advice_parts) if advice_parts else "Operational conditions remain stable."

        explanation = (
            f"Simulating {', '.join(drivers) if drivers else 'current conditions'} "
            f"projects customer demand shifting from {base_customers} to {sim_customers} ({customer_pct_change:+.1f}%), "
            f"with net hourly profit moving from ₹{base_profit:.2f} to ₹{sim_profit:.2f} ({profit_delta:+.2f} INR, {profit_pct_change:+.1f}%)."
        )

        return {
            "scenario_type": scenario_type,
            "baseline": {
                "aov": baseline_aov,
                "vendor_category": vendor_category,
                "hour": now.hour,
                "rainfall_mm": float(baseline_weather.get("precipitation_mm", 0.0)),
                "is_rain": bool(baseline_weather.get("is_rain", False)),
                "competition_score": round(baseline_comp["competition_score"], 3),
                "fuel_price_inr": base_fuel_price,
                "expected_customers": base_customers,
                "prediction_interval": base_quantiles,
                "expected_revenue_inr": base_biz["expected_revenue_inr"],
                "expected_costs_inr": base_biz["expected_operating_cost_inr"],
                "fuel_cost_inr": base_fuel_cost,
                "expected_profit_inr": base_profit,
            },
            "scenario": {
                "aov": sim_aov,
                "vendor_category": sim_category,
                "hour": sim_timestamp.hour,
                "rainfall_mm": float(sim_weather.get("precipitation_mm", 0.0)),
                "is_rain": bool(sim_weather.get("is_rain", False)),
                "competition_score": round(sim_comp["competition_score"], 3),
                "fuel_price_inr": sim_fuel_price,
                "expected_customers": sim_customers,
                "prediction_interval": sim_quantiles,
                "expected_revenue_inr": sim_biz["expected_revenue_inr"],
                "expected_costs_inr": sim_biz["expected_operating_cost_inr"],
                "fuel_cost_inr": sim_fuel_cost,
                "expected_profit_inr": sim_profit,
            },
            "simulation_delta": {
                "customer_delta": customer_delta,
                "customer_pct_change": customer_pct_change,
                "revenue_delta_inr": revenue_delta,
                "revenue_pct_change": revenue_pct_change,
                "fuel_delta_inr": fuel_delta,
                "profit_delta_inr": profit_delta,
                "percentage_change": profit_pct_change,
            },
            "primary_driver": primary_driver,
            "strategic_advice": strategic_advice,
            "explanation": explanation,
        }
