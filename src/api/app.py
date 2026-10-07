"""
GeoDemand AI — FastAPI Live Recommendation Service

Endpoints:
  POST /v1/location/context       — GPS → H3, timezone
  POST /v1/recommendations/live   — Full live recommendation pipeline
  GET  /v1/system/data-health     — Data source freshness & health
  GET  /v1/system/ping            — Liveness check

Run with:
  uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Dashboard served at:
  http://localhost:8000/
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "api"))


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.location import get_location_context, generate_candidates
from api.feature_assembler import FeatureAssembler
from api.recommender import DemandModel, calculate_business_metrics, rank_candidates, generate_explanation
from api.providers.competition import CompetitionProviderFactory
from api.providers.traffic import TrafficProviderFactory
from api.providers.events import EventProviderFactory
from api.providers.fuel import FuelPriceProvider
from api.providers.routing import RoutingProviderFactory
from api.spatial_filter import is_water_location, is_water_cell
from features.weather_client import OpenMeteoClient
from config.settings import (
    STAY_THRESHOLD_INR_PER_HOUR,
    CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR,
    MAX_RECOMMENDED_TRAVEL_MIN,
    MAX_RECOMMENDED_TRAVEL_KM,
    API_KEY,
    REQUIRE_API_KEY,
    DEFAULT_AOV_INR,
    DEFAULT_VARIABLE_COST_RATE,
    DEFAULT_FIXED_COST_PER_DAY_INR,
)
import h3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Security helper to validate API key header if enabled."""
    if REQUIRE_API_KEY:
        if not x_api_key or (API_KEY and x_api_key != API_KEY):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key. Provide a valid X-API-Key header.",
            )
    return x_api_key



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GeoDemand AI API starting up...")
    _load_model()
    try:
        _get_assembler()
    except Exception as e:
        logger.warning(f"Feature assembler init warning: {e}")
    logger.info("Startup complete.")
    yield
    logger.info("GeoDemand AI API shutting down...")


app = FastAPI(
    title="GeoDemand AI",
    description="Production-grade geospatial demand intelligence for mobile vendors",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard
DASHBOARD_DIR = ROOT / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

# Lazy singletons
_weather_client: Optional[OpenMeteoClient] = None
_assembler: Optional[FeatureAssembler] = None
_competition = None
_traffic = None
_events = None
_fuel: Optional[FuelPriceProvider] = None
_model_health = {"status": "not_loaded", "loaded_at": None}


def _get_weather_client() -> OpenMeteoClient:
    global _weather_client
    if _weather_client is None:
        _weather_client = OpenMeteoClient(cache_dir=str(ROOT / "data" / "raw" / "weather"))
    return _weather_client


def _get_assembler() -> FeatureAssembler:
    global _assembler
    if _assembler is None:
        _assembler = FeatureAssembler()
    return _assembler


def _get_providers():
    global _competition, _traffic, _events, _fuel
    if _competition is None:
        _competition = CompetitionProviderFactory.get_provider()
    if _traffic is None:
        _traffic = TrafficProviderFactory.get_provider()
    if _events is None:
        _events = EventProviderFactory.get_provider()
    if _fuel is None:
        _fuel = FuelPriceProvider()
    return _competition, _traffic, _events, _fuel


def _load_model():
    global _model_health
    try:
        DemandModel.get()
        _model_health = {"status": "loaded", "loaded_at": datetime.now(timezone.utc).isoformat()}
    except FileNotFoundError as e:
        _model_health = {"status": "not_trained", "error": str(e)}
        logger.warning(str(e))


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class LocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180, le=180, json_schema_extra={"example": 80.6480})


class RecommendationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180, le=180, json_schema_extra={"example": 80.6480})
    vendor_id: Optional[str] = Field(None, json_schema_extra={"example": "V_001"})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    vendor_name: Optional[str] = None
    average_order_value: float = Field(DEFAULT_AOV_INR, gt=0)
    variable_cost_rate: float = Field(DEFAULT_VARIABLE_COST_RATE, gt=0, lt=1)
    fixed_cost_per_day: float = Field(DEFAULT_FIXED_COST_PER_DAY_INR, gt=0)
    inventory_capacity: int = Field(200, gt=0)
    preparation_time: float = Field(5.0, gt=0)
    search_radius_km: float = Field(3.0, gt=0, le=10)
    top_n: int = Field(5, ge=1, le=10)
    timestamp: Optional[datetime] = Field(None, json_schema_extra={"example": "2026-10-05T12:00:00Z"})


class PredictDemandRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180, le=180, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    average_order_value: float = Field(DEFAULT_AOV_INR, gt=0)
    variable_cost_rate: float = Field(DEFAULT_VARIABLE_COST_RATE, gt=0, lt=1)
    fixed_cost_per_day: float = Field(DEFAULT_FIXED_COST_PER_DAY_INR, gt=0)
    timestamp: Optional[datetime] = Field(None, json_schema_extra={"example": "2026-10-05T12:00:00Z"})



class RecommendZonesRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180, le=180, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    search_radius_km: float = Field(3.0, gt=0, le=10)
    top_n: int = Field(5, ge=1, le=10)


class ScenarioSimulateRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180, le=180, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    baseline_aov: float = Field(DEFAULT_AOV_INR, gt=0)
    simulated_aov: float = Field(DEFAULT_AOV_INR * 1.5, gt=0)
    simulated_weather_condition: Optional[str] = Field(None, json_schema_extra={"example": "rainy"})
    simulated_hour: Optional[int] = Field(None, ge=0, le=23, json_schema_extra={"example": 13})





# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "GeoDemand AI v2.1 — POST /v1/recommendations/live to begin"})


@app.get("/v1/system/ping")
async def ping():
    return {"status": "ok", "service": "GeoDemand AI", "version": "2.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/system/data-health")
async def data_health():
    """Returns provenance and freshness status for all data sources."""
    competition, traffic, events, _ = _get_providers()
    return {
        "model": _model_health,
        "weather": {
            "source_type": "real_live",
            "source_name": "open-meteo",
            "note": "No API key required. Fallback: synthetic_fallback",
            "cache_ttl_sec": 300,
        },
        "poi": {
            "source_type": "real_periodic",
            "source_name": "openstreetmap",
            "note": "Loaded from static_features.parquet (periodically refreshed)",
        },
        "competition": {
            "source_type": competition.source_type,
            "source_name": competition.source_name,
        },
        "traffic": {
            "source_type": traffic.source_type,
            "source_name": traffic.source_name,
            "note": "Set HERE_API_KEY for live traffic",
        },
        "events": {
            "source_type": events.source_type,
            "source_name": events.source_name,
        },
        "holidays": {
            "source_type": "real_offline",
            "source_name": "india_holiday_calendar",
        },
        "customer_demand": {
            "source_type": "derived",
            "source_name": "geodemand_ml_model",
            "note": "Demand is an ML estimate — no real customer count sensor exists",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/routing/route")
@app.post("/v1/routing/route")
async def get_street_route(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float):
    """Returns real street road navigation route coordinates [[lat, lng], ...] for map rendering."""
    provider = RoutingProviderFactory.get_provider()
    return provider.get_route_geometry(origin_lat, origin_lng, dest_lat, dest_lng)


@app.post("/v1/location/context")
async def location_context(req: LocationRequest):
    """GPS → H3 cell, timestamp, and location context."""
    ctx = get_location_context(req.latitude, req.longitude)
    return ctx.to_dict()


@app.post("/v1/predict/demand")
async def predict_demand(req: PredictDemandRequest, api_key: str = Depends(verify_api_key)):
    """Predict customer demand and business economics for a single location."""
    if _model_health["status"] not in ("loaded",):
        raise HTTPException(status_code=503, detail="ML model not loaded")

    loc_ctx = get_location_context(req.latitude, req.longitude)
    weather_client = _get_weather_client()
    weather_forecast = weather_client.get_hourly_forecast(req.latitude, req.longitude, target_hour_offset=1)
    comp_prov, traffic_prov, event_prov, _ = _get_providers()
    now = req.timestamp or datetime.now(timezone.utc)

    comp = comp_prov.get_competition_score(loc_ctx.h3_cell, req.vendor_category)
    event = event_prov.get_event_importance(loc_ctx.h3_cell, now)

    vendor_profile = {
        "vendor_id": "predict_user",
        "vendor_category": req.vendor_category,
        "vendor_name": "Mobile Vendor",
        "average_order_value": req.average_order_value,
        "variable_cost_rate": req.variable_cost_rate,
        "fixed_cost_per_day": req.fixed_cost_per_day,
    }

    assembler = _get_assembler()
    features, freshness = assembler.assemble(
        h3_cell=loc_ctx.h3_cell,
        vendor_profile=vendor_profile,
        weather=weather_forecast,
        competition=comp,
        event=event,
        decision_timestamp=now,
    )

    model = DemandModel.get()
    uncertainty = model.predict_with_uncertainty(features)
    expected_customers = uncertainty["p50"]

    biz = calculate_business_metrics(
        expected_customers=expected_customers,
        aov=req.average_order_value,
        variable_cost_rate=req.variable_cost_rate,
        fixed_cost_per_day=req.fixed_cost_per_day,
    )

    return {
        "location": {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "h3_cell": loc_ctx.h3_cell,
            "is_water": loc_ctx.is_water,
        },
        "vendor_category": req.vendor_category,
        "prediction": {
            "expected_customers": expected_customers,
            "confidence_interval": uncertainty,
        },
        "economics": biz,
        "data_freshness": freshness,
    }


@app.post("/v1/recommend/zones")
async def recommend_zones(req: RecommendZonesRequest, api_key: str = Depends(verify_api_key)):
    """Target high-demand candidate zones within radius."""
    rec_req = RecommendationRequest(
        latitude=req.latitude,
        longitude=req.longitude,
        vendor_category=req.vendor_category,
        search_radius_km=req.search_radius_km,
        top_n=req.top_n,
    )
    return await recommendations_live(rec_req)


@app.post("/v1/scenario/simulate")
async def scenario_simulate(req: ScenarioSimulateRequest, api_key: str = Depends(verify_api_key)):
    """Simulate business impact under varied AOV or weather scenario."""
    base_res = await predict_demand(
        PredictDemandRequest(
            latitude=req.latitude,
            longitude=req.longitude,
            vendor_category=req.vendor_category,
            average_order_value=req.baseline_aov,
        ),
        api_key=api_key,
    )
    sim_res = await predict_demand(
        PredictDemandRequest(
            latitude=req.latitude,
            longitude=req.longitude,
            vendor_category=req.vendor_category,
            average_order_value=req.simulated_aov,
        ),
        api_key=api_key,
    )

    base_profit = base_res["economics"]["expected_profit_inr"]
    sim_profit = sim_res["economics"]["expected_profit_inr"]
    profit_delta = round(sim_profit - base_profit, 2)
    pct_change = round((profit_delta / max(abs(base_profit), 1)) * 100, 1)

    return {
        "baseline": {
            "aov": req.baseline_aov,
            "expected_customers": base_res["prediction"]["expected_customers"],
            "expected_profit_inr": base_profit,
        },
        "scenario": {
            "aov": req.simulated_aov,
            "expected_customers": sim_res["prediction"]["expected_customers"],
            "expected_profit_inr": sim_profit,
        },
        "simulation_delta": {
            "profit_delta_inr": profit_delta,
            "percentage_change": pct_change,
        },
        "explanation": f"Simulating an AOV shift from ₹{req.baseline_aov} to ₹{req.simulated_aov} results in a profit change of ₹{profit_delta:+.2f}/hr ({pct_change:+.1f}%).",
    }



@app.post("/v1/recommendations/live")
async def recommendations_live(req: RecommendationRequest):
    """
    Full live recommendation pipeline:
    GPS → H3 → candidates → features → inference → business metrics → Top N
    """
    if _model_health["status"] not in ("loaded",):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "ML model not loaded",
                "fix": "Run: python3 src/models/train_demand_model.py",
                "model_status": _model_health,
            }
        )

    now = req.timestamp or datetime.now(timezone.utc)
    competition_prov, traffic_prov, event_prov, fuel_prov = _get_providers()
    weather_client = _get_weather_client()
    assembler = _get_assembler()
    model = DemandModel.get()

    vendor_profile = {
        "vendor_id": req.vendor_id or "live_user",
        "vendor_category": req.vendor_category,
        "vendor_name": req.vendor_name or "Mobile Vendor",
        "average_order_value": req.average_order_value,
        "variable_cost_rate": req.variable_cost_rate,
        "fixed_cost_per_day": req.fixed_cost_per_day,
        "inventory_capacity": req.inventory_capacity,
        "preparation_time": req.preparation_time,
    }

    # --- Location context ---
    loc_ctx = get_location_context(req.latitude, req.longitude)

    # --- Fetch weather ONCE for current location (candidate weather = city-grain) ---
    weather_now = weather_client.get_current_weather(req.latitude, req.longitude)
    weather_forecast = weather_client.get_hourly_forecast(req.latitude, req.longitude, target_hour_offset=1)

    # --- Generate candidate cells ---
    candidates_raw = generate_candidates(
        req.latitude, req.longitude, loc_ctx.h3_cell,
        search_radius_km=req.search_radius_km,
    )

    # --- Assemble features + predict for each candidate ---
    candidate_results = []
    global_freshness = None

    for cand in candidates_raw:
        comp = competition_prov.get_competition_score(cand.h3_cell, req.vendor_category)
        event = event_prov.get_event_importance(cand.h3_cell, now)
        traffic = traffic_prov.get_traffic(req.latitude, req.longitude,
                                           cand.centroid_lat, cand.centroid_lng)

        # Traffic-aware driving duration and congestion-adjusted fuel cost
        eff_speed = max(5.0, float(traffic.get("traffic_speed_kmh", 20.0)))
        travel_time_min = 0 if cand.is_current_cell else max(
            1, int(round((cand.distance_km / eff_speed) * 60.0 + float(traffic.get("traffic_delay_min", 0.0))))
        )

        base_fuel = fuel_prov.get_fuel_cost_for_distance(cand.distance_km)
        congestion_penalty = 1.0 + float(traffic.get("congestion_ratio", 0.0)) * 0.35
        actual_fuel_cost = round(base_fuel["fuel_cost_inr"] * congestion_penalty, 2)

        # Features at decision_timestamp t (current weather for current context)
        # For next-hour prediction, use forecast weather
        features, freshness = assembler.assemble(
            h3_cell=cand.h3_cell,
            vendor_profile=vendor_profile,
            weather=weather_forecast,  # next-hour forecast (temporal contract)
            competition=comp,
            event=event,
            decision_timestamp=now,
        )

        if global_freshness is None:
            global_freshness = freshness

        uncertainty = model.predict_with_uncertainty(features)
        expected_customers = uncertainty["p50"]

        biz = calculate_business_metrics(
            expected_customers=expected_customers,
            aov=req.average_order_value,
            variable_cost_rate=req.variable_cost_rate,
            fixed_cost_per_day=req.fixed_cost_per_day,
            fuel_cost=actual_fuel_cost,
            travel_time_minutes=travel_time_min,
        )

        drivers, explanation = generate_explanation(
            {**biz, "distance_km": cand.distance_km, "competition_level": comp.get("competition_level", "medium"),
             "competition_score": comp["competition_score"], "_features": features},
            req.vendor_category,
        )

        candidate_results.append({
            "h3_cell": cand.h3_cell,
            "latitude": cand.centroid_lat,
            "longitude": cand.centroid_lng,
            "distance_km": cand.distance_km,
            "estimated_travel_time_min": travel_time_min,
            "travel_fraction": biz["travel_fraction"],
            "is_current_cell": cand.is_current_cell,
            "is_water": getattr(cand, "is_water", False),
            "expected_customers": expected_customers,
            "prediction_interval": uncertainty,
            "confidence_interval": uncertainty,
            "expected_revenue_inr": biz["expected_revenue_inr"],
            "expected_profit_inr": biz["expected_profit_inr"],
            "candidate_predicted_profit_inr": biz["candidate_predicted_profit_inr"],
            "relocation_adjusted_realized_profit_inr": biz["relocation_adjusted_realized_profit_inr"],
            "realized_next_hour_profit_inr": biz["relocation_adjusted_realized_profit_inr"],
            "expected_operating_cost_inr": biz["expected_operating_cost_inr"],
            "fuel_cost_inr": actual_fuel_cost,
            "competition_score": round(comp["competition_score"], 3),
            "competition_level": comp.get("competition_level", "medium"),
            "traffic_level": traffic.get("traffic_level", "medium"),
            "traffic_speed_kmh": traffic.get("traffic_speed_kmh", 20.0),
            "traffic_source_type": traffic.get("source_type", "synthetic_fallback"),
            "weather_condition": weather_forecast.get("weather_condition", "clear"),
            "top_drivers": drivers,
            "explanation": explanation,
            # for ranking
            "_features": features,
        })

    # --- Step 1: Candidate Ranking (Multi-Factor Scoring) ---
    # Multi-factor score: 60% profit, 25% demand, -10% distance, -5% competition
    ranked_candidates = rank_candidates(candidate_results)

    # Establish baseline for current location cell
    current_cell_data = next((c for c in ranked_candidates if c["is_current_cell"]), ranked_candidates[0] if ranked_candidates else {})
    current_profit = current_cell_data.get("relocation_adjusted_realized_profit_inr", current_cell_data.get("expected_profit_inr", 0))

    # --- Step 2: Compute Net Realized Uplift & Decision Metrics for Valid Destination Candidates ---
    valid_destinations = []
    for c in ranked_candidates:
        if (
            c.get("is_current_cell")
            or c.get("is_water")
            or is_water_cell(c["h3_cell"])
            or is_water_location(c["latitude"], c["longitude"])
            or c.get("distance_km", 0) > req.search_radius_km
            or c.get("expected_profit_inr", 0) <= 0
        ):
            continue

        travel_min = c.get("estimated_travel_time_min", 0)
        dist_km = c.get("distance_km", 0)

        # travel_fraction = max(0, (60 - travel_time_minutes) / 60)
        travel_fraction = round(max(0.0, (60.0 - travel_min) / 60.0), 4)

        # relocation-adjusted realized profit = candidate_predicted_profit * travel_fraction - fuel_cost
        cand_pred_profit = c.get("candidate_predicted_profit_inr", c.get("expected_profit_inr", 0))
        relocation_adjusted_realized_profit = round(cand_pred_profit * travel_fraction - c.get("fuel_cost_inr", 0), 2)

        # Net Realized Uplift vs Staying Put
        realized_net_uplift = round(relocation_adjusted_realized_profit - current_profit, 2)
        improv_pct = round((realized_net_uplift / max(abs(current_profit), 1)) * 100, 1)

        c["travel_fraction"] = travel_fraction
        c["relocation_adjusted_realized_profit_inr"] = relocation_adjusted_realized_profit
        c["realized_next_hour_profit_inr"] = relocation_adjusted_realized_profit
        c["realized_net_uplift_inr"] = realized_net_uplift
        c["profit_improvement_inr"] = realized_net_uplift
        c["profit_improvement_pct"] = improv_pct

        # Multi-factor strategic classification (Net Realized Uplift + Travel Friction Constraints)
        if realized_net_uplift < STAY_THRESHOLD_INR_PER_HOUR:
            c["decision_verdict"] = "STAY_PUT"
            c["decision_action"] = "STAY HERE"
            reason = f"Staying put is optimal; travel time ({travel_min} min) and fuel reduce effective net gain (+₹{realized_net_uplift:.0f}/hr)."
        elif travel_min > MAX_RECOMMENDED_TRAVEL_MIN or dist_km > MAX_RECOMMENDED_TRAVEL_KM:
            c["decision_verdict"] = "CONSIDER_MOVE"
            c["decision_action"] = "CONSIDER"
            reason = f"Positive demand (+₹{realized_net_uplift:.0f}/hr net uplift), but travel friction is high ({travel_min} min / {dist_km:.1f} km)."
        elif realized_net_uplift < CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR:
            c["decision_verdict"] = "CONSIDER_MOVE"
            c["decision_action"] = "CONSIDER"
            reason = f"Moderate move (+₹{realized_net_uplift:.0f}/hr net uplift within {travel_min} min drive); evaluate spot footfall."
        else:
            c["decision_verdict"] = "RECOMMENDED_MOVE"
            c["decision_action"] = "MOVE"
            reason = f"Strong move (+₹{realized_net_uplift:.0f}/hr net realized uplift); high demand cluster {travel_min} min away."

        c["decision_rationale"] = reason
        c["decision_reason"] = reason
        c["decision"] = c["decision_verdict"]
        c["reason"] = reason
        c["current_profit"] = current_profit
        c["recommended_profit"] = relocation_adjusted_realized_profit
        c["profit_uplift"] = realized_net_uplift
        c["travel_time"] = travel_min
        c["distance"] = dist_km

        valid_destinations.append(c)

    # --- Step 3: Final Recommendation Selection (Primarily ordered by Net Realized Uplift with Travel Constraints) ---
    recommendations = sorted(
        valid_destinations,
        key=lambda x: (
            1 if x["decision_verdict"] == "RECOMMENDED_MOVE" else 0,
            x["realized_net_uplift_inr"],
            x.get("recommendation_score", 0),
        ),
        reverse=True,
    )[:req.top_n]

    # Assign rank and add H3 boundary polygons for top recommendations
    for i, c in enumerate(recommendations):
        c["rank"] = i + 1
        try:
            c["boundary"] = [list(pt) for pt in h3.cell_to_boundary(c["h3_cell"])]
        except Exception:
            c["boundary"] = []

    # Build demand heatmap data for all evaluated land candidates
    max_demand = max([c["expected_customers"] for c in candidate_results], default=1)
    min_demand = min([c["expected_customers"] for c in candidate_results], default=0)
    demand_span = max(max_demand - min_demand, 1)

    top_rec_cells = {c["h3_cell"]: c["rank"] for c in recommendations}

    demand_heatmap = []
    for c in candidate_results:
        if c.get("is_water") or is_water_cell(c["h3_cell"]) or is_water_location(c["latitude"], c["longitude"]) or (not c["is_current_cell"] and c["distance_km"] > req.search_radius_km):
            continue
        try:
            bnd = [list(pt) for pt in h3.cell_to_boundary(c["h3_cell"])]
        except Exception:
            bnd = []
        intensity = round((c["expected_customers"] - min_demand) / demand_span, 3)
        demand_heatmap.append({
            "h3_cell": c["h3_cell"],
            "latitude": c["latitude"],
            "longitude": c["longitude"],
            "boundary": bnd,
            "expected_customers": c["expected_customers"],
            "expected_profit_inr": c["expected_profit_inr"],
            "demand_intensity": intensity,
            "is_current_cell": c["is_current_cell"],
            "rank": top_rec_cells.get(c["h3_cell"]),
        })

    # Clean _features from output
    for c in recommendations:
        c.pop("_features", None)
    current_cell_data.pop("_features", None)

    # Top-level strategic decision
    top_rec = recommendations[0] if recommendations else None
    if not top_rec or top_rec["decision_verdict"] == "STAY_PUT":
        top_reason = top_rec["decision_reason"] if top_rec else "No nearby candidate provides enough incremental profit to justify relocation."
        overall_decision = {
            "verdict": "STAY_PUT",
            "action": "STAY HERE",
            "decision": "STAY_PUT",
            "headline": "Stay at Current Location",
            "rationale": top_reason,
            "reason": top_reason,
            "current_profit": current_profit,
            "recommended_profit": top_rec["expected_profit_inr"] if top_rec else current_profit,
            "profit_uplift": top_rec["profit_improvement_inr"] if top_rec else 0.0,
            "travel_time": top_rec.get("estimated_travel_time_min", 0) if top_rec else 0,
            "distance": top_rec.get("distance_km", 0.0) if top_rec else 0.0,
            "top_net_improvement_inr": top_rec["profit_improvement_inr"] if top_rec else 0.0,
            "stay_threshold_inr": STAY_THRESHOLD_INR_PER_HOUR,
            "consider_threshold_inr": CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR,
        }
    elif top_rec["decision_verdict"] == "CONSIDER_MOVE":
        top_reason = top_rec["decision_reason"]
        overall_decision = {
            "verdict": "CONSIDER_MOVE",
            "action": "CONSIDER MOVE",
            "decision": "CONSIDER_MOVE",
            "headline": f"Consider Relocating to #{top_rec['rank']} ({top_rec['distance_km']} km, ~{top_rec['estimated_travel_time_min']} min)",
            "rationale": top_reason,
            "reason": top_reason,
            "current_profit": current_profit,
            "recommended_profit": top_rec["expected_profit_inr"],
            "profit_uplift": top_rec["profit_improvement_inr"],
            "travel_time": top_rec.get("estimated_travel_time_min", 0),
            "distance": top_rec.get("distance_km", 0.0),
            "top_net_improvement_inr": top_rec["profit_improvement_inr"],
            "stay_threshold_inr": STAY_THRESHOLD_INR_PER_HOUR,
            "consider_threshold_inr": CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR,
        }
    else:
        top_reason = top_rec["decision_reason"]
        overall_decision = {
            "verdict": "RECOMMENDED_MOVE",
            "action": "RECOMMENDED MOVE",
            "decision": "RECOMMENDED_MOVE",
            "headline": f"Recommended Move to #{top_rec['rank']} (+₹{top_rec['profit_improvement_inr']:.0f}/hr)",
            "rationale": top_reason,
            "reason": top_reason,
            "current_profit": current_profit,
            "recommended_profit": top_rec["expected_profit_inr"],
            "profit_uplift": top_rec["profit_improvement_inr"],
            "travel_time": top_rec.get("estimated_travel_time_min", 0),
            "distance": top_rec.get("distance_km", 0.0),
            "top_net_improvement_inr": top_rec["profit_improvement_inr"],
            "stay_threshold_inr": STAY_THRESHOLD_INR_PER_HOUR,
            "consider_threshold_inr": CONSIDER_MOVE_THRESHOLD_INR_PER_HOUR,
        }


    # Comprehensive Hybrid Data Quality Summary
    data_quality_summary = {
        "weather": "live" if weather_now.get("source_type") == "real_live" else "fallback",
        "traffic": "live" if traffic_prov.source_type == "real_live" else "proxy",
        "competition": "live" if competition_prov.source_type == "real_live" else "real_periodic",
        "poi": "real_periodic",
        "events": "live" if event_prov.source_type == "real_live" else "simulated",
        "fuel": "manual",
        "demand": "ml_estimate",
        "revenue": "derived",
        "profit": "derived",
    }

    return {
        "request_id": f"rec_{now.strftime('%Y%m%d%H%M%S')}",
        "generated_at": now.isoformat(),
        "search_radius_km": req.search_radius_km,
        "decision": overall_decision,
        "current_location": {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "h3_cell": loc_ctx.h3_cell,
            "is_water": loc_ctx.is_water,
            "boundary": [list(pt) for pt in h3.cell_to_boundary(loc_ctx.h3_cell)],
        },
        "current_estimated_demand": current_cell_data.get("expected_customers"),
        "current_expected_profit_inr": current_cell_data.get("expected_profit_inr"),
        "current_weather": {
            "condition": weather_now.get("weather_condition"),
            "temperature_c": weather_now.get("temperature"),
            "source_type": weather_now.get("source_type"),
            "source_name": weather_now.get("source_name"),
            "retrieved_at": weather_now.get("retrieved_at"),
            "from_cache": weather_now.get("from_cache", False),
        },
        "recommendations": recommendations,
        "demand_heatmap": demand_heatmap,
        "total_candidates_evaluated": len(demand_heatmap),
        "data_freshness": global_freshness or {},
        "data_quality": data_quality_summary,
        "model_info": {
            "model_type": model.meta.get("model_type"),
            "test_mae_customers": model.meta.get("metrics", {}).get("test_mae"),
            "trained_at": model.meta.get("trained_at"),
            "note": "Demand is an ML estimate based on real-world contextual signals — not a live sensor count.",
        },
    }
