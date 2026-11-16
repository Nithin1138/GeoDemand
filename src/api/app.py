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
from typing import Optional, List, Dict, Any

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "api"))
sys.path.insert(0, str(ROOT / "src" / "fleet"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)


from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.location import get_location_context, generate_candidates
from api.feature_assembler import FeatureAssembler
from api.recommender import (
    DemandModel,
    calculate_business_metrics,
    rank_candidates,
    generate_explanation,
    calculate_category_demand,
    compute_category_affinity,
)
from api.providers.competition import CompetitionProviderFactory
from api.providers.traffic import TrafficProviderFactory
from api.providers.events import EventProviderFactory
from api.providers.fuel import FuelPriceProvider
from api.providers.routing import RoutingProviderFactory
from api.spatial_filter import is_water_location, is_water_cell
from api.poi_resolver import resolve_candidate_landmark
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


OPENAPI_TAGS = [
    {
        "name": "System",
        "description": "System liveness, public client config, and data source health & freshness telemetry.",
    },
    {
        "name": "Location",
        "description": "Geospatial resolution, H3 indexing (res 8), local timezone, and water exclusion check.",
    },
    {
        "name": "Demand Forecasting",
        "description": "Point-in-time demand estimation with calibrated quantiles (P10, P50, P90) and unit economics.",
    },
    {
        "name": "Recommendations",
        "description": "Real-time candidate ranking, travel-aware realized net uplift, and strategic relocation decisions.",
    },
    {
        "name": "Scenario Simulation",
        "description": "Sensitivity analysis exploring price, weather, and operational counterfactuals.",
    },
    {
        "name": "Routing",
        "description": "Real street network navigation geometry from OpenStreetMap OSRM.",
    },
    {
        "name": "Shift Optimization",
        "description": "Multi-stop daily shift planning and VRPTW itinerary optimization using Google OR-Tools.",
    },
    {
        "name": "Fleet Management",
        "description": "Multi-vendor fleet dispatch, cell capacity tracking, and atomic H3 reservations.",
    },
]

app = FastAPI(
    title="GeoDemand AI",
    description="Production-grade geospatial demand intelligence for mobile vendors in urban India.",
    version="2.1.0",
    openapi_tags=OPENAPI_TAGS,
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
# Pydantic Request & Response Schemas (Imported & Re-exported from api.schemas)
# ---------------------------------------------------------------------------

from api.schemas import (
    LocationRequest,
    RecommendationRequest,
    PredictDemandRequest,
    BatchPredictDemandRequest,
    RecommendZonesRequest,
    ScenarioSimulateRequest,
    RoutingRequest,
    PingResponse,
    SystemConfigResponse,
    DataHealthResponse,
    LocationContextResponse,
    PredictDemandResponse,
    BatchPredictDemandResponse,
    ScenarioSimulateResponse,
    LiveRecommendationsResponse,
    OverallDecisionResponse,
    RecommendationItem,
    DemandHeatmapItem,
    BusinessEconomicsResponse,
    APICacheStatsResponse,
    StreetRouteResponse,
    ShiftPlanRequest,
    ShiftPlanResponse,
    FleetReserveCellRequest,
    FleetReserveCellResponse,
    FleetCancelReservationRequest,
    FleetCancelReservationResponse,
    CellOccupancyResponse,
    FleetDispatchRequest,
    FleetDispatchResponse,
    FleetDispatchVendorItem,
    DecisionVerdict,
)
from api.cache import api_response_cache, generate_cache_key
from api.scenario_simulation import ScenarioSimulator
from api.business_economics import FUEL_PRICE_INR_PER_LITRE
from optimization.shift_optimizer import ShiftOptimizer
from fleet.fleet_manager import FleetManager, Vendor, VendorStatus, ReservationStatus





# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def dashboard():
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "GeoDemand AI v2.1 — POST /v1/recommendations/live to begin"})


@app.get("/v1/system/ping", response_model=PingResponse, tags=["System"], summary="Operational liveness ping")
async def ping():
    return {"status": "ok", "service": "GeoDemand AI", "version": "2.1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/v1/system/config", response_model=SystemConfigResponse, tags=["System"], summary="Public client configuration")
async def system_config():
    """Returns public client configuration like Mapbox token and default coordinates."""
    import os
    return {
        "mapbox_token": os.getenv("MAPBOX_TOKEN"),
        "live_mode": os.getenv("LIVE_MODE", "true").lower() == "true",
        "default_lat": 16.5062,
        "default_lng": 80.6480,
        "city": "Vijayawada",
        "state": "Andhra Pradesh",
    }


@app.get("/v1/system/data-health", response_model=DataHealthResponse, tags=["System"], summary="Data source health and freshness telemetry")
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


@app.get("/v1/system/cache-stats", response_model=APICacheStatsResponse, tags=["System"], summary="API response cache health and telemetry")
async def cache_stats():
    """Inspect memory cache size, hits, misses, and hit ratio."""
    return api_response_cache.get_stats()


@app.post("/v1/system/cache-clear", tags=["System"], summary="Clear API response cache")
async def cache_clear(api_key: str = Depends(verify_api_key)):
    """Evict all items from the API response cache."""
    api_response_cache.clear()
    return {"status": "ok", "message": "API response cache cleared successfully."}


@app.get("/v1/routing/route", response_model=StreetRouteResponse, tags=["Routing"], summary="Road routing coordinates (GET)")
@app.post("/v1/routing/route", response_model=StreetRouteResponse, tags=["Routing"], summary="Road routing coordinates (POST)")
async def get_street_route(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float):
    """Returns real street road navigation route coordinates [[lat, lng], ...] for map rendering."""
    provider = RoutingProviderFactory.get_provider()
    return provider.get_route_geometry(origin_lat, origin_lng, dest_lat, dest_lng)


@app.post("/v1/routes/shift-plan", response_model=ShiftPlanResponse, tags=["Shift Optimization"], summary="Multi-stop daily shift planning and VRPTW itinerary optimization")
async def shift_plan(req: ShiftPlanRequest, api_key: str = Depends(verify_api_key)):
    """
    Generate an optimal multi-stop itinerary for an 8–12 hour vendor shift using Google OR-Tools:
    - Enforces shift start and end times
    - Enforces minimum dwell times and commercial time windows
    - Accounts for travel time, distance, fuel expenses, and setup/teardown overhead
    - Maximizes expected shift net profit over staying put
    """
    if _model_health["status"] not in ("loaded",):
        raise HTTPException(status_code=503, detail="ML model not loaded")

    cache_key = generate_cache_key("shift_plan", req.model_dump())
    cached_val = api_response_cache.get(cache_key)
    if cached_val is not None:
        return cached_val

    # Generate candidate zones around origin
    rec_req = RecommendationRequest(
        latitude=req.latitude,
        longitude=req.longitude,
        vendor_category=req.vendor_category,
        search_radius_km=req.search_radius_km,
        top_n=10,
    )
    rec_res = await recommendations_live(rec_req, api_key=api_key)
    candidates = rec_res.get("recommendations", [])

    optimizer = ShiftOptimizer()
    plan = optimizer.optimize_shift(
        origin_lat=req.latitude,
        origin_lng=req.longitude,
        candidates=candidates,
        vendor_category=req.vendor_category,
        shift_start_time=req.shift_start_time,
        shift_duration_hours=req.shift_duration_hours,
        max_relocation_stops=req.max_relocation_stops,
        min_dwell_time_minutes=req.min_dwell_time_minutes,
        setup_time_minutes=req.setup_time_minutes,
        teardown_time_minutes=req.teardown_time_minutes,
        average_order_value=req.average_order_value,
        fuel_price_per_litre=req.fuel_price_inr or FUEL_PRICE_INR_PER_LITRE,
    )

    api_response_cache.set(cache_key, plan, ttl_seconds=60)
    return plan


# ---------------------------------------------------------------------------
# Fleet Management Endpoints
# ---------------------------------------------------------------------------

fleet_manager = FleetManager()


@app.post("/v1/fleet/reserve-cell", response_model=FleetReserveCellResponse, tags=["Fleet Management"], summary="Reserve an H3 cell with capacity enforcement")
async def fleet_reserve_cell(req: FleetReserveCellRequest, api_key: str = Depends(verify_api_key)):
    """
    Atomically reserve a trading spot in an H3 cell.
    - Enforces total cell physical capacity and category limits
    - Supports soft reservations with TTL auto-expiration or hard confirmed reservations
    - Prevents double-booking and concurrent conflict races
    """
    ok, res, msg = fleet_manager.reserve_cell(
        vendor_id=req.vendor_id,
        h3_cell=req.h3_cell,
        vendor_category=req.vendor_category,
        duration_minutes=req.duration_minutes,
        is_soft=req.is_soft,
        ttl_minutes=req.ttl_minutes,
        deposit_amount_inr=req.deposit_amount_inr,
        fleet_id=req.fleet_id,
    )
    if not ok or not res:
        return {
            "status": "REJECTED",
            "reservation_id": None,
            "vendor_id": req.vendor_id,
            "h3_cell": req.h3_cell,
            "vendor_category": req.vendor_category,
            "is_soft": req.is_soft,
            "expires_at": None,
            "slot_start": None,
            "slot_end": None,
            "message": msg,
            "rejection_reason": msg,
        }

    return {
        "status": "CONFIRMED",
        "reservation_id": res.reservation_id,
        "vendor_id": res.vendor_id,
        "h3_cell": res.h3_cell,
        "vendor_category": res.category,
        "is_soft": res.is_soft,
        "expires_at": res.expires_at.isoformat(),
        "slot_start": res.slot_start.isoformat(),
        "slot_end": res.slot_end.isoformat(),
        "message": msg,
        "rejection_reason": None,
    }


@app.post("/v1/fleet/cancel-reservation", response_model=FleetCancelReservationResponse, tags=["Fleet Management"], summary="Cancel an active or pending reservation")
async def fleet_cancel_reservation(req: FleetCancelReservationRequest, api_key: str = Depends(verify_api_key)):
    """Cancels a reservation and immediately frees up cell capacity."""
    ok, msg, cell = fleet_manager.cancel_reservation(
        reservation_id=req.reservation_id,
        vendor_id=req.vendor_id,
    )
    return {
        "status": "CANCELLED" if ok else "ERROR",
        "reservation_id": req.reservation_id,
        "released_h3_cell": cell,
        "message": msg,
    }


@app.get("/v1/fleet/cell-occupancy", response_model=CellOccupancyResponse, tags=["Fleet Management"], summary="Get H3 cell occupancy, capacity, and demand multipliers")
async def fleet_cell_occupancy(h3_cell: str, vendor_category: Optional[str] = None):
    """Returns active capacity, current occupancies, and competition multipliers for an H3 cell."""
    if not h3.is_valid_cell(h3_cell):
        raise HTTPException(status_code=400, detail=f"Invalid H3 cell: {h3_cell}")
    return fleet_manager.get_cell_occupancy(h3_cell, vendor_category)


@app.post("/v1/fleet/dispatch", response_model=FleetDispatchResponse, tags=["Fleet Management"], summary="Coordinate multi-vendor dispatch minimizing clustering")
async def fleet_dispatch(req: FleetDispatchRequest, api_key: str = Depends(verify_api_key)):
    """
    Coordinate dispatch across multiple vendors:
    - Maximizes fleet-wide total net profit
    - Applies non-linear demand cannibalization penalty D_eff = D_base / (1 + lambda * N)
    - Minimizes unnecessary clustering across commercial zones
    - Automatically creates soft reservations for dispatched spots
    """
    vendor_objs: List[Vendor] = []
    for vi in req.vendors:
        v = Vendor(
            vendor_id=vi.vendor_id,
            category=vi.vendor_category,
            current_lat=vi.latitude,
            current_lng=vi.longitude,
            current_h3=vi.current_h3_cell or "",
            status=VendorStatus(vi.status) if vi.status in [s.value for s in VendorStatus] else VendorStatus.IDLE,
            average_order_value=vi.average_order_value,
            fuel_efficiency_km_l=vi.fuel_efficiency_km_l,
            speed_km_h=vi.speed_km_h,
            fleet_id=req.fleet_id,
        )
        vendor_objs.append(v)

    res = fleet_manager.dispatch_fleet(
        vendors=vendor_objs,
        candidate_cells=req.candidate_cells,
        fleet_id=req.fleet_id,
        search_radius_km=req.search_radius_km,
        max_travel_km=req.max_travel_km,
        fuel_price_inr=req.fuel_price_inr or FUEL_PRICE_INR_PER_LITRE,
        auto_reserve_soft=req.auto_reserve_soft,
        soft_reservation_ttl_minutes=req.soft_reservation_ttl_minutes,
    )
    return res


@app.get("/v1/fleet/vendors", tags=["Fleet Management"], summary="List registered fleet vendors")
async def fleet_list_vendors(fleet_id: Optional[str] = None, status: Optional[str] = None, api_key: str = Depends(verify_api_key)):
    """Lists registered vendors with status and active reservation references."""
    stat_enum = None
    if status:
        try:
            stat_enum = VendorStatus(status.upper())
        except ValueError:
            pass
    vendors = fleet_manager.list_vendors(fleet_id=fleet_id, status=stat_enum)
    return [v.to_dict() for v in vendors]


@app.post("/v1/location/context", response_model=LocationContextResponse, tags=["Location"], summary="GPS location context and H3 cell")
async def location_context(req: LocationRequest):
    """GPS → H3 cell, timestamp, and location context."""
    ctx = get_location_context(req.latitude, req.longitude)
    return ctx.to_dict()


@app.post("/v1/predict/demand", response_model=PredictDemandResponse, tags=["Demand Forecasting"], summary="Demand prediction and unit economics")
async def predict_demand(req: PredictDemandRequest, api_key: str = Depends(verify_api_key)):
    """Predict customer demand and business economics for a single location."""
    if _model_health["status"] not in ("loaded",):
        raise HTTPException(status_code=503, detail="ML model not loaded")

    cache_key = generate_cache_key("predict", req.model_dump())
    cached_val = api_response_cache.get(cache_key)
    if cached_val is not None:
        return cached_val

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
        "inventory_capacity": getattr(req, "inventory_capacity", 200),
        "preparation_time": getattr(req, "preparation_time", 5.0),
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
    if hasattr(assembler, "validate_features"):
        assembler.validate_features(features)

    model = DemandModel.get()
    raw_uncertainty = model.predict_with_uncertainty(features)
    raw_p50 = raw_uncertainty["p50"]

    # Category-calibrated customer conversion & spatial affinity
    cat_customers, category_affinity = calculate_category_demand(
        raw_p50=raw_p50,
        vendor_category=req.vendor_category,
        features=features,
    )

    scale_ratio = cat_customers / max(1, raw_p50)
    uncertainty = {
        "p10": max(1, int(round(raw_uncertainty.get("p10", raw_p50 * 0.8) * scale_ratio))),
        "p50": cat_customers,
        "p90": max(cat_customers, int(round(raw_uncertainty.get("p90", raw_p50 * 1.2) * scale_ratio))),
    }
    expected_customers = cat_customers

    biz = calculate_business_metrics(
        expected_customers=expected_customers,
        aov=req.average_order_value,
        variable_cost_rate=req.variable_cost_rate,
        fixed_cost_per_day=req.fixed_cost_per_day,
    )

    model_ver = model.meta.get("model_version", "v2.1.0")
    model_arch = model.meta.get("model_type", "lightgbm")

    res = {
        "location": {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "h3_cell": loc_ctx.h3_cell,
            "is_water": loc_ctx.is_water,
        },
        "vendor_category": req.vendor_category,
        "prediction_timestamp": now.isoformat(),
        "model_version": model_ver,
        "model_type": model_arch,
        "prediction": {
            "expected_customers": expected_customers,
            "p10": uncertainty["p10"],
            "p50": uncertainty["p50"],
            "p90": uncertainty["p90"],
            "confidence_interval": uncertainty,
            "prediction_interval": uncertainty,
            "category_affinity": category_affinity,
            "prediction_timestamp": now.isoformat(),
            "model_version": model_ver,
        },
        "economics": biz,
        "data_freshness": freshness,
    }
    api_response_cache.set(cache_key, res, ttl_seconds=60)
    return res


@app.post("/v1/predict/batch", response_model=BatchPredictDemandResponse, tags=["Demand Forecasting"], summary="Batch demand prediction and unit economics")
async def predict_batch(req: BatchPredictDemandRequest, api_key: str = Depends(verify_api_key)):
    """Evaluate customer demand and unit economics for multiple vendor locations or profiles in batch (up to 50 items)."""
    if _model_health["status"] not in ("loaded",):
        raise HTTPException(status_code=503, detail="ML model not loaded")

    now = datetime.now(timezone.utc)
    results = []
    total_customers = 0
    total_revenue = 0.0
    total_profit = 0.0
    best_idx = 0
    best_profit = float("-inf")

    for i, item in enumerate(req.items):
        item_res = await predict_demand(item, api_key=api_key)
        results.append(item_res)
        cust = item_res["prediction"]["expected_customers"]
        rev = item_res["economics"]["expected_revenue_inr"]
        prof = item_res["economics"]["expected_profit_inr"]
        total_customers += cust
        total_revenue += rev
        total_profit += prof
        if prof > best_profit:
            best_profit = prof
            best_idx = i

    avg_profit = round(total_profit / len(req.items), 2)

    return {
        "batch_id": f"batch_{now.strftime('%Y%m%d%H%M%S')}_{len(req.items)}",
        "processed_at": now.isoformat(),
        "count": len(results),
        "results": results,
        "summary": {
            "total_locations": len(req.items),
            "total_expected_customers": total_customers,
            "total_expected_revenue_inr": round(total_revenue, 2),
            "total_expected_profit_inr": round(total_profit, 2),
            "average_profit_per_location_inr": avg_profit,
            "best_location_index": best_idx,
            "best_location_profit_inr": round(best_profit if best_profit != float("-inf") else 0.0, 2),
        },
    }


@app.post("/v1/recommend/zones", response_model=LiveRecommendationsResponse, tags=["Recommendations"], summary="Target candidate zones within search radius")
async def recommend_zones(req: RecommendZonesRequest, api_key: str = Depends(verify_api_key)):
    """Target high-demand candidate zones within radius."""
    rec_req = RecommendationRequest(
        latitude=req.latitude,
        longitude=req.longitude,
        vendor_category=req.vendor_category,
        search_radius_km=req.search_radius_km,
        top_n=req.top_n,
    )
    return await recommendations_live(rec_req, api_key=api_key)


@app.post("/v1/scenario/simulate", response_model=ScenarioSimulateResponse, tags=["Scenario Simulation"], summary="Simulate business impact under varied conditions")
async def scenario_simulate(req: ScenarioSimulateRequest, api_key: str = Depends(verify_api_key)):
    """Simulate business impact under varied AOV, rainfall, competitor density, fuel price, category, or hour scenario without retraining."""
    if _model_health["status"] not in ("loaded",):
        raise HTTPException(status_code=503, detail="ML model not loaded")

    cache_key = generate_cache_key("scenario", req.model_dump())
    cached_val = api_response_cache.get(cache_key)
    if cached_val is not None:
        return cached_val

    simulator = ScenarioSimulator(
        model=DemandModel.get(),
        assembler=_get_assembler(),
    )
    comp_prov, traffic_prov, event_prov, fuel_prov = _get_providers()
    weather_client = _get_weather_client()

    res = simulator.simulate(
        latitude=req.latitude,
        longitude=req.longitude,
        vendor_category=req.vendor_category,
        baseline_aov=req.baseline_aov,
        simulated_aov=req.simulated_aov,
        simulated_rainfall_mm=req.simulated_rainfall_mm,
        simulated_is_rain=req.simulated_is_rain,
        simulated_weather_condition=req.simulated_weather_condition,
        simulated_competitor_count_delta=req.simulated_competitor_count_delta,
        simulated_competition_score=req.simulated_competition_score,
        simulated_fuel_price_inr=req.simulated_fuel_price_inr,
        simulated_vendor_category=req.simulated_vendor_category,
        simulated_hour=req.simulated_hour,
        distance_km=req.distance_km if req.distance_km is not None else 1.5,
        weather_client=weather_client,
        competition_provider=comp_prov,
        event_provider=event_prov,
        fuel_provider=fuel_prov,
    )
    api_response_cache.set(cache_key, res, ttl_seconds=60)
    return res


@app.post("/v1/recommendations/live", response_model=LiveRecommendationsResponse, tags=["Recommendations"], summary="Full live vendor recommendation pipeline")
async def recommendations_live(req: RecommendationRequest, api_key: str = Depends(verify_api_key)):
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

    cache_key = generate_cache_key("recommendations", req.model_dump())
    cached_val = api_response_cache.get(cache_key)
    if cached_val is not None:
        return cached_val

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

    # --- Assemble features + predict for each candidate (concurrently) ---
    from concurrent.futures import ThreadPoolExecutor

    def _eval_candidate(cand):
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
        features, freshness = assembler.assemble(
            h3_cell=cand.h3_cell,
            vendor_profile=vendor_profile,
            weather=weather_forecast,  # next-hour forecast (temporal contract)
            competition=comp,
            event=event,
            decision_timestamp=now,
        )

        raw_uncertainty = model.predict_with_uncertainty(features)
        raw_p50 = raw_uncertainty["p50"]

        # Category-calibrated customer conversion & spatial affinity
        cat_customers, category_affinity = calculate_category_demand(
            raw_p50=raw_p50,
            vendor_category=req.vendor_category,
            features=features,
        )

        scale_ratio = cat_customers / max(1, raw_p50)
        uncertainty = {
            "p10": max(1, int(round(raw_uncertainty.get("p10", raw_p50 * 0.8) * scale_ratio))),
            "p50": cat_customers,
            "p90": max(cat_customers, int(round(raw_uncertainty.get("p90", raw_p50 * 1.2) * scale_ratio))),
        }
        expected_customers = cat_customers

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

        landmark_info = resolve_candidate_landmark(
            cand.centroid_lat, cand.centroid_lng, cand.h3_cell, features=features
        )

        return {
            "h3_cell": cand.h3_cell,
            "latitude": cand.centroid_lat,
            "longitude": cand.centroid_lng,
            "landmark_name": landmark_info["landmark_name"],
            "landmark_display": landmark_info["landmark_display"],
            "landmark_type": landmark_info["landmark_type"],
            "is_college": landmark_info["is_college"],
            "landmark_emoji": landmark_info["emoji"],
            "distance_km": cand.distance_km,
            "estimated_travel_time_min": travel_time_min,
            "travel_fraction": biz["travel_fraction"],
            "is_current_cell": cand.is_current_cell,
            "is_water": getattr(cand, "is_water", False),
            "expected_customers": expected_customers,
            "prediction_interval": uncertainty,
            "confidence_interval": uncertainty,
            "category_affinity": category_affinity,
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
            "_features": features,
            "_freshness": freshness,
        }

    # Parallel evaluation across thread pool ensures sub-2-second response times
    max_workers = min(12, max(1, len(candidates_raw)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        raw_eval_results = list(executor.map(_eval_candidate, candidates_raw))

    candidate_results = []
    global_freshness = None
    for res in raw_eval_results:
        if global_freshness is None and "_freshness" in res:
            global_freshness = res["_freshness"]
        candidate_results.append(res)

    # --- Step 1: Candidate Ranking (Category-Aware Multi-Factor Scoring) ---
    ranked_candidates = rank_candidates(candidate_results, vendor_category=req.vendor_category)

    # Establish baseline for current location cell
    current_cell_data = next((c for c in ranked_candidates if c["is_current_cell"]), ranked_candidates[0] if ranked_candidates else {})
    current_profit = current_cell_data.get("relocation_adjusted_realized_profit_inr", current_cell_data.get("expected_profit_inr", 0))

    # --- Step 2: Compute Net Realized Uplift & Decision Metrics for Valid Destination Candidates ---
    valid_destinations = []
    viable_destinations = []
    session_hours = 4.0
    session_minutes = session_hours * 60.0
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

        # Realistic shift travel fraction (amortizing travel time over a standard 4-hour vendor session)
        travel_fraction = round(max(0.75, (session_minutes - travel_min) / session_minutes), 4)

        cand_pred_profit = c.get("candidate_predicted_profit_inr", c.get("expected_profit_inr", 0))
        hourly_fuel_cost = round(c.get("fuel_cost_inr", 0) / session_hours, 2)
        relocation_adjusted_realized_profit = round(cand_pred_profit * travel_fraction - hourly_fuel_cost, 2)

        # Realized net profit uplift per hour vs staying at current position
        realized_net_uplift = round(relocation_adjusted_realized_profit - current_profit, 2)
        improv_pct = round((realized_net_uplift / max(abs(current_profit), 1)) * 100, 1)

        c["travel_fraction"] = travel_fraction
        c["relocation_adjusted_realized_profit_inr"] = relocation_adjusted_realized_profit
        c["realized_next_hour_profit_inr"] = relocation_adjusted_realized_profit
        c["realized_net_uplift_inr"] = realized_net_uplift
        c["profit_improvement_inr"] = realized_net_uplift
        c["profit_improvement_pct"] = improv_pct

        # Multi-factor strategic classification via Decision Engine
        from api.decision_engine import evaluate_decision
        dec = evaluate_decision(c, current_cell_data)

        c["decision_verdict"] = dec.decision_verdict
        c["decision_action"] = dec.decision_action
        c["decision_rationale"] = dec.decision_reason
        c["decision_reason"] = dec.decision_reason
        c["decision"] = dec.decision_verdict
        c["reason"] = dec.decision_reason
        c["current_profit"] = dec.current_profit
        c["recommended_profit"] = dec.recommended_profit
        c["profit_uplift"] = dec.realized_net_uplift_inr
        c["travel_time"] = dec.travel_time_min
        c["distance"] = dec.distance_km
        c["is_move_recommended"] = dec.is_move_recommended

        valid_destinations.append(c)

        # Commercial viability filter tailored to mobile vendor type
        feat = c.get("_features", {})
        poi_cols = [
            "office_count", "college_count", "school_count", "hospital_count",
            "mall_count", "restaurant_count", "bus_stop_count", "park_count",
        ]
        total_pois = sum(feat.get(col, 0.0) for col in poi_cols)
        bldg_cnt = feat.get("building_count", 0.0)
        comm_ratio = feat.get("commercial_ratio", 0.0)
        ind_ratio = feat.get("industrial_ratio", 0.0)
        road_dens = feat.get("road_density", 0.5)

        cat = (req.vendor_category or "food").lower()
        is_viable = False
        if road_dens >= 0.10:
            if cat == "repair":
                if ind_ratio >= 0.08 or comm_ratio >= 0.15 or total_pois >= 5 or bldg_cnt >= 2:
                    is_viable = True
            elif cat == "fruits":
                hosp_cnt = feat.get("hospital_count", 0.0)
                park_cnt = feat.get("park_count", 0.0)
                if (hosp_cnt >= 3 or park_cnt >= 2) or total_pois >= 8 or (bldg_cnt >= 2 and total_pois >= 3):
                    is_viable = True
            else:
                if (
                    total_pois >= 8
                    or (bldg_cnt >= 2 and total_pois >= 3)
                    or (comm_ratio >= 0.20 and total_pois >= 4)
                ):
                    is_viable = True

        if is_viable:
            viable_destinations.append(c)

    # Prioritize commercially viable destinations over empty land/fields
    target_destinations = viable_destinations if viable_destinations else valid_destinations

    # --- Step 3: Final Recommendation Selection (Ordered by Verdict & Category-Aware Multi-Factor Score) ---
    recommendations = sorted(
        target_destinations,
        key=lambda x: (
            1 if x["decision_verdict"] == "RECOMMENDED_MOVE" else (0.5 if x["decision_verdict"] == "CONSIDER_MOVE" else 0),
            x.get("recommendation_score", 0),
            x["realized_net_uplift_inr"],
            x.get("recommendation_score", 0),
            x["realized_net_uplift_inr"],
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

    res = {
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
        "data_freshness": {
            **(global_freshness or {}),
            "traffic": {
                "source_type": traffic_prov.source_type,
                "source_name": traffic_prov.source_name,
                "retrieved_at": now.isoformat(),
                "note": f"Live traffic from {traffic_prov.source_name}" if traffic_prov.source_type == "real_live" else "Time-of-day proxy model",
            },
            "static_poi": {
                **(global_freshness or {}).get("static_poi", {}),
                "source_type": "real_periodic",
            },
        },
        "data_quality": data_quality_summary,
        "model_info": {
            "model_type": model.meta.get("model_type"),
            "test_mae_customers": model.meta.get("metrics", {}).get("test_mae"),
            "trained_at": model.meta.get("trained_at"),
            "note": "Demand is an ML estimate based on real-world contextual signals — not a live sensor count.",
        },
    }
    api_response_cache.set(cache_key, res, ttl_seconds=60)
    return res
