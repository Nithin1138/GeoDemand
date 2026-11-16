"""
GeoDemand AI — Pydantic Request & Response Schemas

Formal data contract definitions for all GeoDemand AI live API endpoints:
- Location Context
- Demand Prediction
- Zone Recommendations
- Scenario Simulation
- Real Street Routing
- System Liveness & Data Health
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

from config.settings import (
    DEFAULT_AOV_INR,
    DEFAULT_VARIABLE_COST_RATE,
    DEFAULT_FIXED_COST_PER_DAY_INR,
)


# ===========================================================================
# Enums
# ===========================================================================

class DecisionVerdict(str, Enum):
    RECOMMENDED_MOVE = "RECOMMENDED_MOVE"
    CONSIDER_MOVE = "CONSIDER_MOVE"
    STAY_PUT = "STAY_PUT"


class VendorCategory(str, Enum):
    FOOD = "food"
    BEVERAGES = "beverages"
    FRUITS = "fruits"
    VEGETABLES = "vegetables"
    SNACKS = "snacks"
    FLOWERS = "flowers"
    APPAREL = "apparel"
    REPAIR = "repair"
    GENERAL = "general"


# ===========================================================================
# Request Schemas
# ===========================================================================

class LocationRequest(BaseModel):
    """GPS coordinates request to resolve spatial context and H3 cell."""
    model_config = ConfigDict(extra="ignore")

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="WGS84 latitude coordinate in decimal degrees",
        json_schema_extra={"example": 16.5062},
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="WGS84 longitude coordinate in decimal degrees",
        json_schema_extra={"example": 80.6480},
    )


class PredictDemandRequest(BaseModel):
    """Request parameters for point demand prediction and unit economics."""
    model_config = ConfigDict(extra="ignore")

    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field(
        "food",
        description="Vendor commercial category (e.g. food, fruits, tea_coffee, repair)",
        json_schema_extra={"example": "food"},
    )
    average_order_value: float = Field(
        DEFAULT_AOV_INR,
        gt=0,
        description="Average order value (AOV) in INR",
        json_schema_extra={"example": 100.0},
    )
    variable_cost_rate: float = Field(
        DEFAULT_VARIABLE_COST_RATE,
        gt=0,
        lt=1,
        description="Cost of goods sold (COGS) as a fraction of revenue (0 to 1)",
        json_schema_extra={"example": 0.38},
    )
    fixed_cost_per_day: float = Field(
        DEFAULT_FIXED_COST_PER_DAY_INR,
        gt=0,
        description="Total daily fixed overhead costs in INR",
        json_schema_extra={"example": 500.0},
    )
    inventory_capacity: int = Field(
        200,
        gt=0,
        description="Vendor inventory unit capacity",
        json_schema_extra={"example": 200},
    )
    preparation_time: float = Field(
        5.0,
        gt=0,
        description="Order preparation time in minutes",
        json_schema_extra={"example": 5.0},
    )
    timestamp: Optional[datetime] = Field(
        None,
        description="Target prediction timestamp (ISO 8601). Defaults to current UTC time.",
        json_schema_extra={"example": "2026-10-05T12:00:00Z"},
    )


class RecommendZonesRequest(BaseModel):
    """Target high-demand candidate zones within radius."""
    model_config = ConfigDict(extra="ignore")

    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    search_radius_km: float = Field(
        3.0,
        gt=0,
        le=10.0,
        description="Candidate search radius in kilometers (max 10 km)",
        json_schema_extra={"example": 3.0},
    )
    top_n: int = Field(
        5,
        ge=1,
        le=10,
        description="Number of top candidate locations to return (1 to 10)",
        json_schema_extra={"example": 5},
    )


class ScenarioSimulateRequest(BaseModel):
    """Simulate profit and customer sensitivity across varying business assumptions."""
    model_config = ConfigDict(extra="ignore")

    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    baseline_aov: float = Field(
        DEFAULT_AOV_INR,
        gt=0,
        description="Baseline average order value in INR",
        json_schema_extra={"example": 100.0},
    )
    simulated_aov: Optional[float] = Field(
        None,
        gt=0,
        description="Hypothetical average order value in INR to simulate",
        json_schema_extra={"example": 150.0},
    )
    simulated_rainfall_mm: Optional[float] = Field(
        None,
        ge=0.0,
        description="Simulated precipitation in millimeters (e.g. 15.0 for heavy rain)",
        json_schema_extra={"example": 12.5},
    )
    simulated_is_rain: Optional[bool] = Field(
        None,
        description="Simulated rain occurrence flag",
        json_schema_extra={"example": True},
    )
    simulated_weather_condition: Optional[str] = Field(
        None,
        description="Simulated weather scenario (e.g. rainy, clear, heavy_rain)",
        json_schema_extra={"example": "rainy"},
    )
    simulated_competitor_count_delta: Optional[int] = Field(
        None,
        description="Simulated net change in nearby competitor count (e.g. +3 or -2)",
        json_schema_extra={"example": 3},
    )
    simulated_competition_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Simulated normalized competition score (0.0 to 1.0)",
        json_schema_extra={"example": 0.75},
    )
    simulated_fuel_price_inr: Optional[float] = Field(
        None,
        gt=0.0,
        description="Simulated fuel price in INR per liter (e.g. 110.0)",
        json_schema_extra={"example": 108.5},
    )
    simulated_vendor_category: Optional[str] = Field(
        None,
        description="Simulated alternative vendor category (e.g. beverages, fruits, repair)",
        json_schema_extra={"example": "beverages"},
    )
    simulated_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Simulated hour of day (0 to 23)",
        json_schema_extra={"example": 13},
    )
    distance_km: Optional[float] = Field(
        1.5,
        ge=0.0,
        description="Optional relocation distance in km for fuel and travel sensitivity",
        json_schema_extra={"example": 1.5},
    )


class BatchPredictDemandRequest(BaseModel):
    """Batch prediction request for multiple vendor locations or items."""
    model_config = ConfigDict(extra="ignore")

    items: List[PredictDemandRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of location and vendor profiles to score in batch (max 50)",
    )


class RecommendationRequest(BaseModel):
    """Full live recommendation pipeline request."""
    model_config = ConfigDict(extra="ignore")

    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": 80.6480})
    vendor_id: Optional[str] = Field(None, json_schema_extra={"example": "V_001"})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    vendor_name: Optional[str] = Field(None, json_schema_extra={"example": "Vijayawada Chaat Hub"})
    average_order_value: float = Field(DEFAULT_AOV_INR, gt=0, json_schema_extra={"example": 100.0})
    variable_cost_rate: float = Field(DEFAULT_VARIABLE_COST_RATE, gt=0, lt=1, json_schema_extra={"example": 0.38})
    fixed_cost_per_day: float = Field(DEFAULT_FIXED_COST_PER_DAY_INR, gt=0, json_schema_extra={"example": 500.0})
    inventory_capacity: int = Field(200, gt=0, json_schema_extra={"example": 200})
    preparation_time: float = Field(5.0, gt=0, json_schema_extra={"example": 5.0})
    search_radius_km: float = Field(3.0, gt=0, le=10.0, json_schema_extra={"example": 3.0})
    top_n: int = Field(5, ge=1, le=10, json_schema_extra={"example": 5})
    timestamp: Optional[datetime] = Field(None, json_schema_extra={"example": "2026-10-05T12:00:00Z"})


class RoutingRequest(BaseModel):
    """Street routing coordinates request."""
    model_config = ConfigDict(extra="ignore")

    origin_lat: float = Field(..., ge=-90.0, le=90.0)
    origin_lng: float = Field(..., ge=-180.0, le=180.0)
    dest_lat: float = Field(..., ge=-90.0, le=90.0)
    dest_lng: float = Field(..., ge=-180.0, le=180.0)


class ShiftPlanRequest(BaseModel):
    """Multi-stop daily shift planning and VRPTW optimization request."""
    model_config = ConfigDict(extra="ignore")

    latitude: float = Field(..., ge=-90.0, le=90.0, json_schema_extra={"example": 16.5062})
    longitude: float = Field(..., ge=-180.0, le=180.0, json_schema_extra={"example": 80.6480})
    vendor_category: str = Field("food", json_schema_extra={"example": "food"})
    shift_start_time: Optional[datetime] = Field(
        None,
        description="Shift start timestamp (ISO 8601). Defaults to current UTC time.",
        json_schema_extra={"example": "2026-10-05T08:00:00Z"},
    )
    shift_duration_hours: float = Field(
        8.0,
        ge=4.0,
        le=14.0,
        description="Total shift operating duration in hours (supports 8 to 12 hours)",
        json_schema_extra={"example": 8.0},
    )
    max_relocation_stops: int = Field(
        3,
        ge=1,
        le=5,
        description="Maximum number of relocation stops during the shift (1 to 5)",
        json_schema_extra={"example": 3},
    )
    min_dwell_time_minutes: int = Field(
        90,
        ge=30,
        le=240,
        description="Minimum continuous trading dwell time per stop in minutes",
        json_schema_extra={"example": 90},
    )
    setup_time_minutes: int = Field(
        15,
        ge=0,
        le=60,
        description="Time required to unpack canopy and setup burner/cart in minutes",
        json_schema_extra={"example": 15},
    )
    teardown_time_minutes: int = Field(
        15,
        ge=0,
        le=60,
        description="Time required to pack cart and clean up site in minutes",
        json_schema_extra={"example": 15},
    )
    search_radius_km: float = Field(
        4.0,
        gt=0,
        le=10.0,
        description="Search radius for candidate relocation zones in kilometers",
        json_schema_extra={"example": 4.0},
    )
    average_order_value: Optional[float] = Field(
        None,
        gt=0,
        description="Optional override for Average Order Value in INR",
        json_schema_extra={"example": 120.0},
    )
    fuel_price_inr: Optional[float] = Field(
        None,
        gt=0,
        description="Optional override for fuel price in INR per liter",
        json_schema_extra={"example": 102.5},
    )


# ===========================================================================
# Response Schemas
# ===========================================================================

class StreetRouteResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    code: str = Field(..., description="Route calculation status ('Ok' or 'Fallback')")
    coordinates: List[List[float]] = Field(..., description="Polyline coordinates [[lat, lng], ...]")
    distance_km: float = Field(..., description="Driving road distance in kilometers")
    duration_min: int = Field(..., description="Estimated travel duration in minutes")
    is_routable: bool = Field(True, description="Whether route is accessible")
    source: str = Field(..., description="Data provenance (osrm_street_network, grid_road_interpolation, identical_point)")


class PingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    status: str = Field("ok", description="Service health status")
    service: str = Field("GeoDemand AI", description="Service identifier")
    version: str = Field("2.1.0", description="API semantic version")
    timestamp: str = Field(..., description="Current UTC ISO 8601 timestamp")


class SystemConfigResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    mapbox_token: Optional[str] = Field(None, description="Client-side Mapbox access token")
    live_mode: bool = Field(True, description="Whether live API provider integration is active")
    default_lat: float = Field(16.5062, description="Default latitude coordinate")
    default_lng: float = Field(80.6480, description="Default longitude coordinate")
    city: str = Field("Vijayawada", description="Primary deployment city")
    state: str = Field("Andhra Pradesh", description="Primary deployment state")


class DataSourceHealth(BaseModel):
    model_config = ConfigDict(extra="allow")
    source_type: str = Field(..., description="real_live, real_periodic, real_offline, derived, or synthetic_fallback")
    source_name: str = Field(..., description="Provider or system name")
    note: Optional[str] = Field(None, description="Operational notes or fallback triggers")
    cache_ttl_sec: Optional[int] = Field(None, description="Cache time-to-live in seconds")


class DataHealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Dict[str, Any] = Field(..., description="ML model loading status and metadata")
    weather: DataSourceHealth = Field(..., description="Weather provider health")
    poi: DataSourceHealth = Field(..., description="POI static feature store health")
    competition: DataSourceHealth = Field(..., description="Competition provider health")
    traffic: DataSourceHealth = Field(..., description="Traffic provider health")
    events: DataSourceHealth = Field(..., description="Events provider health")
    holidays: DataSourceHealth = Field(..., description="Calendar provider health")
    customer_demand: DataSourceHealth = Field(..., description="Customer demand estimation health")
    timestamp: str = Field(..., description="Health check execution timestamp")


class LocationContextResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    latitude: float = Field(..., description="WGS84 latitude")
    longitude: float = Field(..., description="WGS84 longitude")
    h3_cell: str = Field(..., description="H3 spatial index at resolution 8")
    h3_resolution: int = Field(8, description="Spatial resolution level (fixed at 8)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    timezone_name: str = Field(..., description="IANA timezone name (e.g. Asia/Kolkata)")
    is_water: bool = Field(False, description="Whether location falls within water boundaries")
    source_type: str = Field("real_live", description="Data provenance source type")
    source_name: str = Field("device_gps", description="Data provenance source name")


class QuantileConfidenceInterval(BaseModel):
    p10: int = Field(..., description="10th percentile conservative customer estimate")
    p50: int = Field(..., description="50th percentile median customer estimate")
    p90: int = Field(..., description="90th percentile optimistic customer estimate")


class BusinessEconomicsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    expected_customers: int = Field(..., description="Estimated customer footfall per hour")
    average_spend_inr: float = Field(..., description="Average spend per customer in INR")
    average_order_value_inr: float = Field(..., description="Average order value in INR")
    expected_revenue_inr: float = Field(..., description="Hourly gross revenue in INR")
    expected_costs_inr: float = Field(..., description="Total hourly operating costs including fuel in INR")
    expected_operating_cost_inr: float = Field(..., description="Operating costs in INR")
    base_operating_cost_inr: float = Field(..., description="Operating costs excluding fuel in INR")
    expected_profit_inr: float = Field(..., description="Net expected hourly profit in INR")
    candidate_predicted_profit_inr: float = Field(..., description="Unadjusted candidate profit in INR")
    ingredient_cost_inr: float = Field(..., description="COGS in INR")
    fixed_cost_per_hour_inr: float = Field(..., description="Hourly fixed overhead in INR")
    fuel_cost_inr: float = Field(..., description="Estimated travel fuel expense in INR")
    travel_time_minutes: float = Field(..., description="Travel time to location in minutes")
    travel_fraction: float = Field(..., description="Fraction of hour available after travel")
    relocation_adjusted_realized_profit_inr: float = Field(..., description="Realized profit after relocation adjustment")
    realized_next_hour_profit_inr: float = Field(..., description="Realized profit in next hour in INR")
    variable_cost_rate: float = Field(..., description="Variable cost fraction")


class PredictDemandResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    location: Dict[str, Any] = Field(..., description="Location spatial metadata and water status")
    vendor_category: str = Field(..., description="Vendor category")
    prediction_timestamp: str = Field(..., description="ISO 8601 prediction timestamp")
    model_version: str = Field("v2.1.0", description="Trained model version")
    model_type: str = Field("lightgbm", description="Champion model architecture")
    prediction: Dict[str, Any] = Field(..., description="Expected customers and confidence interval")
    economics: BusinessEconomicsResponse = Field(..., description="Detailed business unit economics")
    data_freshness: Dict[str, Any] = Field(..., description="Timestamp and provenance of input data")


class ScenarioMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")
    aov: float = Field(..., description="Average order value in INR")
    expected_customers: int = Field(..., description="Customer demand")
    expected_profit_inr: float = Field(..., description="Expected net profit in INR")
    vendor_category: Optional[str] = Field(None, description="Vendor category")
    hour: Optional[int] = Field(None, description="Hour of operation")
    rainfall_mm: Optional[float] = Field(None, description="Precipitation in mm")
    is_rain: Optional[bool] = Field(None, description="Rain status flag")
    competition_score: Optional[float] = Field(None, description="Competition score")
    fuel_price_inr: Optional[float] = Field(None, description="Fuel price per liter in INR")
    prediction_interval: Optional[Dict[str, int]] = Field(default_factory=dict, description="P10, P50, P90 interval")
    expected_revenue_inr: Optional[float] = Field(None, description="Gross revenue in INR")
    expected_costs_inr: Optional[float] = Field(None, description="Total costs in INR")
    fuel_cost_inr: Optional[float] = Field(None, description="Travel fuel expense in INR")


class ScenarioDelta(BaseModel):
    model_config = ConfigDict(extra="allow")
    profit_delta_inr: float = Field(..., description="Net hourly profit difference in INR")
    percentage_change: float = Field(..., description="Percentage profit change")
    customer_delta: Optional[int] = Field(None, description="Net customer demand difference")
    customer_pct_change: Optional[float] = Field(None, description="Percentage customer change")
    revenue_delta_inr: Optional[float] = Field(None, description="Hourly revenue difference in INR")
    revenue_pct_change: Optional[float] = Field(None, description="Percentage revenue change")
    fuel_delta_inr: Optional[float] = Field(None, description="Fuel cost difference in INR")


class ScenarioSimulateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    scenario_type: Optional[str] = Field("multi_factor", description="Identified simulation scenario type")
    baseline: ScenarioMetrics = Field(..., description="Baseline scenario performance")
    scenario: ScenarioMetrics = Field(..., description="Simulated scenario performance")
    simulation_delta: ScenarioDelta = Field(..., description="Comparative delta metrics")
    primary_driver: Optional[str] = Field(None, description="Dominant variable driving the profit delta")
    strategic_advice: Optional[str] = Field(None, description="Actionable recommendation for mobile vendor")
    explanation: str = Field(..., description="Human-readable business summary of simulation")


class BatchPredictSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_locations: int = Field(..., description="Number of locations evaluated")
    total_expected_customers: int = Field(..., description="Sum of expected customers across all locations")
    total_expected_revenue_inr: float = Field(..., description="Sum of expected revenue in INR")
    total_expected_profit_inr: float = Field(..., description="Sum of expected profit in INR")
    average_profit_per_location_inr: float = Field(..., description="Mean profit per location in INR")
    best_location_index: int = Field(..., description="Index of location with highest expected profit")
    best_location_profit_inr: float = Field(..., description="Profit of top location in INR")


class BatchPredictDemandResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_id: str = Field(..., description="Batch execution tracking ID")
    processed_at: str = Field(..., description="ISO 8601 processing timestamp")
    count: int = Field(..., description="Number of evaluated items")
    results: List[PredictDemandResponse] = Field(..., description="Individual prediction responses")
    summary: BatchPredictSummary = Field(..., description="Batch summary statistics")


class APICacheStatsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    active_entries: int = Field(..., description="Number of active unexpired cached entries")
    total_capacity: int = Field(..., description="Maximum cache item capacity")
    default_ttl_seconds: int = Field(..., description="Default cache TTL in seconds")
    hits: int = Field(..., description="Total cache hits")
    misses: int = Field(..., description="Total cache misses")
    total_requests: int = Field(..., description="Total cache access requests")
    hit_ratio: float = Field(..., description="Cache hit ratio between 0.0 and 1.0")


class OverallDecisionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    verdict: DecisionVerdict = Field(..., description="RECOMMENDED_MOVE, CONSIDER_MOVE, or STAY_PUT")
    action: str = Field(..., description="RECOMMENDED MOVE, CONSIDER MOVE, or STAY HERE")
    decision: str = Field(..., description="Backward-compatible verdict alias")
    headline: str = Field(..., description="Executive recommendation summary headline")
    rationale: str = Field(..., description="Detailed decision explanation")
    reason: str = Field(..., description="Backward-compatible rationale alias")
    current_profit: float = Field(..., description="Hourly profit at current position in INR")
    recommended_profit: float = Field(..., description="Expected profit at top destination in INR")
    profit_uplift: float = Field(..., description="Net profit uplift per hour in INR")
    travel_time: float = Field(..., description="Travel duration in minutes")
    distance: float = Field(..., description="Distance in kilometers")
    top_net_improvement_inr: float = Field(..., description="Top net improvement in INR")
    stay_threshold_inr: float = Field(..., description="Minimum uplift threshold to consider moving")
    consider_threshold_inr: float = Field(..., description="Threshold for high-confidence recommended move")


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    rank: int = Field(..., description="Recommendation ranking order (1-indexed)")
    h3_cell: str = Field(..., description="H3 resolution 8 cell identifier")
    latitude: float = Field(..., description="Candidate centroid latitude")
    longitude: float = Field(..., description="Candidate centroid longitude")
    landmark_name: str = Field(..., description="Descriptive landmark or street name")
    landmark_display: str = Field(..., description="Formatted display string with distance")
    landmark_type: str = Field(..., description="Type of nearby POI (e.g. college, office)")
    is_college: bool = Field(..., description="Flag indicating educational institute proximity")
    landmark_emoji: str = Field(..., description="Visual icon representation")
    distance_km: float = Field(..., description="Distance from origin in kilometers")
    estimated_travel_time_min: int = Field(..., description="Estimated driving time in minutes")
    travel_fraction: float = Field(..., description="Available trading time fraction")
    is_current_cell: bool = Field(..., description="Whether this is the origin cell")
    is_water: bool = Field(..., description="Water exclusion flag")
    expected_customers: int = Field(..., description="Expected hourly customer count")
    prediction_interval: Dict[str, int] = Field(..., description="P10, P50, P90 confidence interval")
    confidence_interval: Dict[str, int] = Field(..., description="Confidence interval alias")
    category_affinity: float = Field(..., description="Category-specific location affinity score")
    expected_revenue_inr: float = Field(..., description="Gross revenue in INR")
    expected_profit_inr: float = Field(..., description="Base net profit in INR")
    candidate_predicted_profit_inr: float = Field(..., description="Unadjusted candidate profit in INR")
    relocation_adjusted_realized_profit_inr: float = Field(..., description="Net realized profit after travel in INR")
    realized_next_hour_profit_inr: float = Field(..., description="Next-hour realized profit alias")
    realized_net_uplift_inr: float = Field(..., description="Net hourly gain over current location in INR")
    profit_improvement_inr: float = Field(..., description="Profit improvement alias")
    profit_improvement_pct: float = Field(..., description="Percentage profit gain")
    expected_operating_cost_inr: float = Field(..., description="Total operating cost in INR")
    fuel_cost_inr: float = Field(..., description="Fuel travel expense in INR")
    competition_score: float = Field(..., description="Normalized competition index")
    competition_level: str = Field(..., description="low, medium, or high")
    traffic_level: str = Field(..., description="light, medium, or congested")
    traffic_speed_kmh: float = Field(..., description="Current corridor traffic speed")
    traffic_source_type: str = Field(..., description="Data provenance for traffic")
    weather_condition: str = Field(..., description="Current weather descriptor")
    decision_verdict: str = Field(..., description="RECOMMENDED_MOVE, CONSIDER_MOVE, or STAY_PUT")
    decision_action: str = Field(..., description="Action badge text")
    decision_rationale: str = Field(..., description="Explanatory decision sentence")
    decision_reason: str = Field(..., description="Rationale alias")
    top_drivers: List[str] = Field(..., description="Top natural language demand factors")
    explanation: str = Field(..., description="Comprehensive explanatory narrative")
    boundary: List[List[float]] = Field(..., description="H3 hexagon polygon coordinates [[lat, lng], ...]")


class DemandHeatmapItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    h3_cell: str = Field(..., description="H3 hexagon index")
    latitude: float = Field(..., description="Hexagon centroid latitude")
    longitude: float = Field(..., description="Hexagon centroid longitude")
    boundary: List[List[float]] = Field(..., description="Hexagon vertex coordinates")
    expected_customers: int = Field(..., description="Predicted demand")
    expected_profit_inr: float = Field(..., description="Predicted profit in INR")
    demand_intensity: float = Field(..., description="Normalized demand intensity [0, 1]")
    is_current_cell: bool = Field(..., description="Origin cell flag")
    rank: Optional[int] = Field(None, description="Rank if among top recommendations")


class LiveRecommendationsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    request_id: str = Field(..., description="Unique recommendation tracking ID")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")
    search_radius_km: float = Field(..., description="Effective search radius in kilometers")
    decision: OverallDecisionResponse = Field(..., description="Overall strategic recommendation")
    current_location: Dict[str, Any] = Field(..., description="Origin location context and boundary")
    current_estimated_demand: Optional[int] = Field(None, description="Current location hourly demand")
    current_expected_profit_inr: Optional[float] = Field(None, description="Current location hourly profit")
    current_weather: Dict[str, Any] = Field(..., description="Live weather condition and telemetry")
    recommendations: List[RecommendationItem] = Field(..., description="Ranked list of top candidate zones")
    demand_heatmap: List[DemandHeatmapItem] = Field(..., description="Spatial heatmap data across all evaluated zones")
    total_candidates_evaluated: int = Field(..., description="Number of evaluated hexagon cells")
    data_freshness: Dict[str, Any] = Field(..., description="Provenance and retrieval timestamps for all inputs")
    data_quality: Dict[str, str] = Field(..., description="Hybrid data quality ratings per subsystem")
    model_info: Dict[str, Any] = Field(..., description="Model version, MAE metrics, and training metadata")


class ShiftStopItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    stop_number: int = Field(..., description="Itinerary stop sequence number (1-indexed)")
    h3_cell: str = Field(..., description="H3 hexagon cell identifier")
    latitude: float = Field(..., description="Stop latitude")
    longitude: float = Field(..., description="Stop longitude")
    landmark_name: str = Field(..., description="Descriptive landmark or street name")
    landmark_display: str = Field(..., description="Formatted display string")
    landmark_type: str = Field(..., description="POI type")
    landmark_emoji: str = Field(..., description="Visual emoji badge")
    is_current_cell: bool = Field(..., description="Whether this is the starting origin cell")
    arrival_time: str = Field(..., description="ISO 8601 arrival timestamp")
    arrival_formatted: str = Field(..., description="Human-readable arrival time (e.g. '08:00 AM')")
    setup_time_minutes: int = Field(..., description="Site setup overhead duration in minutes")
    trading_start_time: str = Field(..., description="Trading open timestamp")
    trading_start_formatted: str = Field(..., description="Human-readable trading open time")
    dwell_minutes: int = Field(..., description="Continuous trading dwell time in minutes")
    dwell_hours: float = Field(..., description="Continuous trading dwell time in hours")
    trading_end_time: str = Field(..., description="Trading close timestamp")
    trading_end_formatted: str = Field(..., description="Human-readable trading close time")
    teardown_time_minutes: int = Field(..., description="Teardown overhead duration in minutes")
    departure_time: str = Field(..., description="Departure timestamp")
    departure_formatted: str = Field(..., description="Human-readable departure time")
    travel_to_next_stop_minutes: int = Field(..., description="Travel duration to next stop in minutes")
    travel_to_next_stop_km: float = Field(..., description="Road distance to next stop in km")
    fuel_cost_to_next_inr: float = Field(..., description="Travel fuel expense to next stop in INR")
    expected_hourly_customers: int = Field(..., description="Projected hourly customer count")
    customers_served_at_stop: int = Field(..., description="Total customers served during dwell period")
    gross_revenue_inr: float = Field(..., description="Gross revenue earned at stop in INR")
    operating_costs_inr: float = Field(..., description="COGS and operating costs at stop in INR")
    net_profit_inr: float = Field(..., description="Net trading profit earned at stop in INR")


class ShiftSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_stops: int = Field(..., description="Number of itinerary stops")
    total_dwell_minutes: int = Field(..., description="Total active trading dwell time in minutes")
    total_dwell_hours: float = Field(..., description="Total active trading dwell time in hours")
    total_transit_minutes: int = Field(..., description="Total transit travel time in minutes")
    total_overhead_minutes: int = Field(..., description="Total setup and teardown overhead in minutes")
    total_travel_distance_km: float = Field(..., description="Total road distance traveled across shift in km")
    total_fuel_cost_inr: float = Field(..., description="Total fuel consumption expense in INR")
    total_expected_customers: int = Field(..., description="Total expected customers served across shift")
    total_gross_revenue_inr: float = Field(..., description="Total gross shift revenue in INR")
    total_net_profit_inr: float = Field(..., description="Total realized shift net profit after fuel in INR")


class ShiftPlanResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    shift_id: str = Field(..., description="Unique shift plan identifier")
    generated_at: str = Field(..., description="ISO 8601 generation timestamp")
    shift_start: str = Field(..., description="Shift start timestamp")
    shift_end: str = Field(..., description="Shift end timestamp")
    shift_duration_hours: float = Field(..., description="Shift duration in hours")
    itinerary: List[ShiftStopItem] = Field(..., description="Chronologically ordered itinerary stops")
    summary: ShiftSummary = Field(..., description="Aggregate shift operational and financial metrics")
    baseline_stay_profit_inr: float = Field(..., description="Projected shift profit if staying put at origin in INR")
    shift_net_uplift_inr: float = Field(..., description="Net profit uplift of multi-stop plan vs staying put in INR")
    shift_net_uplift_pct: float = Field(..., description="Percentage profit gain vs staying put")
    verdict: str = Field(..., description="RECOMMENDED_SHIFT, CONSIDER_SHIFT, or STAY_PUT_SHIFT")
    decision_reason: str = Field(..., description="Natural language strategic explanation")
    solver_telemetry: Dict[str, Any] = Field(..., description="OR-Tools solver status and solve time")
