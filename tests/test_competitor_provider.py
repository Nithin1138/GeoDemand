"""
Unit and integration tests for Competitor Intelligence Provider Subsystem:
- Common CompetitionProvider ABC interface compliance
- Operating hours parsing & category-specific heuristics
- Google Places provider with rate-limiting & fallback
- OSM Overpass provider with Overpass QL & rate-limiting fallback
- OSM local snapshot periodic provider & spatial H3 aggregation
- Scheduled competitor refresh worker
- FastAPI competitor intelligence endpoints
"""

import os
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient

from api.providers.competition import (
    CompetitorEntity,
    CompetitionProvider,
    SimulatedCompetitionProvider,
    OSMCompetitionProvider,
    OSMOverpassCompetitionProvider,
    GooglePlacesCompetitionProvider,
    ScheduledCompetitorRefresh,
    CompetitionProviderFactory,
    parse_operating_hours_detailed,
    aggregate_competitors_for_cell,
    normalize_vendor_category,
)


# ===========================================================================
# 1. Operating Hours Parser & Heuristics
# ===========================================================================

def test_parse_operating_hours_24_7():
    is_open, is_day, is_night, src = parse_operating_hours_detailed("24/7", "food")
    assert is_open is True
    assert is_day is True
    assert is_night is True
    assert src == "provider_explicit"


def test_parse_operating_hours_closed():
    is_open, is_day, is_night, src = parse_operating_hours_detailed("off", "retail")
    assert is_open is False
    assert is_day is False
    assert is_night is False
    assert src == "provider_explicit"


def test_parse_operating_hours_overnight_span():
    # Vendor operating evening to late night (18:00 - 02:00)
    # At 20:00 (evening): open=True, daytime=False, nighttime=True
    now = datetime(2026, 8, 18, 20, 0, 0, tzinfo=timezone.utc)
    is_open, is_day, is_night, src = parse_operating_hours_detailed("18:00-02:00", "food", now=now)
    assert is_open is True
    assert is_day is False
    assert is_night is True
    assert src == "provider_explicit"


def test_parse_operating_hours_missing_fallback_category():
    # Food category defaults to active daytime and nighttime
    is_open, is_day, is_night, src = parse_operating_hours_detailed(None, "food")
    assert src == "category_default"
    assert is_day is True
    assert is_night is True

    # Salon defaults to daytime only
    is_open, is_day, is_night, src = parse_operating_hours_detailed(None, "salon")
    assert src == "category_default"
    assert is_day is True
    assert is_night is False


# ===========================================================================
# 2. Competitor Entity & Cell Spatial Aggregation
# ===========================================================================

def test_aggregate_competitors_for_cell():
    comps = [
        CompetitorEntity(
            competitor_id="c1",
            name="Biryani House",
            category="food",
            raw_type="restaurant",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell="88619aa6c7fffff",
            opening_hours="10:00-23:00",
            is_open_now=True,
            is_daytime_active=True,
            is_nighttime_active=True,
            operating_hours_source="provider_explicit",
            source="test_source",
        ),
        CompetitorEntity(
            competitor_id="c2",
            name="Tea Point",
            category="tea",
            raw_type="cafe",
            latitude=16.5065,
            longitude=80.6482,
            h3_cell="88619aa6c7fffff",
            opening_hours=None,
            is_open_now=True,
            is_daytime_active=True,
            is_nighttime_active=True,
            operating_hours_source="category_default",
            source="test_source",
        ),
    ]

    agg = aggregate_competitors_for_cell(comps, "88619aa6c7fffff", "food")
    assert agg["h3_cell"] == "88619aa6c7fffff"
    assert agg["total_competitors"] == 2
    assert agg["same_category_competitors"] == 1
    assert agg["category_counts"]["food"] == 1
    assert agg["category_counts"].get("tea", 0) + agg["category_counts"].get("beverages", 0) == 1
    assert 0.0 <= agg["competition_score"] <= 1.0
    assert agg["competition_level"] in ("low", "medium", "high")


# ===========================================================================
# 3. Local OSM Competition Provider
# ===========================================================================

def test_osm_competition_provider_loads_and_scores():
    prov = OSMCompetitionProvider()
    score = prov.get_competition_score("88619aa6c7fffff", "food")
    assert "competition_score" in score
    assert 0.0 <= score["competition_score"] <= 1.0
    assert score["source_name"] in ("openstreetmap", "simulated_competition_model")
    assert score["vendor_category"] == "food"

    comps = prov.get_competitors("88619aa6c7fffff")
    assert isinstance(comps, list)
    cell_comp = prov.get_cell_competition("88619aa6c7fffff", "food")
    assert cell_comp["h3_cell"] == "88619aa6c7fffff"
    assert "freshness" in cell_comp


# ===========================================================================
# 4. OSM Overpass Provider (Mocked QL & Circuit Breaker)
# ===========================================================================

def test_osm_overpass_provider_successful_query():
    mock_overpass_resp = {
        "elements": [
            {
                "type": "node",
                "id": 12345,
                "lat": 16.5062,
                "lon": 80.6480,
                "tags": {
                    "name": "Overpass Cafe",
                    "amenity": "cafe",
                    "opening_hours": "08:00-22:00",
                },
            }
        ]
    }
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_overpass_resp

        prov = OSMOverpassCompetitionProvider()
        comps = prov.get_competitors("88619aa6c7fffff", "beverages")
        assert len(comps) == 1
        assert comps[0].name == "Overpass Cafe"
        assert comps[0].category == "beverages"
        assert comps[0].operating_hours_source == "provider_explicit"


def test_osm_overpass_circuit_breaker_on_429_rate_limit():
    with patch("requests.post") as mock_post:
        # Simulate Overpass HTTP 429 Too Many Requests
        mock_post.return_value.status_code = 429
        mock_post.return_value.raise_for_status.side_effect = Exception("HTTP 429 Too Many Requests")

        prov = OSMOverpassCompetitionProvider()
        # Should gracefully trip circuit breaker and fall back to local OSM
        score = prov.get_competition_score("88619aa6c7fffff", "food")
        assert prov.is_rate_limited() is True
        assert "competition_score" in score
        assert 0.0 <= score["competition_score"] <= 1.0


# ===========================================================================
# 5. Google Places Provider (Mocked & Circuit Breaker)
# ===========================================================================

def test_google_places_search_nearby_success():
    mock_places_resp = {
        "places": [
            {
                "id": "places/ChIJ1234",
                "displayName": {"text": "Google Cafe"},
                "primaryType": "cafe",
                "types": ["cafe", "food"],
                "location": {"latitude": 16.5062, "longitude": 80.6480},
                "regularOpeningHours": {"openNow": True},
                "rating": 4.5,
                "formattedAddress": "MG Road, Vijayawada",
            }
        ]
    }
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_places_resp

        prov = GooglePlacesCompetitionProvider(api_key="mock_google_key")
        comps = prov.get_competitors("88619aa6c7fffff", "tea")
        assert len(comps) == 1
        assert comps[0].name == "Google Cafe"
        assert comps[0].is_open_now is True
        assert comps[0].source == "google_places_api"


def test_google_places_circuit_breaker_fallback():
    with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
        # Simulate Quota Exceeded (HTTP 429) on both endpoints
        mock_post.return_value.status_code = 429
        mock_get.return_value.status_code = 429

        prov = GooglePlacesCompetitionProvider(api_key="mock_google_key")
        score = prov.get_competition_score("88619aa6c7fffff", "food")
        assert prov.is_rate_limited() is True
        assert "competition_score" in score


# ===========================================================================
# 6. Scheduled Competitor Refresh Worker
# ===========================================================================

def test_scheduled_competitor_refresh_worker():
    scheduler = ScheduledCompetitorRefresh(
        provider=OSMCompetitionProvider(),
        refresh_interval_sec=14400.0,
        monitored_cells=["88619aa6c7fffff", "88619aa6c1fffff"],
    )
    res = scheduler.refresh_cells()
    assert res["status"] == "COMPLETED"
    assert res["refreshed_cells_count"] == 2
    assert res["total_competitors_found"] >= 0

    status = scheduler.get_status()
    assert status["monitored_cells_count"] == 2
    assert status["refreshed_cells_count"] == 2
    assert status["last_refresh_time"] is not None


# ===========================================================================
# 7. Provider Factory
# ===========================================================================

def test_competition_provider_factory():
    prov_synth = CompetitionProviderFactory.create("synthetic")
    assert isinstance(prov_synth, SimulatedCompetitionProvider)

    prov_osm = CompetitionProviderFactory.create("osm")
    assert isinstance(prov_osm, OSMCompetitionProvider)

    prov_overpass = CompetitionProviderFactory.create("osm_overpass")
    assert isinstance(prov_overpass, OSMOverpassCompetitionProvider)


# ===========================================================================
# 8. FastAPI Competitor Endpoints
# ===========================================================================

@pytest.fixture(scope="module")
def api_client():
    from api.app import app
    return TestClient(app)


def test_get_cell_competitors_endpoint(api_client):
    r = api_client.get("/v1/competitors/cell/88619aa6c7fffff?vendor_category=food")
    assert r.status_code == 200
    body = r.json()
    assert body["h3_cell"] == "88619aa6c7fffff"
    assert body["vendor_category"] == "food"
    assert "competition_score" in body
    assert "competitors" in body
    assert "freshness" in body


def test_get_cell_competitors_invalid_h3(api_client):
    r = api_client.get("/v1/competitors/cell/invalid_cell_index")
    assert r.status_code == 400
    assert "Invalid H3 cell index" in r.json()["detail"]


def test_competitor_refresh_endpoint(api_client):
    r = api_client.post(
        "/v1/competitors/refresh",
        json={"h3_cells": ["88619aa6c7fffff"], "force": True},
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "COMPLETED"
    assert body["refreshed_cells_count"] == 1


def test_competitor_scheduler_status_endpoint(api_client):
    r = api_client.get(
        "/v1/competitors/scheduler-status",
        headers={"X-API-Key": "dev-local-secret-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "monitored_cells_count" in body
    assert "refresh_interval_hours" in body
