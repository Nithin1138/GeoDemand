"""
Unit and Integration Tests for Competitor and Operating-Hours Pipeline (Phase 12.2).
"""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import pytest

from src.data.competitor_pipeline import (
    CompetitorPipeline,
    CompetitorRecord,
    parse_operating_hours,
    COMPETITOR_CATEGORY_MAP,
)


def test_operating_hours_parser_day_night():
    """Verify operating hours parser accurately segments day vs night (Tasks 6 & 7)."""
    # 24/7 operates both day and night
    is_day, is_night = parse_operating_hours("24/7")
    assert is_day is True and is_night is True

    # Standard daytime office hours (09:00 - 17:00)
    is_day, is_night = parse_operating_hours("09:00-17:00", category="salon")
    assert is_day is True
    assert is_night is False

    # Evening / late night dinner cart (18:00 - 02:00)
    is_day, is_night = parse_operating_hours("18:00-02:00", category="food")
    assert is_day is False
    assert is_night is True

    # Full day restaurant (11:00 - 23:00)
    is_day, is_night = parse_operating_hours("11:00-23:00", category="food")
    assert is_day is True
    assert is_night is True


def test_competitor_category_mapping():
    """Verify amenities map to canonical categories: food, fruits, salon, repair (Tasks 2 & 4)."""
    assert COMPETITOR_CATEGORY_MAP["restaurant"] == "food"
    assert COMPETITOR_CATEGORY_MAP["cafe"] == "food"
    assert COMPETITOR_CATEGORY_MAP["greengrocer"] == "fruits"
    assert COMPETITOR_CATEGORY_MAP["hairdresser"] == "salon"
    assert COMPETITOR_CATEGORY_MAP["bicycle_repair"] == "repair"


def test_stale_competitor_detection():
    """Verify competitor records unverified for > 30 days are flagged as stale (Task 8)."""
    pipeline = CompetitorPipeline()
    now = datetime.now(timezone.utc)
    old_date = (now - timedelta(days=45)).isoformat()

    elements = [
        {
            "id": 101,
            "lat": 16.5062,
            "lon": 80.6480,
            "tags": {"amenity": "restaurant", "name": "Historic Dhaba"},
            "last_verified": old_date,
        },
        {
            "id": 102,
            "lat": 16.5062,
            "lon": 80.6480,
            "tags": {"amenity": "cafe", "name": "Fresh Coffee"},
            "last_verified": now.isoformat(),
        },
    ]

    records = pipeline.process_competitor_records(elements, as_of_time=now)
    assert len(records) == 2
    r_old = next(r for r in records if r.competitor_id == "101")
    r_fresh = next(r for r in records if r.competitor_id == "102")

    assert r_old.is_stale is True
    assert r_fresh.is_stale is False


def test_h3_competitor_aggregation(tmp_path):
    """Verify category-specific and time-aware counts aggregate to H3 cells (Tasks 5, 11, 12)."""
    pipeline = CompetitorPipeline(cache_dir=tmp_path)
    now = datetime.now(timezone.utc)

    records = [
        CompetitorRecord(
            competitor_id="c1",
            name="Food Stall 1",
            category="food",
            raw_amenity="fast_food",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="88619aa6c7fffff",
            is_daytime_active=True,
            is_nighttime_active=True,
            last_verified_at=now,
            is_stale=False,
        ),
        CompetitorRecord(
            competitor_id="c2",
            name="Salon 1",
            category="salon",
            raw_amenity="hairdresser",
            latitude=16.5062,
            longitude=80.6480,
            h3_cell_id="88619aa6c7fffff",
            is_daytime_active=True,
            is_nighttime_active=False,
            last_verified_at=now,
            is_stale=False,
        ),
    ]

    df_agg = pipeline.aggregate_by_h3_cell(records, ["88619aa6c7fffff", "8861892557fffff"])
    assert len(df_agg) == 2

    cell_row = df_agg[df_agg["h3_cell_id"] == "88619aa6c7fffff"].iloc[0]
    assert cell_row["total_competitors"] == 2
    assert cell_row["food_competitor_count"] == 1
    assert cell_row["salon_competitor_count"] == 1
    assert cell_row["daytime_competitor_count"] == 2
    assert cell_row["nighttime_competitor_count"] == 1
    assert cell_row["stale_competitor_count"] == 0

    # Empty cell has 0 counts
    empty_cell = df_agg[df_agg["h3_cell_id"] == "8861892557fffff"].iloc[0]
    assert empty_cell["total_competitors"] == 0


def test_daily_refresh_caching(tmp_path):
    """Verify daily cache refresh job writes to cache and reloads within TTL (Tasks 1 & 9)."""
    pipeline = CompetitorPipeline(cache_dir=tmp_path, cache_ttl_hours=1.0)
    elements = [
        {"id": "c_cache_1", "lat": 16.5062, "lon": 80.6480, "tags": {"amenity": "cafe"}},
    ]

    # First run computes and writes cache
    df1 = pipeline.run_daily_refresh(raw_elements=elements, all_h3_cells=["88619aa6c7fffff"])
    assert len(df1) == 1
    cache_path = tmp_path / "competitors_aggregated_cache.parquet"
    assert cache_path.exists()

    # Second run loads from cache
    df2 = pipeline.run_daily_refresh(raw_elements=elements, all_h3_cells=["88619aa6c7fffff"])
    assert len(df2) == 1
    assert df2["food_competitor_count"].iloc[0] == df1["food_competitor_count"].iloc[0]
