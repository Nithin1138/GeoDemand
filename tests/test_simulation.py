"""
Test Suite for Market Simulation Engine & Anti-Circularity Guards (Member 2).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "src" / "data"))
sys.path.insert(0, str(ROOT_DIR / "src" / "simulation"))

from src.simulation.market_simulation_v2 import generate_historical_transactions
from src.data.h3_cells import build_h3_cells
from src.data.static_features import build_static_features
from src.data.weather import build_weather_hourly
from src.data.calendar_dataset import build_calendar
from src.data.events import build_events
from src.data.competition import build_competition
from src.data.vendor_profiles import build_vendor_profiles
from src.features.feature_store import LEAKAGE_COLUMNS


@pytest.fixture(scope="module")
def small_simulation_data():
    """Builds a lightweight 5-vendor 7-day dataset for rapid deterministic testing."""
    cells = build_h3_cells(radius_km=3.0)
    static = build_static_features(cells, seed=42)
    weather = build_weather_hourly(16.5062, 80.6480, "2025-01-01", "2025-01-07", seed=42)
    cal = build_calendar("2025-01-01", "2025-01-07")
    vendors = build_vendor_profiles(cells["h3_cell_id"].tolist(), n_vendors=5, seed=42)
    events_df = build_events(cells["h3_cell_id"].tolist(), "2025-01-01", "2025-01-07", seed=42)
    comp = build_competition(cells["h3_cell_id"].tolist(), vendors, pd.date_range("2025-01-01", "2025-01-07"))

    tx1 = generate_historical_transactions(cells, static, weather, cal, events_df, comp, vendors, seed=123)
    tx2 = generate_historical_transactions(cells, static, weather, cal, events_df, comp, vendors, seed=123)

    return {"tx1": tx1, "tx2": tx2}


def test_simulation_reproducibility(small_simulation_data):
    """Same random seed must produce byte-for-byte identical simulated transactions."""
    tx1 = small_simulation_data["tx1"]
    tx2 = small_simulation_data["tx2"]

    assert len(tx1) == len(tx2)
    assert (tx1["expected_customer_count"] == tx2["expected_customer_count"]).all()
    assert (tx1["profit"] == tx2["profit"]).all()


def test_simulation_math_and_accounting_integrity(small_simulation_data):
    """Verify customer counts, gross revenues, COGS, and profits are strictly coherent."""
    tx = small_simulation_data["tx1"]

    # Target non-negativity
    assert (tx["expected_customer_count"] >= 0).all()

    # Revenue = customers * AOV
    expected_rev = (tx["expected_customer_count"] * tx["average_order_value"]).round(2)
    diff_rev = (tx["revenue"] - expected_rev).abs()
    assert (diff_rev < 1e-2).all(), "Revenue calculation divergence"

    # Profit = Revenue - Operating Cost
    expected_profit = (tx["revenue"] - tx["operating_cost"]).round(2)
    diff_profit = (tx["profit"] - expected_profit).abs()
    assert (diff_profit < 1e-2).all(), "Profit calculation divergence"


def test_simulation_temporal_lag_contract(small_simulation_data):
    """Verify decision_timestamp t produces predictions strictly for target_timestamp t+1."""
    tx = small_simulation_data["tx1"]
    lags = tx["target_timestamp"] - tx["decision_timestamp"]
    assert (lags == pd.Timedelta(hours=1)).all()


def test_no_latent_simulation_leakage_in_transactions(small_simulation_data):
    """Verify latent variables (skill factor, popularity, quality) are not present in output."""
    tx = small_simulation_data["tx1"]
    for leak in LEAKAGE_COLUMNS:
        assert leak not in tx.columns, f"Latent simulation parameter {leak} found in transactions!"
