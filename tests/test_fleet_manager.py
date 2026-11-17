"""
Comprehensive Test Suite for Fleet Management, Competition Sharing,
Atomic Reservations, Concurrency, and Multi-Vendor Dispatch.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
import pytest
import h3

from fleet.fleet_manager import (
    FleetManager,
    Vendor,
    Reservation,
    VendorStatus,
    ReservationStatus,
    calculate_effective_demand,
    calculate_demand_sharing_penalty,
    CATEGORY_LAMBDA_COMPETITION,
    DEFAULT_LAMBDA,
    DEFAULT_CELL_TOTAL_CAPACITY,
    DEFAULT_CELL_CATEGORY_CAPACITIES,
)
from api.app import (
    fleet_reserve_cell,
    fleet_cancel_reservation,
    fleet_cell_occupancy,
    fleet_dispatch,
    fleet_list_vendors,
)
from api.schemas import (
    FleetReserveCellRequest,
    FleetReserveCellResponse,
    FleetCancelReservationRequest,
    FleetCancelReservationResponse,
    CellOccupancyResponse,
    FleetDispatchRequest,
    FleetDispatchResponse,
    FleetDispatchVendorItem,
)


def run_coroutine(coro):
    """Helper for executing FastAPI async routes without event loop network sockets on Windows."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not complete synchronously")


@pytest.fixture(autouse=True)
def clean_fleet_manager():
    """Ensure clean state before each test."""
    manager = FleetManager(":memory:")
    manager.reset_state_for_testing()
    yield manager
    manager.reset_state_for_testing()


# ===========================================================================
# 1. Vendor Model & State Transitions
# ===========================================================================

def test_vendor_model_and_availability(clean_fleet_manager):
    """Validates vendor identity, default H3 cell computation, and status flags."""
    v = Vendor(
        vendor_id="VND-001",
        category="Food",
        name="Benz Dosa Express",
        current_lat=16.5062,
        current_lng=80.6480,
    )
    assert v.category == "food"
    assert v.current_h3 == h3.latlng_to_cell(16.5062, 80.6480, 8)
    assert v.status == VendorStatus.IDLE
    assert v.is_available is True

    # Register and retrieve
    clean_fleet_manager.register_vendor(v)
    retrieved = clean_fleet_manager.get_vendor("VND-001")
    assert retrieved is not None
    assert retrieved.name == "Benz Dosa Express"

    # Status transition
    v.status = VendorStatus.DISPATCHED
    assert v.is_available is False


# ===========================================================================
# 2. Effective Demand & Competition Sharing Calculation
# ===========================================================================

def test_effective_demand_and_competition_penalties():
    """
    Tests the roadmap's non-linear demand cannibalization formula:
    D_effective(H) = D_base(H) / (1 + lambda * N_same_category(H))
    """
    base_demand = 30.0  # 30 customers/hr

    # N=0: No co-located competition -> full demand
    eff_0 = calculate_effective_demand(base_demand, same_category_count=0, category="food")
    assert eff_0 == 30.0
    assert calculate_demand_sharing_penalty(base_demand, 0, "food") == 0.0

    # Food category (lambda = 0.35)
    # N=1: D_eff = 30 / (1 + 0.35*1) = 30 / 1.35 = 22.22
    eff_food_1 = calculate_effective_demand(base_demand, same_category_count=1, category="food")
    assert eff_food_1 == pytest.approx(22.22, rel=1e-2)
    pen_food_1 = calculate_demand_sharing_penalty(base_demand, 1, "food")
    assert pen_food_1 == pytest.approx(7.78, rel=1e-2)

    # N=2: D_eff = 30 / (1 + 0.35*2) = 30 / 1.70 = 17.65
    eff_food_2 = calculate_effective_demand(base_demand, same_category_count=2, category="food")
    assert eff_food_2 == pytest.approx(17.65, rel=1e-2)

    # Repair category (high cannibalization, lambda = 0.50)
    # N=1: D_eff = 30 / (1 + 0.50*1) = 20.0
    eff_repair_1 = calculate_effective_demand(base_demand, 1, "repair")
    assert eff_repair_1 == 20.0
    assert calculate_demand_sharing_penalty(base_demand, 1, "repair") == 10.0

    # Custom lambda override
    eff_custom = calculate_effective_demand(base_demand, 1, "food", custom_lambda=0.20)
    assert eff_custom == 25.0


# ===========================================================================
# 3. Cell Capacity & Category-Specific Limits
# ===========================================================================

def test_cell_capacity_and_category_limits(clean_fleet_manager):
    """Verifies physical total capacity and category-specific capacity caps."""
    cell = h3.latlng_to_cell(16.5062, 80.6480, 8)

    # Configure custom capacities: total 3, food 2, beverages 1
    clean_fleet_manager.set_cell_capacity(cell, total_capacity=3, category_capacities={"food": 2, "beverages": 1})

    # Reserve 1st food vendor -> Success
    ok1, res1, _ = clean_fleet_manager.reserve_cell("V-F1", cell, "food")
    assert ok1 is True
    assert res1 is not None

    # Reserve 2nd food vendor -> Success
    ok2, res2, _ = clean_fleet_manager.reserve_cell("V-F2", cell, "food")
    assert ok2 is True
    assert res2 is not None

    # Reserve 3rd food vendor -> Rejection (food category limit reached: 2)
    ok3, res3, reason3 = clean_fleet_manager.reserve_cell("V-F3", cell, "food")
    assert ok3 is False
    assert res3 is None
    assert "category capacity exceeded" in reason3.lower()

    # Reserve 1st beverage vendor -> Success (total capacity now 3/3)
    ok4, res4, _ = clean_fleet_manager.reserve_cell("V-B1", cell, "beverages")
    assert ok4 is True
    assert res4 is not None

    # Reserve 1st fruit vendor -> Rejection (total cell capacity 3/3 reached)
    ok5, res5, reason5 = clean_fleet_manager.reserve_cell("V-FR1", cell, "fruits")
    assert ok5 is False
    assert res5 is None
    assert "total capacity exceeded" in reason5.lower()


# ===========================================================================
# 4. Soft Reservation & TTL Expiry
# ===========================================================================

def test_soft_reservation_and_ttl_expiry(clean_fleet_manager):
    """Soft reservations hold capacity temporarily and expire after TTL."""
    cell = h3.latlng_to_cell(16.5062, 80.6480, 8)
    clean_fleet_manager.set_cell_capacity(cell, total_capacity=1, category_capacities={"food": 1})

    now = datetime.now(timezone.utc)
    # Reserve with 10 minute TTL
    ok, res, _ = clean_fleet_manager.reserve_cell(
        vendor_id="VND-SOFT",
        h3_cell=cell,
        vendor_category="food",
        is_soft=True,
        ttl_minutes=10,
    )
    assert ok is True
    assert res.is_soft is True

    # Occupancy is full
    occ_before = clean_fleet_manager.get_cell_occupancy(cell)
    assert occ_before["total_active_occupancy"] == 1
    assert occ_before["is_at_capacity"] is True

    # Fast forward 5 minutes (still valid)
    t_5m = now + timedelta(minutes=5)
    assert res.is_active_at(t_5m) is True

    # Fast forward 12 minutes (expired)
    t_12m = now + timedelta(minutes=12)
    assert res.is_active_at(t_12m) is False

    # Sweep expired
    expired_cnt = clean_fleet_manager.sweep_expired_reservations(ref_time=t_12m)
    assert expired_cnt == 1

    # Capacity is now freed
    occ_after = clean_fleet_manager.get_cell_occupancy(cell)
    assert occ_after["total_active_occupancy"] == 0
    assert occ_after["available_capacity"] == 1

    # Another vendor can now book
    ok_new, res_new, _ = clean_fleet_manager.reserve_cell("VND-NEW", cell, "food")
    assert ok_new is True


# ===========================================================================
# 5. Reservation Cancellation
# ===========================================================================

def test_reservation_cancellation(clean_fleet_manager):
    """Explicit cancellation immediately releases cell capacity."""
    cell = h3.latlng_to_cell(16.5062, 80.6480, 8)
    clean_fleet_manager.set_cell_capacity(cell, total_capacity=1)

    ok, res, _ = clean_fleet_manager.reserve_cell("V-CANCEL", cell, "food")
    assert ok is True
    res_id = res.reservation_id

    # Verify occupied
    occ = clean_fleet_manager.get_cell_occupancy(cell)
    assert occ["available_capacity"] == 0

    # Cancel reservation
    can_ok, can_msg, freed_cell = clean_fleet_manager.cancel_reservation(res_id, vendor_id="V-CANCEL")
    assert can_ok is True
    assert freed_cell == cell

    # Verify freed
    occ_after = clean_fleet_manager.get_cell_occupancy(cell)
    assert occ_after["available_capacity"] == 1
    assert occ_after["total_active_occupancy"] == 0


# ===========================================================================
# 6. Prevention of Conflicting Double-Booking
# ===========================================================================

def test_prevent_conflicting_reservations(clean_fleet_manager):
    """A vendor cannot hold multiple overlapping active reservations."""
    cell1 = "88619aa6c7fffff"  # Benz Circle
    cell2 = "88619aa6c5fffff"  # MG Road

    ok1, res1, _ = clean_fleet_manager.reserve_cell("V-SINGLE", cell1, "food")
    assert ok1 is True

    # Attempt to reserve cell2 while reservation in cell1 is still active
    ok2, res2, reason2 = clean_fleet_manager.reserve_cell("V-SINGLE", cell2, "food")
    assert ok2 is False
    assert res2 is None
    assert "already has active reservation" in reason2


# ===========================================================================
# 7. Atomic Concurrency Stress Test (Race Condition Protection)
# ===========================================================================

def test_atomic_concurrency_race_condition(clean_fleet_manager):
    """
    Stress test concurrency:
    30 simultaneous threads attempt to reserve the last remaining slot in a cell of capacity 1.
    Strict invariant: Exactly 1 reservation succeeds, 29 are rejected, with zero state corruption.
    """
    cell = h3.latlng_to_cell(16.5062, 80.6480, 8)
    clean_fleet_manager.set_cell_capacity(cell, total_capacity=1, category_capacities={"food": 1})

    num_threads = 30
    successes = []
    failures = []

    def attempt_reservation(thread_idx: int):
        v_id = f"VND-CONCUR-{thread_idx}"
        ok, res, msg = clean_fleet_manager.reserve_cell(
            vendor_id=v_id,
            h3_cell=cell,
            vendor_category="food",
            is_soft=True,
            ttl_minutes=15,
        )
        if ok and res:
            successes.append(res.reservation_id)
        else:
            failures.append(msg)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(attempt_reservation, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    # Invariants
    assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
    assert len(failures) == num_threads - 1

    # Verify cell occupancy is exactly 1
    occ = clean_fleet_manager.get_cell_occupancy(cell)
    assert occ["total_active_occupancy"] == 1
    assert occ["available_capacity"] == 0


# ===========================================================================
# 8. Multi-Vendor Fleet Dispatch & Clustering Minimization
# ===========================================================================

def test_multi_vendor_fleet_dispatch_clustering_mitigation(clean_fleet_manager):
    """
    Dispatch 4 vendors across candidate zones.
    Verifies that diminishing returns from D_eff = D_base / (1 + lambda*N)
    and the greedy welfare allocation spread vendors across distinct zones
    rather than clustering all 4 into the same cell.
    """
    vendors = [
        Vendor(vendor_id=f"VND-{i}", category="food", current_lat=16.5062, current_lng=80.6480)
        for i in range(1, 5)
    ]

    # Provide 4 distinct high-demand commercial candidate zones in Vijayawada
    candidates = [
        {"h3_cell": "88619aa6c7fffff", "latitude": 16.5062, "longitude": 80.6480, "landmark_name": "Benz Circle", "base_demand": 24.0},
        {"h3_cell": "88619aa6c5fffff", "latitude": 16.5140, "longitude": 80.6550, "landmark_name": "MG Road", "base_demand": 22.0},
        {"h3_cell": "88619aa6c3fffff", "latitude": 16.5180, "longitude": 80.6400, "landmark_name": "Governorpet", "base_demand": 20.0},
        {"h3_cell": "88619aa6c1fffff", "latitude": 16.5220, "longitude": 80.6350, "landmark_name": "Eluru Road", "base_demand": 19.0},
    ]

    result = clean_fleet_manager.dispatch_fleet(
        vendors=vendors,
        candidate_cells=candidates,
        fleet_id="fleet_vijayawada",
        max_travel_km=5.0,
    )

    assert result["dispatched_count"] == 4
    assert result["idle_count"] == 0
    assert result["fleet_total_net_profit_inr"] > 0
    assert len(result["assignments"]) == 4

    # High cluster dispersion: vendors distributed across at least 3 distinct cells
    assigned_cells = set(a["assigned_h3_cell"] for a in result["assignments"])
    assert len(assigned_cells) >= 3
    assert result["cluster_dispersion_ratio"] >= 0.75

    # Each assigned vendor has auto-created soft reservation
    for a in result["assignments"]:
        assert a["reservation_id"] is not None
        assert a["dispatch_verdict"] == "DISPATCHED"


# ===========================================================================
# 9. Support for Independent Vendors and Organized Fleets
# ===========================================================================

def test_fleet_dispatch_independent_and_organized(clean_fleet_manager):
    """Supports both single independent vendors and multi-cart fleets."""
    # Independent single vendor dispatch
    v_indep = Vendor(vendor_id="INDEP-1", category="beverages", current_lat=16.5062, current_lng=80.6480)
    indep_res = clean_fleet_manager.dispatch_fleet(
        vendors=[v_indep],
        fleet_id=None,
    )
    assert indep_res["fleet_id"] == "independent"
    assert indep_res["total_vendors"] == 1
    assert indep_res["dispatched_count"] == 1

    # Organized fleet
    fleet_vendors = [
        Vendor(vendor_id=f"CORP-{i}", category="snacks", current_lat=16.5062, current_lng=80.6480)
        for i in range(1, 3)
    ]
    corp_res = clean_fleet_manager.dispatch_fleet(
        vendors=fleet_vendors,
        fleet_id="corp_fleet_alpha",
    )
    assert corp_res["fleet_id"] == "corp_fleet_alpha"
    assert corp_res["dispatched_count"] == 2


# ===========================================================================
# 10. Fleet API Endpoint Contract & Validation
# ===========================================================================

def test_fleet_reserve_and_occupancy_endpoints():
    """Validates POST /v1/fleet/reserve-cell and GET /v1/fleet/cell-occupancy."""
    cell = "88619aa6c7fffff"

    req = FleetReserveCellRequest(
        vendor_id="API-VND-1",
        h3_cell=cell,
        vendor_category="food",
        duration_minutes=90,
        is_soft=True,
        ttl_minutes=20,
    )

    coro_res = fleet_reserve_cell(req, api_key="test-key")
    res_body = run_coroutine(coro_res)
    validated_res = FleetReserveCellResponse(**res_body)

    assert validated_res.status == "CONFIRMED"
    assert validated_res.reservation_id is not None
    assert validated_res.vendor_id == "API-VND-1"
    assert validated_res.h3_cell == cell

    # Check cell occupancy endpoint
    coro_occ = fleet_cell_occupancy(h3_cell=cell)
    occ_body = run_coroutine(coro_occ)
    validated_occ = CellOccupancyResponse(**occ_body)

    assert validated_occ.h3_cell == cell
    assert validated_occ.total_active_occupancy >= 1
    assert "food" in validated_occ.categories
    assert validated_occ.categories["food"].active_count >= 1

    # Cancel reservation via endpoint
    cancel_req = FleetCancelReservationRequest(
        reservation_id=validated_res.reservation_id,
        vendor_id="API-VND-1",
    )
    coro_cancel = fleet_cancel_reservation(cancel_req, api_key="test-key")
    cancel_body = run_coroutine(coro_cancel)
    validated_cancel = FleetCancelReservationResponse(**cancel_body)

    assert validated_cancel.status == "CANCELLED"
    assert validated_cancel.released_h3_cell == cell


def test_fleet_dispatch_endpoint_contract():
    """Validates POST /v1/fleet/dispatch pipeline and validated response schema."""
    req = FleetDispatchRequest(
        fleet_id="fleet_test_api",
        vendors=[
            FleetDispatchVendorItem(
                vendor_id="DISP-V1",
                vendor_category="food",
                latitude=16.5062,
                longitude=80.6480,
            ),
            FleetDispatchVendorItem(
                vendor_id="DISP-V2",
                vendor_category="fruits",
                latitude=16.5070,
                longitude=80.6490,
            ),
        ],
        search_radius_km=3.0,
        max_travel_km=4.0,
    )

    coro = fleet_dispatch(req, api_key="test-key")
    body = run_coroutine(coro)
    validated = FleetDispatchResponse(**body)

    assert validated.fleet_id == "fleet_test_api"
    assert validated.total_vendors == 2
    assert validated.dispatched_count >= 1
    assert validated.fleet_total_net_profit_inr > 0
    assert len(validated.assignments) >= 1
    assert validated.cluster_dispersion_ratio > 0.0
