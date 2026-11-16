"""
GeoDemand Fleet Management & Multi-Vendor Dispatch Engine.

Key Capabilities:
- Vendor Identity, Category, Location, and Availability tracking
- H3 Cell Total and Category-Specific Capacity enforcement
- Non-linear competition demand cannibalization modeling:
    D_effective(H) = D_base(H) / (1 + lambda * N_same_category(H))
- Soft (provisional) & Hard H3 reservations with TTL expiration
- Reservation cancellation and conflict prevention (anti-double booking)
- Thread-safe concurrency control via RLock and atomic SQLite persistence
- Multi-vendor fleet dispatch maximizing fleet-wide expected profit
- Minimization of vendor clustering via diminishing returns and dispersion rewards
- Independent vendor and organized multi-cart fleet support
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set

import h3

from api.spatial_filter import is_water_location, is_water_cell
from api.location import get_location_context, generate_candidates
from api.poi_resolver import resolve_candidate_landmark
from api.providers.routing import RoutingProviderFactory
from config.product_pricing import get_category_pricing, PRODUCT_CATEGORIES
from config.settings import DEFAULT_AOV_INR, DEFAULT_VARIABLE_COST_RATE, DEFAULT_FIXED_COST_PER_DAY_INR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Constants & Competition Penalties
# ---------------------------------------------------------------------------

DEFAULT_CELL_TOTAL_CAPACITY = 6

DEFAULT_CELL_CATEGORY_CAPACITIES: Dict[str, int] = {
    "food": 2,
    "fruits": 2,
    "beverages": 2,
    "snacks": 2,
    "repair": 1,
    "salon": 1,
}

# Lambda competition penalty coefficients: D_eff = D_base / (1 + lambda * N)
CATEGORY_LAMBDA_COMPETITION: Dict[str, float] = {
    "food": 0.35,
    "fruits": 0.30,
    "beverages": 0.25,
    "snacks": 0.40,
    "repair": 0.50,
    "salon": 0.45,
}
DEFAULT_LAMBDA = 0.35

DEFAULT_SOFT_RESERVATION_TTL_MINUTES = 15
DEFAULT_DISPATCH_RADIUS_KM = 5.0
DEFAULT_FUEL_PRICE_INR_PER_LITRE = 102.50
DEFAULT_FUEL_EFFICIENCY_KM_PER_L = 35.0


class VendorStatus(str, Enum):
    IDLE = "IDLE"
    RESERVED = "RESERVED"
    DISPATCHED = "DISPATCHED"
    OCCUPIED = "OCCUPIED"
    OFFLINE = "OFFLINE"


class ReservationStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    COMPLETED = "COMPLETED"


# ---------------------------------------------------------------------------
# Core Entities
# ---------------------------------------------------------------------------

@dataclass
class Vendor:
    vendor_id: str
    category: str
    name: str = ""
    current_lat: float = 16.5062
    current_lng: float = 80.6480
    current_h3: str = ""
    status: VendorStatus = VendorStatus.IDLE
    inventory_capacity: int = 100
    current_inventory: int = 100
    average_order_value: Optional[float] = None
    fuel_efficiency_km_l: float = DEFAULT_FUEL_EFFICIENCY_KM_PER_L
    speed_km_h: float = 15.0
    fleet_id: Optional[str] = None
    active_reservation_id: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.category = self.category.lower().strip()
        if not self.current_h3:
            self.current_h3 = h3.latlng_to_cell(self.current_lat, self.current_lng, 8)
        if isinstance(self.status, str):
            self.status = VendorStatus(self.status)

    @property
    def is_available(self) -> bool:
        return self.status in (VendorStatus.IDLE,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vendor_id": self.vendor_id,
            "name": self.name or f"Vendor {self.vendor_id}",
            "category": self.category,
            "current_lat": self.current_lat,
            "current_lng": self.current_lng,
            "current_h3": self.current_h3,
            "status": self.status.value,
            "inventory_capacity": self.inventory_capacity,
            "current_inventory": self.current_inventory,
            "average_order_value": self.average_order_value,
            "fuel_efficiency_km_l": self.fuel_efficiency_km_l,
            "speed_km_h": self.speed_km_h,
            "fleet_id": self.fleet_id,
            "active_reservation_id": self.active_reservation_id,
            "last_heartbeat": self.last_heartbeat.isoformat() if isinstance(self.last_heartbeat, datetime) else str(self.last_heartbeat),
        }


@dataclass
class Reservation:
    reservation_id: str
    vendor_id: str
    h3_cell: str
    category: str
    status: ReservationStatus = ReservationStatus.PENDING
    is_soft: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=DEFAULT_SOFT_RESERVATION_TTL_MINUTES))
    slot_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    slot_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=60))
    deposit_amount_inr: float = 0.0
    fleet_id: Optional[str] = None

    def __post_init__(self):
        self.category = self.category.lower().strip()
        if isinstance(self.status, str):
            self.status = ReservationStatus(self.status)

    def is_active_at(self, ref_time: Optional[datetime] = None) -> bool:
        """Returns True if reservation is currently valid, unexpired, and active/pending."""
        now = ref_time or datetime.now(timezone.utc)
        if self.status not in (ReservationStatus.PENDING, ReservationStatus.ACTIVE):
            return False
        if self.is_soft and now > self.expires_at:
            return False
        if now > self.slot_end:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "vendor_id": self.vendor_id,
            "h3_cell": self.h3_cell,
            "category": self.category,
            "status": self.status.value,
            "is_soft": self.is_soft,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at),
            "expires_at": self.expires_at.isoformat() if isinstance(self.expires_at, datetime) else str(self.expires_at),
            "slot_start": self.slot_start.isoformat() if isinstance(self.slot_start, datetime) else str(self.slot_start),
            "slot_end": self.slot_end.isoformat() if isinstance(self.slot_end, datetime) else str(self.slot_end),
            "deposit_amount_inr": self.deposit_amount_inr,
            "fleet_id": self.fleet_id,
        }


# ---------------------------------------------------------------------------
# Competition & Demand-Sharing Calculation
# ---------------------------------------------------------------------------

def calculate_effective_demand(
    base_demand: float,
    same_category_count: int,
    category: str = "food",
    custom_lambda: Optional[float] = None,
) -> float:
    """
    Roadmap formula for non-linear demand cannibalization:
    D_effective(H) = D_base(H) / (1 + lambda * N_same_category(H))
    """
    lam = custom_lambda if custom_lambda is not None else CATEGORY_LAMBDA_COMPETITION.get(category.lower(), DEFAULT_LAMBDA)
    n = max(0, int(same_category_count))
    if n == 0 or lam <= 0:
        return round(float(base_demand), 2)
    effective = float(base_demand) / (1.0 + lam * n)
    return round(effective, 2)


def calculate_demand_sharing_penalty(
    base_demand: float,
    same_category_count: int,
    category: str = "food",
    custom_lambda: Optional[float] = None,
) -> float:
    """Calculates customer loss due to co-location competition."""
    eff = calculate_effective_demand(base_demand, same_category_count, category, custom_lambda)
    return round(max(0.0, float(base_demand) - eff), 2)


# ---------------------------------------------------------------------------
# FleetManager Engine
# ---------------------------------------------------------------------------

class FleetManager:
    """
    Thread-safe fleet manager with atomic reservations, cell capacity limits,
    demand cannibalization penalties, and multi-vendor dispatch optimization.
    """

    _instance: Optional[FleetManager] = None
    _init_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return
        self._lock = threading.RLock()
        self.db_path = db_path or str(Path(__file__).parent.parent.parent / "data" / "geodemand_fleet.db")
        self._vendors: Dict[str, Vendor] = {}
        self._reservations: Dict[str, Reservation] = {}
        self._cell_capacities: Dict[str, int] = {}
        self._cell_category_capacities: Dict[Tuple[str, str], int] = {}
        self.routing = RoutingProviderFactory.get_provider()

        self._init_sqlite_db()
        self._load_from_sqlite()
        self._initialized = True

    def reset_state_for_testing(self):
        """Helper for test isolation."""
        with self._lock:
            self._vendors.clear()
            self._reservations.clear()
            self._cell_capacities.clear()
            self._cell_category_capacities.clear()
            if self.db_path and self.db_path != ":memory:":
                try:
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM fleet_reservations")
                    cur.execute("DELETE FROM fleet_vendors")
                    conn.commit()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error resetting test sqlite db: {e}")

    def _init_sqlite_db(self):
        """Initializes SQLite schema for fleet reservations and vendor registries."""
        if not self.db_path or self.db_path == ":memory:":
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fleet_vendors (
                    vendor_id TEXT PRIMARY KEY,
                    name TEXT,
                    category TEXT NOT NULL,
                    current_lat REAL NOT NULL,
                    current_lng REAL NOT NULL,
                    current_h3 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    inventory_capacity INTEGER NOT NULL,
                    current_inventory INTEGER NOT NULL,
                    average_order_value REAL,
                    fuel_efficiency_km_l REAL NOT NULL,
                    speed_km_h REAL NOT NULL,
                    fleet_id TEXT,
                    active_reservation_id TEXT,
                    last_heartbeat TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fleet_reservations (
                    reservation_id TEXT PRIMARY KEY,
                    vendor_id TEXT NOT NULL,
                    h3_cell TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_soft INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    slot_start TEXT NOT NULL,
                    slot_end TEXT NOT NULL,
                    deposit_amount_inr REAL NOT NULL,
                    fleet_id TEXT
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_f_res_cell ON fleet_reservations(h3_cell, status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_f_res_vendor ON fleet_reservations(vendor_id, status)")
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite fleet DB: {e}")

    def _load_from_sqlite(self):
        """Loads persistent records from SQLite on startup."""
        if not self.db_path or self.db_path == ":memory:":
            return
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Load vendors
            cur.execute("SELECT * FROM fleet_vendors")
            for row in cur.fetchall():
                v = Vendor(
                    vendor_id=row["vendor_id"],
                    category=row["category"],
                    name=row["name"] or "",
                    current_lat=row["current_lat"],
                    current_lng=row["current_lng"],
                    current_h3=row["current_h3"],
                    status=VendorStatus(row["status"]),
                    inventory_capacity=row["inventory_capacity"],
                    current_inventory=row["current_inventory"],
                    average_order_value=row["average_order_value"],
                    fuel_efficiency_km_l=row["fuel_efficiency_km_l"],
                    speed_km_h=row["speed_km_h"],
                    fleet_id=row["fleet_id"],
                    active_reservation_id=row["active_reservation_id"],
                    last_heartbeat=datetime.fromisoformat(row["last_heartbeat"]),
                )
                self._vendors[v.vendor_id] = v

            # Load active/pending reservations
            cur.execute("SELECT * FROM fleet_reservations WHERE status IN ('PENDING', 'ACTIVE')")
            for row in cur.fetchall():
                r = Reservation(
                    reservation_id=row["reservation_id"],
                    vendor_id=row["vendor_id"],
                    h3_cell=row["h3_cell"],
                    category=row["category"],
                    status=ReservationStatus(row["status"]),
                    is_soft=bool(row["is_soft"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=datetime.fromisoformat(row["expires_at"]),
                    slot_start=datetime.fromisoformat(row["slot_start"]),
                    slot_end=datetime.fromisoformat(row["slot_end"]),
                    deposit_amount_inr=row["deposit_amount_inr"],
                    fleet_id=row["fleet_id"],
                )
                self._reservations[r.reservation_id] = r
            conn.close()
            # Clean any loaded reservations that expired while offline
            self.sweep_expired_reservations()
        except Exception as e:
            logger.warning(f"Error loading state from SQLite: {e}")

    def _sync_vendor_to_db(self, vendor: Vendor):
        if not self.db_path or self.db_path == ":memory:":
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO fleet_vendors (
                    vendor_id, name, category, current_lat, current_lng, current_h3,
                    status, inventory_capacity, current_inventory, average_order_value,
                    fuel_efficiency_km_l, speed_km_h, fleet_id, active_reservation_id, last_heartbeat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                vendor.vendor_id, vendor.name, vendor.category, vendor.current_lat, vendor.current_lng, vendor.current_h3,
                vendor.status.value, vendor.inventory_capacity, vendor.current_inventory, vendor.average_order_value,
                vendor.fuel_efficiency_km_l, vendor.speed_km_h, vendor.fleet_id, vendor.active_reservation_id,
                vendor.last_heartbeat.isoformat()
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to sync vendor {vendor.vendor_id} to DB: {e}")

    def _sync_reservation_to_db(self, reservation: Reservation):
        if not self.db_path or self.db_path == ":memory:":
            return
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO fleet_reservations (
                    reservation_id, vendor_id, h3_cell, category, status, is_soft,
                    created_at, expires_at, slot_start, slot_end, deposit_amount_inr, fleet_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reservation.reservation_id, reservation.vendor_id, reservation.h3_cell, reservation.category,
                reservation.status.value, 1 if reservation.is_soft else 0,
                reservation.created_at.isoformat(), reservation.expires_at.isoformat(),
                reservation.slot_start.isoformat(), reservation.slot_end.isoformat(),
                reservation.deposit_amount_inr, reservation.fleet_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to sync reservation {reservation.reservation_id} to DB: {e}")

    # -----------------------------------------------------------------------
    # Capacity Configuration
    # -----------------------------------------------------------------------

    def set_cell_capacity(self, h3_cell: str, total_capacity: int, category_capacities: Optional[Dict[str, int]] = None):
        """Configure custom total and category capacities for an H3 cell."""
        with self._lock:
            self._cell_capacities[h3_cell] = max(1, total_capacity)
            if category_capacities:
                for cat, cap in category_capacities.items():
                    self._cell_category_capacities[(h3_cell, cat.lower())] = max(0, cap)

    def get_cell_total_capacity(self, h3_cell: str) -> int:
        return self._cell_capacities.get(h3_cell, DEFAULT_CELL_TOTAL_CAPACITY)

    def get_cell_category_capacity(self, h3_cell: str, category: str) -> int:
        cat_clean = category.lower().strip()
        custom = self._cell_category_capacities.get((h3_cell, cat_clean))
        if custom is not None:
            return custom
        return DEFAULT_CELL_CATEGORY_CAPACITIES.get(cat_clean, 2)

    # -----------------------------------------------------------------------
    # Vendor Management
    # -----------------------------------------------------------------------

    def register_vendor(self, vendor: Vendor) -> Vendor:
        """Register or update vendor state."""
        with self._lock:
            self._vendors[vendor.vendor_id] = vendor
            self._sync_vendor_to_db(vendor)
            return vendor

    def get_vendor(self, vendor_id: str) -> Optional[Vendor]:
        with self._lock:
            return self._vendors.get(vendor_id)

    def list_vendors(self, fleet_id: Optional[str] = None, status: Optional[VendorStatus] = None) -> List[Vendor]:
        with self._lock:
            res = list(self._vendors.values())
            if fleet_id:
                res = [v for v in res if v.fleet_id == fleet_id]
            if status:
                res = [v for v in res if v.status == status]
            return res

    # -----------------------------------------------------------------------
    # Reservation Management
    # -----------------------------------------------------------------------

    def sweep_expired_reservations(self, ref_time: Optional[datetime] = None) -> int:
        """
        Sweeps soft reservations that have exceeded their TTL.
        Releases cell capacity and resets vendor status from RESERVED to IDLE.
        """
        now = ref_time or datetime.now(timezone.utc)
        expired_count = 0
        with self._lock:
            for r in list(self._reservations.values()):
                if r.status in (ReservationStatus.PENDING, ReservationStatus.ACTIVE):
                    if r.is_soft and now > r.expires_at:
                        r.status = ReservationStatus.EXPIRED
                        self._sync_reservation_to_db(r)
                        expired_count += 1
                        # Free vendor if bound
                        v = self._vendors.get(r.vendor_id)
                        if v and v.active_reservation_id == r.reservation_id:
                            v.active_reservation_id = None
                            if v.status in (VendorStatus.RESERVED, VendorStatus.DISPATCHED):
                                v.status = VendorStatus.IDLE
                                self._sync_vendor_to_db(v)
                    elif now > r.slot_end:
                        r.status = ReservationStatus.COMPLETED
                        self._sync_reservation_to_db(r)
                        v = self._vendors.get(r.vendor_id)
                        if v and v.active_reservation_id == r.reservation_id:
                            v.active_reservation_id = None
                            if v.status in (VendorStatus.RESERVED, VendorStatus.DISPATCHED, VendorStatus.OCCUPIED):
                                v.status = VendorStatus.IDLE
                                self._sync_vendor_to_db(v)
            return expired_count

    def get_active_reservations_for_cell(self, h3_cell: str, ref_time: Optional[datetime] = None) -> List[Reservation]:
        """Returns all currently active unexpired reservations for cell H3."""
        now = ref_time or datetime.now(timezone.utc)
        self.sweep_expired_reservations(now)
        with self._lock:
            return [
                r for r in self._reservations.values()
                if r.h3_cell == h3_cell and r.is_active_at(now)
            ]

    def get_cell_occupancy(self, h3_cell: str, vendor_category: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns cell capacity, active occupancy, available slots,
        category breakdown, and effective demand multipliers.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self.sweep_expired_reservations(now)

            active_res = self.get_active_reservations_for_cell(h3_cell, now)
            total_cap = self.get_cell_total_capacity(h3_cell)
            total_active = len(active_res)
            avail_total = max(0, total_cap - total_active)

            # Category breakdowns
            cat_counts: Dict[str, int] = {}
            for r in active_res:
                cat_counts[r.category] = cat_counts.get(r.category, 0) + 1

            categories_detail: Dict[str, Any] = {}
            for cat in PRODUCT_CATEGORIES.keys():
                cat_cap = self.get_cell_category_capacity(h3_cell, cat)
                cur_count = cat_counts.get(cat, 0)
                lam = CATEGORY_LAMBDA_COMPETITION.get(cat, DEFAULT_LAMBDA)
                mult = round(1.0 / (1.0 + lam * cur_count), 4)

                categories_detail[cat] = {
                    "category": cat,
                    "active_count": cur_count,
                    "category_capacity": cat_cap,
                    "available_capacity": max(0, cat_cap - cur_count),
                    "competition_lambda": lam,
                    "effective_demand_multiplier": mult,
                }

            return {
                "h3_cell": h3_cell,
                "total_capacity": total_cap,
                "total_active_occupancy": total_active,
                "available_capacity": avail_total,
                "is_at_capacity": avail_total <= 0,
                "categories": categories_detail,
                "active_reservation_ids": [r.reservation_id for r in active_res],
            }

    def reserve_cell(
        self,
        vendor_id: str,
        h3_cell: str,
        vendor_category: str = "food",
        duration_minutes: int = 60,
        is_soft: bool = True,
        ttl_minutes: int = DEFAULT_SOFT_RESERVATION_TTL_MINUTES,
        deposit_amount_inr: float = 0.0,
        fleet_id: Optional[str] = None,
        slot_start_time: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[Reservation], str]:
        """
        Atomically reserve a spot in an H3 cell.
        Enforces total cell capacity and category-specific capacity.
        Prevents duplicate / conflicting reservations for the same vendor.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self.sweep_expired_reservations(now)

            # Check vendor conflict: vendor cannot have an active overlapping reservation
            existing_active = [
                r for r in self._reservations.values()
                if r.vendor_id == vendor_id and r.is_active_at(now)
            ]
            if existing_active:
                cur = existing_active[0]
                return False, None, f"Vendor {vendor_id} already has active reservation {cur.reservation_id} in cell {cur.h3_cell}"

            # Validate H3 cell format & water exclusion
            if not h3.is_valid_cell(h3_cell):
                return False, None, f"Invalid H3 cell: {h3_cell}"
            if is_water_cell(h3_cell):
                return False, None, f"H3 cell {h3_cell} is located in water/unsuitable zone"

            # Check capacity limits
            active_in_cell = self.get_active_reservations_for_cell(h3_cell, now)
            total_cap = self.get_cell_total_capacity(h3_cell)
            if len(active_in_cell) >= total_cap:
                return False, None, f"Cell {h3_cell} total capacity exceeded ({len(active_in_cell)}/{total_cap})"

            cat_clean = vendor_category.lower().strip()
            cat_cap = self.get_cell_category_capacity(h3_cell, cat_clean)
            cat_count = sum(1 for r in active_in_cell if r.category == cat_clean)
            if cat_count >= cat_cap:
                return False, None, f"Cell {h3_cell} category capacity exceeded for '{cat_clean}' ({cat_count}/{cat_cap})"

            # Calculate time slots
            start_t = slot_start_time or now
            end_t = start_t + timedelta(minutes=duration_minutes)
            exp_t = now + timedelta(minutes=ttl_minutes) if is_soft else end_t

            res_id = f"RES-{uuid.uuid4().hex[:10].upper()}"
            res = Reservation(
                reservation_id=res_id,
                vendor_id=vendor_id,
                h3_cell=h3_cell,
                category=cat_clean,
                status=ReservationStatus.PENDING if is_soft else ReservationStatus.ACTIVE,
                is_soft=is_soft,
                created_at=now,
                expires_at=exp_t,
                slot_start=start_t,
                slot_end=end_t,
                deposit_amount_inr=deposit_amount_inr,
                fleet_id=fleet_id,
            )

            # Commit reservation to in-memory state and SQLite
            self._reservations[res_id] = res
            self._sync_reservation_to_db(res)

            # Update or create vendor record
            v = self._vendors.get(vendor_id)
            if v:
                v.status = VendorStatus.RESERVED
                v.active_reservation_id = res_id
                if fleet_id and not v.fleet_id:
                    v.fleet_id = fleet_id
                self._sync_vendor_to_db(v)
            else:
                # Auto-register vendor if not previously known
                lat, lng = h3.cell_to_latlng(h3_cell)
                new_v = Vendor(
                    vendor_id=vendor_id,
                    category=cat_clean,
                    current_lat=lat,
                    current_lng=lng,
                    current_h3=h3_cell,
                    status=VendorStatus.RESERVED,
                    fleet_id=fleet_id,
                    active_reservation_id=res_id,
                )
                self._vendors[vendor_id] = new_v
                self._sync_vendor_to_db(new_v)

            logger.info(f"Confirmed reservation {res_id} for vendor {vendor_id} in {h3_cell} (soft={is_soft})")
            return True, res, "Reservation confirmed successfully"

    def cancel_reservation(self, reservation_id: str, vendor_id: Optional[str] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Cancels an active or pending reservation, immediately releasing cell capacity.
        """
        with self._lock:
            res = self._reservations.get(reservation_id)
            if not res:
                return False, f"Reservation {reservation_id} not found", None

            if vendor_id and res.vendor_id != vendor_id:
                return False, f"Vendor {vendor_id} does not own reservation {reservation_id}", None

            if res.status in (ReservationStatus.CANCELLED, ReservationStatus.EXPIRED, ReservationStatus.COMPLETED):
                return False, f"Reservation {reservation_id} is already in {res.status.value} state", res.h3_cell

            res.status = ReservationStatus.CANCELLED
            self._sync_reservation_to_db(res)

            v = self._vendors.get(res.vendor_id)
            if v and v.active_reservation_id == reservation_id:
                v.active_reservation_id = None
                v.status = VendorStatus.IDLE
                self._sync_vendor_to_db(v)

            logger.info(f"Cancelled reservation {reservation_id} for vendor {res.vendor_id} in {res.h3_cell}")
            return True, f"Reservation {reservation_id} successfully cancelled", res.h3_cell

    # -----------------------------------------------------------------------
    # Multi-Vendor Fleet Dispatch Engine
    # -----------------------------------------------------------------------

    def dispatch_fleet(
        self,
        vendors: List[Vendor],
        candidate_cells: Optional[List[Dict[str, Any]]] = None,
        fleet_id: Optional[str] = "default_fleet",
        search_radius_km: float = DEFAULT_DISPATCH_RADIUS_KM,
        max_travel_km: float = DEFAULT_DISPATCH_RADIUS_KM,
        fuel_price_inr: float = DEFAULT_FUEL_PRICE_INR_PER_LITRE,
        auto_reserve_soft: bool = True,
        soft_reservation_ttl_minutes: int = DEFAULT_SOFT_RESERVATION_TTL_MINUTES,
    ) -> Dict[str, Any]:
        """
        Dispatches multiple vendors to candidate commercial zones.
        - Maximizes fleet-wide net profit: Revenue - COGS - Travel/Fuel Costs - Opex
        - Minimizes unnecessary clustering by incorporating non-linear demand cannibalization:
            D_effective(H) = D_base(H) / (1 + lambda * N_same_category)
        - Prevents duplicate dispatches and enforces cell capacities
        - Creates soft H3 reservations for assigned spots
        """
        with self._lock:
            start_t0 = time.time()
            now = datetime.now(timezone.utc)
            self.sweep_expired_reservations(now)

            if not vendors:
                return {
                    "dispatch_id": f"DISP-{uuid.uuid4().hex[:8].upper()}",
                    "fleet_id": fleet_id or "independent",
                    "timestamp": now.isoformat(),
                    "total_vendors": 0,
                    "dispatched_count": 0,
                    "idle_count": 0,
                    "fleet_total_revenue_inr": 0.0,
                    "fleet_total_net_profit_inr": 0.0,
                    "fleet_total_fuel_cost_inr": 0.0,
                    "assignments": [],
                    "unassigned_vendors": [],
                    "cluster_dispersion_ratio": 1.0,
                    "dispatch_explanation": "No vendors provided for dispatch.",
                }

            # Register/update vendors in internal manager state
            for v in vendors:
                self.register_vendor(v)

            # Filter vendors available for dispatch (not occupied, not already holding active reservation)
            available_vendors: List[Vendor] = []
            unassigned: List[Dict[str, Any]] = []

            for v in vendors:
                active_res = [r for r in self._reservations.values() if r.vendor_id == v.vendor_id and r.is_active_at(now)]
                if active_res:
                    unassigned.append({
                        "vendor_id": v.vendor_id,
                        "category": v.category,
                        "status": v.status.value,
                        "reason": f"Vendor already has active reservation {active_res[0].reservation_id}",
                    })
                elif v.status in (VendorStatus.OCCUPIED, VendorStatus.OFFLINE):
                    unassigned.append({
                        "vendor_id": v.vendor_id,
                        "category": v.category,
                        "status": v.status.value,
                        "reason": f"Vendor status is {v.status.value}",
                    })
                else:
                    available_vendors.append(v)

            if not available_vendors:
                return {
                    "dispatch_id": f"DISP-{uuid.uuid4().hex[:8].upper()}",
                    "fleet_id": fleet_id or "independent",
                    "timestamp": now.isoformat(),
                    "total_vendors": len(vendors),
                    "dispatched_count": 0,
                    "idle_count": len(unassigned),
                    "fleet_total_revenue_inr": 0.0,
                    "fleet_total_net_profit_inr": 0.0,
                    "fleet_total_fuel_cost_inr": 0.0,
                    "assignments": [],
                    "unassigned_vendors": unassigned,
                    "cluster_dispersion_ratio": 1.0,
                    "dispatch_explanation": "All vendors are currently unavailable or already holding active reservations.",
                }

            # Resolve Candidate Pool
            pool: List[Dict[str, Any]] = []
            if candidate_cells:
                for c in candidate_cells:
                    cell_id = c.get("h3_cell")
                    if cell_id and h3.is_valid_cell(cell_id) and not is_water_cell(cell_id):
                        lat, lng = h3.cell_to_latlng(cell_id)
                        pool.append({
                            "h3_cell": cell_id,
                            "latitude": float(c.get("latitude", lat)),
                            "longitude": float(c.get("longitude", lng)),
                            "landmark_name": c.get("landmark_name") or resolve_candidate_landmark(lat, lng).get("name", "Commercial Hub"),
                            "base_demand": float(c.get("base_demand") or c.get("expected_customers", 18)),
                        })
            else:
                # Dynamically generate candidates around centroid of available vendors
                centroid_lat = sum(v.current_lat for v in available_vendors) / len(available_vendors)
                centroid_lng = sum(v.current_lng for v in available_vendors) / len(available_vendors)
                generated = generate_candidates(centroid_lat, centroid_lng, search_radius_km=search_radius_km)
                for g in generated:
                    if not g.is_water:
                        pool.append({
                            "h3_cell": g.h3_cell,
                            "latitude": g.centroid_lat,
                            "longitude": g.centroid_lng,
                            "landmark_name": g.landmark_name,
                            "base_demand": 18.0,
                        })

            if not pool:
                for v in available_vendors:
                    unassigned.append({
                        "vendor_id": v.vendor_id,
                        "category": v.category,
                        "status": v.status.value,
                        "reason": "No accessible candidate zones found within search radius.",
                    })
                return {
                    "dispatch_id": f"DISP-{uuid.uuid4().hex[:8].upper()}",
                    "fleet_id": fleet_id or "independent",
                    "timestamp": now.isoformat(),
                    "total_vendors": len(vendors),
                    "dispatched_count": 0,
                    "idle_count": len(unassigned),
                    "fleet_total_revenue_inr": 0.0,
                    "fleet_total_net_profit_inr": 0.0,
                    "fleet_total_fuel_cost_inr": 0.0,
                    "assignments": [],
                    "unassigned_vendors": unassigned,
                    "cluster_dispersion_ratio": 1.0,
                    "dispatch_explanation": "No eligible candidate cells available.",
                }

            # Map initial cell occupancies from active reservations
            cell_initial_occupancy: Dict[str, Dict[str, int]] = {}
            cell_total_active: Dict[str, int] = {}

            for c in pool:
                cid = c["h3_cell"]
                active_res = self.get_active_reservations_for_cell(cid, now)
                cell_total_active[cid] = len(active_res)
                cat_dict: Dict[str, int] = {}
                for r in active_res:
                    cat_dict[r.category] = cat_dict.get(r.category, 0) + 1
                cell_initial_occupancy[cid] = cat_dict

            # -------------------------------------------------------------------
            # Greedy Marginal Welfare Maximization with Cannibalization Penalty
            # -------------------------------------------------------------------
            # Repeatedly picks the (vendor, cell) assignment that maximizes
            # marginal fleet net profit uplift while respecting capacity constraints.
            # Diminishing returns from D_effective = D_base / (1 + lambda * N)
            # inherently discourages clustering vendors into the same cell.

            assignments: List[Dict[str, Any]] = []
            assigned_vendor_ids: Set[str] = set()

            # Dynamic tracking during allocation loop
            dynamic_cell_total = dict(cell_total_active)
            dynamic_cell_cat: Dict[str, Dict[str, int]] = {
                cid: dict(cats) for cid, cats in cell_initial_occupancy.items()
            }

            while len(assigned_vendor_ids) < len(available_vendors):
                best_pair: Optional[Tuple[Vendor, Dict[str, Any], Dict[str, Any]]] = None
                best_profit_score = -1e9

                for v in available_vendors:
                    if v.vendor_id in assigned_vendor_ids:
                        continue

                    pricing = get_category_pricing(v.category)
                    aov = v.average_order_value or pricing.default_aov_inr
                    cogs_rate = pricing.default_cogs_rate

                    for cand in pool:
                        cid = cand["h3_cell"]
                        tot_cap = self.get_cell_total_capacity(cid)
                        cur_tot = dynamic_cell_total.get(cid, 0)
                        if cur_tot >= tot_cap:
                            continue

                        cat_cap = self.get_cell_category_capacity(cid, v.category)
                        cur_cat = dynamic_cell_cat.get(cid, {}).get(v.category, 0)
                        if cur_cat >= cat_cap:
                            continue

                        # Travel metrics
                        travel_route = self.routing.get_travel_time(
                            v.current_lat, v.current_lng,
                            cand["latitude"], cand["longitude"]
                        )
                        dist_km = travel_route.get("distance_km", 1.5)
                        dur_min = travel_route.get("duration_min", 6.0)

                        if dist_km > max_travel_km:
                            continue

                        fuel_cost = (dist_km / v.fuel_efficiency_km_l) * fuel_price_inr

                        # Effective demand considering current category vendors in cell
                        base_dem = cand["base_demand"]
                        eff_dem = calculate_effective_demand(base_dem, cur_cat, v.category)
                        dem_pen = calculate_demand_sharing_penalty(base_dem, cur_cat, v.category)

                        gross_rev = eff_dem * aov
                        cogs = gross_rev * cogs_rate
                        # Hourly fixed overhead share for a 3-hour dispatch horizon
                        fixed_opex = (DEFAULT_FIXED_COST_PER_DAY_INR / 12.0) * 3.0
                        net_profit = (gross_rev * 3.0) - (cogs * 3.0) - fuel_cost - fixed_opex

                        # Dispersion bonus: small incentive to prefer unpopulated cells if profits are tied
                        dispersion_bonus = 50.0 if cur_tot == 0 else (20.0 if cur_cat == 0 else 0.0)
                        score = net_profit + dispersion_bonus

                        # Only dispatch if expected net profit is positive
                        if net_profit > 0 and score > best_profit_score:
                            best_profit_score = score
                            best_pair = (
                                v,
                                cand,
                                {
                                    "travel_distance_km": round(dist_km, 2),
                                    "travel_time_minutes": round(dur_min, 1),
                                    "fuel_cost_inr": round(fuel_cost, 2),
                                    "base_demand_customers": round(base_dem, 1),
                                    "effective_demand_customers": round(eff_dem, 1),
                                    "demand_sharing_penalty_customers": round(dem_pen, 1),
                                    "same_category_co_located": cur_cat,
                                    "expected_gross_revenue_inr": round(gross_rev * 3.0, 2),
                                    "expected_cogs_inr": round(cogs * 3.0, 2),
                                    "expected_net_profit_inr": round(net_profit, 2),
                                }
                            )

                if best_pair is None:
                    # No further profitable or feasible assignments can be made
                    break

                v_win, c_win, metrics = best_pair
                cid = c_win["h3_cell"]

                # Create soft reservation if requested
                res_id = None
                if auto_reserve_soft:
                    ok, res_obj, _ = self.reserve_cell(
                        vendor_id=v_win.vendor_id,
                        h3_cell=cid,
                        vendor_category=v_win.category,
                        duration_minutes=180,
                        is_soft=True,
                        ttl_minutes=soft_reservation_ttl_minutes,
                        fleet_id=fleet_id,
                    )
                    if ok and res_obj:
                        res_id = res_obj.reservation_id

                v_win.status = VendorStatus.DISPATCHED
                assigned_vendor_ids.add(v_win.vendor_id)
                dynamic_cell_total[cid] = dynamic_cell_total.get(cid, 0) + 1
                dynamic_cell_cat.setdefault(cid, {})[v_win.category] = dynamic_cell_cat.get(cid, {}).get(v_win.category, 0) + 1

                assignments.append({
                    "vendor_id": v_win.vendor_id,
                    "vendor_category": v_win.category,
                    "origin_h3": v_win.current_h3,
                    "assigned_h3_cell": cid,
                    "landmark_name": c_win["landmark_name"],
                    "travel_distance_km": metrics["travel_distance_km"],
                    "travel_time_minutes": metrics["travel_time_minutes"],
                    "fuel_cost_inr": metrics["fuel_cost_inr"],
                    "base_demand_customers": metrics["base_demand_customers"],
                    "effective_demand_customers": metrics["effective_demand_customers"],
                    "demand_sharing_penalty_customers": metrics["demand_sharing_penalty_customers"],
                    "same_category_co_located": metrics["same_category_co_located"],
                    "expected_gross_revenue_inr": metrics["expected_gross_revenue_inr"],
                    "expected_cogs_inr": metrics["expected_cogs_inr"],
                    "expected_net_profit_inr": metrics["expected_net_profit_inr"],
                    "reservation_id": res_id,
                    "dispatch_verdict": "DISPATCHED",
                    "dispatch_reason": (
                        f"Assigned to {c_win['landmark_name']} ({cid}). "
                        f"Expected net profit ₹{metrics['expected_net_profit_inr']} with "
                        f"{metrics['effective_demand_customers']} eff. customers/hr "
                        f"({metrics['same_category_co_located']} same-category co-located)."
                    ),
                })

            # Identify unassigned vendors
            for v in available_vendors:
                if v.vendor_id not in assigned_vendor_ids:
                    unassigned.append({
                        "vendor_id": v.vendor_id,
                        "category": v.category,
                        "status": v.status.value,
                        "reason": "No profitable cell with remaining capacity within travel radius.",
                    })

            # Compute fleet totals
            tot_rev = round(sum(a["expected_gross_revenue_inr"] for a in assignments), 2)
            tot_prof = round(sum(a["expected_net_profit_inr"] for a in assignments), 2)
            tot_fuel = round(sum(a["fuel_cost_inr"] for a in assignments), 2)

            # Dispersion ratio: unique assigned cells / total assignments
            unique_cells = len(set(a["assigned_h3_cell"] for a in assignments))
            dispersion_ratio = round(unique_cells / len(assignments), 2) if assignments else 1.0

            dispatch_id = f"DISP-{uuid.uuid4().hex[:8].upper()}"
            solve_ms = round((time.time() - start_t0) * 1000, 2)

            explanation = (
                f"Fleet dispatch {dispatch_id} deployed {len(assignments)}/{len(vendors)} vendors "
                f"across {unique_cells} commercial zones (dispersion ratio {dispersion_ratio}). "
                f"Fleet expected net profit: ₹{tot_prof} (fuel expense: ₹{tot_fuel}). "
                f"Solved in {solve_ms}ms."
            )

            return {
                "dispatch_id": dispatch_id,
                "fleet_id": fleet_id or "independent",
                "timestamp": now.isoformat(),
                "total_vendors": len(vendors),
                "dispatched_count": len(assignments),
                "idle_count": len(unassigned),
                "fleet_total_revenue_inr": tot_rev,
                "fleet_total_net_profit_inr": tot_prof,
                "fleet_total_fuel_cost_inr": tot_fuel,
                "assignments": assignments,
                "unassigned_vendors": unassigned,
                "cluster_dispersion_ratio": dispersion_ratio,
                "dispatch_explanation": explanation,
            }
