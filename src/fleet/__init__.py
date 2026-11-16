"""
GeoDemand Fleet Management & Multi-Vendor Dispatch Package.
Provides cell capacity enforcement, competition sharing penalties,
atomic soft/hard H3 reservations, and fleet-wide profit optimization.
"""

try:
    from src.fleet.fleet_manager import (
        FleetManager,
        Vendor,
        Reservation,
        VendorStatus,
        ReservationStatus,
        calculate_effective_demand,
        CATEGORY_LAMBDA_COMPETITION,
        DEFAULT_LAMBDA,
        DEFAULT_CELL_TOTAL_CAPACITY,
        DEFAULT_CELL_CATEGORY_CAPACITIES,
    )
except ImportError:
    from fleet.fleet_manager import (
        FleetManager,
        Vendor,
        Reservation,
        VendorStatus,
        ReservationStatus,
        calculate_effective_demand,
        CATEGORY_LAMBDA_COMPETITION,
        DEFAULT_LAMBDA,
        DEFAULT_CELL_TOTAL_CAPACITY,
        DEFAULT_CELL_CATEGORY_CAPACITIES,
    )

__all__ = [
    "FleetManager",
    "Vendor",
    "Reservation",
    "VendorStatus",
    "ReservationStatus",
    "calculate_effective_demand",
    "CATEGORY_LAMBDA_COMPETITION",
    "DEFAULT_LAMBDA",
    "DEFAULT_CELL_TOTAL_CAPACITY",
    "DEFAULT_CELL_CATEGORY_CAPACITIES",
]
