"""
Fuel price provider.

Fuel is NOT real-time in the sense weather is — it changes daily at best.
Treat as real_periodic: configured value with effective_date.

Set env vars:
  FUEL_PRICE_INR_PER_LITRE  (e.g. "95.0")
  FUEL_EFFICIENCY_KM_PER_L  (e.g. "18.0" for a bike/van)

Derived: cost_per_km = price_per_litre / efficiency_km_per_l
"""

from __future__ import annotations

import os
from datetime import datetime, timezone


DEFAULT_FUEL_PRICE_INR_PER_LITRE = 95.0  # ₹ — approximate Vijayawada petrol price
DEFAULT_FUEL_EFFICIENCY_KM_PER_L = 18.0  # km/litre — typical auto/van


class FuelPriceProvider:
    """
    Returns fuel cost parameters from environment config or hardcoded defaults.
    source_type = real_periodic (updated by operator, not scraped live).
    """

    def __init__(self):
        self.price_per_litre = float(os.environ.get("FUEL_PRICE_INR_PER_LITRE", DEFAULT_FUEL_PRICE_INR_PER_LITRE))
        self.efficiency = float(os.environ.get("FUEL_EFFICIENCY_KM_PER_L", DEFAULT_FUEL_EFFICIENCY_KM_PER_L))
        self.cost_per_km = round(self.price_per_litre / self.efficiency, 3)

    def get_fuel_cost_for_distance(self, distance_km: float) -> dict:
        """Returns fuel cost (₹) for a given travel distance."""
        cost = round(distance_km * self.cost_per_km, 2)
        return {
            "fuel_cost_inr": cost,
            "distance_km": round(distance_km, 3),
            "cost_per_km_inr": self.cost_per_km,
            "fuel_price_per_litre_inr": self.price_per_litre,
            "fuel_efficiency_km_per_l": self.efficiency,
            "fuel_type": "petrol",
            "currency": "INR",
            "source_type": "real_periodic",
            "source_name": "configured_value",
            "effective_date": datetime.now(timezone.utc).date().isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "note": "Fuel price is a configured value (real_periodic). "
                    "Set FUEL_PRICE_INR_PER_LITRE env var to update.",
        }
