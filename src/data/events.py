"""
Dataset 5 — Events (events.parquet)

Single responsibility: sparse local events (festival stalls, weddings,
college fests, marathons) that create localized demand spikes independent
of the regular weather/holiday/POI signal.

Real event listing APIs (Eventbrite, Meetup, etc.) generally require paid
tiers or heavy scraping for India-specific hyperlocal coverage, so this
is v1-simulated, sparsely, tagged accordingly — flagged honestly rather
than pretending it's a live feed. A real integration point is documented
below for when/if a usable free event source is found.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_TYPES = ["college_fest", "wedding", "religious_gathering", "marathon", "local_market", "concert"]
EVENT_IMPORTANCE_BY_TYPE = {
    "college_fest": 0.7, "wedding": 0.4, "religious_gathering": 0.6,
    "marathon": 0.5, "local_market": 0.3, "concert": 0.8,
}


def build_events(
    h3_cell_ids: list[str], start_date: str, end_date: str,
    density_pct: float = 0.02, seed: int = 2026,
) -> pd.DataFrame:
    """
    density_pct: fraction of (cell x day) combinations that get an event.
    Kept low and sparse — real event density in a city is low; a naive
    high-density synthetic events table would swamp the weather/POI signal
    and make the model over-rely on a fabricated feature.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n_combinations = len(h3_cell_ids) * len(dates)
    n_events = max(1, int(n_combinations * density_pct))

    event_cells = rng.choice(h3_cell_ids, size=n_events)
    event_dates = rng.choice(dates, size=n_events)
    event_types = rng.choice(EVENT_TYPES, size=n_events)
    start_hours = rng.integers(9, 20, size=n_events)
    durations = rng.integers(2, 8, size=n_events)
    attendance = np.array([
        int(rng.gamma(shape=2.0, scale=EVENT_IMPORTANCE_BY_TYPE[t] * 400))
        for t in event_types
    ])

    return pd.DataFrame({
        "event_id": [str(uuid.uuid4())[:8] for _ in range(n_events)],
        "h3_cell_id": event_cells,
        "timestamp": [pd.Timestamp(d) + pd.Timedelta(hours=int(h)) for d, h in zip(event_dates, start_hours)],
        "event_type": event_types,
        "expected_attendance": attendance,
        "event_importance": [EVENT_IMPORTANCE_BY_TYPE[t] for t in event_types],
        "duration_hours": durations,
        "source": "simulated:sparse",
    })


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from h3_cells import build_h3_cells

    cells = build_h3_cells()
    df = build_events(cells["h3_cell_id"].tolist(), "2025-01-01", "2025-12-31")
    out = Path(__file__).parent.parent.parent / "data" / "processed" / "events.parquet"
    df.to_parquet(out, index=False)
    print(f"events.parquet: {len(df):,} events -> {out}")
    print(df.head())
