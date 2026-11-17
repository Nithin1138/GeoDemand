"""
GeoDemand AI — Dynamic Time Window Engine

Determines location- and category-specific commercial trading time windows:
- College / University zones: Morning rush (08:30–10:30), Lunch rush (12:30–14:30), Evening dispersal (16:00–18:30)
- Office / Tech parks: Breakfast (08:00–10:30), Lunch peak (12:30–14:30), Evening tea/snacks (16:30–19:30)
- Transit hubs (Railway/Bus): Early morning (06:00–10:00), Evening peak (17:00–21:30)
- Commercial markets / Bazaars: Late morning (10:30–13:30), Evening peak (17:00–21:30)
- Parks / Recreation: Morning walkers (06:00–08:30), Evening families (16:30–20:00)
- Hospital zones: Morning (07:30–09:30), Lunch (12:00–14:00), Dinner (19:00–21:00)

Vendor category alignments:
- food: Lunch (11:30–14:30) & Dinner (19:00–22:30)
- fruits: Morning (07:00–11:00) & Evening (16:00–20:00)
- beverages: Morning rush (07:00–11:00) & Afternoon tea (15:00–19:30)
- snacks: Evening street chaat (16:00–22:00)
- repair: Business hours (09:30–18:30)
- salon: All-day groom (09:00–20:00)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeWindow:
    """A viable commercial trading interval (in minutes from shift start or local clock time)."""
    start_hour: float  # e.g. 8.5 = 08:30
    end_hour: float    # e.g. 14.5 = 14:30
    peak_hour: float   # e.g. 13.0 = 13:00
    demand_multiplier: float  # 1.0 = base, 1.4 = peak
    window_name: str   # e.g. "Lunch Rush"

    @property
    def duration_hours(self) -> float:
        return max(0.0, self.end_hour - self.start_hour)


# Category default baseline windows (hours in 24-hr local time)
CATEGORY_WINDOWS: Dict[str, List[TimeWindow]] = {
    "food": [
        TimeWindow(start_hour=7.5, end_hour=10.5, peak_hour=9.0, demand_multiplier=1.2, window_name="Breakfast Rush"),
        TimeWindow(start_hour=11.5, end_hour=15.0, peak_hour=13.0, demand_multiplier=1.5, window_name="Lunch Peak"),
        TimeWindow(start_hour=18.5, end_hour=22.5, peak_hour=20.5, demand_multiplier=1.6, window_name="Dinner Feast"),
    ],
    "fruits": [
        TimeWindow(start_hour=6.5, end_hour=11.5, peak_hour=8.5, demand_multiplier=1.4, window_name="Morning Market"),
        TimeWindow(start_hour=15.5, end_hour=20.5, peak_hour=18.0, demand_multiplier=1.3, window_name="Evening Groceries"),
    ],
    "beverages": [
        TimeWindow(start_hour=6.5, end_hour=11.0, peak_hour=8.5, demand_multiplier=1.5, window_name="Morning Chai/Coffee"),
        TimeWindow(start_hour=11.5, end_hour=14.5, peak_hour=13.0, demand_multiplier=1.1, window_name="Lunch Coolers"),
        TimeWindow(start_hour=15.5, end_hour=20.0, peak_hour=17.5, demand_multiplier=1.6, window_name="Evening Tea Break"),
    ],
    "snacks": [
        TimeWindow(start_hour=11.0, end_hour=14.0, peak_hour=12.5, demand_multiplier=1.1, window_name="Mid-day Bites"),
        TimeWindow(start_hour=15.5, end_hour=22.0, peak_hour=18.5, demand_multiplier=1.7, window_name="Evening Chaat & Street Food"),
    ],
    "repair": [
        TimeWindow(start_hour=9.0, end_hour=18.5, peak_hour=12.0, demand_multiplier=1.2, window_name="Standard Commercial Day"),
    ],
    "salon": [
        TimeWindow(start_hour=8.5, end_hour=20.5, peak_hour=17.0, demand_multiplier=1.3, window_name="Grooming Operating Hours"),
    ],
}


# Landmark / POI-specific window adjustments
POI_WINDOW_MODIFIERS: Dict[str, List[TimeWindow]] = {
    "college": [
        TimeWindow(start_hour=8.0, end_hour=10.5, peak_hour=9.0, demand_multiplier=1.4, window_name="Campus Morning Arrival"),
        TimeWindow(start_hour=12.0, end_hour=14.5, peak_hour=13.0, demand_multiplier=1.7, window_name="Campus Lunch Break"),
        TimeWindow(start_hour=15.5, end_hour=18.5, peak_hour=17.0, demand_multiplier=1.5, window_name="Campus Evening Dispersal"),
    ],
    "office": [
        TimeWindow(start_hour=8.0, end_hour=10.5, peak_hour=9.0, demand_multiplier=1.3, window_name="Office Arrival & Breakfast"),
        TimeWindow(start_hour=12.3, end_hour=14.5, peak_hour=13.2, demand_multiplier=1.8, window_name="Corporate Lunch Peak"),
        TimeWindow(start_hour=16.5, end_hour=19.5, peak_hour=18.0, demand_multiplier=1.6, window_name="Evening Commute & Snacks"),
    ],
    "transit": [
        TimeWindow(start_hour=6.0, end_hour=11.0, peak_hour=8.0, demand_multiplier=1.5, window_name="Morning Transit Rush"),
        TimeWindow(start_hour=11.5, end_hour=16.0, peak_hour=13.5, demand_multiplier=1.1, window_name="Mid-day Travelers"),
        TimeWindow(start_hour=16.5, end_hour=22.0, peak_hour=19.0, demand_multiplier=1.6, window_name="Evening Transit Rush"),
    ],
    "hospital": [
        TimeWindow(start_hour=7.0, end_hour=10.0, peak_hour=8.3, demand_multiplier=1.3, window_name="Hospital Morning Shift"),
        TimeWindow(start_hour=11.5, end_hour=14.5, peak_hour=12.8, demand_multiplier=1.4, window_name="Hospital Lunch"),
        TimeWindow(start_hour=17.5, end_hour=21.0, peak_hour=19.0, demand_multiplier=1.3, window_name="Hospital Evening Visitors"),
    ],
    "park": [
        TimeWindow(start_hour=5.5, end_hour=9.0, peak_hour=7.0, demand_multiplier=1.6, window_name="Morning Walkers & Fitness"),
        TimeWindow(start_hour=16.0, end_hour=20.5, peak_hour=18.0, demand_multiplier=1.5, window_name="Evening Park Visitors"),
    ],
}


def resolve_candidate_time_windows(
    candidate: Dict[str, Any],
    vendor_category: str,
    shift_start: datetime,
    shift_end: datetime,
) -> List[Dict[str, Any]]:
    """
    Resolve active commercial time windows for a candidate zone clipped to the vendor's shift duration.
    
    Returns:
        List of dicts with:
          - window_name: str
          - window_start: datetime
          - window_end: datetime
          - peak_time: datetime
          - demand_multiplier: float
          - duration_minutes: int
    """
    cat = (vendor_category or "food").lower().strip()
    base_windows = CATEGORY_WINDOWS.get(cat, CATEGORY_WINDOWS["food"])

    # Check POI type
    landmark_type = (candidate.get("landmark_type") or "").lower()
    is_college = candidate.get("is_college", False)

    poi_windows = None
    if is_college or "college" in landmark_type or "university" in landmark_type:
        poi_windows = POI_WINDOW_MODIFIERS["college"]
    elif "office" in landmark_type or "tech" in landmark_type or "commercial" in landmark_type:
        poi_windows = POI_WINDOW_MODIFIERS["office"]
    elif "railway" in landmark_type or "bus" in landmark_type or "transit" in landmark_type:
        poi_windows = POI_WINDOW_MODIFIERS["transit"]
    elif "hospital" in landmark_type or "clinic" in landmark_type:
        poi_windows = POI_WINDOW_MODIFIERS["hospital"]
    elif "park" in landmark_type or "garden" in landmark_type:
        poi_windows = POI_WINDOW_MODIFIERS["park"]

    active_templates = poi_windows if poi_windows else base_windows
    shift_date = shift_start.date()
    resolved = []

    for tmpl in active_templates:
        start_min = int(tmpl.start_hour * 60)
        end_min = int(tmpl.end_hour * 60)
        peak_min = int(tmpl.peak_hour * 60)

        w_start = datetime.combine(shift_date, time(0, 0), tzinfo=shift_start.tzinfo) + timedelta(minutes=start_min)
        w_end = datetime.combine(shift_date, time(0, 0), tzinfo=shift_start.tzinfo) + timedelta(minutes=end_min)
        w_peak = datetime.combine(shift_date, time(0, 0), tzinfo=shift_start.tzinfo) + timedelta(minutes=peak_min)

        # Clip to shift bounds
        clipped_start = max(shift_start, w_start)
        clipped_end = min(shift_end, w_end)

        if clipped_end > clipped_start:
            dur_min = int((clipped_end - clipped_start).total_seconds() / 60)
            # Only include windows that allow at least 30 min of trading
            if dur_min >= 30:
                resolved.append({
                    "window_name": tmpl.window_name,
                    "window_start": clipped_start,
                    "window_end": clipped_end,
                    "peak_time": w_peak if clipped_start <= w_peak <= clipped_end else clipped_start + timedelta(minutes=dur_min // 2),
                    "demand_multiplier": tmpl.demand_multiplier,
                    "duration_minutes": dur_min,
                })

    # If no window overlaps (e.g. night shift), provide a default continuous window over the shift
    if not resolved:
        dur = int((shift_end - shift_start).total_seconds() / 60)
        resolved.append({
            "window_name": f"{cat.capitalize()} Standard Operating Window",
            "window_start": shift_start,
            "window_end": shift_end,
            "peak_time": shift_start + timedelta(minutes=dur // 2),
            "demand_multiplier": 1.0,
            "duration_minutes": dur,
        })

    return resolved
