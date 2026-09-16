"""
Dataset 4 — Calendar (calendar.parquet)

Single responsibility: time/holiday/season context per hour.
Real holidays via src/features/holiday_client.py (offline `holidays` lib —
correctly dated, no hardcoded lunar-calendar mistakes). School vacation
is a documented approximation (India academic-calendar summer break +
winter break windows) since there's no single authoritative free API for
region-specific school calendars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "features"))
from holiday_client import HolidayClient  # noqa: E402


def _season(month: int) -> str:
    # rough India seasonal bucket
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def _is_school_vacation(date: pd.Timestamp) -> bool:
    # Approximation: mid-May to mid-June (summer break), late Dec (winter break)
    m, d = date.month, date.day
    if m == 5 and d >= 15:
        return True
    if m == 6 and d <= 15:
        return True
    if m == 12 and d >= 20:
        return True
    return False


def build_calendar(start_date: str, end_date: str, country: str = "IN") -> pd.DataFrame:
    dates = pd.date_range(start_date, end_date, freq="D")
    years = sorted(dates.year.unique().tolist())
    holiday_client = HolidayClient(country=country, years=years)
    holiday_df = holiday_client.get_holiday_flags(dates).set_index("date")

    timestamps = pd.date_range(start_date, pd.Timestamp(end_date) + pd.Timedelta(hours=23), freq="h")
    rows = []
    for ts in timestamps:
        d = pd.Timestamp(ts.date())
        h_row = holiday_df.loc[d] if d in holiday_df.index else None
        is_holiday = bool(h_row["is_holiday"]) if h_row is not None else False
        holiday_name = h_row["holiday_name"] if h_row is not None else ""

        rows.append({
            "timestamp": ts,
            "hour": ts.hour,
            "weekday": ts.day_name(),
            "month": ts.month,
            "season": _season(ts.month),
            "quarter": ts.quarter,
            "is_weekend": ts.weekday() >= 5,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "festival_name": holiday_name,  # `holidays` lib doesn't distinguish civic vs. festival; same source
            "school_vacation": _is_school_vacation(ts),
            "source": "real:holidays_lib",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_calendar("2025-01-01", "2025-12-31")
    out = Path(__file__).parent.parent.parent / "data" / "processed" / "calendar.parquet"
    df.to_parquet(out, index=False)
    print(f"calendar.parquet: {len(df):,} hourly rows -> {out}")
    print(f"Holidays flagged: {df[df['is_holiday']]['timestamp'].dt.date.nunique()} days")
    print(df.head())
