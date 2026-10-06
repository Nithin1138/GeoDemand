"""
Real holiday/festival calendar — the `holidays` Python library.

NO API KEY, NO NETWORK REQUIRED. `holidays` ships accurate official holiday
tables for 100+ countries, computed offline (handles lunar/solar calendar
festivals correctly, e.g. Diwali, Eid, which shift dates every year — a
hardcoded {(month, day)} set, like the placeholder in market_simulator.py's
first draft, gets these wrong).

This is real government-published holiday data, not a simulation — it
just happens not to need a network call to retrieve.

USAGE:
    client = HolidayClient(country="IN", years=[2025])
    holiday_series = client.get_holiday_flags(dates)
"""

from __future__ import annotations

import pandas as pd

try:
    import holidays as holidays_lib
except ImportError as e:
    raise ImportError(
        "holidays package required. Install with: pip install holidays --break-system-packages"
    ) from e


class HolidayClient:
    def __init__(self, country: str = "IN", years: list[int] | None = None, subdiv: str | None = "AP"):
        """
        country: ISO country code, e.g. "IN" for India.
        subdiv: State subdivision (defaults to "AP" for Andhra Pradesh).
        """
        self.country = country
        self.subdiv = subdiv
        self.calendar = holidays_lib.country_holidays(country, years=years, subdiv=subdiv)

    def is_holiday(self, date_val) -> tuple[bool, str]:
        """Check if a single date is an official holiday."""
        d = pd.to_datetime(date_val).date()
        name = self.calendar.get(d)
        return (name is not None, name or "")

    def is_weekend(self, date_val) -> bool:
        """Check if a single date is a weekend (Saturday or Sunday)."""
        dt = pd.to_datetime(date_val)
        return bool(dt.dayofweek >= 5)

    def get_holiday_flags(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Returns a DataFrame: date, is_holiday, holiday_name, is_weekend, source, source_type."""
        rows = []
        for d in dates:
            d_date = d.date() if hasattr(d, "date") else pd.to_datetime(d).date()
            name = self.calendar.get(d_date)
            dt = pd.to_datetime(d)
            rows.append({
                "date": d,
                "is_holiday": name is not None,
                "holiday_name": name or "",
                "is_weekend": bool(dt.dayofweek >= 5),
                "source": "real:holidays-lib",
                "source_type": "real_offline",
                "source_name": f"holidays-{self.country.lower()}-{self.subdiv.lower() if self.subdiv else 'all'}",
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    client = HolidayClient(country="IN", years=[2025])
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    df = client.get_holiday_flags(dates)
    print(f"Real holidays found in 2025: {df['is_holiday'].sum()}")
    print(df[df["is_holiday"]][["date", "holiday_name"]].to_string(index=False))
