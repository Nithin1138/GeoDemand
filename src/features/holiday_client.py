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
    def __init__(self, country: str = "IN", years: list[int] | None = None, subdiv: str | None = None):
        """
        country: ISO country code, e.g. "IN" for India.
        subdiv: optional state/region subdivision if the library supports it
                for finer-grained regional holidays (e.g. state-specific festivals).
        """
        self.country = country
        self.calendar = holidays_lib.country_holidays(country, years=years, subdiv=subdiv)

    def get_holiday_flags(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Returns a DataFrame: date, is_holiday, holiday_name, source."""
        rows = []
        for d in dates:
            d_date = d.date() if hasattr(d, "date") else d
            name = self.calendar.get(d_date)
            rows.append({
                "date": d,
                "is_holiday": name is not None,
                "holiday_name": name or "",
                "source": "real:holidays-lib",
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    client = HolidayClient(country="IN", years=[2025])
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    df = client.get_holiday_flags(dates)
    print(f"Real holidays found in 2025: {df['is_holiday'].sum()}")
    print(df[df["is_holiday"]][["date", "holiday_name"]].to_string(index=False))
