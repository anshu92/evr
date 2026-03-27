"""Trading-day calendar: weekdays minus market holidays (NYSE / TSX).

Self-contained — no external dependency required.
"""
from __future__ import annotations

from datetime import date as _date, datetime, timedelta

_HOLIDAY_CACHE: dict[tuple[str, int], set[_date]] = {}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _date:
    """Return the Nth occurrence of a weekday in a month (1-indexed)."""
    first = _date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> _date:
    """Return the last occurrence of a weekday in a month."""
    if month == 12:
        last_day = _date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = _date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed(d: _date) -> _date:
    """Shift a fixed holiday to the observed weekday (Fri if Sat, Mon if Sun)."""
    if d.weekday() == 5:  # Saturday → Friday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday → Monday
        return d + timedelta(days=1)
    return d


def _easter(year: int) -> _date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return _date(year, month, day + 1)


def nyse_holidays(year: int) -> set[_date]:
    """Compute NYSE market holidays for a given year."""
    key = ("US", year)
    if key in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[key]

    holidays = set()

    # New Year's Day (Jan 1)
    holidays.add(_observed(_date(year, 1, 1)))

    # Martin Luther King Jr. Day (3rd Monday in January)
    holidays.add(_nth_weekday(year, 1, 0, 3))  # 0 = Monday

    # Presidents' Day (3rd Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday (2 days before Easter Sunday)
    holidays.add(_easter(year) - timedelta(days=2))

    # Memorial Day (last Monday in May)
    holidays.add(_last_weekday(year, 5, 0))

    # Juneteenth (June 19) — observed since 2022
    if year >= 2022:
        holidays.add(_observed(_date(year, 6, 19)))

    # Independence Day (July 4)
    holidays.add(_observed(_date(year, 7, 4)))

    # Labor Day (1st Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving (4th Thursday in November)
    holidays.add(_nth_weekday(year, 11, 3, 4))  # 3 = Thursday

    # Christmas (December 25)
    holidays.add(_observed(_date(year, 12, 25)))

    _HOLIDAY_CACHE[key] = holidays
    return holidays


def _canada_observed(d: _date) -> _date:
    """Canada-style observed holiday (Sat/Sun -> Monday)."""
    if d.weekday() == 5:  # Saturday -> Monday
        return d + timedelta(days=2)
    if d.weekday() == 6:  # Sunday -> Monday
        return d + timedelta(days=1)
    return d


def tsx_holidays(year: int) -> set[_date]:
    """Compute core TSX holidays for a given year."""
    key = ("CA", year)
    if key in _HOLIDAY_CACHE:
        return _HOLIDAY_CACHE[key]

    holidays: set[_date] = set()

    # New Year's Day
    holidays.add(_canada_observed(_date(year, 1, 1)))

    # Family Day (3rd Monday in February)
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday
    holidays.add(_easter(year) - timedelta(days=2))

    # Victoria Day (Monday preceding May 25)
    victoria = _date(year, 5, 24)
    while victoria.weekday() != 0:
        victoria -= timedelta(days=1)
    holidays.add(victoria)

    # Canada Day
    holidays.add(_canada_observed(_date(year, 7, 1)))

    # Civic Holiday (1st Monday in August)
    holidays.add(_nth_weekday(year, 8, 0, 1))

    # Labour Day (1st Monday in September)
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving (2nd Monday in October)
    holidays.add(_nth_weekday(year, 10, 0, 2))

    # Christmas + Boxing Day (ensure distinct observed dates)
    christmas_obs = _canada_observed(_date(year, 12, 25))
    holidays.add(christmas_obs)
    boxing_obs = _canada_observed(_date(year, 12, 26))
    while boxing_obs in holidays or boxing_obs.weekday() >= 5:
        boxing_obs += timedelta(days=1)
    holidays.add(boxing_obs)

    _HOLIDAY_CACHE[key] = holidays
    return holidays


def market_for_ticker(ticker: str | None) -> str:
    t = str(ticker or "").strip().upper()
    if t.endswith(".TO") or t.endswith(".V"):
        return "CA"
    return "US"


def is_trading_day(d, market: str = "US") -> bool:
    """Check if a date is a trading day for the requested market."""
    if hasattr(d, "date"):
        d = d.date()
    if d.weekday() >= 5:
        return False
    m = str(market or "US").upper()
    holidays = tsx_holidays(d.year) if m == "CA" else nyse_holidays(d.year)
    return d not in holidays


def trading_days_between(start: datetime, end: datetime, market: str = "US") -> int:
    """Count trading days between two dates, excluding start, including end."""
    if end <= start:
        return 0
    count = 0
    d = start.date() + timedelta(days=1)
    end_d = end.date()
    while d <= end_d:
        if is_trading_day(d, market=market):
            count += 1
        d += timedelta(days=1)
    return count


def add_trading_days(start: datetime, trading_days: int, market: str = "US") -> datetime:
    """Return the datetime that is N trading days after start."""
    if trading_days <= 0:
        return start
    d = start
    added = 0
    while added < trading_days:
        d += timedelta(days=1)
        if is_trading_day(d, market=market):
            added += 1
    return d
