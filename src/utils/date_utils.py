"""
Date and time utilities for Market Decoder
"""

from datetime import datetime, timedelta, date
from typing import Optional, List, Tuple
import pytz
import holidays

def get_ist_now() -> datetime:
    """
    Get current datetime in Indian Standard Time
    
    Returns:
        Current datetime in IST
    """
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def parse_nse_date(date_str: str) -> Optional[datetime]:
    """
    Parse NSE date string to datetime
    
    Args:
        date_str: Date string in NSE format (e.g., '25-Jan-2024')
        
    Returns:
        Datetime object or None if parsing fails
    """
    try:
        # Try different NSE date formats
        formats = [
            '%d-%b-%Y',      # 25-Jan-2024
            '%d-%B-%Y',      # 25-January-2024
            '%Y-%m-%d',      # 2024-01-25
            '%d/%m/%Y',      # 25/01/2024
            '%d-%m-%Y',      # 25-01-2024
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If none of the formats work, return None
        return None
        
    except Exception:
        return None

def format_date_for_nse(dt: datetime) -> str:
    """
    Format datetime for NSE API
    
    Args:
        dt: Datetime object
        
    Returns:
        Date string in NSE format
    """
    return dt.strftime('%d-%b-%Y').upper()

def is_market_open() -> bool:
    """
    Check if Indian market is currently open
    
    Returns:
        True if market is open, False otherwise
    """
    now = get_ist_now()
    
    # Check if weekday (Monday=0, Friday=4)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check market hours (9:15 AM to 3:30 PM IST)
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def is_pre_market() -> bool:
    """
    Check if it's pre-market hours
    
    Returns:
        True if pre-market hours, False otherwise
    """
    now = get_ist_now()
    
    # Pre-market: 9:00 AM to 9:15 AM IST
    pre_market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    pre_market_end = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    return pre_market_start <= now < pre_market_end

def is_post_market() -> bool:
    """
    Check if it's post-market hours
    
    Returns:
        True if post-market hours, False otherwise
    """
    now = get_ist_now()
    
    # Post-market: 3:30 PM to 4:00 PM IST
    post_market_start = now.replace(hour=15, minute=30, second=0, microsecond=0)
    post_market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return post_market_start <= now < post_market_end

def get_next_market_open() -> datetime:
    """
    Get next market opening time
    
    Returns:
        Next market opening datetime
    """
    now = get_ist_now()
    
    # If market is open today and hasn't closed yet
    if is_market_open():
        return now
    
    # Find next trading day
    next_day = now
    while True:
        next_day += timedelta(days=1)
        
        # Check if it's a trading day (not weekend and not holiday)
        if is_trading_day(next_day):
            # Set to market open time
            market_open = next_day.replace(hour=9, minute=15, second=0, microsecond=0)
            return market_open

def is_trading_day(dt: datetime) -> bool:
    """
    Check if a given date is a trading day
    
    Args:
        dt: Date to check
        
    Returns:
        True if trading day, False otherwise
    """
    # Check if weekend
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if holiday (using India holidays)
    india_holidays = holidays.India()
    if dt.date() in india_holidays:
        return False
    
    return True

def get_trading_days_between(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    Get all trading days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of trading days
    """
    trading_days = []
    current_date = start_date
    
    while current_date <= end_date:
        if is_trading_day(current_date):
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    return trading_days

def days_to_expiry(expiry_date: datetime, current_date: Optional[datetime] = None) -> int:
    """
    Calculate trading days to expiry
    
    Args:
        expiry_date: Expiry date
        current_date: Current date (defaults to now)
        
    Returns:
        Number of trading days to expiry
    """
    if current_date is None:
        current_date = get_ist_now()
    
    # Get trading days between current date and expiry date
    trading_days = get_trading_days_between(current_date, expiry_date)
    
    # Exclude current day if market is closed or after market hours
    if not is_market_open():
        trading_days = [day for day in trading_days if day.date() > current_date.date()]
    
    return len(trading_days)

def trading_days_to_expiry(current_date: datetime, expiry_date: datetime) -> int:
    """
    Calculate trading days to expiry (alias for days_to_expiry)
    
    Args:
        current_date: Current date
        expiry_date: Expiry date
        
    Returns:
        Number of trading days to expiry
    """
    return days_to_expiry(expiry_date, current_date)

def get_current_expiry() -> datetime:
    """
    Get current month's expiry date (last Thursday of the month)
    
    Returns:
        Current month's expiry datetime
    """
    now = get_ist_now()
    
    # Get last Thursday of current month
    return get_last_thursday_of_month(now.year, now.month)

def get_next_expiry() -> datetime:
    """
    Get next month's expiry date
    
    Returns:
        Next month's expiry datetime
    """
    now = get_ist_now()
    
    # If current month's expiry has passed, get next month's
    current_expiry = get_current_expiry()
    if now > current_expiry:
        # Move to next month
        if now.month == 12:
            next_year = now.year + 1
            next_month = 1
        else:
            next_year = now.year
            next_month = now.month + 1
        
        return get_last_thursday_of_month(next_year, next_month)
    else:
        return current_expiry

def get_last_thursday_of_month(year: int, month: int) -> datetime:
    """
    Get last Thursday of a given month
    
    Args:
        year: Year
        month: Month
        
    Returns:
        Last Thursday datetime
    """
    # Get last day of month
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)
    
    # Find last Thursday
    while last_day.weekday() != 3:  # Thursday is 3
        last_day -= timedelta(days=1)
    
    # Set to market close time (3:30 PM)
    return last_day.replace(hour=15, minute=30, second=0, microsecond=0)

def get_expiry_series(count: int = 3) -> List[datetime]:
    """
    Get series of upcoming expiry dates
    
    Args:
        count: Number of expiry dates to return
        
    Returns:
        List of expiry dates
    """
    expiries = []
    now = get_ist_now()
    
    # Start from current month
    year = now.year
    month = now.month
    
    for _ in range(count):
        expiry = get_last_thursday_of_month(year, month)
        
        # If expiry has passed, skip to next
        if expiry < now:
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            expiry = get_last_thursday_of_month(year, month)
        
        expiries.append(expiry)
        
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    
    return expiries

def time_to_expiry_years(expiry_date: datetime, current_date: Optional[datetime] = None) -> float:
    """
    Calculate time to expiry in years
    
    Args:
        expiry_date: Expiry date
        current_date: Current date (defaults to now)
        
    Returns:
        Time to expiry in years
    """
    if current_date is None:
        current_date = get_ist_now()
    
    # Calculate trading days to expiry
    trading_days = days_to_expiry(expiry_date, current_date)
    
    # Convert to years (252 trading days in a year)
    return trading_days / 252.0

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human readable duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def get_market_holidays(year: int) -> List[date]:
    """
    Get market holidays for a given year
    
    Args:
        year: Year
        
    Returns:
        List of holiday dates
    """
    india_holidays = holidays.India(years=year)
    return list(india_holidays.keys())

def is_holiday(dt: datetime) -> bool:
    """
    Check if a date is a market holiday
    
    Args:
        dt: Date to check
        
    Returns:
        True if holiday, False otherwise
    """
    india_holidays = holidays.India()
    return dt.date() in india_holidays

def get_previous_trading_day(current_date: Optional[datetime] = None) -> datetime:
    """
    Get previous trading day
    
    Args:
        current_date: Current date (defaults to now)
        
    Returns:
        Previous trading day datetime
    """
    if current_date is None:
        current_date = get_ist_now()
    
    prev_day = current_date - timedelta(days=1)
    
    while not is_trading_day(prev_day):
        prev_day -= timedelta(days=1)
    
    return prev_day

def get_next_trading_day(current_date: Optional[datetime] = None) -> datetime:
    """
    Get next trading day
    
    Args:
        current_date: Current date (defaults to now)
        
    Returns:
        Next trading day datetime
    """
    if current_date is None:
        current_date = get_ist_now()
    
    next_day = current_date + timedelta(days=1)
    
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    
    return next_day

def is_intraday() -> bool:
    """
    Check if current time is during intraday trading hours
    
    Returns:
        True if intraday hours, False otherwise
    """
    return is_market_open()

def is_eod_time() -> bool:
    """
    Check if it's end of day (last 30 minutes of trading)
    
    Returns:
        True if last 30 minutes, False otherwise
    """
    now = get_ist_now()
    
    if not is_market_open():
        return False
    
    # Last 30 minutes: 3:00 PM to 3:30 PM
    eod_start = now.replace(hour=15, minute=0, second=0, microsecond=0)
    eod_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return eod_start <= now <= eod_end

def get_time_until_market_close() -> timedelta:
    """
    Get time until market closes
    
    Returns:
        Time until market close
    """
    now = get_ist_now()
    
    if not is_market_open():
        return timedelta(0)
    
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_close - now

def get_time_since_market_open() -> timedelta:
    """
    Get time since market opened
    
    Returns:
        Time since market open
    """
    now = get_ist_now()
    
    if not is_market_open():
        return timedelta(0)
    
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    
    return now - market_open