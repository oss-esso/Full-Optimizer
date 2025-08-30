"""
Time utility functions for the EPDT heuristic solver.

This module contains utility functions for formatting and handling time-related calculations.
"""

def format_duration_detailed(minutes: float) -> str:
    """
    Format duration in minutes to a human-readable string with days, hours, and minutes.
    
    Args:
        minutes: Duration in minutes
    
    Returns:
        Formatted string like "1d 5h 30m", "2h 15m", or "45m"
    """
    if minutes < 0:
        return "0m"
    
    total_minutes = int(minutes)
    days = total_minutes // 1440
    remaining_minutes = total_minutes % 1440
    hours = remaining_minutes // 60
    mins = remaining_minutes % 60
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0 or len(parts) == 0:
        parts.append(f"{mins}m")
    
    return " ".join(parts)


def format_time_hhmm(minutes: float) -> str:
    """Formats minutes into a hh:mm string."""
    if minutes < 0:
        return "00:00"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def format_date_from_minutes(minutes: float, start_date=None) -> str:
    """
    Format minutes since start into a date string.
    
    Args:
        minutes: Minutes since the start of operations
        start_date: Starting date (optional)
    
    Returns:
        Formatted date string in dd/MM format
    """
    import datetime
    # Use provided start date or default to current date (August 26, 2025)
    if start_date is None:
        start_date = datetime.date(2025, 8, 26)
    days_offset = int(minutes // 1440)  # 1440 minutes per day
    target_date = start_date + datetime.timedelta(days=days_offset)
    return target_date.strftime("%d/%m")
