"""
Modular Hours of Service (HoS) Simulation Module

This module addresses the feedback about making HoS simulation more modular
and resolving circular import issues. It provides a clean separation of
HoS logic from the main routing algorithms.

Key improvements:
1. Modular design with separate functions for breaks, daily rests, and weekly rests
2. No circular imports - standalone module
3. Enhanced event tracking integration
4. Improved readability and maintainability
5. Better testing capabilities
"""

from typing import List, Tuple, Optional, Dict, Any
import copy
from dataclasses import dataclass

# Import only the data structures we need to avoid circular imports
try:
    from epdt_data_structures import DriverState, HoSEvent
except ImportError:
    # Fallback for testing or standalone use
    print("Warning: Could not import data structures. Using local definitions.")
    from typing import List as HoSEventList
    
    @dataclass
    class DriverState:
        # Minimal definition for standalone use
        drive_since_break: float = 0.0
        work_since_break: float = 0.0
        drive_today: float = 0.0
        work_today: float = 0.0
        drive_this_week: float = 0.0
        drive_last_week: float = 0.0


@dataclass
class HoSSimulationResult:
    """
    Result of HoS simulation containing feasibility and cost information.
    """
    is_feasible: bool
    total_duration: float  # Total time including breaks and rests
    driving_time: float    # Actual driving time
    working_time: float    # Total working time
    break_time: float      # Time spent on breaks
    rest_time: float       # Time spent on daily/weekly rests
    violations: List[str]  # List of regulation violations
    events: List[Dict[str, Any]]  # Detailed event log


class HoSRegulations:
    """
    European Hours of Service regulations constants and validation rules.
    
    This centralizes all regulation constants for easy maintenance and updates.
    """
    
    # Time limits in minutes
    MAX_DRIVE_WITHOUT_BREAK = 4.5 * 60      # 4.5 hours
    MAX_WORK_WITHOUT_BREAK = 6 * 60         # 6 hours  
    MAX_DRIVE_PER_DAY = 9 * 60              # 9 hours (extendable to 10)
    MAX_WORK_PER_DAY = 13 * 60              # 13 hours (extendable to 14)
    MAX_DRIVE_PER_WEEK = 56 * 60            # 56 hours
    MAX_DRIVE_TWO_WEEKS = 90 * 60           # 90 hours in any two consecutive weeks
    
    # Rest requirements in minutes
    MIN_BREAK_DURATION = 45                 # 45 minutes
    MIN_PARTIAL_BREAK = 15                  # 15 minutes (for split breaks)
    MIN_DAILY_REST = 11 * 60                # 11 hours (reducible to 9)
    MIN_REDUCED_DAILY_REST = 9 * 60         # 9 hours (reduced rest)
    MIN_WEEKLY_REST = 45 * 60               # 45 hours
    
    # Extension limits
    MAX_DRIVING_EXTENSIONS_PER_WEEK = 2     # Can extend to 10h twice per week
    MAX_WORK_EXTENSIONS_PER_WEEK = 2        # Can extend to 14h twice per week
    MAX_REDUCED_RESTS_PER_WEEK = 3          # Max 3 reduced daily rests per week
    
    @classmethod
    def get_max_daily_drive(cls, extensions_used: int) -> float:
        """Get maximum daily driving time considering extensions used."""
        if extensions_used < cls.MAX_DRIVING_EXTENSIONS_PER_WEEK:
            return 10 * 60  # Can extend to 10 hours
        return cls.MAX_DRIVE_PER_DAY
    
    @classmethod
    def get_max_daily_work(cls, extensions_used: int) -> float:
        """Get maximum daily work time considering extensions used."""
        if extensions_used < cls.MAX_WORK_EXTENSIONS_PER_WEEK:
            return 14 * 60  # Can extend to 14 hours
        return cls.MAX_WORK_PER_DAY
    
    @classmethod
    def validate_break_duration(cls, duration: float, has_recent_partial: bool = False) -> bool:
        """Validate if break duration meets regulations."""
        if duration >= cls.MIN_BREAK_DURATION:
            return True
        if duration >= cls.MIN_PARTIAL_BREAK and has_recent_partial:
            return True  # Split break system
        return False


def calculate_travel_time_between_tasks(task1, task2, vehicle) -> float:
    """
    Calculate travel time between two tasks using proper Haversine distance.
    
    Args:
        task1: Starting task
        task2: Ending task
        vehicle: Vehicle object for speed calculation
        
    Returns:
        Travel time in minutes
    """
    if not hasattr(task1, 'lat') or not hasattr(task2, 'lat'):
        return 15.0  # Default 15 minutes if no coordinates
    
    # Import the proper Haversine calculation function
    try:
        from second_level import calculate_travel_time_haversine
        
        # Use vehicle-specific average speed or default to more realistic truck speed
        avg_speed_kmh = getattr(vehicle, 'average_speed', 60.0)  # 60 km/h for realistic European truck travel
        
        # Use proper Haversine calculation
        travel_time_minutes = calculate_travel_time_haversine(
            task1.lat, task1.lon, 
            task2.lat, task2.lon, 
            avg_speed_kmh
        )
        
        return travel_time_minutes
        
    except ImportError:
        # Fallback to improved calculation if import fails
        import math
        
        # Improved Haversine distance calculation (same as in second_level.py)
        lat1_rad = math.radians(task1.lat)
        lon1_rad = math.radians(task1.lon)
        lat2_rad = math.radians(task2.lat)
        lon2_rad = math.radians(task2.lon)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        # Earth's radius in kilometers
        R = 6371.0
        distance_km = R * c
        
        # Use vehicle-specific average speed or default to realistic truck speed
        avg_speed_kmh = getattr(vehicle, 'average_speed', 60.0)  # 60 km/h for European truck travel
        travel_time_hours = distance_km / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60.0
        
        return travel_time_minutes


def check_break_requirement(driver_state: DriverState, upcoming_drive_time: float) -> Tuple[bool, float]:
    """
    Check if a break is required before driving for the specified duration.
    
    Args:
        driver_state: Current driver state
        upcoming_drive_time: Planned driving time in minutes
        
    Returns:
        Tuple of (break_required, break_duration_needed)
    """
    # Check if driving time would exceed continuous driving limit
    total_drive_time = driver_state.drive_since_break + upcoming_drive_time
    
    if total_drive_time > HoSRegulations.MAX_DRIVE_WITHOUT_BREAK:
        # Break is required
        break_duration = HoSRegulations.MIN_BREAK_DURATION
        return True, break_duration
    
    # Check if work time would exceed continuous work limit
    total_work_time = driver_state.work_since_break + upcoming_drive_time
    
    if total_work_time > HoSRegulations.MAX_WORK_WITHOUT_BREAK:
        # Break is required
        break_duration = HoSRegulations.MIN_BREAK_DURATION
        return True, break_duration
    
    return False, 0.0


def check_daily_rest_requirement(driver_state: DriverState) -> Tuple[bool, float]:
    """
    Check if daily rest is required based on current state.
    
    Args:
        driver_state: Current driver state
        
    Returns:
        Tuple of (rest_required, rest_duration_needed)
    """
    max_daily_drive = HoSRegulations.get_max_daily_drive(driver_state.daily_driving_extensions_used)
    max_daily_work = HoSRegulations.get_max_daily_work(driver_state.daily_work_extensions_used)
    
    # Check if daily limits are reached
    if (driver_state.drive_today >= max_daily_drive or 
        driver_state.work_today >= max_daily_work):
        
        # Determine rest duration needed
        if driver_state.reduced_rest_used < HoSRegulations.MAX_REDUCED_RESTS_PER_WEEK:
            rest_duration = HoSRegulations.MIN_REDUCED_DAILY_REST
        else:
            rest_duration = HoSRegulations.MIN_DAILY_REST
        
        return True, rest_duration
    
    return False, 0.0


def check_weekly_rest_requirement(driver_state: DriverState) -> Tuple[bool, float]:
    """
    Check if weekly rest is required based on current state.
    
    Args:
        driver_state: Current driver state
        
    Returns:
        Tuple of (rest_required, rest_duration_needed)
    """
    # Check weekly driving limits
    if driver_state.drive_this_week >= HoSRegulations.MAX_DRIVE_PER_WEEK:
        return True, HoSRegulations.MIN_WEEKLY_REST
    
    # Check two-week driving limits
    total_two_weeks = driver_state.drive_this_week + driver_state.drive_last_week
    if total_two_weeks >= HoSRegulations.MAX_DRIVE_TWO_WEEKS:
        return True, HoSRegulations.MIN_WEEKLY_REST
    
    return False, 0.0


def apply_break_to_driver_state(driver_state: DriverState, break_duration: float, 
                               current_time: float = None, location: str = None) -> None:
    """
    Apply break effects to driver state with proper event tracking.
    
    Args:
        driver_state: Driver state to update
        break_duration: Duration of break in minutes
        current_time: Current time for event tracking
        location: Location where break is taken
    """
    # Enhanced break handling with event tracking
    if hasattr(driver_state, 'take_break'):
        try:
            # Try with enhanced signature first
            driver_state.take_break(break_duration, current_time, location)
        except TypeError:
            # Fallback to basic signature
            driver_state.take_break(break_duration)
    else:
        # Fallback for basic DriverState
        if break_duration >= HoSRegulations.MIN_BREAK_DURATION:
            driver_state.drive_since_break = 0
            driver_state.work_since_break = 0
        elif break_duration >= HoSRegulations.MIN_PARTIAL_BREAK:
            # Partial break effect
            driver_state.drive_since_break = max(0, driver_state.drive_since_break - break_duration * 0.5)


def apply_daily_rest_to_driver_state(driver_state: DriverState, rest_duration: float,
                                   current_time: float = None, location: str = None) -> None:
    """
    Apply daily rest effects to driver state with proper event tracking.
    
    Args:
        driver_state: Driver state to update
        rest_duration: Duration of rest in minutes
        current_time: Current time for event tracking
        location: Location where rest is taken
    """
    if hasattr(driver_state, 'take_daily_rest'):
        driver_state.take_daily_rest(rest_duration, current_time, location)
    else:
        # Fallback for basic DriverState
        # Track extensions
        if driver_state.drive_today > HoSRegulations.MAX_DRIVE_PER_DAY:
            driver_state.daily_driving_extensions_used += 1
        if driver_state.work_today > HoSRegulations.MAX_WORK_PER_DAY:
            driver_state.daily_work_extensions_used += 1
        
        # Reset daily counters
        driver_state.drive_since_break = 0
        driver_state.work_since_break = 0
        driver_state.drive_today = 0
        driver_state.work_today = 0


def apply_weekly_rest_to_driver_state(driver_state: DriverState, rest_duration: float,
                                    current_time: float = None, location: str = None) -> None:
    """
    Apply weekly rest effects to driver state with proper event tracking.
    
    Args:
        driver_state: Driver state to update
        rest_duration: Duration of rest in minutes
        current_time: Current time for event tracking
        location: Location where rest is taken
    """
    if hasattr(driver_state, 'take_weekly_rest'):
        driver_state.take_weekly_rest(rest_duration, current_time, location)
    else:
        # Fallback for basic DriverState
        driver_state.drive_last_week = driver_state.drive_this_week
        driver_state.drive_this_week = 0
        driver_state.daily_driving_extensions_used = 0
        driver_state.daily_work_extensions_used = 0
        driver_state.reduced_rest_used = 0
        apply_daily_rest_to_driver_state(driver_state, rest_duration, current_time, location)


def sort_tasks_chronologically(tasks: List) -> List:
    """
    Sort tasks in strict chronological order: yesterday, today, tomorrow.
    
    Args:
        tasks: List of tasks to sort
        
    Returns:
        List of tasks sorted chronologically by day
    """
    if not tasks:
        return []
    
    # Group tasks by day
    tasks_by_day = {}
    for task in tasks:
        day = getattr(task, 'day', 0)  # Default to today (0) if no day attribute
        if day not in tasks_by_day:
            tasks_by_day[day] = []
        tasks_by_day[day].append(task)
    
    # Sort days and concatenate tasks
    sorted_tasks = []
    for day in sorted(tasks_by_day.keys()):
        sorted_tasks.extend(tasks_by_day[day])
    
    return sorted_tasks


def simulate_hos_advanced(route, driver_state: DriverState, sorted_tasks: List) -> Tuple[bool, float]:
    """
    Advanced HoS simulation with modular design and detailed event tracking.
    
    This function addresses the feedback about making HoS simulation more modular
    by breaking down the logic into smaller, testable functions.
    
    Args:
        route: Route object containing vehicle and tasks
        driver_state: Current driver state (will be modified)
        sorted_tasks: Tasks sorted chronologically
        
    Returns:
        Tuple of (is_feasible, total_duration_including_rests)
    """
    if not sorted_tasks:
        return True, 0.0
    
    # Initialize tracking variables
    current_time = 0.0
    total_break_time = 0.0
    total_rest_time = 0.0
    violations = []
    events = []
    
    # Simulate each task execution
    for i, task in enumerate(sorted_tasks):
        task_start_time = current_time
        
        # 1. Handle service time at current location
        service_time = getattr(task, 'service_time', 0)
        
        # Check if work time limits allow service
        if not driver_state.can_work(service_time):
            # Need a break before service
            break_required, break_duration = check_break_requirement(driver_state, 0)
            if break_required:
                apply_break_to_driver_state(driver_state, break_duration, current_time)
                current_time += break_duration
                total_break_time += break_duration
                events.append({
                    'type': 'break',
                    'start_time': current_time - break_duration,
                    'duration': break_duration,
                    'reason': 'Work time limit reached'
                })
        
        # Perform service
        driver_state.work_since_break += service_time
        driver_state.work_today += service_time
        driver_state.work_this_week += service_time
        current_time += service_time
        
        # 2. Handle travel to next task (if not the last task)
        if i < len(sorted_tasks) - 1:
            next_task = sorted_tasks[i + 1]
            
            # Calculate travel time
            travel_time = calculate_travel_time_between_tasks(task, next_task, route.vehicle)
            
            # Check if driving time limits allow travel
            if not driver_state.can_drive(travel_time):
                # Check what type of rest is needed
                daily_rest_required, daily_rest_duration = check_daily_rest_requirement(driver_state)
                weekly_rest_required, weekly_rest_duration = check_weekly_rest_requirement(driver_state)
                
                if weekly_rest_required:
                    apply_weekly_rest_to_driver_state(driver_state, weekly_rest_duration, current_time)
                    current_time += weekly_rest_duration
                    total_rest_time += weekly_rest_duration
                    events.append({
                        'type': 'weekly_rest',
                        'start_time': current_time - weekly_rest_duration,
                        'duration': weekly_rest_duration,
                        'reason': 'Weekly driving limit reached'
                    })
                elif daily_rest_required:
                    apply_daily_rest_to_driver_state(driver_state, daily_rest_duration, current_time)
                    current_time += daily_rest_duration
                    total_rest_time += daily_rest_duration
                    events.append({
                        'type': 'daily_rest',
                        'start_time': current_time - daily_rest_duration,
                        'duration': daily_rest_duration,
                        'reason': 'Daily driving/work limit reached'
                    })
                else:
                    # Just need a break
                    break_required, break_duration = check_break_requirement(driver_state, travel_time)
                    if break_required:
                        apply_break_to_driver_state(driver_state, break_duration, current_time)
                        current_time += break_duration
                        total_break_time += break_duration
                        events.append({
                            'type': 'break',
                            'start_time': current_time - break_duration,
                            'duration': break_duration,
                            'reason': 'Continuous driving limit reached'
                        })
            
            # Check again if we can drive after rest/break
            if not driver_state.can_drive(travel_time):
                violations.append(f"Cannot complete travel from task {i} to {i+1}: HoS violation")
                return False, current_time
            
            # Perform travel
            driver_state.drive_since_break += travel_time
            driver_state.drive_today += travel_time
            driver_state.drive_this_week += travel_time
            driver_state.work_since_break += travel_time
            driver_state.work_today += travel_time
            driver_state.work_this_week += travel_time
            current_time += travel_time
        
        # 3. Check time window constraints (if applicable)
        if hasattr(task, 'latest_time') and task.latest_time is not None:
            if current_time > task.latest_time and not getattr(task, 'soft_time_window', False):
                violations.append(f"Task {i} violates hard time window: arrival={current_time:.1f}, latest={task.latest_time}")
                return False, current_time
    
    # Simulation completed successfully
    total_duration = current_time
    
    return True, total_duration


def validate_route_hos_feasibility(route, driver_state: DriverState = None) -> HoSSimulationResult:
    """
    Comprehensive HoS feasibility validation for a route.
    
    This function provides a complete analysis of route feasibility including
    detailed event tracking and violation reporting.
    
    Args:
        route: Route object to validate
        driver_state: Optional driver state (creates new if None)
        
    Returns:
        HoSSimulationResult with detailed analysis
    """
    # Use provided driver state or route's driver state or create new
    if driver_state is None:
        if hasattr(route, 'driver') and route.driver and route.driver.hos_state:
            working_state = copy.deepcopy(route.driver.hos_state)
        else:
            working_state = DriverState()
    else:
        working_state = copy.deepcopy(driver_state)
    
    # Sort tasks chronologically
    sorted_tasks = sort_tasks_chronologically(route.tasks)
    
    # Run simulation
    is_feasible, total_duration = simulate_hos_advanced(route, working_state, sorted_tasks)
    
    # Calculate detailed metrics
    driving_time = sum(
        calculate_travel_time_between_tasks(sorted_tasks[i], sorted_tasks[i+1], route.vehicle)
        for i in range(len(sorted_tasks) - 1)
    ) if len(sorted_tasks) > 1 else 0.0
    
    working_time = sum(getattr(task, 'service_time', 0) for task in sorted_tasks) + driving_time
    break_time = max(0, total_duration - working_time)
    rest_time = 0.0  # Would need to track this in simulation
    
    # Get events if driver state supports it
    events = []
    if hasattr(working_state, 'export_events_to_dict'):
        events = working_state.export_events_to_dict()
    
    # Get violations if driver state supports it
    violations = []
    if hasattr(working_state, 'get_compliance_report'):
        compliance_report = working_state.get_compliance_report()
        violations = compliance_report.get('violations', [])
    
    return HoSSimulationResult(
        is_feasible=is_feasible,
        total_duration=total_duration,
        driving_time=driving_time,
        working_time=working_time,
        break_time=break_time,
        rest_time=rest_time,
        violations=violations,
        events=events
    )


# Backward compatibility functions for existing code
def _simulate_hos_advanced(route, driver_state: DriverState, sorted_tasks: List) -> Tuple[bool, float]:
    """Backward compatibility wrapper for existing code."""
    return simulate_hos_advanced(route, driver_state, sorted_tasks)


def _sort_tasks_chronologically(tasks: List) -> List:
    """Backward compatibility wrapper for existing code."""
    return sort_tasks_chronologically(tasks)


def _calculate_travel_time_between_tasks(task1, task2, vehicle) -> float:
    """Backward compatibility wrapper for existing code."""
    return calculate_travel_time_between_tasks(task1, task2, vehicle)


if __name__ == "__main__":
    # Test the modular HoS system
    print("Testing Modular HoS Simulation System")
    print("="*50)
    
    # Create test driver state
    test_state = DriverState()
    print(f"✓ Created test driver state")
    
    # Test regulation validation
    regulations = HoSRegulations()
    max_drive = regulations.get_max_daily_drive(0)
    print(f"✓ Max daily drive with 0 extensions: {max_drive/60:.1f} hours")
    
    # Test break requirement check
    test_state.drive_since_break = 4 * 60  # 4 hours
    break_needed, duration = check_break_requirement(test_state, 60)  # 1 hour more
    print(f"✓ Break needed after 4h + 1h driving: {break_needed}, duration: {duration} min")
    
    print("\nModular HoS system ready for use!")
    print("- No circular imports")
    print("- Modular design for better testing")
    print("- Enhanced event tracking")
    print("- Backward compatibility maintained")
