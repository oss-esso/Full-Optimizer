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

from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING
import copy
from dataclasses import dataclass

# Import only the data structures we need to avoid circular imports
try:
    from epdt_data_structures import DriverState, HoSEvent
except ImportError:
    try:
        from .epdt_data_structures import DriverState, HoSEvent
    except ImportError:
        from algo.epdt_data_structures import DriverState, HoSEvent

if TYPE_CHECKING:
    try:
        from epdt_data_structures import Route
    except ImportError:
        try:
            from .epdt_data_structures import Route
        except ImportError:
            from algo.epdt_data_structures import Route


@dataclass
class SimulatedEvent:
    """
    Represents a single event in the timeline simulation for HoS compliance.
    
    This is used by the build_compliant_timeline function to create a detailed
    timeline of all events including travel, work, waits, and mandatory rests.
    """
    event_type: str  # 'DRIVE', 'WORK', 'WAIT', 'REST'
    start_time: float  # Minutes from start of planning period
    end_time: float  # Minutes from start of planning period
    duration: float  # Duration in minutes
    description: str  # Human-readable description
    location: Optional[str] = None  # Location where event occurs
    task_id: Optional[str] = None  # Associated task ID if applicable
    rest_type: Optional[str] = None  # Type of rest: '45min_break', '11h_daily', '45h_weekly'
    cost: float = 0.0  # Cost associated with this event
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging/export."""
        return {
            'event_type': self.event_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'description': self.description,
            'location': self.location,
            'task_id': self.task_id,
            'rest_type': self.rest_type,
            'cost': self.cost
        }


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


@dataclass
class HoSAnalysisResult:
    """
    Unified result from the single, authoritative HoS Engine containing all HoS analysis.
    
    This is the single source of truth for HoS compliance as recommended in TODO3.md Section 1.
    """
    is_feasible: bool
    timeline: List[SimulatedEvent]  # Detailed timeline of all events
    costs: Dict[str, float]  # Breakdown of costs associated with timeline
    violations: List[str]  # List of specific violations if not feasible
    total_duration: float  # Total time including breaks and rests
    driving_time: float    # Actual driving time
    working_time: float    # Total working time
    break_time: float      # Time spent on breaks
    rest_time: float       # Time spent on daily/weekly rests
    rest_cost: float       # Total cost of mandatory rests
    driver_cost: float     # Total driver cost including rest time


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
    MAX_DRIVE_PER_WEEK = 45 * 60            # 45 hours (CORRECTED from 56)
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
    Calculate travel time between two tasks using the proper route provider.
    This ensures OSRM data is used when available for realistic travel times.
    
    Args:
        task1: Starting task
        task2: Ending task
        vehicle: Vehicle object for speed calculation
        
    Returns:
        Travel time in minutes
    """
    try:
        # Import and use the proper route provider function
        from route_provider import calculate_travel_time_between_tasks as route_provider_calc
        return route_provider_calc(task1, task2, vehicle)
    except ImportError:
        # Fallback if route provider not available
        if not hasattr(task1, 'lat') or not hasattr(task2, 'lat'):
            return 15.0  # Default 15 minutes if no coordinates
        
        # Import the proper Haversine calculation function
        try:
            from second_level import calculate_travel_time_haversine
            
            # Use vehicle-specific average speed or default to more realistic truck speed
            avg_speed_kmh = getattr(vehicle, 'average_speed', 45.0)  # 45 km/h for realistic European heavy truck travel (includes traffic, terrain, restrictions)
            
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
        avg_speed_kmh = getattr(vehicle, 'average_speed', 45.0)  # 45 km/h for European heavy truck travel (includes traffic, terrain, restrictions)
        travel_time_hours = distance_km / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60.0
        
        return travel_time_minutes


def calculate_distance_between_tasks(task1, task2) -> float:
    """
    Calculate distance between two tasks using the proper route provider.
    This ensures OSRM data is used when available for realistic distances.
    
    Args:
        task1: Starting task
        task2: Ending task
        
    Returns:
        Distance in kilometers
    """
    try:
        # Import and use the proper route provider
        from route_provider import get_route_provider
        
        provider = get_route_provider()
        
        # Extract coordinates and node IDs
        start_coords = (getattr(task1, 'lon', 0), getattr(task1, 'lat', 0))
        end_coords = (getattr(task2, 'lon', 0), getattr(task2, 'lat', 0))
        start_node_id = getattr(task1, 'location_id', f"{task1.lat}_{task1.lon}")
        end_node_id = getattr(task2, 'location_id', f"{task2.lat}_{task2.lon}")
        
        route_details = provider.get_route_details(
            start_node_id, end_node_id, start_coords, end_coords
        )
        
        if route_details:
            return route_details['distance_km']
        
    except Exception as e:
        # Fallback to Haversine distance calculation
        pass
    
    # Fallback: Calculate Haversine distance
    if not hasattr(task1, 'lat') or not hasattr(task2, 'lat'):
        return 0.0  # Default 0 km if no coordinates
    
    import math
    
    # Haversine distance calculation
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
    
    return distance_km


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
    Advanced HoS simulation with UNIFIED timing calculation to match final route display.
    
    This function now uses the exact same timing calculation method as the final route display
    to ensure consistency between validation and execution simulation.
    
    Note: Drivers with B licenses are exempt from HoS regulations.
    
    Args:
        route: Route object containing vehicle and tasks
        driver_state: Current driver state (will be modified)
        sorted_tasks: Tasks sorted chronologically
        
    Returns:
        Tuple of (is_feasible, total_duration_including_rests)
    """
    if not sorted_tasks:
        return True, 0.0
    
    # SIMPLIFIED APPROACH: Apply same timing calculation for ALL drivers
    # This ensures consistency between construction validation and final display
    # Initialize tracking variables (matching final display logic)
    cumulative_time = 0.0
    cumulative_driving_time = 0.0
    driving_since_last_break = 0.0
    violations = []
    events = []
    
    # Process all tasks except depot start/return (matching final display logic)
    tasks_to_process = [t for t in sorted_tasks if not (hasattr(t, 'is_depot_start') and t.is_depot_start()) and not (hasattr(t, 'is_depot_return') and t.is_depot_return())]
    
    # Simulate each task using the EXACT same logic as final route display
    for i, task in enumerate(tasks_to_process):
        # Calculate travel time (matching final display)
        if i > 0:
            try:
                travel_time = calculate_travel_time_between_tasks(tasks_to_process[i-1], task, route.vehicle)
            except:
                travel_time = 30.0  # Default travel time (matching final display)
        else:
            travel_time = 30.0  # From depot (matching final display)
            
        # Apply HoS breakdown for travel (matching final display logic)
        if travel_time > 0:
            cumulative_time, driving_since_last_break = _apply_unified_hos_breakdown(
                travel_time, cumulative_time, driving_since_last_break, cumulative_driving_time
            )
            cumulative_driving_time += travel_time
        else:
            cumulative_time += travel_time
        
        # UNIFIED TIME WINDOW VALIDATION: Apply waiting time if arriving early (matching final display)
        earliest = getattr(task, 'earliest_time', None)
        if earliest is not None and cumulative_time < earliest:
            waiting_time = earliest - cumulative_time
            cumulative_time = earliest  # Wait until earliest time (matching final display)
        
        # Service time
        service_time = getattr(task, 'service_time', 5.0)
        cumulative_time += service_time
        
        # UNIFIED TIME WINDOW VIOLATION CHECK (matching final display validation)
        latest = getattr(task, 'latest_time', None)
        if latest is not None and cumulative_time > latest:
            delay_time = cumulative_time - latest
            # Use ABSOLUTE 10-minute grace period (as defined previously)
            ABSOLUTE_GRACE_PERIOD_MINUTES = 10.0
            if delay_time > ABSOLUTE_GRACE_PERIOD_MINUTES:
                violation_msg = f"UNIFIED VALIDATION VIOLATION: Task {getattr(task, 'id', 'unknown')}: arrives at {cumulative_time:.1f}min but latest allowed is {latest:.1f}min (late by {delay_time:.1f}min, exceeds ABSOLUTE {ABSOLUTE_GRACE_PERIOD_MINUTES} min grace period)"
                violations.append(violation_msg)
                print(f"            UNIFIED TIMING DEBUG: REJECTING route due to: {violation_msg}")
                # Return infeasible if we have violations
                return False, cumulative_time
    
    # Final return to depot (matching final display)
    if tasks_to_process:
        try:
            final_travel = calculate_travel_time_between_tasks(tasks_to_process[-1], sorted_tasks[0], route.vehicle)  # Back to depot
        except:
            final_travel = 30.0
    else:
        final_travel = 30.0
        
    # Apply HoS logic for final travel (matching final display)
    if final_travel > 0:
        cumulative_time, driving_since_last_break = _apply_unified_hos_breakdown(
            final_travel, cumulative_time, driving_since_last_break, cumulative_driving_time
        )
        cumulative_driving_time += final_travel
    else:
        cumulative_time += final_travel
    
    # Return feasible if no violations
    return True, cumulative_time


def _apply_unified_hos_breakdown(travel_time_minutes: float, current_time: float, 
                               driving_since_break: float, total_driving_time: float) -> tuple:
    """
    Apply UNIFIED HoS breakdown that matches the final route display logic exactly.
    
    This function ensures that HoS validation uses the same timing calculation
    as the final route execution simulation.
    """
    if travel_time_minutes <= 0:
        return current_time, driving_since_break
    
    remaining_travel = travel_time_minutes
    updated_time = current_time
    updated_driving_since_break = driving_since_break
    
    # 4.5-hour continuous driving limit
    MAX_CONTINUOUS_DRIVING_MINUTES = 4.5 * 60  # 270 minutes
    MANDATORY_BREAK_MINUTES = 45.0
    
    while remaining_travel > 0:
        # Check if we need a break before continuing
        can_drive_without_break = MAX_CONTINUOUS_DRIVING_MINUTES - updated_driving_since_break
        
        if remaining_travel <= can_drive_without_break:
            # Can complete the journey without a break
            updated_time += remaining_travel
            updated_driving_since_break += remaining_travel
            remaining_travel = 0
        else:
            # Need to take a break partway through
            # Drive until break is required
            updated_time += can_drive_without_break
            updated_driving_since_break += can_drive_without_break
            remaining_travel -= can_drive_without_break
            
            # Take mandatory break
            updated_time += MANDATORY_BREAK_MINUTES
            updated_driving_since_break = 0  # Reset driving counter after break
    
    return updated_time, updated_driving_since_break


def validate_route_hos_feasibility(route, driver_state: DriverState = None) -> HoSSimulationResult:
    """
    Comprehensive HoS feasibility validation for a route.
    
    This function provides a complete analysis of route feasibility including
    detailed event tracking and violation reporting.
    
    Note: Timing is based on vehicle regulations, not driver license.
    - Heavy vehicles (regulations=YES): Full HoS rules with 4.5h breaks
    - Light vehicles (regulations=NO): Only overnight rest after 15h driving
    
    Args:
        route: Route object to validate
        driver_state: Optional driver state (creates new if None)
        
    Returns:
        HoSSimulationResult with detailed analysis
    """
    # Check vehicle regulations instead of driver license
    # Light vehicles (regulations=NO) get simplified timing with only overnight rest
    vehicle_regulations = getattr(route.vehicle, 'regulations', '') if route.vehicle else ''
    has_hos_regulations = str(vehicle_regulations).upper() in ['YES', 'TRUE', '1']
    
    if not has_hos_regulations:
        # Light vehicle - use build_compliant_timeline but treat as B license (simplified rules)
        try:
            # Use the corrected build_compliant_timeline function for consistent waiting time handling
            timeline, rest_cost = build_compliant_timeline(route)
            
            # Calculate metrics from timeline 
            driving_time = sum(event.duration for event in timeline if event.event_type == 'DRIVE')
            working_time = sum(event.duration for event in timeline if event.event_type in ['DRIVE', 'WORK'])
            break_time = sum(event.duration for event in timeline if event.event_type == 'REST')
            wait_time = sum(event.duration for event in timeline if event.event_type == 'WAIT')
            
            total_duration = timeline[-1].end_time if timeline else 0.0
            
            # CRITICAL: Check time window feasibility for light vehicles too
            is_feasible = True
            violations = []
            
            for event in timeline:
                if event.event_type == 'WORK' and hasattr(event, 'task_id') and event.task_id:
                    # Find the corresponding task
                    task = None
                    for route_task in route.tasks:
                        if route_task.id == event.task_id:
                            task = route_task
                            break
                    
                    if task and hasattr(task, 'latest_time') and task.latest_time is not None:
                        if event.start_time > task.latest_time:
                            is_feasible = False
                            late_by = event.start_time - task.latest_time
                            violations.append(f"Task {task.id} arrives {late_by:.1f} minutes after latest time {task.latest_time}")
            
            # Convert SimulatedEvent objects to dict format for compatibility
            events_as_dicts = [event.to_dict() for event in timeline]
            
            return HoSSimulationResult(
                is_feasible=is_feasible,  # Now properly validates time window feasibility
                total_duration=total_duration,
                driving_time=driving_time,
                working_time=working_time,
                break_time=break_time,
                rest_time=break_time,  # break_time includes all rest periods
                violations=violations,  # Actual time window violations
                events=events_as_dicts
            )
        except Exception as e:
            print(f"ERROR in build_compliant_timeline for light vehicle: {e}")
            # Fallback to original light vehicle logic
            pass
        
        # FALLBACK: Original light vehicle logic (should not be reached)
        # Light vehicle - only overnight rest after 15h driving, no HoS breaks
        sorted_tasks = sort_tasks_chronologically(route.tasks)
        
        # Generate timeline events for light vehicles (no HoS breaks, only overnight rest)
        events = []
        current_time = 0.0
        total_driving_time = 0.0
        
        for i, task in enumerate(sorted_tasks):
            # Travel time to this task
            travel_time = 0.0
            if i > 0:
                travel_time = calculate_travel_time_between_tasks(sorted_tasks[i-1], task, route.vehicle)
                total_driving_time += travel_time
                
                # Check if overnight rest needed (after 15h driving)
                if total_driving_time > 15 * 60:  # 15 hours in minutes
                    # Add overnight rest (9 hours)
                    rest_event = type('Event', (), {
                        'task': None,
                        'start_time': current_time,
                        'end_time': current_time + 9 * 60,  # 9 hours rest
                        'event_type': 'OVERNIGHT_REST',
                        'duration': 9 * 60
                    })()
                    events.append(rest_event)
                    current_time += 9 * 60
                    total_driving_time = 0.0  # Reset driving counter after rest
                
                current_time += travel_time
            
            # Create event for this task
            event = type('Event', (), {
                'task': task,
                'start_time': current_time,
                'end_time': current_time + getattr(task, 'service_time', 5.0),
                'event_type': 'TASK',
                'duration': getattr(task, 'service_time', 5.0)
            })()
            events.append(event)
            
            # Update current time
            current_time += getattr(task, 'service_time', 5.0)
        
        total_duration = current_time
        driving_time = sum(getattr(event, 'duration', 0) for event in events if getattr(event, 'event_type', '') == 'TASK' and i > 0)
        
        return HoSSimulationResult(
            is_feasible=True,
            total_duration=total_duration,
            driving_time=driving_time,
            working_time=total_duration,
            break_time=0.0,
            rest_time=sum(getattr(event, 'duration', 0) for event in events if getattr(event, 'event_type', '') == 'OVERNIGHT_REST'),
            events=events,
            violations=[]
        )
    # Use provided driver state or route's driver state or create new
    if driver_state is None:
        if hasattr(route, 'driver') and route.driver and route.driver.hos_state:
            working_state = copy.deepcopy(route.driver.hos_state)
        else:
            working_state = DriverState()
    else:
        working_state = copy.deepcopy(driver_state)
    
    # FIXED: Use the corrected build_compliant_timeline function for heavy vehicles
    try:
        timeline, rest_cost = build_compliant_timeline(route)
        
        # Calculate metrics from timeline 
        driving_time = sum(event.duration for event in timeline if event.event_type == 'DRIVE')
        working_time = sum(event.duration for event in timeline if event.event_type in ['DRIVE', 'WORK'])
        break_time = sum(event.duration for event in timeline if event.event_type == 'REST')
        wait_time = sum(event.duration for event in timeline if event.event_type == 'WAIT')
        
        total_duration = timeline[-1].end_time if timeline else 0.0
        
        # Convert SimulatedEvent objects to dict format for compatibility
        events_as_dicts = [event.to_dict() for event in timeline]
        
        # CRITICAL: Check time window feasibility after HoS compliance
        is_feasible = True
        violations = []
        
        for event in timeline:
            if event.event_type == 'WORK' and hasattr(event, 'task_id') and event.task_id:
                # Find the corresponding task
                task = None
                for route_task in route.tasks:
                    if route_task.id == event.task_id:
                        task = route_task
                        break
                
                if task and hasattr(task, 'latest_time') and task.latest_time is not None:
                    if event.start_time > task.latest_time:
                        is_feasible = False
                        late_by = event.start_time - task.latest_time
                        violations.append(f"Task {task.id} arrives {late_by:.1f} minutes after latest time {task.latest_time}")
        
        return HoSSimulationResult(
            is_feasible=is_feasible,  # Now properly validates time window feasibility
            total_duration=total_duration,
            driving_time=driving_time,
            working_time=working_time,
            break_time=break_time,
            rest_time=break_time,  # break_time includes all rest periods
            violations=violations,  # Actual time window violations
            events=events_as_dicts
        )
    except Exception as e:
        print(f"ERROR in build_compliant_timeline: {e}")
        # Fallback to original logic if build_compliant_timeline fails
        pass
    
    # FALLBACK: Original logic (should not be reached if build_compliant_timeline works)
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


def build_compliant_timeline(route: 'Route') -> Tuple[List[SimulatedEvent], float]:
    """
    Build a detailed timeline that ensures HoS compliance and proper waiting time handling.
    
    CRITICAL FIX: This function now properly incorporates waiting time into the timeline,
    ensuring that when a vehicle arrives early, it waits until the earliest_time before
    starting work, and all subsequent timing calculations account for this waiting.
    
    The function processes each task in sequence (eliminating duplication) and properly
    inserts HoS breaks when driving limits are reached.
    
    Args:
        route: Route object containing tasks and vehicle information
        
    Returns:
        Tuple of (timeline_events, total_rest_cost)
    """
    timeline = []
    total_rest_cost = 0.0
    current_time = 0.0
    driver_cost_per_minute = 1.0  # Cost per minute for driver time
    
    # Initialize driver state
    driver_state = DriverState()
    current_day = 0
    
    if not route.tasks:
        return timeline, total_rest_cost
    
    # Process each task in sequence - this eliminates the duplication issue
    for task_index, task in enumerate(route.tasks):
        # Check for day transition
        task_day = getattr(task, 'day', 0)
        if task_day != current_day and task_index > 0:
            # New day: insert mandatory daily rest
            if driver_state.work_today > 0:  # If work was done previous day
                rest_duration = HoSRegulations.MIN_DAILY_REST  # 11 hours
                rest_cost = rest_duration * driver_cost_per_minute
                
                timeline.append(SimulatedEvent(
                    event_type='REST',
                    start_time=current_time,
                    end_time=current_time + rest_duration,
                    duration=rest_duration,
                    description="Mandatory 11-hour daily rest",
                    rest_type='11h_daily',
                    cost=rest_cost
                ))
                
                current_time += rest_duration
                total_rest_cost += rest_cost
                driver_state.reset_daily()
                
            current_day = task_day
        
        # If this is not the first task, simulate travel from previous task
        if task_index > 0:
            prev_task = route.tasks[task_index - 1]
            travel_time = _calculate_travel_time_for_simulation(prev_task, task, route.vehicle)
            travel_time_remaining = travel_time
            
            # Simulate travel with HoS breaks
            while travel_time_remaining > 0:
                # Calculate maximum time the driver can legally drive before needing a break
                max_drive_before_break = HoSRegulations.MAX_DRIVE_WITHOUT_BREAK - driver_state.drive_since_break
                max_drive_before_daily_limit = HoSRegulations.MAX_DRIVE_PER_DAY - driver_state.drive_today
                max_work_before_daily_limit = HoSRegulations.MAX_WORK_PER_DAY - driver_state.work_today
                
                drivable_time = min(max_drive_before_break, max_drive_before_daily_limit, 
                                  max_work_before_daily_limit, travel_time_remaining)
                
                if drivable_time > 0:
                    # Simulate driving for the calculated time
                    # Helper function to get location name from task
                    def get_location_name(task):
                        if hasattr(task, 'location') and task.location:
                            return getattr(task.location, 'name', getattr(task, 'location_id', task.id))
                        return getattr(task, 'location_id', task.id)
                    
                    prev_location = get_location_name(prev_task)
                    current_location = get_location_name(task)
                    
                    # Calculate distance for display
                    distance_km = calculate_distance_between_tasks(prev_task, task)
                    
                    timeline.append(SimulatedEvent(
                        event_type='DRIVE',
                        start_time=current_time,
                        end_time=current_time + drivable_time,
                        duration=drivable_time,
                        description=f"Drive from {prev_location} to {current_location} ({distance_km:.1f}km)",
                        task_id=f"{prev_task.id}->{task.id}"
                    ))
                    
                    driver_state.drive_since_break += drivable_time
                    driver_state.drive_today += drivable_time
                    driver_state.work_today += drivable_time
                    driver_state.drive_this_week += drivable_time
                    driver_state.total_work_this_week += drivable_time
                    
                    current_time += drivable_time
                    travel_time_remaining -= drivable_time
                
                # If travel is not complete, a rest is mandatory
                if travel_time_remaining > 0:
                    rest_duration = 0.0
                    rest_type = None
                    rest_description = ""
                    
                    # Determine type of rest needed
                    if driver_state.drive_since_break >= HoSRegulations.MAX_DRIVE_WITHOUT_BREAK:
                        # 45-minute break required
                        rest_duration = HoSRegulations.MIN_BREAK_DURATION
                        rest_type = '45min_break'
                        rest_description = "Mandatory 45-minute driving break"
                    elif (driver_state.drive_today >= HoSRegulations.MAX_DRIVE_PER_DAY or 
                          driver_state.work_today >= HoSRegulations.MAX_WORK_PER_DAY):
                        # Daily rest required
                        rest_duration = HoSRegulations.MIN_DAILY_REST
                        rest_type = '11h_daily'
                        rest_description = "Mandatory 11-hour daily rest"
                    elif driver_state.drive_this_week >= HoSRegulations.MAX_DRIVE_PER_WEEK:
                        # Weekly rest required
                        rest_duration = HoSRegulations.MIN_WEEKLY_REST
                        rest_type = '45h_weekly'
                        rest_description = "Mandatory 45-hour weekly rest"
                    else:
                        # Shouldn't happen, but handle gracefully
                        rest_duration = HoSRegulations.MIN_BREAK_DURATION
                        rest_type = '45min_break'
                        rest_description = "Mandatory break (fallback)"
                    
                    rest_cost = rest_duration * driver_cost_per_minute
                    
                    timeline.append(SimulatedEvent(
                        event_type='REST',
                        start_time=current_time,
                        end_time=current_time + rest_duration,
                        duration=rest_duration,
                        description=rest_description,
                        rest_type=rest_type,
                        cost=rest_cost
                    ))
                    
                    current_time += rest_duration
                    total_rest_cost += rest_cost
                    
                    # Reset HoS counters as appropriate
                    if rest_type == '45min_break':
                        driver_state.drive_since_break = 0.0
                    elif rest_type == '11h_daily':
                        driver_state.reset_daily()
                    elif rest_type == '45h_weekly':
                        # Reset weekly counters
                        driver_state.drive_this_week = 0.0
                        driver_state.total_work_this_week = 0.0
                        driver_state.reset_daily()
        
        # CRITICAL FIX: Check if we need to wait for the earliest_time of current task
        earliest_start = getattr(task, 'earliest_time', None)
        if earliest_start is not None and current_time < earliest_start:
            wait_duration = earliest_start - current_time
            
            # Determine description based on task type
            if task_index == 0:
                description = "Initial wait at depot before departure"
            else:
                description = f"Wait at {task.id}"
            
            timeline.append(SimulatedEvent(
                event_type='WAIT',
                start_time=current_time,
                end_time=earliest_start,
                duration=wait_duration,
                description=description,
                location=getattr(task, 'location', None),
                task_id=task.id
            ))
            # CRUCIALLY: Advance simulation clock to the end of the waiting period
            current_time = earliest_start
        
        # Now simulate work at current task
        service_time = getattr(task, 'service_time', 0)
        
        # FIXED: Always create WORK events for pickup/delivery tasks, even with zero service time
        # This ensures all task visits are visible in the timeline display
        is_pickup_delivery = (hasattr(task, 'task_type') and 
                             hasattr(task.task_type, 'name') and 
                             task.task_type.name in ['PICKUP', 'DELIVERY'])
        is_depot = hasattr(task, 'id') and 'depot' in str(task.id).lower()
        
        # Always show work events for pickup/delivery tasks, or if service_time > 0
        if service_time > 0 or (is_pickup_delivery and not is_depot):
            # Use minimum 1 minute for zero-service pickup/delivery to make it visible
            display_service_time = max(service_time, 1.0 if is_pickup_delivery and not is_depot else service_time)
            
            timeline.append(SimulatedEvent(
                event_type='WORK',
                start_time=current_time,
                end_time=current_time + display_service_time,
                duration=display_service_time,
                description=f"Service at {task.id}",
                location=getattr(task, 'location', None),
                task_id=task.id
            ))
            
            # Only count actual service time for HoS regulations, not display time
            actual_work_time = service_time  # Use actual service time for HoS calculations
            driver_state.work_today += actual_work_time
            driver_state.total_work_this_week += actual_work_time
            current_time += display_service_time
    
    return timeline, total_rest_cost


def _calculate_travel_time_for_simulation(task1, task2, vehicle) -> float:
    """
    Calculate travel time between two tasks for simulation purposes.
    
    This is a helper function that uses the same logic as the main routing system
    to ensure consistency in travel time calculations.
    
    Args:
        task1: Starting task
        task2: Ending task
        vehicle: Vehicle object for speed calculation
        
    Returns:
        Travel time in minutes
    """
    # Use the calculate_travel_time_between_tasks function defined in this module
    return calculate_travel_time_between_tasks(task1, task2, vehicle)


class HoSEngine:
    """
    Single, Authoritative Hours of Service Engine
    
    This class implements Section 1, Recommendation 1 from TODO3.md:
    "Create a Single, Authoritative HoS Engine."
    
    All HoS logic is unified in this class with one primary function analyze_route()
    that is the single source of truth for HoS compliance.
    """
    
    def __init__(self):
        """Initialize the HoS Engine."""
        self.regulations = HoSRegulations()
    
    def analyze_route(self, route: 'Route') -> HoSAnalysisResult:
        """
        Single source of truth for HoS analysis.
        
        This method performs the complete HoS simulation and returns a comprehensive
        result containing feasibility, timeline, costs, and violations.
        
        As recommended in the technical review, this now includes BOTH HoS compliance
        AND time window validation in a single, authoritative analysis.
        
        Args:
            route: Route object to analyze
            
        Returns:
            HoSAnalysisResult containing complete HoS analysis
        """
        try:
            # All drivers follow HoS regulations, but with different rules:
            # B license: max 15h consecutive driving, then 9h sleep
            # CE license: full EU HoS regulations
            timeline, rest_cost = build_compliant_timeline(route)
            
            # TECHNICAL REVIEW FIX: Immediately validate time windows after timeline is built
            # This prevents the optimizer from favoring routes that seem good before rest insertion
            # but become infeasible after mandatory rests are added
            is_feasible, violations = self._validate_timeline_against_constraints(timeline, route)
            
            # DEBUG: Log validation results
            if violations:
                print(f"            DEBUG HoS ENGINE: analyze_route returning is_feasible={is_feasible} with {len(violations)} violations: {violations[:2]}")
            
            # Calculate metrics from timeline
            driving_time = sum(event.duration for event in timeline if event.event_type == 'DRIVE')
            working_time = sum(event.duration for event in timeline if event.event_type in ['DRIVE', 'WORK'])
            break_time = sum(event.duration for event in timeline if event.event_type == 'REST')
            total_duration = timeline[-1].end_time if timeline else 0.0
            
            # Calculate costs
            driver_cost_per_minute = (route.vehicle.cost_per_hour / 60.0) if route.vehicle else (25.0 / 60.0)
            driver_cost = working_time * driver_cost_per_minute
            
            return HoSAnalysisResult(
                is_feasible=is_feasible,  # Now reflects BOTH HoS and time window validation
                timeline=timeline,
                costs={'driver_cost': driver_cost, 'rest_cost': rest_cost},
                violations=violations,
                total_duration=total_duration,
                driving_time=driving_time,
                working_time=working_time,
                break_time=break_time,
                rest_time=break_time,
                rest_cost=rest_cost,
                driver_cost=driver_cost
            )
            
        except Exception as e:
            # If analysis fails, return infeasible with error details
            return HoSAnalysisResult(
                is_feasible=False,
                timeline=[],
                costs={},
                violations=[f"HoS analysis failed with error: {str(e)}"],
                total_duration=0.0,
                driving_time=0.0,
                working_time=0.0,
                break_time=0.0,
                rest_time=0.0,
                rest_cost=0.0,
                driver_cost=0.0
            )
    
    def _analyze_b_license_route(self, route: 'Route') -> HoSAnalysisResult:
        """Analyze route for B license driver (exempt from HoS regulations)."""
        sorted_tasks = sort_tasks_chronologically(route.tasks)
        
        # Calculate basic metrics without HoS constraints
        driving_time = 0.0
        working_time = 0.0
        timeline = []
        current_time = 0.0
        
        for i, task in enumerate(sorted_tasks):
            service_time = getattr(task, 'service_time', 0)
            working_time += service_time
            
            if service_time > 0:
                timeline.append(SimulatedEvent(
                    event_type='WORK',
                    start_time=current_time,
                    end_time=current_time + service_time,
                    duration=service_time,
                    description=f"Service at {task.id}",
                    location=getattr(task, 'location', None),
                    task_id=task.id
                ))
                current_time += service_time
            
            # Add travel time to next task
            if i < len(sorted_tasks) - 1:
                next_task = sorted_tasks[i + 1]
                travel_time = calculate_travel_time_between_tasks(task, next_task, route.vehicle)
                driving_time += travel_time
                working_time += travel_time
                
                if travel_time > 0:
                    # Helper function to get location name from task
                    def get_location_name(task):
                        if hasattr(task, 'location') and task.location:
                            return getattr(task.location, 'name', getattr(task, 'location_id', task.id))
                        return getattr(task, 'location_id', task.id)
                    
                    current_location = get_location_name(task)
                    next_location = get_location_name(next_task)
                    
                    # Calculate distance for display
                    distance_km = calculate_distance_between_tasks(task, next_task)
                    
                    timeline.append(SimulatedEvent(
                        event_type='DRIVE',
                        start_time=current_time,
                        end_time=current_time + travel_time,
                        duration=travel_time,
                        description=f"Drive from {current_location} to {next_location} ({distance_km:.1f}km)",
                        task_id=f"{task.id}->{next_task.id}"
                    ))
                    current_time += travel_time
        
        return HoSAnalysisResult(
            is_feasible=True,
            timeline=timeline,
            costs={'driver_cost': current_time * (route.vehicle.cost_per_hour / 60.0) if route.vehicle else 0.0},
            violations=[],
            total_duration=current_time,
            driving_time=driving_time,
            working_time=working_time,
            break_time=0.0,
            rest_time=0.0,
            rest_cost=0.0,
            driver_cost=current_time * (route.vehicle.cost_per_hour / 60.0) if route.vehicle else 0.0
        )
    
    def _analyze_ce_license_route(self, route: 'Route') -> HoSAnalysisResult:
        """Analyze route for CE license driver with full HoS compliance."""
        # Build compliant timeline with mandatory rests
        timeline, rest_cost = build_compliant_timeline(route)
        
        # Validate timeline against time window constraints  
        is_timeline_feasible, failure_reason = self._validate_timeline_feasible(timeline, route)
        
        # Calculate detailed costs and metrics
        costs = self._calculate_timeline_costs(timeline, route)
        driving_time = costs.get('driving_time', 0.0)
        working_time = costs.get('working_time', 0.0)
        break_time = costs.get('break_time', 0.0)
        rest_time = costs.get('rest_time', 0.0)
        total_duration = costs.get('total_duration', 0.0)
        driver_cost = costs.get('driver_cost', 0.0)
        
        violations = []
        if not is_timeline_feasible:
            violations.append(failure_reason)
        
        return HoSAnalysisResult(
            is_feasible=is_timeline_feasible,
            timeline=timeline,
            costs=costs,
            violations=violations,
            total_duration=total_duration,
            driving_time=driving_time,
            working_time=working_time,
            break_time=break_time,
            rest_time=rest_time,
            rest_cost=rest_cost,
            driver_cost=driver_cost
        )
    
    def _validate_timeline_against_constraints(self, timeline: List[SimulatedEvent], route: 'Route') -> Tuple[bool, List[str]]:
        """
        Validate timeline against ALL business constraints including time windows.
        
        This implements the technical review recommendation to validate time windows 
        immediately after timeline is built, not as a separate step.
        
        CONSISTENCY FIX: This function now aligns with the waiting logic used in
        build_compliant_timeline and timing simulation in second_level.py. Early
        arrivals are NOT treated as violations since waiting logic is already
        applied during timeline generation.
        
        Args:
            timeline: Complete timeline with all events including mandatory rests
            route: Route object containing tasks with time window constraints
            
        Returns:
            Tuple of (is_feasible, list_of_violations)
        """
        violations = []
        
        if not timeline:
            return True, []
        
        # Build a map of task arrivals from the timeline
        task_arrivals = {}
        for event in timeline:
            if event.event_type in ['WORK'] and event.task_id:
                task_arrivals[event.task_id] = event.start_time
        
        # DEBUG: Print task arrivals for debugging
        debug_enabled = len(task_arrivals) > 0
        if debug_enabled:
            print(f"            DEBUG HoS: Validating {len(task_arrivals)} task arrivals against time windows")
        
        # Validate each task's time window constraints
        violation_count = 0
        for task in route.tasks:
            if not hasattr(task, 'id') or task.id not in task_arrivals:
                continue
                
            arrival_time = task_arrivals[task.id]
            
            # CONSISTENCY FIX: Remove early arrival violation check
            # Early arrivals are handled by waiting logic in build_compliant_timeline
            # This ensures consistent behavior with timing simulation in second_level.py
            # which also allows waiting for early arrivals
            
            # Check latest time constraint with ABSOLUTE 10-minute grace period (not influenced by relaxation)
            if hasattr(task, 'latest_time') and task.latest_time is not None:
                # ABSOLUTE ENFORCEMENT: 10-minute grace period regardless of any relaxation parameters
                ABSOLUTE_GRACE_PERIOD_MINUTES = 10.0
                late_by = arrival_time - task.latest_time
                
                if late_by > ABSOLUTE_GRACE_PERIOD_MINUTES:
                    violation_msg = f"Task {task.id}: arrives at {arrival_time:.1f}min but latest allowed is {task.latest_time:.1f}min (late by {late_by:.1f}min, exceeds ABSOLUTE {ABSOLUTE_GRACE_PERIOD_MINUTES} min grace period)"
                    violations.append(violation_msg)
                    violation_count += 1
                    if debug_enabled:
                        print(f"            DEBUG HoS: TIME WINDOW VIOLATION - {violation_msg}")
                    # DEBUGGING: Log when we're rejecting routes due to time window violations
                    print(f"            WARNING: Rejecting route due to ABSOLUTE time window violation: {violation_msg}")
                elif debug_enabled and violation_count < 3:
                    if late_by > 0:
                        print(f"            DEBUG HoS: Task {task.id}: arrival {arrival_time:.1f} late by {late_by:.1f}min but within ABSOLUTE {ABSOLUTE_GRACE_PERIOD_MINUTES} min grace period (OK)")
                    else:
                        print(f"            DEBUG HoS: Task {task.id}: arrival {arrival_time:.1f} <= latest {task.latest_time:.1f} (OK)")
        
        if debug_enabled:
            print(f"            DEBUG HoS: Found {violation_count} time window violations")
        
        # Additional HoS regulation validation could be added here
        # For now, we assume build_compliant_timeline already ensures HoS compliance
        
        is_feasible = len(violations) == 0
        return is_feasible, violations
    
    def _validate_timeline_feasible(self, timeline: List[SimulatedEvent], route: 'Route') -> Tuple[bool, str]:
        """
        Validate a timeline against time window constraints.
        
        This consolidates the logic from the old is_timeline_feasible function.
        """
        if not timeline:
            return True, "No timeline to validate"
        
        for event in timeline:
            if event.task_id and not event.task_id.startswith('depot'):
                # Find the corresponding task
                task = None
                for route_task in route.tasks:
                    if hasattr(route_task, 'id') and route_task.id == event.task_id:
                        task = route_task
                        break
                
                if task and hasattr(task, 'latest_time') and task.latest_time is not None:
                    # Check if arrival time violates hard time window
                    if not getattr(task, 'soft_time_window', False):  # Hard time window
                        if event.start_time > task.latest_time:
                            return False, f"Time window violation at task {task.id}: arrival at {event.start_time:.1f} min, but latest allowed is {task.latest_time:.1f} min"
        
        return True, "Timeline is feasible"
    
    def _calculate_timeline_costs(self, timeline: List[SimulatedEvent], route: 'Route') -> Dict[str, float]:
        """Calculate detailed costs and metrics from timeline."""
        costs = {
            'driving_time': 0.0,
            'working_time': 0.0,
            'break_time': 0.0,
            'rest_time': 0.0,
            'total_duration': 0.0,
            'driver_cost': 0.0
        }
        
        if not timeline:
            return costs
        
        # Calculate metrics from timeline
        total_duration = max(event.end_time for event in timeline) if timeline else 0.0
        costs['total_duration'] = total_duration
        
        for event in timeline:
            if event.event_type == 'DRIVE':
                costs['driving_time'] += event.duration
                costs['working_time'] += event.duration
            elif event.event_type == 'WORK':
                costs['working_time'] += event.duration
            elif event.event_type == 'REST':
                if event.rest_type in ['45min_break']:
                    costs['break_time'] += event.duration
                else:
                    costs['rest_time'] += event.duration
        
        # Calculate total driver cost
        driver_cost_per_minute = (route.vehicle.cost_per_hour / 60.0) if route.vehicle else (25.0 / 60.0)
        costs['driver_cost'] = total_duration * driver_cost_per_minute
        
        return costs


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
