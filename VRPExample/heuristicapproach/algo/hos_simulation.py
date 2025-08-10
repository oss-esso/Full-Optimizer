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
from epdt_data_structures import DriverState, HoSEvent

if TYPE_CHECKING:
    from epdt_data_structures import Route


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
    
    # B license drivers are exempt from HoS regulations
    if route.driver and hasattr(route.driver, 'license') and route.driver.license == 'B':
        # Calculate simple total duration without HoS constraints
        total_duration = 0.0
        for i, task in enumerate(sorted_tasks):
            total_duration += getattr(task, 'service_time', 0)
            if i < len(sorted_tasks) - 1:
                next_task = sorted_tasks[i + 1]
                travel_time = calculate_travel_time_between_tasks(task, next_task, route.vehicle)
                total_duration += travel_time
        return True, total_duration
    
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
    
    Note: Drivers with B licenses are exempt from HoS regulations.
    
    Args:
        route: Route object to validate
        driver_state: Optional driver state (creates new if None)
        
    Returns:
        HoSSimulationResult with detailed analysis
    """
    # B license drivers are exempt from HoS regulations
    if route.driver and hasattr(route.driver, 'license') and route.driver.license == 'B':
        # Calculate simple metrics without HoS constraints
        sorted_tasks = sort_tasks_chronologically(route.tasks)
        total_duration = sum(getattr(task, 'service_time', 0) for task in sorted_tasks)
        if len(sorted_tasks) > 1:
            for i in range(len(sorted_tasks) - 1):
                travel_time = calculate_travel_time_between_tasks(sorted_tasks[i], sorted_tasks[i+1], route.vehicle)
                total_duration += travel_time
        
        return HoSSimulationResult(
            is_feasible=True,
            total_duration=total_duration,
            driving_time=total_duration - sum(getattr(task, 'service_time', 0) for task in sorted_tasks),
            working_time=total_duration,
            break_time=0.0,
            rest_time=0.0,
            events=[],
            violations=[],
            reason="B license - exempt from HoS regulations"
        )
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


def build_compliant_timeline(route: 'Route') -> Tuple[List[SimulatedEvent], float]:
    """
    Build a legally compliant timeline by simulating a route and inserting all mandatory rests.
    
    This is the core of the new two-stage HoS simulation engine. It creates a detailed
    timeline of all events including travel, work, waits, and mandatory rests without
    making any feasibility judgments about time windows.
    
    Args:
        route: The route to simulate
        
    Returns:
        Tuple of (timeline_events, total_rest_cost)
        - timeline_events: List of SimulatedEvent objects representing the complete timeline
        - total_rest_cost: Total cost of all inserted rests
    """
    if not route.tasks or len(route.tasks) < 2:
        return [], 0.0
    
    # Initialize simulation state
    timeline = []
    total_rest_cost = 0.0
    driver_cost_per_minute = (route.vehicle.cost_per_hour / 60.0) if route.vehicle else (25.0 / 60.0)
    
    # Initialize driver state - use route's driver state if available, otherwise create new
    if hasattr(route, 'driver_state') and route.driver_state:
        driver_state = copy.deepcopy(route.driver_state)
    else:
        driver_state = DriverState()
    
    # B license drivers are exempt from HoS regulations
    if route.driver and hasattr(route.driver, 'license') and route.driver.license == 'B':
        # For B license drivers, just create basic timeline without HoS constraints
        current_time = 0.0
        for i in range(len(route.tasks) - 1):
            start_task = route.tasks[i]
            end_task = route.tasks[i + 1]
            
            # Service time at start task
            if start_task.service_time > 0:
                timeline.append(SimulatedEvent(
                    event_type='WORK',
                    start_time=current_time,
                    end_time=current_time + start_task.service_time,
                    duration=start_task.service_time,
                    description=f"Service at {start_task.id}",
                    location=getattr(start_task, 'location', None),
                    task_id=start_task.id
                ))
                current_time += start_task.service_time
            
            # Travel time to next task
            travel_time = _calculate_travel_time_for_simulation(start_task, end_task, route.vehicle)
            if travel_time > 0:
                timeline.append(SimulatedEvent(
                    event_type='DRIVE',
                    start_time=current_time,
                    end_time=current_time + travel_time,
                    duration=travel_time,
                    description=f"Drive from {start_task.id} to {end_task.id}",
                    task_id=f"{start_task.id}->{end_task.id}"
                ))
                current_time += travel_time
                
        return timeline, 0.0
    
    # For CE license drivers, perform full HoS simulation
    current_time = 0.0
    current_day = getattr(route.tasks[0], 'day', 0) if route.tasks else 0
    
    # --- START: NEW LOGIC TO INSERT ---
    # Correctly initialize current_time by handling the initial depot wait.
    if len(route.tasks) > 1:
        first_real_task = route.tasks[1] # tasks[0] is the depot start
        
        # Use 'earliest_start_time' if available, fall back to 'earliest_time'
        task_start_time = getattr(first_real_task, 'earliest_start_time', getattr(first_real_task, 'earliest_time', 0))

        if task_start_time and task_start_time > 0:
            # Calculate travel time from depot to the first real task
            travel_to_first_task = _calculate_travel_time_for_simulation(route.tasks[0], first_real_task, route.vehicle)
            
            # The driver must depart to arrive on time
            required_departure_time = task_start_time - travel_to_first_task
            
            if required_departure_time > current_time:
                wait_duration = required_departure_time - current_time
                
                # This is a non-working, unpaid wait at the depot
                timeline.append(SimulatedEvent(
                    event_type='WAIT',
                    start_time=current_time,
                    end_time=required_departure_time,
                    duration=wait_duration,
                    description="Initial wait at depot before departure",
                    location=getattr(route.tasks[0], 'location', 'Depot'),
                    task_id=route.tasks[0].id,
                    cost=0.0 # Depot wait has no cost
                ))
                
                # Advance the simulation clock to the required departure time
                current_time = required_departure_time
    # --- END: NEW LOGIC TO INSERT ---
    
    for i in range(len(route.tasks) - 1):
        start_task = route.tasks[i]
        end_task = route.tasks[i + 1]
        
        # Check for day transition
        task_day = getattr(start_task, 'day', 0)
        if task_day != current_day:
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
        
        # Simulate waiting time if needed
        earliest_start = getattr(start_task, 'earliest_start_time', None) or getattr(start_task, 'earliest_time', None)
        
        if (earliest_start is not None and current_time < earliest_start):
            
            wait_duration = earliest_start - current_time
            
            # Determine if this is depot waiting or customer waiting
            is_depot_waiting = (hasattr(start_task, 'is_depot_start') and start_task.is_depot_start()) or \
                              (hasattr(start_task, 'task_type') and 'depot' in str(start_task.task_type).lower())
            
            if not is_depot_waiting:
                # Customer waiting - check if it can count as a break
                rest_applied = False
                remaining_wait = wait_duration
                wait_start_time = current_time
                
                # For long waits (11+ hours), always use as daily rest if driver has done significant work
                if wait_duration >= HoSRegulations.MIN_DAILY_REST and driver_state.work_today > 0:
                    # Use wait time as 11-hour daily rest
                    rest_duration = HoSRegulations.MIN_DAILY_REST  # Only take 11 hours, not the full wait
                    timeline.append(SimulatedEvent(
                        event_type='REST',
                        start_time=current_time,
                        end_time=current_time + rest_duration,
                        duration=rest_duration,
                        description=f"Daily rest during wait at {start_task.id}",
                        rest_type='11h_daily',
                        location=getattr(start_task, 'location', None),
                        task_id=start_task.id,
                        cost=rest_duration * driver_cost_per_minute
                    ))
                    
                    driver_state.reset_daily()
                    current_time += rest_duration
                    remaining_wait -= rest_duration
                    total_rest_cost += rest_duration * driver_cost_per_minute
                    rest_applied = True
                
                # For moderate waits (45+ minutes), use as break if driver needs a driving break
                elif (wait_duration >= HoSRegulations.MIN_BREAK_DURATION and 
                      driver_state.drive_since_break > 0):
                    # Use wait time as 45-minute break
                    rest_duration = HoSRegulations.MIN_BREAK_DURATION  # Only take 45 minutes, not the full wait
                    timeline.append(SimulatedEvent(
                        event_type='REST',
                        start_time=current_time,
                        end_time=current_time + rest_duration,
                        duration=rest_duration,
                        description=f"45-minute break during wait at {start_task.id}",
                        rest_type='45min_break',
                        location=getattr(start_task, 'location', None),
                        task_id=start_task.id,
                        cost=rest_duration * driver_cost_per_minute
                    ))
                    
                    driver_state.drive_since_break = 0.0
                    current_time += rest_duration
                    remaining_wait -= rest_duration
                    total_rest_cost += rest_duration * driver_cost_per_minute
                    rest_applied = True
                
                # If there's remaining wait time after rest, count it as regular wait (not work time)
                if remaining_wait > 0:
                    timeline.append(SimulatedEvent(
                        event_type='WAIT',
                        start_time=current_time,
                        end_time=earliest_start,
                        duration=remaining_wait,
                        description=f"Wait at customer location {start_task.id}",
                        location=getattr(start_task, 'location', None),
                        task_id=start_task.id
                    ))
                    # Note: Only short waits (under 45 minutes) count as work time
                    if not rest_applied and wait_duration < HoSRegulations.MIN_BREAK_DURATION:
                        driver_state.work_today += remaining_wait
                        driver_state.total_work_this_week += remaining_wait
                
                # If no rest was applied and wait is short, count as work time
                elif not rest_applied:
                    timeline.append(SimulatedEvent(
                        event_type='WAIT',
                        start_time=current_time,
                        end_time=earliest_start,
                        duration=wait_duration,
                        description=f"Wait at customer location {start_task.id}",
                        location=getattr(start_task, 'location', None),
                        task_id=start_task.id
                    ))
                    
                    driver_state.work_today += wait_duration
                    driver_state.total_work_this_week += wait_duration
            else:
                # Depot waiting - driver shift hasn't started yet
                timeline.append(SimulatedEvent(
                    event_type='WAIT',
                    start_time=current_time,
                    end_time=earliest_start,
                    duration=wait_duration,
                    description=f"Wait at depot before shift start",
                    location=getattr(start_task, 'location', None),
                    task_id=start_task.id
                ))
            
            current_time = earliest_start
        
        # Simulate service time at start task
        if start_task.service_time > 0:
            timeline.append(SimulatedEvent(
                event_type='WORK',
                start_time=current_time,
                end_time=current_time + start_task.service_time,
                duration=start_task.service_time,
                description=f"Service at {start_task.id}",
                location=getattr(start_task, 'location', None),
                task_id=start_task.id
            ))
            
            driver_state.work_today += start_task.service_time
            driver_state.total_work_this_week += start_task.service_time
            current_time += start_task.service_time
        
        # Simulate travel time with potential interruptions for mandatory rests
        travel_time = _calculate_travel_time_for_simulation(start_task, end_task, route.vehicle)
        travel_time_remaining = travel_time
        
        while travel_time_remaining > 0:
            # Calculate maximum time the driver can legally drive before needing a break
            max_drive_before_break = HoSRegulations.MAX_DRIVE_WITHOUT_BREAK - driver_state.drive_since_break
            max_drive_before_daily_limit = HoSRegulations.MAX_DRIVE_PER_DAY - driver_state.drive_today
            max_work_before_daily_limit = HoSRegulations.MAX_WORK_PER_DAY - driver_state.work_today
            
            drivable_time = min(max_drive_before_break, max_drive_before_daily_limit, 
                              max_work_before_daily_limit, travel_time_remaining)
            
            if drivable_time > 0:
                # Simulate driving for the calculated time
                timeline.append(SimulatedEvent(
                    event_type='DRIVE',
                    start_time=current_time,
                    end_time=current_time + drivable_time,
                    duration=drivable_time,
                    description=f"Drive from {start_task.id} to {end_task.id}",
                    task_id=f"{start_task.id}->{end_task.id}"
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
                    # work_since_break would also reset but we don't track it in this simple version
                elif rest_type == '11h_daily':
                    driver_state.reset_daily()
                elif rest_type == '45h_weekly':
                    # Reset weekly counters
                    driver_state.drive_this_week = 0.0
                    driver_state.total_work_this_week = 0.0
                    driver_state.reset_daily()
    
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
    try:
        # Try to use the route provider if available
        from route_provider import calculate_travel_time_between_tasks
        return calculate_travel_time_between_tasks(task1, task2, vehicle)
    except ImportError:
        # Fallback to the same calculation used in hos_simulation.py
        return calculate_travel_time_between_tasks(task1, task2, vehicle)


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
