"""
EPDT Heuristic Algorithm Data Structures

This module defines the core data structures used by the EPDT (Enhanced Parallel 
Diversified Tabu) algorithm for Vehicle Routing Problem with Pickup and Delivery.

These structures are designed to work with the L1 (First-Level) and L2 (Second-Level)
heuristics as described in the EPDT academic paper.

Key Classes:
- Task: Individual pickup or delivery action
- Order: Collection of related pickup and delivery tasks  
- Vehicle: Fleet vehicle with capacity and regulatory constraints
- Route: Sequence of tasks assigned to a vehicle
- Solution: Complete assignment of orders to vehicle routes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Any
from enum import Enum
from datetime import datetime, timedelta


@dataclass
class HoSEvent:
    """
    Represents a single Hours of Service event for detailed tracking and auditing.
    
    This addresses the feedback about enhancing DriverState with detailed tracking
    of HoS-related events including start/end times of breaks and rests.
    """
    event_type: str  # "break", "daily_rest", "weekly_rest", "drive_start", "drive_end", "work_start", "work_end"
    start_time: float  # Minutes from start of planning period
    end_time: Optional[float] = None  # Minutes from start of planning period
    duration: float = 0.0  # Duration in minutes
    location: Optional[str] = None  # Location where event occurred
    notes: str = ""  # Additional notes for auditing
    regulation_compliance: bool = True  # Whether event meets regulatory requirements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging/export."""
        return {
            'event_type': self.event_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'location': self.location,
            'notes': self.notes,
            'regulation_compliance': self.regulation_compliance
        }


@dataclass
class DriverState:
    """
    Enhanced class to track driver's hours of service state according to European regulations.
    
    Enhanced with detailed event tracking for auditing and compliance reporting,
    addressing the feedback about more comprehensive HoS event tracking.
    
    European HoS Regulations:
    - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins)
    - Maximum 9 hours of driving per day (extendable to 10 hours twice a week)
    - Maximum 13 hours of work per day (extendable to 14 hours twice a week)  
    - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions)
    - Maximum 56 hours driving in a week (90 hours in any two consecutive weeks)
    """
    
    # Current state counters
    drive_since_break: float = 0.0      # Accumulated driving time since last break
    work_since_break: float = 0.0       # Accumulated working time since last break
    drive_today: float = 0.0             # Total driving time in current 24-hour period
    work_today: float = 0.0              # Total duty time in current 24-hour period
    drive_this_week: float = 0.0         # Driving time this week
    drive_last_week: float = 0.0         # Driving time last week
    
    # Additional comprehensive state variables for advanced HoS simulation
    time_in_daily_period: float = 0.0   # Time elapsed since end of last daily rest (max 24h)
    work_this_week: float = 0.0          # Accumulated working time from Monday 00:00
    time_since_weekly_rest: float = 0.0  # Time elapsed since last weekly rest ended (max 144h)
    
    # Extension tracking
    daily_driving_extensions_used: int = 0      # Extensions to 10 hours used this week
    daily_work_extensions_used: int = 0         # Extensions to 14 hours used this week
    reduced_rest_used: int = 0                  # Reduced rest periods used this week
    daily_rest_reductions_used: int = 0         # Count of 9h daily rests between weekly rests (max 3)
    is_weekly_rest_reduction_taken: bool = False  # Flag if reduced weekly rest taken in last two weeks
    
    # Enhanced event tracking (addresses feedback)
    hos_events: List[HoSEvent] = field(default_factory=list)  # Detailed event log
    current_activity: Optional[str] = None  # Current activity: "driving", "working", "resting", "break"
    current_activity_start: Optional[float] = None  # When current activity started
    last_break_start: Optional[float] = None  # When last break started
    last_rest_start: Optional[float] = None   # When last rest started
    
    # Regulation limits in minutes (European HoS)
    MAX_DRIVE_WITHOUT_BREAK = 4.5 * 60         # 4.5 hours
    MAX_WORK_WITHOUT_BREAK = 6 * 60            # 6 hours
    MAX_DRIVE_PER_DAY = 9 * 60                 # 9 hours (extendable to 10)
    MAX_WORK_PER_DAY = 13 * 60                 # 13 hours (extendable to 14)
    MAX_DRIVE_PER_WEEK = 56 * 60               # 56 hours
    MAX_DRIVE_TWO_WEEKS = 90 * 60              # 90 hours in any two consecutive weeks
    MIN_DAILY_REST = 11 * 60                   # 11 hours (reducible to 9)
    MIN_WEEKLY_REST = 45 * 60                  # 45 hours
    
    def start_activity(self, activity_type: str, current_time: float, location: str = None):
        """
        Start a new activity and record the event.
        
        Args:
            activity_type: Type of activity ("driving", "working", "break", "daily_rest", "weekly_rest")
            current_time: Current time in minutes from start of planning period
            location: Optional location information
        """
        # End previous activity if any
        if self.current_activity:
            self.end_activity(current_time)
        
        self.current_activity = activity_type
        self.current_activity_start = current_time
        
        # Record event
        event = HoSEvent(
            event_type=f"{activity_type}_start",
            start_time=current_time,
            location=location,
            notes=f"Started {activity_type}"
        )
        self.hos_events.append(event)
        
        # Update specific tracking
        if activity_type == "break":
            self.last_break_start = current_time
        elif activity_type in ["daily_rest", "weekly_rest"]:
            self.last_rest_start = current_time
    
    def end_activity(self, current_time: float, location: str = None):
        """
        End the current activity and record the event.
        
        Args:
            current_time: Current time in minutes from start of planning period
            location: Optional location information
        """
        if not self.current_activity or self.current_activity_start is None:
            return
        
        duration = current_time - self.current_activity_start
        
        # Record event
        event = HoSEvent(
            event_type=f"{self.current_activity}_end",
            start_time=self.current_activity_start,
            end_time=current_time,
            duration=duration,
            location=location,
            notes=f"Completed {self.current_activity} for {duration:.1f} minutes"
        )
        self.hos_events.append(event)
        
        # Update counters based on activity
        if self.current_activity == "driving":
            self.drive_since_break += duration
            self.drive_today += duration
            self.drive_this_week += duration
            self.work_since_break += duration
            self.work_today += duration
            self.work_this_week += duration
        elif self.current_activity == "working":
            self.work_since_break += duration
            self.work_today += duration
            self.work_this_week += duration
        
        # Clear current activity
        self.current_activity = None
        self.current_activity_start = None
    
    def can_drive(self, duration: float) -> bool:
        """Check if driver can drive for the specified duration without violating HoS."""
        # Check break requirements
        if self.drive_since_break + duration > self.MAX_DRIVE_WITHOUT_BREAK:
            return False
            
        # Check daily limits (considering extensions)
        max_daily = self.get_current_max_daily_drive()
        if self.drive_today + duration > max_daily:
            return False
            
        # Check weekly limits
        if self.drive_this_week + duration > self.MAX_DRIVE_PER_WEEK:
            return False
            
        # Check two-week limits
        if self.drive_this_week + self.drive_last_week + duration > self.MAX_DRIVE_TWO_WEEKS:
            return False
            
        return True
    
    def can_work(self, duration: float) -> bool:
        """Check if driver can work for the specified duration without violating HoS."""
        # Check work time since last break
        if self.work_since_break + duration > self.MAX_WORK_WITHOUT_BREAK:
            return False
            
        # Check daily work limits (considering extensions)
        max_daily_work = self.get_current_max_daily_work()
        if self.work_today + duration > max_daily_work:
            return False
            
        return True
    
    def get_current_max_daily_drive(self) -> float:
        """Get current maximum daily driving time considering extensions."""
        if self.daily_driving_extensions_used < 2:
            return 10 * 60  # Can extend to 10 hours
        return self.MAX_DRIVE_PER_DAY  # 9 hours
        
    def get_current_max_daily_work(self) -> float:
        """Get current maximum daily work time considering extensions."""
        if self.daily_work_extensions_used < 2:
            return 14 * 60  # Can extend to 14 hours
        return self.MAX_WORK_PER_DAY  # 13 hours

    def take_break(self, break_duration: float, current_time: float = None, location: str = None):
        """
        Enhanced break handling with event tracking.
        
        Args:
            break_duration: Duration of break in minutes
            current_time: Current time for event tracking
            location: Location where break is taken
        """
        # Record break event with compliance check
        compliance = break_duration >= 45 or (break_duration >= 15 and self._has_recent_partial_break())
        
        if current_time is not None:
            event = HoSEvent(
                event_type="break",
                start_time=current_time,
                end_time=current_time + break_duration,
                duration=break_duration,
                location=location,
                notes=f"Break taken: {break_duration} minutes",
                regulation_compliance=compliance
            )
            self.hos_events.append(event)
        
        # Apply break effects
        if break_duration >= 45:  # 45-minute break resets driving
            self.drive_since_break = 0
            if break_duration >= 45:  # Also resets work counter
                self.work_since_break = 0
        elif break_duration >= 15:  # Partial break (split break system)
            # European regulations allow split breaks (15 + 30 minutes)
            self.drive_since_break = max(0, self.drive_since_break - break_duration * 0.5)
    
    def _has_recent_partial_break(self) -> bool:
        """Check if there was a recent partial break for split break validation."""
        # Look for breaks in last 2 hours
        recent_breaks = [e for e in self.hos_events[-10:] if e.event_type == "break" and e.duration >= 15]
        return len(recent_breaks) > 0
        
    def take_daily_rest(self, rest_duration: float = None, current_time: float = None, location: str = None):
        """
        Enhanced daily rest handling with event tracking.
        
        Args:
            rest_duration: Duration of rest in minutes (default: minimum required)
            current_time: Current time for event tracking
            location: Location where rest is taken
        """
        if rest_duration is None:
            rest_duration = self.MIN_DAILY_REST
        
        # Record rest event with compliance check
        compliance = rest_duration >= 9 * 60  # Minimum reduced rest
        
        if current_time is not None:
            event = HoSEvent(
                event_type="daily_rest",
                start_time=current_time,
                end_time=current_time + rest_duration,
                duration=rest_duration,
                location=location,
                notes=f"Daily rest: {rest_duration/60:.1f} hours",
                regulation_compliance=compliance
            )
            self.hos_events.append(event)
            
        # Check if this was an extension day
        if self.drive_today > self.MAX_DRIVE_PER_DAY:
            self.daily_driving_extensions_used += 1
        if self.work_today > self.MAX_WORK_PER_DAY:
            self.daily_work_extensions_used += 1
            
        # Track reduced rest usage
        if rest_duration < self.MIN_DAILY_REST:
            self.reduced_rest_used += 1
            
        # Reset daily counters
        self.drive_since_break = 0
        self.work_since_break = 0
        self.drive_today = 0
        self.work_today = 0
        
    def take_weekly_rest(self, rest_duration: float = None, current_time: float = None, location: str = None):
        """
        Enhanced weekly rest handling with event tracking.
        
        Args:
            rest_duration: Duration of rest in minutes (default: minimum required)
            current_time: Current time for event tracking
            location: Location where rest is taken
        """
        if rest_duration is None:
            rest_duration = self.MIN_WEEKLY_REST
        
        # Record rest event
        if current_time is not None:
            event = HoSEvent(
                event_type="weekly_rest",
                start_time=current_time,
                end_time=current_time + rest_duration,
                duration=rest_duration,
                location=location,
                notes=f"Weekly rest: {rest_duration/60:.1f} hours",
                regulation_compliance=rest_duration >= self.MIN_WEEKLY_REST
            )
            self.hos_events.append(event)
        
        # Reset weekly counters
        self.drive_last_week = self.drive_this_week
        self.drive_this_week = 0
        
        # Reset weekly extension counters
        self.daily_driving_extensions_used = 0
        self.daily_work_extensions_used = 0
        self.reduced_rest_used = 0
        
        self.take_daily_rest(rest_duration, current_time, location)
    
    def get_event_log(self, event_types: Optional[List[str]] = None) -> List[HoSEvent]:
        """
        Get filtered event log for auditing and reporting.
        
        Args:
            event_types: Optional list of event types to filter by
            
        Returns:
            List of HoS events matching the filter criteria
        """
        if event_types is None:
            return self.hos_events.copy()
        
        return [event for event in self.hos_events if event.event_type in event_types]
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate a compliance report based on tracked events.
        
        Returns:
            Dictionary containing compliance statistics and violations
        """
        violations = []
        warnings = []
        
        # Check for break violations
        break_events = self.get_event_log(["break"])
        insufficient_breaks = [e for e in break_events if not e.regulation_compliance]
        if insufficient_breaks:
            violations.append(f"Insufficient breaks: {len(insufficient_breaks)} events")
        
        # Check for rest violations
        rest_events = self.get_event_log(["daily_rest", "weekly_rest"])
        insufficient_rests = [e for e in rest_events if not e.regulation_compliance]
        if insufficient_rests:
            violations.append(f"Insufficient rest periods: {len(insufficient_rests)} events")
        
        # Check current state against limits
        if self.drive_today > self.get_current_max_daily_drive():
            violations.append("Daily driving limit exceeded")
        if self.work_today > self.get_current_max_daily_work():
            violations.append("Daily work limit exceeded")
        if self.drive_this_week > self.MAX_DRIVE_PER_WEEK:
            violations.append("Weekly driving limit exceeded")
        
        # Warnings for approaching limits
        if self.drive_today > self.MAX_DRIVE_PER_DAY * 0.9:
            warnings.append("Approaching daily driving limit")
        if self.drive_since_break > self.MAX_DRIVE_WITHOUT_BREAK * 0.9:
            warnings.append("Break needed soon")
        
        return {
            'total_events': len(self.hos_events),
            'violations': violations,
            'warnings': warnings,
            'compliance_score': max(0, 100 - len(violations) * 20 - len(warnings) * 5),
            'extensions_used': {
                'daily_driving': self.daily_driving_extensions_used,
                'daily_work': self.daily_work_extensions_used,
                'reduced_rest': self.reduced_rest_used
            }
        }
    
    def export_events_to_dict(self) -> List[Dict[str, Any]]:
        """Export all events to a list of dictionaries for external reporting."""
        return [event.to_dict() for event in self.hos_events]


@dataclass
class DriverState:
    """
    Enhanced class to track driver's hours of service state according to European regulations.
    
    European HoS Regulations:
    - After 4.5 hours of driving, a 45-minute break is mandatory (can be split into 15 + 30 mins)
    - Maximum 9 hours of driving per day (extendable to 10 hours twice a week)
    - Maximum 13 hours of work per day (extendable to 14 hours twice a week)  
    - Minimum 11 hours of daily rest (can be reduced to 9 hours under certain conditions)
    - Maximum 56 hours driving in a week (90 hours in any two consecutive weeks)
    """
    
    # Current state counters
    drive_since_break: float = 0.0      # Accumulated driving time since last break
    work_since_break: float = 0.0       # Accumulated working time since last break
    drive_today: float = 0.0             # Total driving time in current 24-hour period
    work_today: float = 0.0              # Total duty time in current 24-hour period
    drive_this_week: float = 0.0         # Driving time this week
    drive_last_week: float = 0.0         # Driving time last week
    
    # Additional comprehensive state variables for advanced HoS simulation
    time_in_daily_period: float = 0.0   # Time elapsed since end of last daily rest (max 24h)
    work_this_week: float = 0.0          # Accumulated working time from Monday 00:00
    time_since_weekly_rest: float = 0.0  # Time elapsed since last weekly rest ended (max 144h)
    
    # Extension tracking
    daily_driving_extensions_used: int = 0      # Extensions to 10 hours used this week
    daily_work_extensions_used: int = 0         # Extensions to 14 hours used this week
    reduced_rest_used: int = 0                  # Reduced rest periods used this week
    daily_rest_reductions_used: int = 0         # Count of 9h daily rests between weekly rests (max 3)
    is_weekly_rest_reduction_taken: bool = False  # Flag if reduced weekly rest taken in last two weeks
    
    # Regulation limits in minutes (European HoS)
    MAX_DRIVE_WITHOUT_BREAK = 4.5 * 60         # 4.5 hours
    MAX_WORK_WITHOUT_BREAK = 6 * 60            # 6 hours
    MAX_DRIVE_PER_DAY = 9 * 60                 # 9 hours (extendable to 10)
    MAX_WORK_PER_DAY = 13 * 60                 # 13 hours (extendable to 14)
    MAX_DRIVE_PER_WEEK = 56 * 60               # 56 hours
    MAX_DRIVE_TWO_WEEKS = 90 * 60              # 90 hours in any two consecutive weeks
    MIN_DAILY_REST = 11 * 60                   # 11 hours (reducible to 9)
    MIN_WEEKLY_REST = 45 * 60                  # 45 hours
    
    def can_drive(self, duration: float) -> bool:
        """Check if driver can drive for the specified duration without violating HoS."""
        # Check break requirements
        if self.drive_since_break + duration > self.MAX_DRIVE_WITHOUT_BREAK:
            return False
            
        # Check daily limits (considering extensions)
        max_daily = self.get_current_max_daily_drive()
        if self.drive_today + duration > max_daily:
            return False
            
        # Check weekly limits
        if self.drive_this_week + duration > self.MAX_DRIVE_PER_WEEK:
            return False
            
        # Check two-week limits
        if self.drive_this_week + self.drive_last_week + duration > self.MAX_DRIVE_TWO_WEEKS:
            return False
            
        return True
    
    def can_work(self, duration: float) -> bool:
        """Check if driver can work for the specified duration without violating HoS."""
        # Check work time since last break
        if self.work_since_break + duration > self.MAX_WORK_WITHOUT_BREAK:
            return False
            
        # Check daily work limits (considering extensions)
        max_daily_work = self.get_current_max_daily_work()
        if self.work_today + duration > max_daily_work:
            return False
            
        return True
    
    def get_current_max_daily_drive(self) -> float:
        """Get current maximum daily driving time considering extensions."""
        if self.daily_driving_extensions_used < 2:
            return 10 * 60  # Can extend to 10 hours
        return self.MAX_DRIVE_PER_DAY  # 9 hours
        
    def get_current_max_daily_work(self) -> float:
        """Get current maximum daily work time considering extensions."""
        if self.daily_work_extensions_used < 2:
            return 14 * 60  # Can extend to 14 hours
        return self.MAX_WORK_PER_DAY  # 13 hours

    def take_break(self, break_duration: float):
        """Reset counters after taking a break."""
        if break_duration >= 45:  # 45-minute break resets driving
            self.drive_since_break = 0
            if break_duration >= 45:  # Also resets work counter
                self.work_since_break = 0
        elif break_duration >= 15:  # Partial break (split break system)
            # European regulations allow split breaks (15 + 30 minutes)
            self.drive_since_break = max(0, self.drive_since_break - break_duration * 0.5)
        
    def take_daily_rest(self, rest_duration: float = None):
        """Reset daily counters after taking a daily rest."""
        if rest_duration is None:
            rest_duration = self.MIN_DAILY_REST
            
        # Check if this was an extension day
        if self.drive_today > self.MAX_DRIVE_PER_DAY:
            self.daily_driving_extensions_used += 1
        if self.work_today > self.MAX_WORK_PER_DAY:
            self.daily_work_extensions_used += 1
            
        # Track reduced rest usage
        if rest_duration < self.MIN_DAILY_REST:
            self.reduced_rest_used += 1
            
        # Reset daily counters
        self.drive_since_break = 0
        self.work_since_break = 0
        self.drive_today = 0
        self.work_today = 0
        
    def take_weekly_rest(self):
        """Reset weekly counters after taking a weekly rest."""
        self.drive_last_week = self.drive_this_week
        self.drive_this_week = 0
        
        # Reset weekly extension counters
        self.daily_driving_extensions_used = 0
        self.daily_work_extensions_used = 0
        self.reduced_rest_used = 0
        
        self.take_daily_rest()


@dataclass
class Driver:
    """
    Represents a driver with qualifications, hours of service state, and assignment preferences.
    
    This class decouples driver-specific constraints from vehicles, allowing proper modeling
    of real-world scenarios where drivers can operate multiple vehicles and vice versa.
    """
    id: str
    name: str
    license: str  # 'B' or 'CE'
    default_vehicle_id: str
    cost_per_hour: float = 25.0  # Hourly cost for this driver (wage)
    max_shift_hours: float = 13.0  # Maximum total work hours for a single shift
    max_driving_hours: float = 9.0  # Maximum driving hours within a shift
    home_depot_id: str = "main_depot"  # Default depot
    qualifications: Set[str] = field(default_factory=set)  # Driver capabilities (e.g., ADR_CERTIFIED, FORKLIFT_LICENSE)
    hos_state: DriverState = field(default_factory=DriverState)
    
    def can_operate_vehicle(self, vehicle: 'Vehicle') -> bool:
        """Check if driver can legally operate the given vehicle."""
        # License check for heavy trucks
        if vehicle.vehicle_type == 'heavy' and self.license != 'CE':
            return False
        elif vehicle.vehicle_type in ['standard', 'car'] and self.license not in ['B', 'CE']:
            return False
        
        # Check qualifications for special vehicle capabilities
        required_qualifications = vehicle.capabilities
        if not self.qualifications.issuperset(required_qualifications):
            return False
            
        return True


class TaskType(Enum):
    """Types of tasks in the VRP."""
    PICKUP = "pickup"
    DELIVERY = "delivery"
    DEPOT_START = "depot_start"
    DEPOT_RETURN = "depot_return"


@dataclass
class Task:
    """
    Represents an individual pickup or delivery task.
    
    A task is the atomic unit of work in the EPDT algorithm - it represents
    either picking up cargo from a location or delivering cargo to a location.
    """
    id: str
    location_id: str
    task_type: TaskType
    order_id: str  # Which order this task belongs to
    
    # Location and timing
    lat: float
    lon: float
    service_time: float  # Time required to perform the task (minutes)
    
    # Time window constraints
    earliest_time: Optional[float] = None  # Earliest allowed start time (minutes from start)
    latest_time: Optional[float] = None    # Latest allowed start time (minutes from start)
    soft_time_window: bool = False          # Whether time window is soft (allows violations with penalty)
    
    # Cargo information
    demand: float = 0.0      # Cargo weight (kg) - positive for pickup, negative for delivery
    volume: float = 0.0      # Cargo volume (m³) - positive for pickup, negative for delivery
    pallets: int = 0         # Number of pallets
    
    # Task properties
    is_fixed: bool = False   # Whether task position in route can be changed
    priority: int = 1        # Task priority (higher = more important)
    requires_low_temp: bool = False # Requires low temperature vehicle
    requires_loader: bool = False   # Requires a vehicle with a loader

    # Day of week constraints
    day: int = 0  # Day of week (0=Today, 1=Tomorrow, -1= Yesterday, etc.)
    
    def is_pickup(self) -> bool:
        """Returns True if this is a pickup task."""
        return self.task_type == TaskType.PICKUP
    
    def is_delivery(self) -> bool:
        """Returns True if this is a delivery task."""
        return self.task_type == TaskType.DELIVERY
    
    def is_depot_return(self) -> bool:
        """Returns True if this is a depot return task."""
        return self.task_type == TaskType.DEPOT_RETURN
    
    def is_depot_start(self) -> bool:
        """Returns True if this is a depot start task."""
        return self.task_type == TaskType.DEPOT_START
    
    @staticmethod
    def create_depot_return_task(vehicle_id: str, depot_location_id: str, 
                                depot_lat: float, depot_lon: float,
                                service_time: float = 5.0) -> 'Task':
        """
        Create a depot return task for a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle returning to depot
            depot_location_id: ID of the depot location
            depot_lat: Depot latitude
            depot_lon: Depot longitude
            service_time: Time to check in at depot (default 5 minutes)
            
        Returns:
            Task: A depot return task
        """
        return Task(
            id=f"depot_return_{vehicle_id}",
            location_id=depot_location_id,
            task_type=TaskType.DEPOT_RETURN,
            order_id=f"depot_return_order_{vehicle_id}",  # Special order for depot returns
            lat=depot_lat,
            lon=depot_lon,
            service_time=service_time,
            demand=0.0,  # No cargo change
            volume=0.0,  # No volume change
            earliest_time=None,  # No time constraints for depot return
            latest_time=None,
            soft_time_window=True  # Flexible timing for depot returns
        )

    @staticmethod
    def create_depot_start_task(vehicle_id: str, depot_location_id: str, 
                               depot_lat: float, depot_lon: float,
                               service_time: float = 5.0) -> 'Task':
        """
        Create a depot start task for a vehicle.
        
        Args:
            vehicle_id: ID of the vehicle starting from depot
            depot_location_id: ID of the depot location
            depot_lat: Depot latitude
            depot_lon: Depot longitude
            service_time: Time to check out from depot (default 5 minutes)
            
        Returns:
            Task: A depot start task
        """
        return Task(
            id=f"depot_start_{vehicle_id}",
            location_id=depot_location_id,
            task_type=TaskType.DEPOT_START,
            order_id=f"depot_start_order_{vehicle_id}",  # Special order for depot starts
            lat=depot_lat,
            lon=depot_lon,
            service_time=service_time,
            demand=0.0,  # No cargo change
            volume=0.0,  # No volume change
            earliest_time=None,  # No time constraints for depot start
            latest_time=None,
            soft_time_window=True  # Flexible timing for depot starts
        )


@dataclass  
class Order:
    """
    Represents a complete order with pickup and delivery requirements.
    
    An order consists of one or more pickup tasks and one or more delivery tasks.
    The EPDT algorithm ensures precedence constraints: all pickups must occur
    before their corresponding deliveries.
    """
    id: str
    pickup_tasks: List[Task] = field(default_factory=list)
    delivery_tasks: List[Task] = field(default_factory=list)
    
    # Order properties
    priority: int = 1           # Order priority for scheduling
    revenue: float = 0.0        # Revenue generated by completing this order
    is_urgent: bool = False     # Whether this is an urgent order
    is_mandatory: bool = True   # Whether this order must be assigned
    is_fixed: bool = False      # Whether this order's assignment is fixed (for multi-day planning)
    
    # Preferred assignment constraints
    preferred_vehicle_ids: Set[str] = field(default_factory=set)  # Preferred vehicles
    forbidden_vehicle_ids: Set[str] = field(default_factory=set)  # Forbidden vehicles
    
    # Time constraints
    earliest_pickup: Optional[float] = None  # Earliest pickup time across all tasks
    latest_delivery: Optional[float] = None  # Latest delivery time across all tasks
    
    def get_pickups(self) -> List[Task]:
        """Get all pickup tasks for this order."""
        return self.pickup_tasks
    
    def get_deliveries(self) -> List[Task]:
        """Get all delivery tasks for this order."""
        return self.delivery_tasks
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks (pickups + deliveries) for this order."""
        return self.pickup_tasks + self.delivery_tasks
    
    def get_total_demand(self) -> float:
        """Get total cargo weight for this order."""
        return sum(task.demand for task in self.pickup_tasks)
    
    def get_total_volume(self) -> float:
        """Get total cargo volume for this order."""
        return sum(task.volume for task in self.pickup_tasks)


@dataclass
class Vehicle:
    """
    Represents a vehicle in the fleet with capacity and regulatory constraints.
    
    This matches the vehicle constraints from the EPDT paper including
    capacity limits, time restrictions, and driver regulation compliance.
    """
    id: str
    depot_id: str
    
    # Capacity constraints
    weight_capacity: float      # Maximum weight capacity (kg)
    volume_capacity: float      # Maximum volume capacity (m³)
    pallet_capacity: Optional[int] = None # Maximum pallet capacity
    
    # Time and distance constraints  
    max_time: Optional[float] = None       # Maximum route time (minutes)
    max_distance: Optional[float] = None   # Maximum route distance (km)
    
    # Cost parameters
    cost_per_km: float = 1.0              # Cost per kilometer
    cost_per_hour: float = 50.0           # Cost per hour (for break time calculation)
    fixed_cost: float = 0.0               # Fixed cost for using this vehicle
    
    # Vehicle type and capabilities
    vehicle_type: str = "standard"         # Vehicle type: "car", "standard" (light truck), or "heavy" (heavy truck)
    capabilities: Set[str] = field(default_factory=set)  # Special capabilities like 'loader', 'low_temp', 'hangers'
    
    # Driver regulation constraints (Hours of Service)
    max_driving_time: float = 540.0        # Max continuous driving (minutes) - 9 hours
    required_break_time: float = 45.0      # Required break duration (minutes)
    max_work_time: float = 780.0           # Max total work time per day (minutes) - 13 hours
    break_frequency: float = 270.0         # Break required every X minutes - 4.5 hours
    
    # Vehicle preferences
    preferred_order_types: Set[str] = field(default_factory=set)  # Preferred order types

    # Loading and unloading constraints
    lifo_required: bool = False  # Last In First Out loading/unloading for vehicles without side doors
    
    # Multi-day planning support
    initial_state: Optional[Dict[str, Any]] = None  # Vehicle's position and pending tasks from previous day


@dataclass
class Route:
    """
    Represents a sequence of tasks assigned to a vehicle.
    
    This is the core data structure for the L2 (Second-Level) heuristic.
    The route maintains the sequence of tasks and provides methods for
    feasibility checking and cost calculation.
    """
    vehicle: Vehicle
    driver: Optional[Driver] = None  # Initially unassigned
    tasks: List[Task] = field(default_factory=list)
    
    # Route metadata
    route_id: Optional[str] = None
    
    # Cached calculations (reset when route changes)
    _cached_score: Optional[float] = field(default=None, init=False)
    _cached_distance: Optional[float] = field(default=None, init=False) 
    _cached_time: Optional[float] = field(default=None, init=False)
    _is_feasible_cached: Optional[bool] = field(default=None, init=False)
    
    def _enforce_pickup_first_ordering(self) -> None:
        """
        IMPLEMENTING THESIS REQUIREMENT (Chapter 3, Section 3.3.4 and Figure 3.2):
        Enforce strict "all pickups before all deliveries" precedence constraint for the entire route.
        
        This implementation follows the EPDT problem definition from the thesis:
        - All pickup operations for all assigned orders must be completed first
        - Only after ALL pickups are done can delivery operations begin
        - This ensures efficient vehicle utilization and proper LIFO/FIFO constraints
        """
        pickups = []
        deliveries = []
        others = []
        
        # Separate tasks by type
        for task in self.tasks:
            if hasattr(task, 'task_type'):
                if task.task_type == TaskType.PICKUP:
                    pickups.append(task)
                elif task.task_type == TaskType.DELIVERY:
                    deliveries.append(task)
                else:
                    others.append(task)
            else:
                # If no task_type, assume it's a depot task or other
                others.append(task)
        
        # Apply strict ordering: ALL pickups first, then ALL deliveries
        # This follows the thesis requirement and prevents bouncing between depot and delivery locations
        if pickups or deliveries:
            # Create optimized sequence: others + all_pickups + all_deliveries
            optimized_sequence = others + pickups + deliveries
            self.tasks = optimized_sequence
            
            # Debug: print reordering information
            # print(f"DEBUG: Applied strict pickup-first ordering: {len(pickups)} pickups → {len(deliveries)} deliveries")
    
    def _validate_order_precedence(self, task_sequence: List) -> bool:
        """
        Validate that each order's pickup comes before its delivery in the given sequence.
        
        Args:
            task_sequence: List of tasks to validate
            
        Returns:
            True if all order precedence constraints are satisfied
        """
        # Build order precedence map
        order_positions = {}
        for i, task in enumerate(task_sequence):
            order_id = getattr(task, 'order_id', None)
            if order_id is None:
                continue
            
            if order_id not in order_positions:
                order_positions[order_id] = {'pickups': [], 'deliveries': []}
            
            if hasattr(task, 'task_type'):
                if task.task_type == TaskType.PICKUP:
                    order_positions[order_id]['pickups'].append(i)
                elif task.task_type == TaskType.DELIVERY:
                    order_positions[order_id]['deliveries'].append(i)
        
        # Check that each order's pickups come before its deliveries
        for order_id, positions in order_positions.items():
            if positions['pickups'] and positions['deliveries']:
                last_pickup_pos = max(positions['pickups'])
                first_delivery_pos = min(positions['deliveries'])
                
                if last_pickup_pos >= first_delivery_pos:
                    return False  # Precedence violated
        
        return True  # All precedence constraints satisfied
    
    def insert_task(self, position: int, task: Task) -> None:
        """Insert a task at the specified position and invalidate caches."""
        self.tasks.insert(position, task)
        self._enforce_pickup_first_ordering()  # Maintain pickup-first order
        self._invalidate_caches()
    
    def insert_task_without_reordering(self, position: int, task: Task) -> None:
        """Insert a task at the specified position without reordering and invalidate caches."""
        self.tasks.insert(position, task)
        # Do NOT call _enforce_pickup_first_ordering() to allow flexible task sequencing
        self._invalidate_caches()
    
    def remove_task(self, position: int) -> Task:
        """Remove and return the task at the specified position."""
        task = self.tasks.pop(position)
        # Removed automatic reordering to allow flexible task sequencing
        self._invalidate_caches()
        return task
    
    def swap_tasks(self, pos1: int, pos2: int) -> None:
        """Swap tasks at two positions."""
        self.tasks[pos1], self.tasks[pos2] = self.tasks[pos2], self.tasks[pos1]
        # Removed automatic reordering to allow flexible task sequencing
        self._invalidate_caches()
    
    def copy(self) -> 'Route':
        """Create a deep copy of this route."""
        import copy
        return copy.deepcopy(self)
    
    def ensure_pickup_first_ordering(self) -> None:
        """Public method to ensure pickup-first ordering."""
        self._enforce_pickup_first_ordering()
        self._invalidate_caches()
    
    def _invalidate_caches(self) -> None:
        """Invalidate all cached calculations."""
        self._cached_score = None
        self._cached_distance = None
        self._cached_time = None
        self._is_feasible_cached = None
    
    def is_feasible(self) -> bool:
        """
        Check if this route is feasible (respects all hard constraints).
        
        This is a placeholder - actual implementation is in second_level.py
        """
        if self._is_feasible_cached is not None:
            return self._is_feasible_cached
        
        # Import here to avoid circular imports
        from second_level import is_feasible
        self._is_feasible_cached = is_feasible(self, debug_feasibility=False)
        return self._is_feasible_cached
    
    def get_orders(self) -> Set[str]:
        """Get set of order IDs in this route (excludes depot start/return tasks)."""
        return {task.order_id for task in self.tasks if not task.is_depot_return() and not task.is_depot_start()}
    
    def get_total_distance(self) -> float:
        """Calculate the total distance of the route in kilometers."""
        if self._cached_distance is not None:
            return self._cached_distance

        if not self.tasks:
            self._cached_distance = 0.0
            return 0.0

        total_distance = 0.0
        
        # Distance from depot to first task
        if hasattr(self.vehicle, 'depot_lat') and hasattr(self.vehicle, 'depot_lon'):
            if self.tasks:
                first_task = self.tasks[0]
                total_distance += self._calculate_distance_between_points(
                    self.vehicle.depot_lat, self.vehicle.depot_lon,
                    first_task.lat, first_task.lon
                )
        
        # Distance between consecutive tasks
        for i in range(len(self.tasks) - 1):
            current_task = self.tasks[i]
            next_task = self.tasks[i + 1]
            
            # Add travel distance to next task
            total_distance += self._calculate_distance_between_points(
                current_task.lat, current_task.lon,
                next_task.lat, next_task.lon
            )
        
        # Return to depot from last task
        if hasattr(self.vehicle, 'depot_lat') and hasattr(self.vehicle, 'depot_lon'):
            if self.tasks:
                last_task = self.tasks[-1]
                total_distance += self._calculate_distance_between_points(
                    last_task.lat, last_task.lon,
                    self.vehicle.depot_lat, self.vehicle.depot_lon
                )
        
        self._cached_distance = total_distance
        return total_distance

    def calculate_travel_time(self, from_task=None, to_task=None) -> float:
        """
        Calculate travel time for this route or between two tasks.
        
        Args:
            from_task: Optional starting task (if None, calculates full route time)
            to_task: Optional ending task (if None, calculates full route time)
            
        Returns:
            Total travel time in minutes
        """
        # If specific tasks provided, calculate travel time between them
        if from_task is not None and to_task is not None:
            return self._calculate_travel_time_between_points(
                from_task.lat, from_task.lon,
                to_task.lat, to_task.lon
            )
        
        # Otherwise calculate full route travel time
        if self._cached_time is not None:
            return self._cached_time
        
        if not self.tasks:
            self._cached_time = 0.0
            return 0.0
        
        total_time = 0.0
        
        # Start from depot to first task
        if hasattr(self.vehicle, 'depot_lat') and hasattr(self.vehicle, 'depot_lon'):
            if self.tasks:
                first_task = self.tasks[0]
                total_time += self._calculate_travel_time_between_points(
                    self.vehicle.depot_lat, self.vehicle.depot_lon,
                    first_task.lat, first_task.lon
                )
        
        # Travel between consecutive tasks
        for i in range(len(self.tasks) - 1):
            current_task = self.tasks[i]
            next_task = self.tasks[i + 1]
            
            # Add service time for current task
            total_time += current_task.service_time
            
            # Add travel time to next task
            total_time += self._calculate_travel_time_between_points(
                current_task.lat, current_task.lon,
                next_task.lat, next_task.lon
            )
        
        # Add service time for last task
        if self.tasks:
            total_time += self.tasks[-1].service_time
        
        # Return to depot from last task
        if hasattr(self.vehicle, 'depot_lat') and hasattr(self.vehicle, 'depot_lon'):
            if self.tasks:
                last_task = self.tasks[-1]
                total_time += self._calculate_travel_time_between_points(
                    last_task.lat, last_task.lon,
                    self.vehicle.depot_lat, self.vehicle.depot_lon
                )
        
        self._cached_time = total_time
        return total_time
    
    def _calculate_travel_time_between_points(self, lat1: float, lon1: float, 
                                           lat2: float, lon2: float) -> float:
        """
        Calculate travel time between two geographic points.
        
        Uses simple Euclidean distance with average speed assumption.
        """
        import math
        
        # Simple Euclidean distance (could be improved with proper geo calculations)
        distance_km = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111.0  # Rough km per degree
        
        # Assume average speed of 50 km/h
        average_speed_kmh = 50.0
        travel_time_hours = distance_km / average_speed_kmh
        travel_time_minutes = travel_time_hours * 60.0
        
        return travel_time_minutes
    
    def _calculate_distance_between_points(self, lat1: float, lon1: float, 
                                         lat2: float, lon2: float) -> float:
        """
        Calculate distance between two geographic points in kilometers.
        
        Uses proper Haversine distance calculation and increments distance counter.
        """
        # Import the proper Haversine function from second_level
        try:
            from second_level import haversine_distance
            return haversine_distance(lat1, lon1, lat2, lon2)
        except ImportError:
            try:
                from .second_level import haversine_distance
                return haversine_distance(lat1, lon1, lat2, lon2)
            except ImportError:
                # Fallback to simple calculation if import fails
                import math
                distance_km = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111.0
                return distance_km


@dataclass
class Solution:
    """
    Represents a complete solution to the VRP.
    
    This is the main data structure for the L1 (First-Level) heuristic.
    It maintains the assignment of orders to vehicle routes and tracks
    unassigned orders.
    """
    routes: Dict[str, Route] = field(default_factory=dict)  # vehicle_id -> Route
    unassigned_orders: Set[str] = field(default_factory=set)  # order_ids not assigned
    
    # Solution metadata
    solution_id: Optional[str] = None
    creation_time: Optional[float] = None
    
    # Cached solution-wide calculations
    _cached_z1_score: Optional[float] = field(default=None, init=False)
    
    def add_route(self, vehicle_id: str, route: Route) -> None:
        """Add or update a route for a vehicle."""
        self.routes[vehicle_id] = route
        self._invalidate_caches()
    
    def remove_route(self, vehicle_id: str) -> Optional[Route]:
        """Remove and return the route for a vehicle."""
        route = self.routes.pop(vehicle_id, None)
        if route:
            self._invalidate_caches()
        return route
    
    def assign_order(self, order_id: str, vehicle_id: str) -> None:
        """Assign an order to a vehicle (removes from unassigned)."""
        self.unassigned_orders.discard(order_id)
        self._invalidate_caches()
    
    def unassign_order(self, order_id: str) -> None:
        """Mark an order as unassigned."""
        self.unassigned_orders.add(order_id)
        self._invalidate_caches()
    
    def copy(self) -> 'Solution':
        """Create a deep copy of this solution."""
        import copy
        return copy.deepcopy(self)
    
    def _invalidate_caches(self) -> None:
        """Invalidate cached solution calculations."""
        self._cached_z1_score = None
    
    def get_z1_score(self) -> float:
        """
        Get the Z1 score for this solution.
        
        This is a placeholder - actual implementation is in first_level.py
        """
        if self._cached_z1_score is not None:
            return self._cached_z1_score
        
        # Import here to avoid circular imports
        try:
            from first_level import calculate_z1_score
            self._cached_z1_score = calculate_z1_score(self)
        except ImportError:
            # Fallback calculation if first_level module not available
            self._cached_z1_score = len(self.get_assigned_orders()) * 100.0
        return self._cached_z1_score
    
    def get_assigned_orders(self) -> Set[str]:
        """Get set of all assigned order IDs."""
        assigned = set()
        for route in self.routes.values():
            assigned.update(route.get_orders())
        return assigned
    
    def get_total_vehicles_used(self) -> int:
        """Get number of vehicles with non-empty routes."""
        return sum(1 for route in self.routes.values() if route.tasks)
    
    def get_vehicle_utilization(self) -> Dict[str, Dict[str, float]]:
        """Get capacity utilization for each vehicle."""
        utilization = {}
        for vehicle_id, route in self.routes.items():
            if not route.tasks:
                utilization[vehicle_id] = {"weight": 0.0, "volume": 0.0}
                continue
                
            # Calculate peak load during route
            current_weight = 0.0
            current_volume = 0.0
            max_weight = 0.0
            max_volume = 0.0
            
            for task in route.tasks:
                current_weight += task.demand
                current_volume += task.volume
                max_weight = max(max_weight, current_weight)
                max_volume = max(max_volume, current_volume)
            
            utilization[vehicle_id] = {
                "weight": max_weight / route.vehicle.weight_capacity,
                "volume": max_volume / route.vehicle.volume_capacity
            }
        
        return utilization


import json

# Algorithm configuration parameters
@dataclass
class EPDTParameters:
    """Configuration parameters for the EPDT algorithm."""
    
    # Tabu Search parameters
    tabu_tenure: int = 10                    # Size of tabu list
    max_non_improving_iterations: int = 50   # M1: Max non-improving iterations
    max_total_iterations: int = 500          # M2: Max total iterations
    
    # Exploration strategy
    exploration_strategy: str = "vnd"        # "vnd" or "cyclic"
    
    # Enhancement flags
    enable_advanced_neighborhoods: bool = True
    enable_granular_search: bool = True
    enable_parallelization: bool = False
    parallel_strategy: str = "PE"            # "PE" (Parallel Evaluation) or "PN" (Parallel Neighborhood)
    
    # Scoring penalties
    vehicle_penalty_per_vehicle: float = 100.0
    unassigned_order_base_penalty: float = 1000.0
    time_window_violation_penalty: float = 50.0
    capacity_violation_penalty: float = 100.0
    Lo: float = 2000.0
    wk_ID: float = 100.0
    wk_IE: float = 100.0
    wk_IF: float = 50.0
    wk_IH: float = 50.0
    wk_IJ: float = 20.0
    M: float = 10000.0
    P_task: float = 100000.0
    P_fleet: float = 100000.0

    # Local search parameters
    local_search_strategy: str = "first_improvement"  # "first_improvement" or "best_improvement"
    
    # Initialization method
    initialization_method: str = "best_insertion"     # "best_insertion" or "round_robin"

    @classmethod
    def load_from_json(cls, file_path: str = "config/epdt_params.json"):
        """Load parameters from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                params_from_json = json.load(f)
            
            # Create a new instance with the loaded parameters
            return cls(**params_from_json)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {file_path}. Using default parameters.")
            return cls()
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Using default parameters.")
            return cls()

    @classmethod
    def load_from_json(cls, file_path: str = "config/epdt_params.json"):
        """Load parameters from a JSON file."""
        import json
        try:
            with open(file_path, 'r') as f:
                params_from_json = json.load(f)
            
            # Create a new instance with the loaded parameters
            return cls(**params_from_json)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {file_path}. Using default parameters.")
            return cls()
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Using default parameters.")
            return cls()
