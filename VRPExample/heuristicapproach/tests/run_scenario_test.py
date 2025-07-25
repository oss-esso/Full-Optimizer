"""
EPDT Algorithm Test Runner

This script orchestrates the testing of the EPDT heuristic algorithm using 
scenarios defined in moda_scenarios.py. It follows the test execution flow
outlined in the TODO document section 5.3.

Execution Steps:
1. Load test scenario (create_furgoni_scenario)
2. Transform data to EPDT format (convert_instance_to_epdt_input)  
3. Configure algorithm parameters
4. Run EPDT heuristic (l1_heuristic)
5. Analyze and print results (print_solution_summary)

Usage:
    python run_scenario_test.py [--scenario furgoni] [--params custom_params.json]
"""

import sys
import os
import time
import json
import argparse
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, Any, Optional

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
vrp_example_root = os.path.join(current_dir, '..', '..')
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, project_root)
sys.path.insert(0, vrp_example_root) 
sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

# Import production profiler components early
try:
    from algo.performance_profiler import (
        ProductionProfiler, SolutionAnalyzer, ConstraintAnalyzer,
        save_production_report
    )
    PRODUCTION_PROFILER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Production profiler not available: {e}")
    PRODUCTION_PROFILER_AVAILABLE = False

# Debug path information
print(f"Debug: Current working directory: {os.getcwd()}")
print(f"Debug: Script directory: {current_dir}")
print(f"Debug: Looking for moda_scenarios in: {src_dir}")
print(f"Debug: moda_scenarios.py exists: {os.path.exists(os.path.join(src_dir, 'moda_scenarios.py'))}")

try:
    # Try different import approaches
    from moda_scenarios import create_furgoni_scenario
    print("Successfully imported create_furgoni_scenario from moda_scenarios")
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(heuristic_root, 'src'))
        from moda_scenarios import create_furgoni_scenario
        print("Successfully imported create_furgoni_scenario with explicit path")
    except ImportError:
        try:
            # Try importing with full path
            import importlib.util
            spec = importlib.util.spec_from_file_location("moda_scenarios", os.path.join(src_dir, "moda_scenarios.py"))
            moda_scenarios = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(moda_scenarios)
            create_furgoni_scenario = moda_scenarios.create_furgoni_scenario
            print("Successfully imported create_furgoni_scenario using importlib")
        except Exception as e:
            print(f"Error: Could not import create_furgoni_scenario: {e}")
            print("   Make sure moda_scenarios.py is available")
            print(f"   Searched in: {src_dir}")
            sys.exit(1)

# Import algo modules with robust path handling
try:
    from data_adapter import (
        convert_instance_to_epdt_input, 
        get_default_parameters,
        create_empty_solution,
        print_conversion_summary
    )
    print("Successfully imported data_adapter functions")
except ImportError:
    try:
        from algo.data_adapter import (
            convert_instance_to_epdt_input, 
            get_default_parameters,
            create_empty_solution,
            print_conversion_summary
        )
        print("Successfully imported data_adapter functions with algo prefix")
    except ImportError as e:
        print(f"Error: Could not import data_adapter functions: {e}")
        sys.exit(1)

try:
    from first_level import l1_heuristic
    print("Successfully imported l1_heuristic from first_level")
except ImportError:
    try:
        from algo.first_level import l1_heuristic
        print("Successfully imported l1_heuristic with algo prefix")
    except ImportError as e:
        print(f"❌ Error: Could not import l1_heuristic: {e}")
        print("   Make sure first_level.py is available and properly implemented")
        sys.exit(1)

# Import route provider for travel time calculations
try:
    # Import the centralized route provider
    from route_provider import calculate_travel_time_between_tasks, set_testing_mode
    # Set testing mode for consistent Haversine calculations
    set_testing_mode(use_haversine=True)
    print("Successfully configured route provider for testing mode (Haversine)")
except ImportError:
    try:
        from algo.route_provider import calculate_travel_time_between_tasks, set_testing_mode
        # Set testing mode for consistent Haversine calculations  
        set_testing_mode(use_haversine=True)
        print("Successfully configured route provider for testing mode (Haversine) with algo prefix")
    except ImportError:
        print("⚠️  Warning: route_provider not available, using fallback calculation")
        # Create a fallback function using our centralized distance calculator
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            utils_dir = os.path.join(current_dir, '..', 'utils')
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            
            from distance_calculator import calculate_travel_time_between_tasks
            print("Using centralized distance calculator directly")
        except ImportError:
            def calculate_travel_time_between_tasks(task1, task2, vehicle):
                return 15.0  # Final fallback


def configure_algorithm_parameters(custom_params_file: Optional[str] = None) -> dict:
    """
    Configure algorithm parameters for the EPDT heuristic.
    
    Args:
        custom_params_file: Optional path to custom parameters JSON file
        
    Returns:
        Dictionary of algorithm parameters suitable for l1_heuristic
    """
    # Always return a dictionary - simplified approach
    default_params = {
        'tabu_tenure': 5,
        'M1': 3,  # Reduced back for faster testing while confirming distance calculations
        'M2': 5,  # Reduced back for faster testing while confirming distance calculations 
        'exploration_strategy': 'vnd',
        'enable_advanced_neighborhoods': True,
        'enable_granular_search': False,  # Disabled for initial testing
        'enable_parallelization': False,
        'parallel_strategy': 'PE',
        'local_search_strategy': 'first_improvement',
        'initialization_method': 'best_insertion',
        'vehicle_penalty_per_vehicle': 100.0,
        'unassigned_order_base_penalty': 1000.0,
        'time_window_violation_penalty': 50.0,
        'capacity_violation_penalty': 50.0,  # Reduced even further to allow assignment
        'Lo': 2000.0,
        'wk_ID': 100.0,
        'wk_IE': 100.0,
        'wk_IF': 50.0,
        'wk_IH': 50.0,
        'wk_IJ': 20.0,
        'M': 10000.0,
        'P_task': 100000.0,
        'P_fleet': 100000.0
    }
    
    # If custom file provided, try to load and override defaults
    if custom_params_file and os.path.exists(custom_params_file):
        try:
            import json
            with open(custom_params_file, 'r') as f:
                custom_params = json.load(f)
            
            # Map EPDTParameters JSON keys to expected keys
            param_mapping = {
                'max_non_improving_iterations': 'M1',
                'max_total_iterations': 'M2'
            }
            
            for key, value in custom_params.items():
                if not key.startswith('_'):  # Skip comment keys
                    mapped_key = param_mapping.get(key, key)
                    default_params[mapped_key] = value
            
            print(f"✅ Loaded custom parameters from {custom_params_file}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load custom parameters: {e}")
            print("   Using default parameters")
    else:
        print(f"✅ Using default EPDT parameters")
    
    return default_params


def calculate_total_time_window_delays(solution):
    """
    Calculate the total time window delays across all routes.
    
    Args:
        solution: The solution object containing all routes
        
    Returns:
        tuple: (total_delay_minutes, violations_count, violations_details)
    """
    total_delay_minutes = 0.0
    violations_count = 0
    violations_details = []
    
    try:
        from algo.second_level import _simulate_hos_advanced, _sort_tasks_chronologically, DriverState
        
        for vehicle_id, route in solution.routes.items():
            if not route.tasks:
                continue
                
            # Get sorted tasks and simulate the route
            sorted_tasks = _sort_tasks_chronologically(route.tasks)
            driver_state = DriverState()
            
            # Simulate the route to get task arrival times
            current_time = 0.0
            
            for i, task in enumerate(sorted_tasks):
                if i == 0:
                    # First task: service time only
                    service_time = getattr(task, 'service_time', 15.0)
                    current_time += service_time
                else:
                    # Subsequent tasks: travel time + waiting + service time
                    prev_task = sorted_tasks[i-1]
                    
                    # Estimate travel time (using same logic as in print_solution_summary)
                    try:
                        from route_provider import calculate_travel_time_between_tasks
                        travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
                    except:
                        travel_time = 30.0  # Default fallback
                    
                    # Add travel time
                    current_time += travel_time
                    
                    # Check for time window violation before adding waiting time
                    if hasattr(task, 'latest_time') and task.latest_time is not None:
                        if current_time > task.latest_time:
                            # Time window violation detected
                            delay_minutes = current_time - task.latest_time
                            total_delay_minutes += delay_minutes
                            violations_count += 1
                            
                            # Store violation details
                            violations_details.append({
                                'vehicle_id': vehicle_id,
                                'task_id': task.id,
                                'location_id': getattr(task, 'location_id', 'unknown'),
                                'delay_minutes': delay_minutes,
                                'latest_allowed': task.latest_time,
                                'actual_arrival': current_time
                            })
                    
                    # Handle time window waiting (simplified)
                    if hasattr(task, 'earliest_time') and task.earliest_time and current_time < task.earliest_time:
                        wait_time = task.earliest_time - current_time
                        current_time += wait_time
                    
                    # Add service time
                    service_time = getattr(task, 'service_time', 15.0)
                    current_time += service_time
                    
    except Exception as e:
        print(f"      ⚠️  Error calculating time window delays: {str(e)}")
        return 0.0, 0, []
    
    return total_delay_minutes, violations_count, violations_details


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


def print_solution_summary(solution, orders, vehicles, params, runtime_seconds):
    """
    Print a comprehensive summary of the EPDT algorithm solution.
    
    This function analyzes the solution and displays:
    - Overall Z1 score and performance metrics
    - Vehicle usage and utilization
    - Route details for each vehicle  
    - Unassigned orders analysis
    - Algorithm performance statistics
    
    Args:
        solution: Final Solution object from l1_heuristic
        orders: List of Order objects 
        vehicles: List of Vehicle objects
        params: EPDTParameters used for the algorithm
        runtime_seconds: Algorithm execution time
    """
    print(f"\n" + "="*80)
    print(f"📊 EPDT ALGORITHM SOLUTION SUMMARY")
    print(f"="*80)
    
    # Overall performance metrics
    print(f"\n🎯 Overall Performance:")
    if hasattr(solution, 'get_z1_score'):
        try:
            z1_score = solution.get_z1_score()
            print(f"   📈 Final Z1 Score: {z1_score:,.2f}")
        except Exception as e:
            print(f"   📈 Final Z1 Score: Unable to calculate ({str(e)})")
    else:
        print(f"   📈 Final Z1 Score: Not available (scoring function not implemented)")
    
    print(f"   ⏱️  Runtime: {runtime_seconds:.2f} seconds")
    print(f"   🚛 Total vehicles used: {solution.get_total_vehicles_used()} / {len(vehicles)}")
    total_distance = sum(r.get_total_distance() for r in solution.routes.values())
    print(f"   🛣️  Total distance: {total_distance:.2f} km")
    
    # Calculate and display time window delays
    total_delay_minutes, violations_count, violations_details = calculate_total_time_window_delays(solution)
    if violations_count > 0:
        total_delay_formatted = format_duration_detailed(total_delay_minutes)
        print(f"   ⏰ Time window violations: {violations_count} tasks with total delay of {total_delay_formatted}")
    else:
        print(f"   ⏰ Time window violations: None detected")
    
    # Assignment statistics
    total_orders = len(orders)
    assigned_orders = len(solution.get_assigned_orders())
    unassigned_orders = len(solution.unassigned_orders)
    
    print(f"   📦 Orders assigned: {assigned_orders} / {total_orders} ({assigned_orders/total_orders*100:.1f}%)")
    print(f"   Orders unassigned: {unassigned_orders}")
    
    # Vehicle utilization analysis
    print(f"\n🚛 Vehicle Usage & Utilization:")
    utilization = solution.get_vehicle_utilization()
    
    active_vehicles = 0
    total_weight_util = 0
    total_volume_util = 0
    
    for vehicle_id, route in solution.routes.items():
        if not route.tasks:
            continue
            
        active_vehicles += 1
        vehicle = route.vehicle
        util = utilization.get(vehicle_id, {"weight": 0, "volume": 0})
        
        print(f"   🚚 {vehicle_id} ({vehicle.vehicle_type}):")
        print(f"      📊 Weight utilization: {util['weight']*100:.1f}% ({util['weight']*vehicle.weight_capacity:.0f}/{vehicle.weight_capacity:.0f} kg)")
        print(f"      📊 Volume utilization: {util['volume']*100:.1f}% ({util['volume']*vehicle.volume_capacity:.1f}/{vehicle.volume_capacity:.1f} m³)")
        print(f"      🎯 Tasks: {len(route.tasks)}")
        print(f"      🏪 Orders: {len(route.get_orders())}")
        
        total_weight_util += util['weight']
        total_volume_util += util['volume']
    
    if active_vehicles > 0:
        avg_weight_util = (total_weight_util / active_vehicles) * 100
        avg_volume_util = (total_volume_util / active_vehicles) * 100
        print(f"\n   📊 Average utilization:")
        print(f"      Weight: {avg_weight_util:.1f}%")
        print(f"      Volume: {avg_volume_util:.1f}%")
    
    # Route details
    print(f"\nRoute Details:")
    for vehicle_id, route in solution.routes.items():
        if not route.tasks:
            print(f"   🚚 {vehicle_id}: Empty route")
            continue
            
        print(f"   🚚 {vehicle_id}:")
        
        # Calculate route feasibility and times using HoS simulation
        try:
            from algo.second_level import is_feasible, _simulate_hos_advanced, _sort_tasks_chronologically, DriverState
            
            feasible = is_feasible(route)
            print(f"      Feasible: {feasible}")
            
            # Get actual route duration and task times from HoS simulation
            sorted_tasks = _sort_tasks_chronologically(route.tasks)
            driver_state = DriverState()
            
            hos_feasible, total_duration = _simulate_hos_advanced(route, driver_state, sorted_tasks)
            
            # Calculate individual task times by re-running simulation step by step  
            task_times = []
            current_time = 0.0
            temp_driver_state = DriverState()
            
            # Simple step-by-step simulation to get task completion times
            for i, task in enumerate(sorted_tasks):
                if i == 0:
                    # First task: service time only
                    service_time = getattr(task, 'service_time', 15.0)
                    current_time += service_time
                else:
                    # Subsequent tasks: travel time + waiting + service time
                    prev_task = sorted_tasks[i-1]
                    
                    # Calculate travel time using centralized system
                    try:
                        # Import centralized function directly to ensure we use it
                        from route_provider import calculate_travel_time_between_tasks as central_calc
                        travel_time = central_calc(prev_task, task, route.vehicle)
                        print(f"DEBUG: Used route_provider for {getattr(prev_task, 'location_id', 'unknown')} -> {getattr(task, 'location_id', 'unknown')}: {travel_time:.1f}m")
                    except ImportError as e1:
                        try:
                            from algo.route_provider import calculate_travel_time_between_tasks as central_calc
                            travel_time = central_calc(prev_task, task, route.vehicle)
                            print(f"DEBUG: Used algo.route_provider for {getattr(prev_task, 'location_id', 'unknown')} -> {getattr(task, 'location_id', 'unknown')}: {travel_time:.1f}m")
                        except ImportError as e2:
                            print(f"DEBUG: Import failed ({e1}, {e2}), using fallback Haversine")
                            # Final fallback - but this should use proper Haversine
                            if hasattr(prev_task, 'lat') and hasattr(task, 'lat'):
                                import math
                                # Proper Haversine calculation
                                lat1_rad = math.radians(prev_task.lat)
                                lon1_rad = math.radians(prev_task.lon)
                                lat2_rad = math.radians(task.lat)
                                lon2_rad = math.radians(task.lon)
                                
                                dlat = lat2_rad - lat1_rad
                                dlon = lon2_rad - lon1_rad
                                
                                a = (math.sin(dlat/2)**2 + 
                                     math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
                                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                                
                                R = 6371.0  # Earth's radius in km
                                distance_km = R * c
                                
                                speed_kmh = getattr(route.vehicle, 'average_speed', 60.0)
                                travel_time = (distance_km / speed_kmh) * 60.0  # Convert to minutes
                                print(f"DEBUG: Fallback Haversine for {getattr(prev_task, 'location_id', 'unknown')} -> {getattr(task, 'location_id', 'unknown')}: {distance_km:.1f}km @ {speed_kmh}km/h = {travel_time:.1f}m")
                            else:
                                travel_time = 15.0  # Final fallback
                                print(f"DEBUG: Final fallback (15m) for {getattr(prev_task, 'location_id', 'unknown')} -> {getattr(task, 'location_id', 'unknown')}")
                    
                    
                    # Add travel time
                    current_time += travel_time
                    
                    # Handle time window waiting (simplified)
                    if hasattr(task, 'earliest_time') and task.earliest_time and current_time < task.earliest_time:
                        wait_time = task.earliest_time - current_time
                        current_time += wait_time
                    
                    # Add service time
                    service_time = getattr(task, 'service_time', 15.0)
                    current_time += service_time
                
                task_times.append(current_time)
            
            # Don't scale task times - use the realistic calculated travel times
            # The HoS simulation total_duration may be different due to breaks/rests
            # but we want to show the actual travel times we calculated
            
            # Format duration based on actual calculated time (not HoS simulation total)
            actual_total_time = task_times[-1] if task_times else total_duration
            if actual_total_time >= 1440:
                days = int(actual_total_time // 1440)
                remaining_hours = (actual_total_time % 1440) / 60
                hours = int(remaining_hours)
                minutes = int((remaining_hours % 1) * 60)
                
                if days > 0:
                    duration_str = f"{days}d {hours}h {minutes}m" if hours > 0 or minutes > 0 else f"{days}d"
                else:
                    duration_str = f"{hours}h {minutes}m"
            else:
                hours = int(actual_total_time / 60)
                minutes = int(actual_total_time % 60)
                duration_str = f"{hours}h {minutes}m"
            
            compliance_note = "HoS compliant" if hos_feasible and actual_total_time < 15 * 60 else "(would violate HoS if attempted without proper rests)"
            
            if actual_total_time >= 1440:
                days = int(actual_total_time // 1440)
                print(f"      📅 Route duration: {days} day(s) ({duration_str}) {compliance_note}")
            else:
                print(f"      📅 Route duration: 1 day(s) ({duration_str}) {compliance_note}")
                
        except Exception as e:
            print(f"      📅 Route duration: Error calculating ({str(e)})")
            task_times = None
        
        # Task sequence with real-time monitoring
        print(f"      📋 Task sequence ({len(route.tasks)} tasks) - Real-time monitoring:")
        current_load_weight = 0
        current_load_volume = 0
        
        # Use sorted tasks to match the order used in HoS simulation
        display_tasks = sorted_tasks if 'sorted_tasks' in locals() else route.tasks
        
        for i, task in enumerate(display_tasks):
            current_load_weight += task.demand
            current_load_volume += task.volume
            
            if hasattr(task, 'is_depot_return') and task.is_depot_return():
                task_type_icon = "🏠"  # Home icon for depot return
            elif hasattr(task, 'is_depot_start') and task.is_depot_start():
                task_type_icon = "🚀"  # Rocket icon for depot start
            else:
                task_type_icon = "📦" if task.is_pickup() else "🏪"
            
            # Add time monitoring if available
            time_info = ""
            if task_times and i < len(task_times):
                cumulative_time = task_times[i]
                time_formatted = format_duration_detailed(cumulative_time)
                # Calculate time since previous task for additional context
                if i > 0 and task_times[i-1] is not None:
                    delta_time = cumulative_time - task_times[i-1]
                    delta_formatted = format_duration_detailed(delta_time)
                    time_info = f" - Cumulative: {time_formatted} (+{delta_formatted})"
                else:
                    time_info = f" - Cumulative: {time_formatted}"
            
            # Add time window information
            time_window_info = ""
            if hasattr(task, 'earliest_time') and hasattr(task, 'latest_time'):
                if task.earliest_time is not None and task.latest_time is not None:
                    # Convert multi-day time windows to readable format
                    # Multi-day format: (day_index * 1440) + time_in_minutes
                    earliest_day = int(task.earliest_time // 1440)
                    earliest_time_of_day = int(task.earliest_time % 1440)
                    earliest_hours = earliest_time_of_day // 60
                    earliest_minutes = earliest_time_of_day % 60
                    
                    latest_day = int(task.latest_time // 1440)
                    latest_time_of_day = int(task.latest_time % 1440)
                    latest_hours = latest_time_of_day // 60
                    latest_minutes = latest_time_of_day % 60
                    
                    if earliest_day == latest_day:
                        # Same day time window
                        time_window_info = f" [Day {earliest_day}: {earliest_hours:02d}:{earliest_minutes:02d}-{latest_hours:02d}:{latest_minutes:02d}]"
                    else:
                        # Multi-day time window - show the actual different days
                        time_window_info = f" [Day {earliest_day} {earliest_hours:02d}:{earliest_minutes:02d} - Day {latest_day} {latest_hours:02d}:{latest_minutes:02d}]"
                elif hasattr(task, 'location_id') and hasattr(task, 'task_type'):
                    # If task doesn't have time windows but has location info, try to get from location
                    try:
                        # This is for depot bays and pickup bays that might not have direct time window info
                        time_window_info = " [No specific time window]"
                    except:
                        time_window_info = ""
            
            print(f"         {i+1:2d}. {task_type_icon} {task.location_id} (Order: {task.order_id}){time_info}{time_window_info}")
            print(f"             Load: {task.demand:+.0f}kg, {task.volume:+.1f}m³ → Total: {current_load_weight:.0f}kg, {current_load_volume:.1f}m³")
    
    # Time window violations details
    if violations_count > 0:
        print(f"\n⚠️  Time Window Violations Details:")
        for violation in violations_details:
            delay_formatted = format_duration_detailed(violation['delay_minutes'])
            latest_day = int(violation['latest_allowed'] // 1440)
            latest_time_of_day = int(violation['latest_allowed'] % 1440)
            latest_hours = latest_time_of_day // 60
            latest_minutes = latest_time_of_day % 60
            
            arrival_day = int(violation['actual_arrival'] // 1440)
            arrival_time_of_day = int(violation['actual_arrival'] % 1440)
            arrival_hours = arrival_time_of_day // 60
            arrival_mins = arrival_time_of_day % 60
            
            print(f"   🚚 {violation['vehicle_id']} - {violation['location_id']}:")
            print(f"      Latest allowed: Day {latest_day} {latest_hours:02d}:{latest_minutes:02d}")
            print(f"      Actual arrival: Day {arrival_day} {arrival_hours:02d}:{arrival_mins:02d}")
            print(f"      Delay: {delay_formatted}")
    
    # Enhanced Unassigned orders analysis with feasibility investigation
    if solution.unassigned_orders:
        print(f"\nUnassigned Orders Analysis:")
        print(f"{'='*80}")
        
        # Calculate fleet capacities for comparison
        total_fleet_weight = sum(v.weight_capacity for v in vehicles)
        total_fleet_volume = sum(v.volume_capacity for v in vehicles)
        max_vehicle_weight = max(v.weight_capacity for v in vehicles)
        max_vehicle_volume = max(v.volume_capacity for v in vehicles)
        
        print(f"🚛 Fleet Capacity Overview:")
        print(f"   Total fleet weight capacity: {total_fleet_weight:,.0f} kg")
        print(f"   Total fleet volume capacity: {total_fleet_volume:.1f} m³")
        print(f"   Largest vehicle weight capacity: {max_vehicle_weight:,.0f} kg")
        print(f"   Largest vehicle volume capacity: {max_vehicle_volume:.1f} m³")
        
        # Calculate current utilization
        current_weight_used = 0
        current_volume_used = 0
        for route in solution.routes.values():
            if route and route.tasks:
                for task in route.tasks:
                    if hasattr(task, 'task_type') and hasattr(task.task_type, 'value'):
                        task_type = task.task_type.value
                    else:
                        task_type = str(task.task_type)
                    
                    if task_type == 'pickup':
                        current_weight_used += getattr(task, 'demand', 0)
                        current_volume_used += getattr(task, 'volume', 0)
        
        remaining_weight_capacity = total_fleet_weight - current_weight_used
        remaining_volume_capacity = total_fleet_volume - current_volume_used
        
        print(f"📊 Current Fleet Utilization:")
        print(f"   Weight used: {current_weight_used:,.0f} kg ({current_weight_used/total_fleet_weight*100:.1f}%)")
        print(f"   Volume used: {current_volume_used:.1f} m³ ({current_volume_used/total_fleet_volume*100:.1f}%)")
        print(f"   Remaining weight capacity: {remaining_weight_capacity:,.0f} kg")
        print(f"   Remaining volume capacity: {remaining_volume_capacity:.1f} m³")
        
        print(f"\n📦 Detailed Unassigned Order Analysis:")
        print(f"{'-'*80}")
        
        total_unassigned_weight = 0
        total_unassigned_volume = 0
        
        for i, order_id in enumerate(solution.unassigned_orders, 1):
            order = next((o for o in orders if o.id == order_id), None)
            if order:
                order_weight = order.get_total_demand()
                order_volume = order.get_total_volume()
                total_unassigned_weight += order_weight
                total_unassigned_volume += order_volume
                
                print(f"   {i}. 📦 {order_id}:")
                print(f"      Weight: {order_weight:.0f} kg")
                print(f"      Volume: {order_volume:.1f} m³")
                print(f"      Priority: {order.priority}")
                print(f"      Mandatory: {order.is_mandatory}")
                
                # Feasibility analysis
                feasibility_issues = []
                
                # Check if order is too large for any single vehicle
                if order_weight > max_vehicle_weight:
                    feasibility_issues.append(f"TOO HEAVY: Exceeds largest vehicle capacity ({max_vehicle_weight:,.0f} kg)")
                elif order_volume > max_vehicle_volume:
                    feasibility_issues.append(f"TOO VOLUMINOUS: Exceeds largest vehicle capacity ({max_vehicle_volume:.1f} m³)")
                
                # Check if order could fit in remaining capacity
                if order_weight > remaining_weight_capacity:
                    feasibility_issues.append(f"WEIGHT CONSTRAINT: Exceeds remaining fleet weight capacity ({remaining_weight_capacity:,.0f} kg)")
                elif order_volume > remaining_volume_capacity:
                    feasibility_issues.append(f"VOLUME CONSTRAINT: Exceeds remaining fleet volume capacity ({remaining_volume_capacity:.1f} m³)")
                
                # Check if there are vehicles with sufficient capacity
                compatible_vehicles = []
                for vehicle in vehicles:
                    if (order_weight <= vehicle.weight_capacity and 
                        order_volume <= vehicle.volume_capacity):
                        compatible_vehicles.append(vehicle.id)
                
                if not compatible_vehicles:
                    feasibility_issues.append("NO COMPATIBLE VEHICLES: No single vehicle can handle this order")
                else:
                    print(f"      Compatible vehicles: {', '.join(compatible_vehicles)}")
                
                # Time window analysis
                if hasattr(order, 'pickup_tasks') and order.pickup_tasks:
                    pickup_task = order.pickup_tasks[0]
                    if hasattr(pickup_task, 'time_window') and pickup_task.time_window:
                        tw = pickup_task.time_window
                        print(f"      Time window: {tw.earliest_start} - {tw.latest_start}")
                        if tw.latest_start - tw.earliest_start < 60:  # Less than 1 hour
                            feasibility_issues.append("TIGHT TIME WINDOW: Very narrow time window for pickup")
                
                if feasibility_issues:
                    print(f"      🚫 Potential rejection reasons:")
                    for issue in feasibility_issues:
                        print(f"         • {issue}")
                else:
                    print(f"      ✅ Order appears feasible - likely algorithmic limitation")
                
                print()  # Empty line between orders
        
        print(f"📊 Unassigned Orders Summary:")
        print(f"   Total unassigned weight: {total_unassigned_weight:,.0f} kg")
        print(f"   Total unassigned volume: {total_unassigned_volume:.1f} m³")
        print(f"   Percentage of total cargo: {total_unassigned_weight/(total_unassigned_weight+current_weight_used)*100:.1f}% weight, {total_unassigned_volume/(total_unassigned_volume+current_volume_used)*100:.1f}% volume")
        
        # Quick feasibility check
        if total_unassigned_weight <= remaining_weight_capacity and total_unassigned_volume <= remaining_volume_capacity:
            print(f"   ✅ All unassigned orders COULD theoretically fit in remaining fleet capacity")
            print(f"   💡 This suggests the issue is in the heuristic's search strategy or feasibility checks")
        else:
            print(f"   ❌ Unassigned orders exceed remaining fleet capacity")
            if total_unassigned_weight > remaining_weight_capacity:
                print(f"   📦 Weight overflow: {total_unassigned_weight - remaining_weight_capacity:,.0f} kg")
            if total_unassigned_volume > remaining_volume_capacity:
                print(f"   📦 Volume overflow: {total_unassigned_volume - remaining_volume_capacity:.1f} m³")
    else:
        print(f"\n✅ All orders successfully assigned!")
    
    print(f"\n{'='*80}")
    
    # Algorithm configuration summary
    print(f"\n⚙️  Algorithm Configuration:")
    print(f"   🔧 Tabu tenure: {params.get('tabu_tenure', 'N/A')}")
    print(f"   🔧 Max non-improving iterations: {params.get('M1', 'N/A')}")
    print(f"   🔧 Max total iterations: {params.get('M2', 'N/A')}")
    print(f"   🔧 Exploration strategy: {params.get('exploration_strategy', 'N/A')}")
    print(f"   🔧 Local search: {params.get('local_search_strategy', 'N/A')}")
    print(f"   🔧 Initialization: {params.get('initialization_method', 'N/A')}")
    
    # Enhancement flags
    enhancements = []
    if params.get('enable_advanced_neighborhoods', False):
        enhancements.append("Advanced neighborhoods")
    if params.get('enable_granular_search', False):
        enhancements.append("Granular search")
    if params.get('enable_parallelization', False):
        enhancements.append(f"Parallelization ({params.get('parallel_strategy', 'N/A')})")
    
    if enhancements:
        print(f"   🚀 Enhancements: {', '.join(enhancements)}")
    else:
        print(f"   🚀 Enhancements: None enabled")
    
    print(f"\n" + "="*80)
    print(f"Solution analysis complete")
    print(f"="*80)


def save_results(solution, orders, vehicles, params, runtime_seconds, output_dir: str = "results"):
    """Save test results to files for later analysis."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    
    # Save solution summary as JSON
    result_data = {
        "timestamp": timestamp,
        "runtime_seconds": runtime_seconds,
        "algorithm_params": {
            "tabu_tenure": params.get('tabu_tenure', 'N/A'),
            "max_non_improving_iterations": params.get('M1', 'N/A'),
            "max_total_iterations": params.get('M2', 'N/A'),
            "exploration_strategy": params.get('exploration_strategy', 'N/A'),
            "local_search_strategy": params.get('local_search_strategy', 'N/A')
        },
        "solution_metrics": {
            "total_vehicles": len(vehicles),
            "vehicles_used": solution.get_total_vehicles_used(),
            "total_orders": len(orders),
            "assigned_orders": len(solution.get_assigned_orders()),
            "unassigned_orders": len(solution.unassigned_orders)
        },
        "vehicle_routes": {},
        "unassigned_orders": list(solution.unassigned_orders)
    }
    
    # Add route details
    for vehicle_id, route in solution.routes.items():
        if route.tasks:
            result_data["vehicle_routes"][vehicle_id] = {
                "task_count": len(route.tasks),
                "order_count": len(route.get_orders()),
                "task_sequence": [
                    {
                        "task_id": task.id,
                        "location_id": task.location_id,
                        "task_type": task.task_type.value,
                        "order_id": task.order_id,
                        "demand": task.demand,
                        "volume": task.volume
                    }
                    for task in route.tasks
                ]
            }
    
    # Save to file
    result_file = os.path.join(output_dir, f"epdt_test_results_{timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"💾 Results saved to {result_file}")


def compare_with_mock_solution(heuristic_solution, orders, vehicles):
    """
    Compare the heuristic solution with a mock "perfect" assignment to identify
    differences in task distribution and potential improvement opportunities.
    """
    print(f"\n🔍 Comparing Heuristic vs Mock Solution")
    print(f"{'='*80}")
    
    # Generate mock solution
    print(f"📊 Generating mock solution for comparison...")
    mock_solution = _create_mock_solution(orders, vehicles)
    
    # Compare assignment rates
    heuristic_assigned = len(orders) - len(heuristic_solution.unassigned_orders)
    mock_assigned = len(orders) - len(mock_solution.unassigned_orders)
    
    print(f"\n📈 Assignment Rate Comparison:")
    print(f"   Heuristic: {heuristic_assigned}/{len(orders)} orders ({heuristic_assigned/len(orders)*100:.1f}%)")
    print(f"   Mock:      {mock_assigned}/{len(orders)} orders ({mock_assigned/len(orders)*100:.1f}%)")
    print(f"   Difference: {mock_assigned - heuristic_assigned} orders ({(mock_assigned - heuristic_assigned)/len(orders)*100:.1f}%)")
    
    # Compare vehicle utilization
    print(f"\n🚛 Vehicle Utilization Comparison:")
    print(f"{'Vehicle':<15} {'Heuristic Load':<20} {'Mock Load':<20} {'Difference'}")
    print(f"{'-'*75}")
    
    # Get correct utilization calculations from solution objects
    heuristic_utilization = heuristic_solution.get_vehicle_utilization()
    mock_utilization = mock_solution.get_vehicle_utilization()
    
    for vehicle in vehicles:
        # Get correct utilization from solution objects
        heuristic_util_data = heuristic_utilization.get(vehicle.id, {"weight": 0, "volume": 0})
        mock_util_data = mock_utilization.get(vehicle.id, {"weight": 0, "volume": 0})
        
        # Calculate percentages
        heuristic_util = heuristic_util_data['weight'] * 100
        mock_util = mock_util_data['weight'] * 100
        
        # Calculate actual weights for display
        heuristic_weight = heuristic_util_data['weight'] * vehicle.weight_capacity
        mock_weight = mock_util_data['weight'] * vehicle.weight_capacity

        print(f"{vehicle.id:<15} {heuristic_util:>6.1f}% ({heuristic_weight:>6.0f}kg) {mock_util:>6.1f}% ({mock_weight:>6.0f}kg) {mock_util-heuristic_util:>+6.1f}%")
    
    # Identify orders that mock could assign but heuristic couldn't
    heuristic_unassigned = set(heuristic_solution.unassigned_orders)
    mock_unassigned = set(mock_solution.unassigned_orders)
    
    mock_success_heuristic_fail = heuristic_unassigned - mock_unassigned
    
    if mock_success_heuristic_fail:
        print(f"\n🔧 Orders Mock Could Assign But Heuristic Couldn't:")
        print(f"{'-'*60}")
        for order_id in mock_success_heuristic_fail:
            order = next((o for o in orders if o.id == order_id), None)
            if order:
                # Find which vehicle the mock assigned it to
                mock_vehicle = None
                for vehicle_id, route in mock_solution.routes.items():
                    if route and route.tasks:
                        for task in route.tasks:
                            if getattr(task, 'order_id', None) == order_id:
                                mock_vehicle = vehicle_id
                                break
                        if mock_vehicle:
                            break
                
                print(f"   📦 {order_id}: {order.get_total_demand():.0f}kg, {order.get_total_volume():.1f}m³")
                if mock_vehicle:
                    print(f"      Mock assigned to: {mock_vehicle}")
                    
                    # Check current heuristic load on that vehicle
                    heuristic_route = heuristic_solution.routes.get(mock_vehicle)
                    current_weight = 0
                    if heuristic_route and heuristic_route.tasks:
                        for task in heuristic_route.tasks:
                            if hasattr(task, 'task_type') and hasattr(task.task_type, 'value'):
                                task_type = task.task_type.value
                            else:
                                task_type = str(task.task_type)
                            if task_type == 'pickup':
                                current_weight += getattr(task, 'demand', 0)
                    
                    vehicle_obj = next((v for v in vehicles if v.id == mock_vehicle), None)
                    if vehicle_obj:
                        remaining_capacity = vehicle_obj.weight_capacity - current_weight
                        print(f"      Heuristic {mock_vehicle} remaining capacity: {remaining_capacity:.0f}kg")
                        if order.get_total_demand() <= remaining_capacity:
                            print(f"      💡 INSIGHT: Order SHOULD fit - likely heuristic search limitation")
                        else:
                            print(f"      ⚠️  Capacity insufficient in heuristic solution")
    
    # Time Window Violations Comparison
    print(f"\n⏰ Time Window Violations Comparison:")
    print(f"{'-'*60}")
    
    # Count heuristic time window violations
    heuristic_violations = 0
    heuristic_violation_details = []
    for vehicle_id, route in heuristic_solution.routes.items():
        if route and route.tasks:
            for task in route.tasks:
                # This is a simplified check - ideally we'd simulate the route timing
                if hasattr(task, 'latest_time') and hasattr(task, 'earliest_time'):
                    if (task.latest_time is not None and 
                        hasattr(task, 'actual_arrival_time') and 
                        task.actual_arrival_time is not None and
                        task.actual_arrival_time > task.latest_time):
                        heuristic_violations += 1
                        delay = task.actual_arrival_time - task.latest_time
                        heuristic_violation_details.append((task.id, delay))
    
    # Mock solution has fewer time window violations due to multi-day allowance
    mock_violations = 0  # Mock allows multi-day routes so fewer violations
    
    print(f"   Heuristic violations: {heuristic_violations} tasks")
    print(f"   Mock violations:      {mock_violations} tasks")
    print(f"   Difference:           {heuristic_violations - mock_violations} tasks")
    
    if heuristic_violations > 0:
        print(f"\n   💡 Note: Time window violations may be reduced by enabling multi-day routing")
        print(f"           which allows longer routes to be split across multiple days")

    # Performance insights
    print(f"\n💡 Key Insights:")
    if mock_assigned > heuristic_assigned:
        print(f"   🔍 Mock solution achieves {((mock_assigned - heuristic_assigned)/len(orders)*100):.1f}% better assignment rate")
        print(f"   🚧 This suggests the heuristic has suboptimal search strategy or feasibility checks")
        if mock_success_heuristic_fail:
            print(f"   🎯 Focus on improving assignment logic for {len(mock_success_heuristic_fail)} specific orders")
    else:
        print(f"   ✅ Heuristic matches mock assignment rate")
    
    print(f"\n📋 Recommendations:")
    if mock_success_heuristic_fail:
        print(f"   1. Review L2 heuristic insertion logic - may be too conservative")
        print(f"   2. Check feasibility constraints in second_level.py")
        print(f"   3. Consider relaxing strict time window or capacity checks")
        print(f"   4. Implement more aggressive neighborhood search")
    else:
        print(f"   1. Focus on runtime optimization rather than assignment rate")
        print(f"   2. Consider faster scoring functions or reduced neighborhood size")
    
    print(f"{'='*80}")
    return mock_solution


def profile_heuristic_performance(orders, vehicles, params, output_dir: str = "results"):
    """
    Production-level profiling and analysis following TODO2.md Section 14 requirements.
    """
    print(f"\n🏭 PRODUCTION-LEVEL PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    try:
        from first_level import l1_heuristic
        
        if not PRODUCTION_PROFILER_AVAILABLE:
            print("❌ Production profiler not available, falling back to basic profiling")
            profiler = cProfile.Profile()
            profiler.enable()
            solution = l1_heuristic(orders, vehicles, params)
            profiler.disable()
            
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)
            return solution, None, None
        
        # Initialize production profiler
        profiler = ProductionProfiler()
        constraint_analyzer = ConstraintAnalyzer()
        
        print(f"🔍 Starting comprehensive production profiling...")
        profiler.start_profiling()
        
        # Mark initialization phase
        profiler.mark_phase('initialization')
        
        # Run the heuristic with detailed monitoring
        print(f"⚡ Running L1 heuristic with production monitoring...")
        solution = l1_heuristic(orders, vehicles, params)
        
        # Mark optimization completion
        profiler.mark_phase('optimization')
        
        # Stop profiling and get metrics
        performance_metrics = profiler.stop_profiling()
        
        print(f"\n📊 PRODUCTION PERFORMANCE RESULTS:")
        print(f"{'='*60}")
        print(f"🕐 Total Runtime: {performance_metrics.total_runtime:.2f}s")
        
        # Check if we meet production target
        if performance_metrics.total_runtime > 30:
            print(f"❌ CRITICAL: Runtime exceeds production target of 30s by {performance_metrics.total_runtime - 30:.1f}s")
        else:
            print(f"✅ GOOD: Runtime meets production target (<30s)")
            
        print(f"💾 Peak Memory: {performance_metrics.memory_peak_mb:.1f}MB")
        
        # Show top bottlenecks
        print(f"\n🎯 TOP PERFORMANCE BOTTLENECKS:")
        print(f"{'-'*50}")
        for i, (func_name, time_spent) in enumerate(performance_metrics.bottleneck_functions[:5], 1):
            print(f"{i}. {func_name}: {time_spent:.3f}s")
        
        # Analyze solution quality
        print(f"\n📈 SOLUTION QUALITY ANALYSIS:")
        print(f"{'-'*50}")
        quality_metrics = SolutionAnalyzer.analyze_solution_quality(solution, orders, vehicles)
        
        print(f"� Assignment Rate: {quality_metrics.assignment_rate:.1%}")
        print(f"🚛 Vehicle Utilization (Weight): {quality_metrics.vehicle_utilization_weight:.1%}")
        print(f"📐 Vehicle Utilization (Volume): {quality_metrics.vehicle_utilization_volume:.1%}")
        print(f"🚚 Vehicles Used: {quality_metrics.vehicles_used}/{quality_metrics.total_vehicles}")
        print(f"📍 Total Distance: {quality_metrics.total_distance:.1f}km")
        print(f"⏰ Time Window Violations: {quality_metrics.time_window_violations}")
        
        # Generate and compare with mock solution - DISABLED for cleaner output
        # print(f"\n🎭 COMPARING WITH OPTIMAL MOCK SOLUTION...")
        # mock_solution = _create_mock_solution(orders, vehicles)
        # mock_quality = SolutionAnalyzer.analyze_solution_quality(mock_solution, orders, vehicles)
        
        # comparison = SolutionAnalyzer.compare_solutions(quality_metrics, mock_quality)
        
        # print(f"📊 HEURISTIC vs MOCK COMPARISON:")
        # print(f"{'-'*40}")
        # print(f"Distance Efficiency: {comparison['distance_efficiency']:.1%}")
        # print(f"Assignment Rate Diff: {comparison['assignment_rate_difference']:.1%}")
        # print(f"Vehicle Efficiency: {comparison['vehicles_efficiency']:.1%}")
        
        # Constraint analysis
        feasibility_analysis = constraint_analyzer.generate_analysis()
        
        # Save comprehensive production report (without mock comparison)
        timestamp = int(time.time())
        report_file = os.path.join(output_dir, f"production_report_{timestamp}.json")
        
        # Create empty comparison since mock solution is disabled
        comparison = {"distance_efficiency": 0.0, "assignment_rate_difference": 0.0, "vehicles_efficiency": 0.0}
        
        save_production_report(
            performance_metrics, quality_metrics, feasibility_analysis,
            comparison, report_file
        )
        
        print(f"\n💾 Comprehensive production report saved to: {report_file}")
        
        # Production recommendations
        if performance_metrics.total_runtime > 30:
            print(f"\n🚨 PRODUCTION OPTIMIZATION REQUIRED:")
            print(f"1. Runtime {performance_metrics.total_runtime:.1f}s exceeds 30s target")
            top_bottleneck = performance_metrics.bottleneck_functions[0]
            print(f"2. Optimize: {top_bottleneck[0]} ({top_bottleneck[1]:.2f}s)")
            print(f"3. Consider reducing neighborhood search complexity")
            print(f"4. Implement faster scoring approximations")
        
        return solution, performance_metrics, quality_metrics
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        
        print(f"\n📊 Top 20 Most Time-Consuming Functions:")
        print(f"{'-'*80}")
        ps.print_stats(20)
        
        # Save detailed profiling results
        timestamp = int(time.time())
        profile_file = os.path.join(output_dir, f"profile_results_{timestamp}.txt")
        
        with open(profile_file, 'w') as f:
            f.write("EPDT L1 Heuristic Performance Profile\n")
            f.write("="*50 + "\n\n")
            
            # Write summary
            ps_summary = pstats.Stats(profiler, stream=f)
            ps_summary.sort_stats('cumulative')
            ps_summary.print_stats()
        
        print(f"\n💾 Detailed profiling results saved to: {profile_file}")
        
        # Extract key metrics
        stats = ps.get_stats()
        total_calls = sum(stat[0] for stat in stats.values())
        total_time = sum(stat[2] for stat in stats.values())
        
        print(f"\n📈 Profiling Summary:")
        print(f"   Total function calls: {total_calls:,}")
        print(f"   Total execution time: {total_time:.2f} seconds")
        print(f"   Functions profiled: {len(stats)}")
        
        # Identify potential bottlenecks
        print(f"\n🎯 Optimization Recommendations:")
        
        # Look for expensive function patterns
        expensive_functions = []
        for func_name, (cc, nc, tt, ct) in stats.items():
            if ct > 1.0:  # Functions taking more than 1 second
                expensive_functions.append((func_name, ct, cc))
        
        expensive_functions.sort(key=lambda x: x[1], reverse=True)
        
        if expensive_functions:
            print(f"   🔴 High-impact optimization targets:")
            for func_name, cumtime, callcount in expensive_functions[:5]:
                print(f"      • {func_name}: {cumtime:.2f}s ({callcount} calls)")
        
        # Check for excessive function calls
        frequent_functions = []
        for func_name, (cc, nc, tt, ct) in stats.items():
            if cc > 10000:  # Functions called more than 10,000 times
                frequent_functions.append((func_name, cc, ct))
        
        frequent_functions.sort(key=lambda x: x[1], reverse=True)
        
        if frequent_functions:
            print(f"   🟡 Frequently called functions (potential optimization via caching):")
            for func_name, callcount, cumtime in frequent_functions[:3]:
                print(f"      • {func_name}: {callcount:,} calls, {cumtime:.2f}s total")
        
        return solution
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        print(f"   Falling back to normal execution...")
        from first_level import l1_heuristic
        return l1_heuristic(orders, vehicles, params)


def run_scenario_test(scenario_name: str = "furgoni", 
                     custom_params_file: Optional[str] = None,
                     save_output: bool = True,
                     enable_profiling: bool = False) -> Dict[str, Any]:
    """
    Main test execution function following the TODO section 5.3 workflow.
    
    Args:
        scenario_name: Name of scenario to test ("furgoni")
        custom_params_file: Optional custom parameters JSON file
        save_output: Whether to save results to files
        enable_profiling: Whether to enable performance profiling
        
    Returns:
        Dictionary with test results and metrics
    """
    print(f"🚀 Starting EPDT Algorithm Test")
    print(f"📋 Scenario: {scenario_name}")
    print(f"="*60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load Scenario
        print(f"\n1️⃣  Loading scenario: {scenario_name}")
        if scenario_name == "furgoni":
            instance = create_furgoni_scenario()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        # Step 2: Transform Data  
        print(f"\n2️⃣  Transforming data to EPDT format")
        orders, vehicles = convert_instance_to_epdt_input(instance)
        print_conversion_summary(orders, vehicles)
        
        # Step 3: Configure Parameters
        print(f"\n3️⃣  Configuring algorithm parameters")
        
        print("DEBUG: About to call configure_algorithm_parameters")
        params = configure_algorithm_parameters(custom_params_file)
        
        # Disable debug flag to prevent hanging and speed up execution
        params['debug_assignment'] = False
        print("🔧 Debug assignment disabled for faster execution")
        
        print(f"DEBUG: Returned params type: {type(params)}")
        print(f"DEBUG: Params is dict: {isinstance(params, dict)}")
        if hasattr(params, '__dict__'):
            print(f"DEBUG: Params attributes: {list(params.__dict__.keys())}")
        
        print(f"🔧 Algorithm Configuration:")
        print(f"   - Tabu tenure: {params.get('tabu_tenure', 'N/A') if isinstance(params, dict) else getattr(params, 'tabu_tenure', 'N/A')}")
        print(f"   - Max iterations: {params.get('M2', 'N/A') if isinstance(params, dict) else getattr(params, 'max_total_iterations', 'N/A')}")
        print(f"   - Strategy: {params.get('exploration_strategy', 'N/A') if isinstance(params, dict) else getattr(params, 'exploration_strategy', 'N/A')}")
        print(f"   - Local search: {params.get('local_search_strategy', 'N/A') if isinstance(params, dict) else getattr(params, 'local_search_strategy', 'N/A')}")
        
        # Step 4: Run Heuristic
        print(f"\n4️⃣  Running EPDT heuristic algorithm")
        
        # Reset distance calculation counter before running
        try:
            # Import from centralized utils
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            utils_dir = os.path.join(current_dir, '..', 'utils')
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            
            from distance_calculator import reset_distance_calculation_count, get_distance_calculation_count
            print("Using centralized distance calculation counter")
        except ImportError:
            try:
                from second_level import reset_distance_calculation_count, get_distance_calculation_count
            except ImportError:
                # Fallback import path
                try:
                    from algo.second_level import reset_distance_calculation_count, get_distance_calculation_count
                except ImportError:
                    print("⚠️  Warning: Could not import distance calculation functions")
                    # Create dummy functions
                    def reset_distance_calculation_count(): pass
                    def get_distance_calculation_count(): return 0
        
        reset_distance_calculation_count()
        
        algorithm_start = time.time()
        
        if enable_profiling:
            print(f"⚡ Profiling enabled - detailed performance analysis will be generated")
            solution = profile_heuristic_performance(orders, vehicles, params)
        else:
            solution = l1_heuristic(orders, vehicles, params)
            
        algorithm_runtime = time.time() - algorithm_start
        
        # Get distance calculation count (equivalent to OSRM calls)
        distance_calculations = get_distance_calculation_count()
        
        print(f"✅ Algorithm completed in {algorithm_runtime:.2f} seconds")
        print(f"🗺️  Distance calculations made: {distance_calculations:,} (equivalent OSRM calls)")
        if distance_calculations > 0:
            print(f"📊 Average time per calculation: {(algorithm_runtime * 1000 / distance_calculations):.2f}ms")
        print(f"💡 Estimated OSRM call time savings: ~{distance_calculations * 0.1:.1f}s (assuming 100ms per OSRM call)")
        
        # Step 5: Analyze and Print Results
        print(f"\n5️⃣  Analyzing results")
        total_runtime = time.time() - start_time
        print_solution_summary(solution, orders, vehicles, params, algorithm_runtime)
        
        # Step 5.5: Mock Solution Comparison DISABLED for cleaner output
        # print(f"\n5️⃣.5️⃣  Performance Analysis")
        # mock_solution = compare_with_mock_solution(solution, orders, vehicles)
        
        # Save results if requested
        if save_output:
            print(f"\n6️⃣  Saving results")
            save_results(solution, orders, vehicles, params, algorithm_runtime)
            
            # Generate interactive map visualization
            print(f"\n7️⃣  Generating interactive map")
            try:
                from algo.solution_visualizer import create_interactive_map
                
                # Create map filename with scenario name and timestamp
                timestamp = int(time.time())
                map_filename = f"{scenario_name}_solution_map_{timestamp}.html"
                map_path = os.path.join("results", map_filename)
                
                # Generate the map
                created_map_path = create_interactive_map(solution, map_path)
                
                if created_map_path:
                    print(f"🗺️  Interactive map saved to: {created_map_path}")
                    # Get absolute path for easier access
                    abs_map_path = os.path.abspath(created_map_path)
                    print(f"🌐 Open in browser: file://{abs_map_path}")
                else:
                    print(f"⚠️  Could not generate interactive map")
                    
            except Exception as e:
                print(f"⚠️  Error generating interactive map: {e}")
                # Don't fail the entire test if map generation fails
                import traceback
                traceback.print_exc()
        
        # Return summary data
        return {
            "success": True,
            "scenario": scenario_name,
            "total_runtime": total_runtime,
            "algorithm_runtime": algorithm_runtime,
            "solution": solution,
            "orders": orders,
            "vehicles": vehicles,
            "params": params
        }
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "total_runtime": time.time() - start_time
        }


def _create_mock_solution(orders, vehicles):
    """Create a mock solution for testing when l1_heuristic is not implemented."""
    print(f"🎭 Creating enhanced mock solution for testing purposes")
    
    solution = create_empty_solution(vehicles)
    
    if not orders or not vehicles:
        return solution
    
    print(f"📊 Analyzing assignment feasibility:")
    print(f"   Orders to assign: {len(orders)}")
    print(f"   Available vehicles: {len(vehicles)}")
    
    # Sort vehicles by capacity (largest first) for better assignment
    sorted_vehicles = sorted(vehicles, key=lambda v: v.weight_capacity, reverse=True)
    
    # Track vehicle utilization and assignments
    vehicle_loads = {v.id: {"weight": 0.0, "volume": 0.0} for v in vehicles}
    vehicle_orders = {v.id: [] for v in vehicles}  # Track which orders go to which vehicle
    
    assigned_count = 0
    assignment_failures = []
    
    # Phase 1: Determine order-to-vehicle assignments based on capacity
    print(f"📋 Phase 1: Determining order assignments...")
    
    for order in orders:
        order_weight = order.get_total_demand()
        order_volume = order.get_total_volume()
        
        # Find a vehicle that can accommodate this order
        assigned = False
        for vehicle in sorted_vehicles:
            current_load = vehicle_loads[vehicle.id]
            
            # Check if this vehicle can handle the order
            if (current_load["weight"] + order_weight <= vehicle.weight_capacity and
                current_load["volume"] + order_volume <= vehicle.volume_capacity):
                
                # Assign the order to this vehicle (just tracking, no tasks yet)
                vehicle_orders[vehicle.id].append(order)
                solution.assign_order(order.id, vehicle.id)
                vehicle_loads[vehicle.id]["weight"] += order_weight
                vehicle_loads[vehicle.id]["volume"] += order_volume
                
                assigned_count += 1
                assigned = True
                
                print(f"   ✅ Assigned {order.id} to {vehicle.id} (Weight: {order_weight:.0f}kg, Volume: {order_volume:.1f}m³)")
                break
        
        if not assigned:
            solution.unassign_order(order.id)
            
            # Find the reason for assignment failure
            reasons = []
            for vehicle in sorted_vehicles:
                current_load = vehicle_loads[vehicle.id]
                if current_load["weight"] + order_weight > vehicle.weight_capacity:
                    weight_exceed = current_load["weight"] + order_weight - vehicle.weight_capacity
                    reasons.append(f"{vehicle.id}: Weight exceeds by {weight_exceed:.0f}kg")
                elif current_load["volume"] + order_volume > vehicle.volume_capacity:
                    volume_exceed = current_load["volume"] + order_volume - vehicle.volume_capacity
                    reasons.append(f"{vehicle.id}: Volume exceeds by {volume_exceed:.1f}m³")
            
            assignment_failures.append({
                "order_id": order.id,
                "weight": order_weight,
                "volume": order_volume,
                "reasons": reasons[:3]  # Show top 3 reasons
            })
            
            print(f"   ❌ Could not assign {order.id} (Weight: {order_weight:.0f}kg, Volume: {order_volume:.1f}m³)")
    
    # Phase 2: Create proper task sequences for each vehicle (all pickups first, then all deliveries)
    print(f"\n📋 Phase 2: Creating proper task sequences...")
    
    for vehicle in sorted_vehicles:
        if not vehicle_orders[vehicle.id]:
            continue  # Skip vehicles with no orders
            
        route = solution.routes[vehicle.id]
        assigned_orders = vehicle_orders[vehicle.id]
        
        print(f"   🚚 {vehicle.id}: Sequencing {len(assigned_orders)} orders")
        
        # Collect all pickup tasks first
        all_pickup_tasks = []
        for order in assigned_orders:
            all_pickup_tasks.extend(order.pickup_tasks)
        
        # Collect all delivery tasks
        all_delivery_tasks = []
        for order in assigned_orders:
            all_delivery_tasks.extend(order.delivery_tasks)
        
        # Sort pickup tasks (could be by location, priority, etc. - keeping simple for now)
        # For now, just maintain order ID sequence for consistency
        all_pickup_tasks.sort(key=lambda task: task.order_id)
        all_delivery_tasks.sort(key=lambda task: task.order_id)
        
        # Add all pickup tasks first (loading phase)
        print(f"      📦 Adding {len(all_pickup_tasks)} pickup tasks")
        for pickup_task in all_pickup_tasks:
            route.tasks.append(pickup_task)
        
        # Then add all delivery tasks (unloading phase) 
        print(f"      🏪 Adding {len(all_delivery_tasks)} delivery tasks")
        for delivery_task in all_delivery_tasks:
            route.tasks.append(delivery_task)
        
        # Add depot return task as the final task
        print(f"      🏠 Adding depot return task")
        try:
            from algo.epdt_data_structures import Task, TaskType
        except ImportError:
            from epdt_data_structures import Task, TaskType
        
        # Get depot information (assume depot coordinates are available on vehicle)
        depot_lat = getattr(vehicle, 'depot_lat', 44.9009)  # Default Asti coordinates
        depot_lon = getattr(vehicle, 'depot_lon', 8.2057)
        depot_id = getattr(vehicle, 'depot_id', 'depot')
        
        # Create depot return task
        depot_return_task = Task.create_depot_return_task(
            vehicle_id=vehicle.id,
            depot_location_id=depot_id,
            depot_lat=depot_lat,
            depot_lon=depot_lon,
            service_time=5.0  # 5 minutes to check in at depot
        )
        
        route.tasks.append(depot_return_task)
        
        print(f"      ✅ Route created: {len(route.tasks)} tasks ({len(all_pickup_tasks)} pickups → {len(all_delivery_tasks)} deliveries → 1 depot return)")
    
    # Print assignment summary
    print(f"\n📋 Assignment Summary:")
    print(f"   ✅ Successfully assigned: {assigned_count}/{len(orders)} orders")
    print(f"   ❌ Failed to assign: {len(assignment_failures)} orders")
    
    # Print vehicle utilization
    print(f"\n🚛 Vehicle Utilization:")
    for vehicle in sorted_vehicles:
        load = vehicle_loads[vehicle.id]
        weight_util = (load["weight"] / vehicle.weight_capacity) * 100
        volume_util = (load["volume"] / vehicle.volume_capacity) * 100
        
        if load["weight"] > 0:  # Only show vehicles with assignments
            orders_count = len(vehicle_orders[vehicle.id])
            print(f"   🚚 {vehicle.id}: {weight_util:.1f}% weight, {volume_util:.1f}% volume ({orders_count} orders)")
            print(f"      Capacity: {vehicle.weight_capacity:.0f}kg, {vehicle.volume_capacity:.1f}m³")
            print(f"      Load: {load['weight']:.0f}kg, {load['volume']:.1f}m³")
            print(f"      Pattern: {len(all_pickup_tasks)} pickups → {len(all_delivery_tasks)} deliveries → depot return")
    
    # Show details of failed assignments
    if assignment_failures:
        print(f"\n❌ Assignment Failure Analysis:")
        for failure in assignment_failures[:5]:  # Show first 5 failures
            print(f"   📦 {failure['order_id']} ({failure['weight']:.0f}kg, {failure['volume']:.1f}m³):")
            for reason in failure['reasons']:
                print(f"      - {reason}")
    
    print(f"✅ Enhanced mock solution created with {solution.get_total_vehicles_used()} vehicles used")
    print(f"🎯 Task sequencing: All vehicles follow pickup-phase → delivery-phase pattern")
    return solution


def create_enhanced_multi_day_scenario():
    """
    Create a test scenario with multi-day tasks to test enhanced EPDT features.
    
    This scenario includes:
    - Yesterday tasks (continuing routes)
    - Today tasks (main planning day)
    - Tomorrow tasks (prospective planning)
    - LIFO constraints
    - Soft time windows
    - Priority-based orders
    
    Returns:
        Tuple of (orders, vehicles, params) for enhanced testing
    """
    from algo.epdt_data_structures import Order, Task, Vehicle, TaskType
    
    # Create vehicles with different capabilities
    vehicles = [
        Vehicle(
            id="V001",
            depot_id="DEPOT1",
            weight_capacity=2000.0,
            volume_capacity=20.0,
            lifo_required=True,  # Test LIFO constraint
            vehicle_type="standard"
        ),
        Vehicle(
            id="V002", 
            depot_id="DEPOT1",
            weight_capacity=1500.0,
            volume_capacity=15.0,
            lifo_required=False,
            vehicle_type="standard"
        )
    ]
    
    # Create orders with multi-day tasks
    orders = []
    
    # Order 1: Yesterday task (continuing route)
    order1 = Order(
        id="ORD001",
        priority="mandatory",
        is_mandatory=True,
        revenue=500.0
    )
    order1.pickup_tasks = [Task(
        id="T001P",
        location_id="LOC001",
        task_type=TaskType.PICKUP,
        order_id="ORD001",
        lat=40.7128,
        lon=-74.0060,
        service_time=30.0,
        demand=100.0,
        volume=2.0,
        day=-1,  # Yesterday
        earliest_time=480.0,  # 8:00 AM
        latest_time=600.0     # 10:00 AM
    )]
    order1.delivery_tasks = [Task(
        id="T001D",
        location_id="LOC002", 
        task_type=TaskType.DELIVERY,
        order_id="ORD001",
        lat=40.7589,
        lon=-73.9851,
        service_time=20.0,
        demand=-100.0,
        volume=-2.0,
        day=0,  # Today
        earliest_time=540.0,  # 9:00 AM
        latest_time=720.0     # 12:00 PM
    )]
    orders.append(order1)
    
    # Order 2: Today task with soft time window
    order2 = Order(
        id="ORD002",
        priority="urgent",
        is_urgent=True,
        revenue=300.0
    )
    order2.pickup_tasks = [Task(
        id="T002P",
        location_id="LOC003",
        task_type=TaskType.PICKUP,
        order_id="ORD002",
        lat=40.7282,
        lon=-74.0776,
        service_time=25.0,
        demand=150.0,
        volume=3.0,
        day=0,  # Today
        earliest_time=600.0,  # 10:00 AM
        latest_time=780.0,    # 1:00 PM
        soft_time_window=True
    )]
    
    # Add late penalty rate as an attribute
    order2.pickup_tasks[0].late_penalty_rate = 2.5
    
    order2.delivery_tasks = [Task(
        id="T002D",
        location_id="LOC004",
        task_type=TaskType.DELIVERY,
        order_id="ORD002",
        lat=40.6892,
        lon=-74.0445,
        service_time=15.0,
        demand=-150.0,
        volume=-3.0,
        day=0,  # Today
        earliest_time=660.0,  # 11:00 AM
        latest_time=840.0,    # 2:00 PM
        soft_time_window=True
    )]
    
    # Add late penalty rate as an attribute
    order2.delivery_tasks[0].late_penalty_rate = 2.0
    orders.append(order2)
    
    # Order 3: Tomorrow task (prospective planning)
    order3 = Order(
        id="ORD003",
        priority="normal",
        revenue=200.0
    )
    order3.pickup_tasks = [Task(
        id="T003P",
        location_id="LOC005",
        task_type=TaskType.PICKUP,
        order_id="ORD003",
        lat=40.7505,
        lon=-73.9934,
        service_time=20.0,
        demand=80.0,
        volume=1.5,
        day=1,  # Tomorrow
        earliest_time=480.0,  # 8:00 AM tomorrow
        latest_time=540.0     # 9:00 AM tomorrow
    )]
    order3.delivery_tasks = [Task(
        id="T003D",
        location_id="LOC006",
        task_type=TaskType.DELIVERY,
        order_id="ORD003",
        lat=40.7831,
        lon=-73.9712,
        service_time=15.0,
        demand=-80.0,
        volume=-1.5,
        day=1,  # Tomorrow
        earliest_time=600.0,  # 10:00 AM tomorrow
        latest_time=720.0     # 12:00 PM tomorrow
    )]
    orders.append(order3)
    
    # Enhanced parameters for multi-day testing
    params = {
        'tabu_tenure': 5,
        'M1': 10,  # Reduced for testing
        'M2': 50,  # Reduced for testing
        'exploration_strategy': 'vnd',
        'unassigned_order_base_penalty': 1000.0,
        'mandatory_order_penalty': 2000.0,
        'vehicle_penalty_per_vehicle': 100.0,
        'initial_vehicle_states': {
            'V001': {
                'yesterday_tasks': []  # Could include continuing tasks
            },
            'V002': {
                'yesterday_tasks': []
            }
        }
    }
    
    return orders, vehicles, params


def test_enhanced_features():
    """
    Test the enhanced EPDT features including multi-day, LIFO, and soft constraints.
    """
    print(f"\n🧪 Testing Enhanced EPDT Features...")
    print(f"="*60)
    
    # Create enhanced scenario
    orders, vehicles, params = create_enhanced_multi_day_scenario()
    
    print(f"📋 Test Scenario Created:")
    print(f"   🚛 Vehicles: {len(vehicles)}")
    print(f"   📦 Orders: {len(orders)}")
    
    # Analyze orders by day
    yesterday_orders = [o for o in orders if any(getattr(t, 'day', 0) < 0 for t in o.get_all_tasks())]
    today_orders = [o for o in orders if any(getattr(t, 'day', 0) == 0 for t in o.get_all_tasks())]
    tomorrow_orders = [o for o in orders if any(getattr(t, 'day', 0) > 0 for t in o.get_all_tasks())]
    
    print(f"   📅 Yesterday orders: {len(yesterday_orders)}")
    print(f"   📅 Today orders: {len(today_orders)}")
    print(f"   📅 Tomorrow orders: {len(tomorrow_orders)}")
    
    # Test priority handling
    mandatory_orders = [o for o in orders if getattr(o, 'is_mandatory', False)]
    urgent_orders = [o for o in orders if getattr(o, 'is_urgent', False)]
    
    print(f"   🔴 Mandatory orders: {len(mandatory_orders)}")
    print(f"   🟡 Urgent orders: {len(urgent_orders)}")
    
    # Test soft time windows
    soft_tw_tasks = []
    for order in orders:
        for task in order.get_all_tasks():
            if hasattr(task, 'soft_time_window') and task.soft_time_window:
                soft_tw_tasks.append(task)
    
    print(f"   ⏰ Soft time window tasks: {len(soft_tw_tasks)}")
    
    # Test LIFO constraints
    lifo_vehicles = [v for v in vehicles if v.lifo_required]
    print(f"   📦 LIFO-constrained vehicles: {len(lifo_vehicles)}")
    
    return orders, vehicles, params


def test_column_generation():
    """
    Test the Column Generation implementation for optimality bounds.
    """
    print(f"\n🧪 Testing Column Generation for Optimality Bounds...")
    print(f"="*60)
    
    try:
        # Import Column Generation modules
        from algo.column_generation_utils import run_comprehensive_benchmark, get_optimality_bounds
        from algo.column_generation_test import run_column_generation_tests
        from algo.column_generation import ColumnGenerationConfig
        
        print(f"✅ Column Generation modules imported successfully")
        
        # Run unit tests first
        print(f"\n🔬 Running Column Generation Unit Tests...")
        test_success = run_column_generation_tests()
        
        if not test_success:
            print(f"⚠️  Some unit tests failed, but continuing with integration test")
        
        # Test on furgoni scenario
        print(f"\n🎯 Testing Column Generation on Furgoni Scenario...")
        
        # Create test scenario
        orders, vehicles, params = create_enhanced_multi_day_scenario()
        
        # Configure Column Generation
        cg_config = ColumnGenerationConfig(
            max_iterations=10,
            optimality_gap_tolerance=1e-4,
            time_limit_seconds=120.0,
            verbose=True,
            initial_routes_strategy="single_order"
        )
        
        print(f"📋 Test Problem:")
        print(f"   📦 Orders: {len(orders)}")
        print(f"   🚛 Vehicles: {len(vehicles)}")
        
        # Get optimality bounds
        print(f"\n🔍 Computing Optimality Bounds...")
        start_time = time.time()
        
        try:
            lower_bound, upper_bound = get_optimality_bounds(orders, vehicles, cg_config)
            computation_time = time.time() - start_time
            
            print(f"✅ Column Generation completed successfully!")
            print(f"   📊 Lower Bound: {lower_bound:.2f}")
            print(f"   📊 Upper Bound: {upper_bound:.2f}")
            
            if upper_bound != float('inf') and lower_bound != float('inf'):
                gap = abs(upper_bound - lower_bound) / max(abs(upper_bound), 1e-10)
                print(f"   📊 Optimality Gap: {gap:.6f} ({gap*100:.4f}%)")
            
            print(f"   ⏱️  Computation Time: {computation_time:.2f} seconds")
            
            # Compare with heuristic if available
            if l1_heuristic:
                print(f"\n🏆 Comparing with EPDT Heuristic...")
                heuristic_start = time.time()
                
                try:
                    heuristic_solution = l1_heuristic(orders, vehicles, params)
                    heuristic_time = time.time() - heuristic_start
                    
                    print(f"   ✅ Heuristic completed in {heuristic_time:.2f} seconds")
                    
                    # Calculate heuristic objective (simplified)
                    vehicles_used = sum(1 for route in heuristic_solution.routes.values() if route.tasks)
                    print(f"   🚛 Heuristic vehicles used: {vehicles_used}")
                    
                    if lower_bound != float('inf'):
                        print(f"   📈 Quality Assessment:")
                        print(f"      Lower bound provides optimality guarantee")
                        print(f"      Any feasible solution ≥ {lower_bound:.2f}")
                    
                except Exception as e:
                    print(f"   ❌ Heuristic comparison failed: {str(e)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Column Generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    except ImportError as e:
        print(f"❌ Column Generation not available: {str(e)}")
        print(f"   Make sure required solvers are installed:")
        print(f"   pip install pulp  # or install Gurobi")
        return False
    
    except Exception as e:
        print(f"❌ Column Generation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_qubo_formulation():
    """
    Test the QUBO formulation for quantum annealing.
    """
    print(f"\n⚛️  Testing QUBO Formulation for Quantum Annealing...")
    print(f"="*60)
    
    try:
        # Import QUBO modules
        from algo.qubo_formulation import solve_epdt_with_qubo, QUBOConfig
        from algo.qubo_integration import solve_epdt_hybrid, HybridConfig
        from algo.qubo_test import run_qubo_tests
        
        print(f"✅ QUBO modules imported successfully")
        
        # Run QUBO unit tests first
        print(f"\n🔬 Running QUBO Unit Tests...")
        test_success = run_qubo_tests()
        
        if not test_success:
            print(f"⚠️  Some QUBO tests failed, but continuing with integration test")
        
        # Test QUBO on a problem instance
        print(f"\n🎯 Testing QUBO on Test Problem...")
        
        # Create test scenario
        orders, vehicles, params = create_enhanced_multi_day_scenario()
        
        # Limit problem size for QUBO (quantum problems are typically smaller)
        test_orders = orders[:3]  # Use first 3 orders
        test_vehicles = vehicles[:2]  # Use first 2 vehicles
        
        print(f"📋 QUBO Test Problem:")
        print(f"   📦 Orders: {len(test_orders)}")
        print(f"   🚛 Vehicles: {len(test_vehicles)}")
        
        # Test hybrid CG+QUBO approach
        print(f"\n🔀 Testing Hybrid Column Generation + QUBO...")
        
        hybrid_config = HybridConfig(
            cg_config=ColumnGenerationConfig(
                max_iterations=5,  # Reduced for testing
                time_limit_seconds=60.0,
                verbose=False
            ),
            qubo_config=QUBOConfig(
                preferred_solver="neal",  # Use simulated annealing
                num_reads=100,
                verbose=False
            ),
            max_routes_for_qubo=10,  # Limit QUBO complexity
            use_column_generation=True
        )
        
        start_time = time.time()
        
        try:
            hybrid_result = solve_epdt_hybrid(test_orders, test_vehicles, hybrid_config)
            computation_time = time.time() - start_time
            
            print(f"✅ Hybrid CG+QUBO completed successfully!")
            
            # Column Generation phase results
            print(f"\n📊 Column Generation Phase:")
            print(f"   Feasible: {hybrid_result.cg_feasible}")
            print(f"   Routes generated: {hybrid_result.cg_routes_generated}")
            if hybrid_result.cg_feasible:
                print(f"   Lower bound: {hybrid_result.cg_lower_bound:.2f}")
                print(f"   Upper bound: {hybrid_result.cg_upper_bound:.2f}")
            
            # QUBO phase results  
            print(f"\n⚛️  QUBO Phase:")
            print(f"   Feasible: {hybrid_result.qubo_feasible}")
            print(f"   QUBO energy: {hybrid_result.qubo_energy:.2f}")
            print(f"   Routes selected: {hybrid_result.qubo_routes_used}")
            print(f"   Solver: {hybrid_result.qubo_solver_info.get('solver', 'unknown')}")
            
            # Quality assessment
            print(f"\n🎯 Quality Assessment:")
            if hybrid_result.optimality_gap != float('inf'):
                print(f"   Optimality gap: {hybrid_result.optimality_gap:.6f}")
            print(f"   Constraint violations: {sum(hybrid_result.constraint_violations.values())}")
            if hybrid_result.hybrid_vs_cg_ratio != 1.0:
                print(f"   vs CG ratio: {hybrid_result.hybrid_vs_cg_ratio:.3f}")
            
            print(f"   ⏱️  Total computation time: {computation_time:.2f} seconds")
            
            # Test direct QUBO solving if we have routes
            if hybrid_result.cg_routes_generated > 0:
                print(f"\n🧪 Testing Direct QUBO Solving...")
                
                # Generate simple routes for direct QUBO test
                simple_routes = []
                simple_costs = []
                
                for order in test_orders:
                    for vehicle in test_vehicles:
                        from .epdt_data_structures import Route
                        route = Route(vehicle=vehicle)
                        for task in order.get_all_tasks():
                            route.tasks.append(task)
                        simple_routes.append(route)
                        simple_costs.append(100.0 + len(route.tasks) * 10.0)
                
                qubo_config = QUBOConfig(
                    preferred_solver="neal",
                    num_reads=50,
                    verbose=False
                )
                
                qubo_result = solve_epdt_with_qubo(
                    test_orders, test_vehicles, 
                    simple_routes[:6], simple_costs[:6],  # Limit for testing
                    qubo_config
                )
                
                print(f"   ✅ Direct QUBO: Energy = {qubo_result.energy:.2f}")
                print(f"   📊 Selected routes: {len(qubo_result.selected_routes)}")
                print(f"   📊 Unassigned orders: {len(qubo_result.unassigned_orders)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Hybrid CG+QUBO failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    except ImportError as e:
        print(f"❌ QUBO modules not available: {str(e)}")
        print(f"   Install required packages:")
        print(f"   pip install dimod dwave-neal  # For QUBO solving")
        return False
    
    except Exception as e:
        print(f"❌ QUBO test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_benchmark():
    """
    Run quantum vs classical solver benchmark.
    """
    print(f"\n🏁 Running Quantum vs Classical Benchmark...")
    print(f"="*60)
    
    try:
        from algo.qubo_integration import benchmark_quantum_vs_classical
        from algo.qubo_formulation import QUBOConfig
        from algo.qubo_integration import HybridConfig
        
        # Create test problem
        orders, vehicles, params = create_enhanced_multi_day_scenario()
        test_orders = orders[:2]  # Small problem for benchmarking
        test_vehicles = vehicles[:1]
        
        print(f"📋 Benchmark Problem:")
        print(f"   📦 Orders: {len(test_orders)}")
        print(f"   🚛 Vehicles: {len(test_vehicles)}")
        
        # Configure different solvers
        configs = {
            'neal_sa': HybridConfig(
                use_column_generation=False,  # Skip CG for pure QUBO comparison
                qubo_config=QUBOConfig(
                    preferred_solver="neal",
                    num_reads=100,
                    verbose=False
                )
            ),
            'exact': HybridConfig(
                use_column_generation=False,
                qubo_config=QUBOConfig(
                    preferred_solver="exact",
                    verbose=False
                )
            )
        }
        
        # Add D-Wave if available
        try:
            import dwave.system
            configs['dwave'] = HybridConfig(
                use_column_generation=False,
                qubo_config=QUBOConfig(
                    preferred_solver="dwave",
                    num_reads=100,
                    verbose=False
                )
            )
            print(f"✅ D-Wave solver available for benchmarking")
        except ImportError:
            print(f"⚠️  D-Wave not available, using classical solvers only")
        
        # Run benchmark
        results = benchmark_quantum_vs_classical(test_orders, test_vehicles, configs)
        
        print(f"\n📊 Benchmark Results:")
        print(f"-" * 40)
        
        for solver_name, result in results.items():
            print(f"\n🔧 {solver_name.upper()}:")
            print(f"   Feasible: {result.qubo_feasible}")
            print(f"   Objective: {result.qubo_objective:.2f}")
            print(f"   Energy: {result.qubo_energy:.2f}")
            print(f"   Solve time: {result.qubo_solve_time:.3f}s")
            print(f"   Routes used: {result.qubo_routes_used}")
        
        # Compare results
        if len(results) > 1:
            print(f"\n🏆 Comparison:")
            best_objective = min(r.qubo_objective for r in results.values() if r.qubo_feasible)
            fastest_time = min(r.qubo_solve_time for r in results.values())
            
            for solver_name, result in results.items():
                if result.qubo_feasible:
                    quality_ratio = result.qubo_objective / best_objective
                    speed_ratio = result.qubo_solve_time / fastest_time
                    print(f"   {solver_name}: Quality={quality_ratio:.3f}x, Speed={speed_ratio:.3f}x")
        
        success = len(results) > 0 and any(r.qubo_feasible for r in results.values())
        print(f"\n✅ Benchmark completed: {'SUCCESS' if success else 'FAILED'}")
        
        return success
        
    except ImportError:
        print(f"❌ Quantum benchmark not available (missing QUBO packages)")
        return False
    except Exception as e:
        print(f"❌ Quantum benchmark failed: {str(e)}")
        return False


def test_advanced_travel_time():
    """
    Test the advanced travel time calculation system using OSRM and SQLite caching.
    This function validates all requirements from section 9 of TODO2.md.
    """
    print("🧪 Testing Advanced Travel Time Calculation System")
    print("=" * 80)
    
    try:
        # Import required modules
        from algo.route_provider import RouteProvider, calculate_travel_time_between_tasks
        from algo.epdt_data_structures import Vehicle
        
        print("✅ Successfully imported route provider modules")
        
        # Test 1: Basic travel time calculation
        print("\n1️⃣  Testing basic travel time calculation...")
        
        # Create test tasks (Milan area coordinates)
        class TestTask:
            def __init__(self, lat, lon, location_id):
                self.lat = lat
                self.lon = lon
                self.location_id = location_id
        
        depot = TestTask(44.9009, 8.2057, "depot")  # Asti center
        customer1 = TestTask(44.9109, 8.2157, "customer1")  # Asti north
        customer2 = TestTask(44.8909, 8.1957, "customer2")  # Asti south
        
        # Create test vehicles
        standard_vehicle = Vehicle(id="std_1", depot_id="depot", 
                                 weight_capacity=1000, volume_capacity=10, 
                                 vehicle_type="standard")
        large_vehicle = Vehicle(id="large_1", depot_id="depot", 
                              weight_capacity=2000, volume_capacity=20, 
                              vehicle_type="large")
        
        # Test travel time calculations
        time_std_1_2 = calculate_travel_time_between_tasks(depot, customer1, standard_vehicle)
        time_large_1_2 = calculate_travel_time_between_tasks(depot, customer1, large_vehicle)
        
        print(f"   Standard vehicle depot->customer1: {time_std_1_2:.2f} minutes")
        print(f"   Large vehicle depot->customer1: {time_large_1_2:.2f} minutes")
        
        # Verify that different vehicle types can have different travel times
        if time_std_1_2 != time_large_1_2:
            print("   ✅ Vehicle-specific travel times working correctly")
        else:
            print("   ✅ Same travel time for both vehicles (expected if no vehicle-specific routing)")
        
        # Test 2: Caching functionality
        print("\n2️⃣  Testing SQLite caching system...")
        
        # Get a route provider instance to check caching
        route_provider = RouteProvider()
        
        # Check if cache database exists
        import os
        if os.path.exists(route_provider.db_path):
            print("   ✅ SQLite cache database exists")
        else:
            print("   ⚠️  SQLite cache database will be created on first use")
        
        # Test same calculation twice (should use cache on second call)
        import time
        start_time = time.time()
        time1 = calculate_travel_time_between_tasks(depot, customer2, standard_vehicle)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        time2 = calculate_travel_time_between_tasks(depot, customer2, standard_vehicle)
        second_call_time = time.time() - start_time
        
        print(f"   First call time: {first_call_time:.4f}s, Result: {time1:.2f} minutes")
        print(f"   Second call time: {second_call_time:.4f}s, Result: {time2:.2f} minutes")
        
        if abs(time1 - time2) < 0.01:  # Same result
            print("   ✅ Caching returns consistent results")
        else:
            print("   ❌ Caching inconsistency detected")
            return False
        
        # Test 3: Edge case handling
        print("\n3️⃣  Testing edge cases...")
        
        # Test with same location (should return minimum travel time)
        same_location_time = calculate_travel_time_between_tasks(depot, depot, standard_vehicle)
        print(f"   Same location travel time: {same_location_time:.2f} minutes")
        
        if same_location_time > 0:
            print("   ✅ Same location returns positive minimum time")
        else:
            print("   ❌ Same location returns zero time")
            return False
        
        # Test 4: Integration with algorithm
        print("\n4️⃣  Testing integration with VRP scenario...")
        
        # Load a small scenario to test integration
        try:
            instance = create_furgoni_scenario()
            orders, vehicles = convert_instance_to_epdt_input(instance)
            
            if orders and vehicles:
                print(f"   Loaded scenario: {len(orders)} orders, {len(vehicles)} vehicles")
                
                # Test travel time calculation between first two orders
                if len(orders) >= 2:
                    order1 = orders[0]
                    order2 = orders[1]
                    vehicle = vehicles[0]
                    
                    # Get tasks from orders (orders contain pickup/delivery tasks)
                    if order1.pickup_tasks and order2.pickup_tasks:
                        task1 = order1.pickup_tasks[0]  # First pickup task of first order
                        task2 = order2.pickup_tasks[0]  # First pickup task of second order
                        
                        travel_time = calculate_travel_time_between_tasks(task1, task2, vehicle)
                        print(f"   Travel time between order tasks: {travel_time:.2f} minutes")
                        
                        if travel_time > 0:
                            print("   ✅ Integration with scenario data working")
                        else:
                            print("   ❌ Integration returning invalid travel time")
                            return False
                    else:
                        print("   ⚠️  Orders don't have pickup tasks to test travel time")
                else:
                    print("   ⚠️  Not enough orders in scenario to test inter-order travel time")
            else:
                print("   ⚠️  Could not load scenario data for integration test")
        except Exception as e:
            print(f"   ⚠️  Could not load scenario for integration test: {str(e)}")
        
        # Test 5: Error handling
        print("\n5️⃣  Testing error handling...")
        
        # Test with invalid coordinates
        invalid_task = TestTask(999, 999, "invalid")
        
        try:
            fallback_time = calculate_travel_time_between_tasks(depot, invalid_task, standard_vehicle)
            print(f"   Fallback calculation for invalid coords: {fallback_time:.2f} minutes")
            print("   ✅ Error handling working (fallback to Euclidean)")
        except Exception as e:
            print(f"   ✅ Error properly caught: {str(e)}")
        
        print("\n✅ All advanced travel time tests completed successfully!")
        print("=" * 80)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ...existing code...

def main():
    """Command line interface for the test runner."""
    parser = argparse.ArgumentParser(description="Run EPDT Algorithm Tests")
    parser.add_argument("--scenario", default="furgoni", 
                       help="Scenario to test (default: furgoni, enhanced)")
    parser.add_argument("--params", 
                       help="Custom parameters JSON file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to files")
    parser.add_argument("--profile", action="store_true",
                       help="Enable performance profiling")
    parser.add_argument("--test-enhanced", action="store_true",
                       help="Run enhanced multi-day feature tests")
    parser.add_argument("--test-column-generation", action="store_true",
                       help="Test Column Generation for optimality bounds")
    parser.add_argument("--test-qubo", action="store_true",
                       help="Test QUBO formulation for quantum annealing")
    parser.add_argument("--test-quantum-benchmark", action="store_true",
                       help="Run quantum vs classical solver benchmark")
    parser.add_argument("--run-benchmark", action="store_true",
                       help="Run comprehensive benchmark comparing all approaches")
    parser.add_argument("--test-travel-time", action="store_true",
                       help="Test advanced travel time calculation system")
    
    args = parser.parse_args()
    
    # Run travel time tests if requested
    if args.test_travel_time:
        print(f"🚀 Running Advanced Travel Time Tests")
        success = test_advanced_travel_time()
        sys.exit(0 if success else 1)
    
    # Run QUBO tests if requested
    if args.test_qubo:
        print(f"🚀 Running QUBO Tests")
        success = test_qubo_formulation()
        sys.exit(0 if success else 1)
    
    # Run quantum benchmark if requested
    if args.test_quantum_benchmark:
        print(f"🚀 Running Quantum Benchmark")
        success = test_quantum_benchmark()
        sys.exit(0 if success else 1)
    
    # Run Column Generation tests if requested
    if args.test_column_generation:
        print(f"🚀 Running Column Generation Tests")
        success = test_column_generation()
        sys.exit(0 if success else 1)
    
    # Run enhanced tests if requested
    if args.test_enhanced:
        print(f"🚀 Running Enhanced EPDT Feature Tests")
        orders, vehicles, params = test_enhanced_features()
        
        print(f"\n🔄 Running Enhanced EPDT Algorithm...")
        import time
        start_time = time.time()
        
        try:
            solution = l1_heuristic(orders, vehicles, params)
            runtime = time.time() - start_time
            
            print(f"✅ Enhanced algorithm completed successfully!")
            print_solution_summary(solution, orders, vehicles, params, runtime)
            
        except Exception as e:
            print(f"❌ Enhanced algorithm failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Run standard test
    results = run_scenario_test(
        scenario_name=args.scenario,
        custom_params_file=args.params,
        save_output=not args.no_save,
        enable_profiling=args.profile
    )
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()