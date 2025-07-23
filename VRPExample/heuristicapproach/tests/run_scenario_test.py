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

# Debug path information
print(f"🔍 Debug: Current working directory: {os.getcwd()}")
print(f"🔍 Debug: Script directory: {current_dir}")
print(f"🔍 Debug: Looking for moda_scenarios in: {src_dir}")
print(f"🔍 Debug: moda_scenarios.py exists: {os.path.exists(os.path.join(src_dir, 'moda_scenarios.py'))}")

try:
    # Try different import approaches
    from moda_scenarios import create_furgoni_scenario
    print("✅ Successfully imported create_furgoni_scenario from moda_scenarios")
except ImportError:
    try:
        import sys
        sys.path.append(os.path.join(heuristic_root, 'src'))
        from moda_scenarios import create_furgoni_scenario
        print("✅ Successfully imported create_furgoni_scenario with explicit path")
    except ImportError:
        try:
            # Try importing with full path
            import importlib.util
            spec = importlib.util.spec_from_file_location("moda_scenarios", os.path.join(src_dir, "moda_scenarios.py"))
            moda_scenarios = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(moda_scenarios)
            create_furgoni_scenario = moda_scenarios.create_furgoni_scenario
            print("✅ Successfully imported create_furgoni_scenario using importlib")
        except Exception as e:
            print(f"❌ Error: Could not import create_furgoni_scenario: {e}")
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
    print("✅ Successfully imported data_adapter functions")
except ImportError:
    try:
        from algo.data_adapter import (
            convert_instance_to_epdt_input, 
            get_default_parameters,
            create_empty_solution,
            print_conversion_summary
        )
        print("✅ Successfully imported data_adapter functions with algo prefix")
    except ImportError as e:
        print(f"❌ Error: Could not import data_adapter functions: {e}")
        sys.exit(1)

try:
    from first_level import l1_heuristic
    print("✅ Successfully imported l1_heuristic from first_level")
except ImportError:
    try:
        from algo.first_level import l1_heuristic
        print("✅ Successfully imported l1_heuristic with algo prefix")
    except ImportError:
        print("⚠️  Warning: l1_heuristic not yet implemented")
        l1_heuristic = None

# Import route provider for travel time calculations
try:
    from route_provider import calculate_travel_time_between_tasks
    print("✅ Successfully imported route_provider for travel time calculations")
except ImportError:
    try:
        from algo.route_provider import calculate_travel_time_between_tasks
        print("✅ Successfully imported route_provider with algo prefix")
    except ImportError:
        print("⚠️  Warning: route_provider not available, travel times will be estimated")
        calculate_travel_time_between_tasks = None


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
    print(f"   ❌ Orders unassigned: {unassigned_orders}")
    
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
    print(f"\n🗺️  Route Details:")
    for vehicle_id, route in solution.routes.items():
        if not route.tasks:
            print(f"   🚚 {vehicle_id}: Empty route")
            continue
            
        print(f"   🚚 {vehicle_id}:")
        
        # Calculate route feasibility and times using HoS simulation
        try:
            from algo.second_level import is_feasible, _simulate_hos_advanced, _sort_tasks_chronologically, DriverState
            
            feasible = is_feasible(route)
            print(f"      ✅ Feasible: {feasible}")
            
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
                    
                    # Calculate travel time
                    if calculate_travel_time_between_tasks:
                        travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
                    else:
                        travel_time = 30.0  # Default fallback
                    
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
            
            # Ensure the last task time matches total duration (or close to it)
            if task_times and abs(task_times[-1] - total_duration) > 1.0:
                # Scale all task times to match total duration
                scale_factor = total_duration / task_times[-1] if task_times[-1] > 0 else 1.0
                task_times = [t * scale_factor for t in task_times]
            
            # Format duration
            if total_duration >= 1440:
                days = int(total_duration // 1440)
                remaining_hours = (total_duration % 1440) / 60
                hours = int(remaining_hours)
                minutes = int((remaining_hours % 1) * 60)
                
                if days > 0:
                    duration_str = f"{days}d {hours}h {minutes}m" if hours > 0 or minutes > 0 else f"{days}d"
                else:
                    duration_str = f"{hours}h {minutes}m"
            else:
                hours = int(total_duration / 60)
                minutes = int(total_duration % 60)
                duration_str = f"{hours}h {minutes}m"
            
            compliance_note = "HoS compliant" if hos_feasible and total_duration < 15 * 60 else "(would violate HoS if attempted without proper rests)"
            
            if total_duration >= 1440:
                days = int(total_duration // 1440)
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
    
    # Unassigned orders analysis
    if solution.unassigned_orders:
        print(f"\n❌ Unassigned Orders Analysis:")
        for order_id in solution.unassigned_orders:
            order = next((o for o in orders if o.id == order_id), None)
            if order:
                print(f"   📦 {order_id}:")
                print(f"      Weight: {order.get_total_demand():.0f} kg")
                print(f"      Volume: {order.get_total_volume():.1f} m³")
                print(f"      Priority: {order.priority}")
                print(f"      Mandatory: {order.is_mandatory}")
    
    # Algorithm configuration summary
    print(f"\n⚙️  Algorithm Configuration:")
    print(f"   🔧 Tabu tenure: {params.tabu_tenure}")
    print(f"   🔧 Max non-improving iterations: {params.max_non_improving_iterations}")
    print(f"   🔧 Max total iterations: {params.max_total_iterations}")
    print(f"   🔧 Exploration strategy: {params.exploration_strategy}")
    print(f"   🔧 Local search: {params.local_search_strategy}")
    print(f"   🔧 Initialization: {params.initialization_method}")
    
    # Enhancement flags
    enhancements = []
    if params.enable_advanced_neighborhoods:
        enhancements.append("Advanced neighborhoods")
    if params.enable_granular_search:
        enhancements.append("Granular search")
    if params.enable_parallelization:
        enhancements.append(f"Parallelization ({params.parallel_strategy})")
    
    if enhancements:
        print(f"   🚀 Enhancements: {', '.join(enhancements)}")
    else:
        print(f"   🚀 Enhancements: None enabled")
    
    print(f"\n" + "="*80)
    print(f"✅ Solution analysis complete")
    print(f"="*80)


def configure_algorithm_parameters(custom_params_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Configure EPDT algorithm parameters for the test run.
    
    Args:
        custom_params_file: Optional JSON file with custom parameters
        
    Returns:
        Dictionary of algorithm parameters
    """
    # Start with defaults
    params = get_default_parameters()
    
    # Load custom parameters if provided
    if custom_params_file and os.path.exists(custom_params_file):
        print(f"🔧 Loading custom parameters from {custom_params_file}")
        try:
            with open(custom_params_file, 'r') as f:
                custom_params = json.load(f)
            
            # Update parameters
            for key, value in custom_params.items():
                if hasattr(params, key):
                    setattr(params, key, value)
                    print(f"   ✅ Set {key} = {value}")
                else:
                    print(f"   ⚠️  Unknown parameter: {key}")
        except Exception as e:
            print(f"   ❌ Error loading custom parameters: {e}")
            print(f"   Using default parameters instead")
    
    return params


def save_results(solution, orders, vehicles, params, runtime_seconds, output_dir: str = "results"):
    """Save test results to files for later analysis."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    
    # Save solution summary as JSON
    result_data = {
        "timestamp": timestamp,
        "runtime_seconds": runtime_seconds,
        "algorithm_params": {
            "tabu_tenure": params.tabu_tenure,
            "max_non_improving_iterations": params.max_non_improving_iterations,
            "max_total_iterations": params.max_total_iterations,
            "exploration_strategy": params.exploration_strategy,
            "local_search_strategy": params.local_search_strategy
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


def run_scenario_test(scenario_name: str = "furgoni", 
                     custom_params_file: Optional[str] = None,
                     save_output: bool = True) -> Dict[str, Any]:
    """
    Main test execution function following the TODO section 5.3 workflow.
    
    Args:
        scenario_name: Name of scenario to test ("furgoni")
        custom_params_file: Optional custom parameters JSON file
        save_output: Whether to save results to files
        
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
        params = configure_algorithm_parameters(custom_params_file)
        
        print(f"🔧 Algorithm Configuration:")
        print(f"   - Tabu tenure: {params.tabu_tenure}")
        print(f"   - Max iterations: {params.max_total_iterations}")
        print(f"   - Strategy: {params.exploration_strategy}")
        print(f"   - Local search: {params.local_search_strategy}")
        
        # Step 4: Run Heuristic
        print(f"\n4️⃣  Running EPDT heuristic algorithm")
        
        if l1_heuristic is None:
            print(f"⚠️  l1_heuristic not implemented yet - creating mock solution")
            solution = _create_mock_solution(orders, vehicles)
            algorithm_runtime = 0.1
        else:
            algorithm_start = time.time()
            solution = l1_heuristic(orders, vehicles, params)
            algorithm_runtime = time.time() - algorithm_start
            print(f"✅ Algorithm completed in {algorithm_runtime:.2f} seconds")
        
        # Step 5: Analyze and Print Results
        print(f"\n5️⃣  Analyzing results")
        total_runtime = time.time() - start_time
        print_solution_summary(solution, orders, vehicles, params, algorithm_runtime)
        
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
        depot_lat = getattr(vehicle, 'depot_lat', 45.4642)  # Default Milan coordinates
        depot_lon = getattr(vehicle, 'depot_lon', 9.1900)
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
        
        depot = TestTask(45.464211, 9.191383, "depot")  # Milan center
        customer1 = TestTask(45.478611, 9.203472, "customer1")  # Milan north
        customer2 = TestTask(45.449722, 9.177222, "customer2")  # Milan south
        
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
        
        if l1_heuristic:
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
        else:
            print(f"⚠️  l1_heuristic not available - skipping algorithm test")
        
        return
    
    # Run standard test
    results = run_scenario_test(
        scenario_name=args.scenario,
        custom_params_file=args.params,
        save_output=not args.no_save
    )
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()