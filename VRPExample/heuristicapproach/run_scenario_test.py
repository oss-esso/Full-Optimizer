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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'VRPInstance'))

try:
    from src.moda_scenarios import create_furgoni_scenario
except ImportError:
    try:
        from moda_scenarios import create_furgoni_scenario
    except ImportError:
        print("❌ Error: Could not import create_furgoni_scenario")
        print("   Make sure moda_scenarios.py is available")
        sys.exit(1)

from algo.data_adapter import (
    convert_instance_to_epdt_input, 
    get_default_parameters,
    create_empty_solution,
    print_conversion_summary
)

try:
    from algo.first_level import l1_heuristic
except ImportError:
    print("⚠️  Warning: l1_heuristic not yet implemented")
    l1_heuristic = None


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
        
        # Calculate route Z2 score if available
        if hasattr(route, 'is_feasible') and hasattr(route, '_cached_score'):
            try:
                feasible = route.is_feasible()
                print(f"      ✅ Feasible: {feasible}")
                
                if route._cached_score is not None:
                    print(f"      📈 Z2 Score: {route._cached_score:.2f}")
            except Exception as e:
                print(f"      ⚠️  Route validation skipped: {str(e)}")
                # Continue without failing - this is just for analysis
        
        # Task sequence
        print(f"      📋 Task sequence ({len(route.tasks)} tasks):")
        current_load_weight = 0
        current_load_volume = 0
        
        for i, task in enumerate(route.tasks):
            current_load_weight += task.demand
            current_load_volume += task.volume
            
            task_type_icon = "📦" if task.is_pickup() else "🏪"
            print(f"         {i+1:2d}. {task_type_icon} {task.location_id} (Order: {task.order_id})")
            print(f"             Load: {task.demand:+.0f}kg, {task.volume:+.1f}m³ → Total: {current_load_weight:.0f}kg, {current_load_volume:.1f}m³")
    
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
        
        print(f"      ✅ Route created: {len(route.tasks)} tasks ({len(all_pickup_tasks)} pickups → {len(all_delivery_tasks)} deliveries)")
    
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
            print(f"      Pattern: {len(vehicle_orders[vehicle.id])} pickups → {len(vehicle_orders[vehicle.id])} deliveries")
    
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


def main():
    """Command line interface for the test runner."""
    parser = argparse.ArgumentParser(description="Run EPDT Algorithm Tests")
    parser.add_argument("--scenario", default="furgoni", 
                       help="Scenario to test (default: furgoni)")
    parser.add_argument("--params", 
                       help="Custom parameters JSON file")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to files")
    
    args = parser.parse_args()
    
    # Run the test
    results = run_scenario_test(
        scenario_name=args.scenario,
        custom_params_file=args.params,
        save_output=not args.no_save
    )
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
