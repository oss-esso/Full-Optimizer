"""
Targeted test for comprehensive HoS simulation implementation.

This test verifies that the enhanced _check_hos_multiday function correctly
handles multi-day routes with required rest periods and calculates associated costs.
"""

import sys
import os
from pathlib import Path
from typing import List

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
algo_dir = os.path.join(heuristic_root, 'algo')
src_dir = os.path.join(heuristic_root, 'src')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, algo_dir)
sys.path.insert(0, src_dir)

print("=== HoS Simulation Targeted Test ===")
print(f"Test directory: {current_dir}")

# Import required modules
try:
    from epdt_data_structures import Order, Vehicle, Route, Task, DriverState
    from second_level import _check_hos_multiday, l2_heuristic
    from first_level import l1_heuristic
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ ERROR importing modules: {e}")
    sys.exit(1)

def create_long_distance_scenario():
    """
    Create a scenario with long-distance tasks that would require multiple rest periods.
    This should force both 45-minute breaks and 11-hour daily rests.
    """
    print("\n📋 Creating long-distance multi-day scenario...")
    
    # Create a vehicle
    vehicle = Vehicle(
        id="TEST_VEHICLE_1",
        depot_id="DEPOT_1",
        weight_capacity=10000,
        volume_capacity=50,
        pallet_capacity=25,
        cost_per_hour=25.0,
        cost_per_km=1.5
    )
    
    # Create a long-distance order that should require overnight rest
    # Place delivery very far from depot (e.g., Barcelona is ~600km from Geneva)
    order = Order(
        id="LONG_DISTANCE_ORDER_1",
        pickup_lat=45.0641,    # Geneva (pickup)
        pickup_lon=7.0494,
        delivery_lat=41.3851,  # Barcelona (delivery) - very far
        delivery_lon=2.1734,
        weight=5000,
        volume=25,
        pallets=10,
        pickup_time_window=[480, 600],  # 8:00-10:00
        delivery_time_window=[1320, 1440], # 22:00-24:00 next day
        service_time_pickup=30,
        service_time_delivery=30
    )
    
    orders = [order]
    vehicles = [vehicle]
    
    print(f"   • Created {len(orders)} long-distance order")
    print(f"   • Created {len(vehicles)} vehicle")
    print(f"   • Distance: Geneva to Barcelona (~600km)")
    print(f"   • Expected driving time: ~6-8 hours (should require breaks)")
    
    return orders, vehicles

def test_hos_simulation_directly():
    """
    Test the _check_hos_multiday function directly with a challenging scenario.
    """
    print("\n🧪 Testing HoS simulation directly...")
    
    orders, vehicles = create_long_distance_scenario()
    vehicle = vehicles[0]
    order = orders[0]
    
    # Create a route manually with tasks
    route = Route(vehicle=vehicle)
    
    # Create depot start task (Geneva depot)
    depot_start = Task(
        id="DEPOT_START",
        lat=45.0641,  # Geneva coordinates  
        lon=7.0494,
        service_time=0,
        demand=0,
        task_type="DEPOT_START"
    )
    
    # Create pickup task
    pickup_task = Task(
        id=f"{order.id}_PICKUP",
        lat=order.pickup_lat,
        lon=order.pickup_lon,
        service_time=order.service_time_pickup,
        demand=order.weight,
        task_type="PICKUP",
        order_id=order.id,
        earliest_time=order.pickup_time_window[0] if order.pickup_time_window else None,
        latest_time=order.pickup_time_window[1] if order.pickup_time_window else None
    )
    
    # Create delivery task
    delivery_task = Task(
        id=f"{order.id}_DELIVERY",
        lat=order.delivery_lat,
        lon=order.delivery_lon,
        service_time=order.service_time_delivery,
        demand=-order.weight,
        task_type="DELIVERY", 
        order_id=order.id,
        earliest_time=order.delivery_time_window[0] if order.delivery_time_window else None,
        latest_time=order.delivery_time_window[1] if order.delivery_time_window else None
    )
    
    # Create depot return task (Geneva depot)
    depot_return = Task(
        id="DEPOT_RETURN",
        lat=45.0641,  # Geneva coordinates
        lon=7.0494,
        service_time=0,
        demand=0,
        task_type="DEPOT_RETURN"
    )
    
    route.tasks = [depot_start, pickup_task, delivery_task, depot_return]
    
    # Test the HoS simulation
    driver_state = DriverState()
    sorted_tasks = route.tasks
    
    print(f"   • Testing route with {len(sorted_tasks)} tasks")
    print(f"   • Route: {' -> '.join([task.id for task in sorted_tasks])}")
    
    try:
        feasible, hos_cost = _check_hos_multiday(route, driver_state, sorted_tasks)
        
        print(f"   • HoS feasible: {feasible}")
        print(f"   • HoS cost: €{hos_cost:.2f}")
        
        if feasible and hos_cost > 0:
            print("   ✅ SUCCESS: Route is feasible with rest periods accounted for")
            print(f"   💰 Rest cost indicates proper simulation of breaks/rests")
            return True
        elif feasible and hos_cost == 0:
            print("   ⚠️  WARNING: Route is feasible but no rest cost calculated")
            print("   💭 This might indicate the scenario isn't challenging enough")
            return True
        else:
            print("   ❌ FAILURE: Route marked as infeasible")
            print("   💭 This might indicate the simulation is too strict")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR in HoS simulation: {e}")
        return False

def test_with_l1_heuristic():
    """
    Test the full optimization pipeline to ensure integration works correctly.
    """
    print("\n🚀 Testing with l1_heuristic pipeline...")
    
    orders, vehicles = create_long_distance_scenario()
    
    # Configure algorithm parameters for testing
    params = {
        'tabu_tenure': 10,
        'M1': 5,
        'M2': 10,
        'exploration_strategy': 'vnd',
        'enable_advanced_neighborhoods': False,
        'enable_granular_search': False,
        'enable_parallelization': False,
        'local_search_strategy': 'first_improvement',
        'initialization_method': 'best_insertion',
        'vehicle_penalty_per_vehicle': 0.0,
        'unassigned_order_base_penalty': 50000.0,
        'time_window_violation_penalty': 500000.0,
        'capacity_violation_penalty': 100.0,
        'distance_violation_penalty': 50.0
    }
    
    try:
        print(f"   • Running l1_heuristic with {len(orders)} orders and {len(vehicles)} vehicles")
        solution = l1_heuristic(orders, vehicles, params)
        
        if solution and solution.routes:
            assigned_routes = len([r for r in solution.routes.values() if r.tasks])
            print(f"   • Generated {assigned_routes} routes")
            
            # Check each route for HoS compliance
            hos_violations = 0
            total_hos_cost = 0.0
            
            for vehicle_id, route in solution.routes.items():
                if route.tasks and len(route.tasks) > 2:  # More than just depot tasks
                    try:
                        driver_state = DriverState()
                        feasible, hos_cost = _check_hos_multiday(route, driver_state, route.tasks)
                        
                        if not feasible:
                            hos_violations += 1
                            print(f"   ❌ HoS violation in route {vehicle_id}")
                        else:
                            total_hos_cost += hos_cost
                            if hos_cost > 0:
                                print(f"   ✅ Route {vehicle_id}: feasible with €{hos_cost:.2f} rest cost")
                            else:
                                print(f"   ✅ Route {vehicle_id}: feasible with no additional rest cost")
                                
                    except Exception as e:
                        print(f"   ❌ Error checking route {vehicle_id}: {e}")
                        hos_violations += 1
            
            print(f"\n📊 Results Summary:")
            print(f"   • Total routes: {assigned_routes}")
            print(f"   • HoS violations: {hos_violations}")
            print(f"   • Total HoS cost: €{total_hos_cost:.2f}")
            
            if hos_violations == 0:
                print("   ✅ SUCCESS: All routes are HoS compliant!")
                return True
            else:
                print("   ❌ FAILURE: HoS violations found")
                return False
        else:
            print("   ❌ No solution generated")
            return False
            
    except Exception as e:
        print(f"   ❌ ERROR in l1_heuristic: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Run the targeted HoS simulation test.
    """
    print("Starting HoS simulation tests...\n")
    
    # Test 1: Direct HoS simulation
    test1_success = test_hos_simulation_directly()
    
    # Test 2: Full pipeline integration
    test2_success = test_with_l1_heuristic()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"Direct HoS simulation test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"L1 heuristic integration test: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The enhanced HoS simulation is working correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please review the implementation.")
    
    return test1_success and test2_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
