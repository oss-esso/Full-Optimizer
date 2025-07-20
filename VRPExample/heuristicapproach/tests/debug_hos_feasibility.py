#!/usr/bin/env python3
"""
Debug HoS Feasibility Issues

This script helps debug why routes are showing as infeasible
when the advanced HoS simulation is used.
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
sys.path.insert(0, heuristic_root)

try:
    from algo.second_level import _simulate_hos_advanced, DriverState, is_feasible
    from algo.epdt_data_structures import Task, TaskType, Route, Vehicle
    print("✅ Successfully imported HoS simulation components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_simple_test_route():
    """Create a simple test route to debug HoS feasibility."""
    
    # Create a test vehicle
    vehicle = Vehicle(
        id="test_vehicle",
        depot_id="depot",
        weight_capacity=3500,
        volume_capacity=20.0,
        vehicle_type="standard"
    )
    
    # Create simple tasks
    pickup_task = Task(
        id="pickup_1",
        location_id="pickup_loc",
        task_type=TaskType.PICKUP,
        order_id="order_1",
        lat=40.7128,
        lon=-74.0060,
        service_time=30.0,  # 30 minutes
        demand=100.0,
        volume=2.0,
        day=0,  # Today
        earliest_time=480.0,  # 8:00 AM
        latest_time=600.0     # 10:00 AM
    )
    
    delivery_task = Task(
        id="delivery_1", 
        location_id="delivery_loc",
        task_type=TaskType.DELIVERY,
        order_id="order_1",
        lat=40.7589,
        lon=-73.9851,
        service_time=20.0,  # 20 minutes
        demand=-100.0,
        volume=-2.0,
        day=0,  # Today
        earliest_time=540.0,  # 9:00 AM
        latest_time=720.0     # 12:00 PM
    )
    
    # Create route
    route = Route(
        vehicle=vehicle,
        tasks=[pickup_task, delivery_task]
    )
    
    return route

def test_hos_simulation():
    """Test the HoS simulation with a simple route."""
    print("🧪 Testing HoS Simulation with Simple Route")
    print("=" * 50)
    
    # Create test route
    route = create_simple_test_route()
    
    print(f"📋 Test Route:")
    print(f"   🚚 Vehicle: {route.vehicle.id} ({route.vehicle.vehicle_type})")
    print(f"   📦 Tasks: {len(route.tasks)}")
    for i, task in enumerate(route.tasks):
        print(f"      {i+1}. {task.task_type.value} at {task.location_id} (day {getattr(task, 'day', 0)})")
        print(f"         Service time: {task.service_time} min, Load: {task.demand} kg")
    
    # Create initial driver state
    driver_state = DriverState()
    print(f"\n🚗 Initial Driver State:")
    print(f"   Drive since break: {driver_state.drive_since_break} min")
    print(f"   Work since break: {driver_state.work_since_break} min")
    print(f"   Drive today: {driver_state.drive_today} min")
    print(f"   Work today: {driver_state.work_today} min")
    
    # Test advanced HoS simulation
    print(f"\n🔬 Running Advanced HoS Simulation...")
    try:
        feasible, total_time = _simulate_hos_advanced(route, driver_state, route.tasks)
        print(f"   ✅ Simulation completed successfully")
        print(f"   📊 Result: Feasible = {feasible}")
        print(f"   ⏱️  Total time: {total_time:.2f} minutes")
        
        if not feasible:
            print(f"   ❌ Route failed HoS feasibility check")
            print(f"   🔍 This indicates the HoS simulation detected a violation")
        else:
            print(f"   ✅ Route passed HoS feasibility check")
            
    except Exception as e:
        print(f"   ❌ Error during HoS simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test overall route feasibility
    print(f"\n🔬 Testing Overall Route Feasibility...")
    try:
        overall_feasible = is_feasible(route)
        print(f"   📊 Overall feasibility: {overall_feasible}")
        
        if feasible and not overall_feasible:
            print(f"   ⚠️  HoS passed but overall feasibility failed")
            print(f"   🔍 This suggests another constraint is failing")
        elif not feasible and not overall_feasible:
            print(f"   ❌ Both HoS and overall feasibility failed")
        elif feasible and overall_feasible:
            print(f"   ✅ Both HoS and overall feasibility passed")
            
    except Exception as e:
        print(f"   ❌ Error during overall feasibility check: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_complex_route():
    """Test with a more complex route that might trigger HoS limits."""
    print("\n🧪 Testing HoS Simulation with Complex Route")
    print("=" * 50)
    
    # Create a test vehicle
    vehicle = Vehicle(
        id="test_vehicle_complex",
        depot_id="depot", 
        weight_capacity=3500,
        volume_capacity=20.0,
        vehicle_type="standard"
    )
    
    # Create multiple tasks that might require long driving times
    tasks = []
    for i in range(6):  # 6 orders = 12 tasks
        pickup_task = Task(
            id=f"pickup_{i+1}",
            location_id=f"pickup_loc_{i+1}",
            task_type=TaskType.PICKUP,
            order_id=f"order_{i+1}",
            lat=40.7128 + i * 0.1,  # Spread out locations
            lon=-74.0060 + i * 0.1,
            service_time=45.0,  # 45 minutes service time
            demand=300.0,
            volume=3.0,
            day=0,
            earliest_time=480.0 + i * 120,  # Spread out time windows
            latest_time=600.0 + i * 120
        )
        
        delivery_task = Task(
            id=f"delivery_{i+1}",
            location_id=f"delivery_loc_{i+1}",
            task_type=TaskType.DELIVERY,
            order_id=f"order_{i+1}",
            lat=40.7589 + i * 0.15,  # Different locations for deliveries
            lon=-73.9851 + i * 0.15,
            service_time=30.0,  # 30 minutes service time
            demand=-300.0,
            volume=-3.0,
            day=0,
            earliest_time=540.0 + i * 120,  # Later than pickup
            latest_time=720.0 + i * 120
        )
        
        tasks.append(pickup_task)
        tasks.append(delivery_task)
    
    # Create route
    route = Route(vehicle=vehicle, tasks=tasks)
    
    print(f"📋 Complex Test Route:")
    print(f"   🚚 Vehicle: {route.vehicle.id}")
    print(f"   📦 Tasks: {len(route.tasks)} ({len(route.tasks)//2} orders)")
    print(f"   ⏰ Time span: {tasks[0].earliest_time:.0f} - {tasks[-1].latest_time:.0f} minutes")
    
    # Create initial driver state
    driver_state = DriverState()
    
    # Test advanced HoS simulation
    print(f"\n🔬 Running Advanced HoS Simulation on Complex Route...")
    try:
        feasible, total_time = _simulate_hos_advanced(route, driver_state, route.tasks)
        print(f"   📊 Result: Feasible = {feasible}")
        print(f"   ⏱️  Total time: {total_time:.2f} minutes ({total_time/60:.1f} hours)")
        
        if total_time > 8 * 60:  # More than 8 hours
            print(f"   ⚠️  Route requires more than 8 hours - may trigger HoS limits")
        if total_time > 13 * 60:  # More than 13 hours (max work time)
            print(f"   ❌ Route exceeds maximum work time (13 hours)")
            
    except Exception as e:
        print(f"   ❌ Error during complex route HoS simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main testing function."""
    print("🔍 HoS Feasibility Debug Tool")
    print("=" * 60)
    
    # Test simple route
    if not test_hos_simulation():
        print("❌ Simple route test failed")
        return
    
    # Test complex route  
    if not test_complex_route():
        print("❌ Complex route test failed")
        return
    
    print("\n" + "=" * 60)
    print("✅ HoS Debug Testing Complete")

if __name__ == "__main__":
    main()
