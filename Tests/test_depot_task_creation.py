"""
Targeted test for Issue 5: Ensure Depot Tasks are Added to All Final Routes

This test creates a small scenario with just a few orders and verifies that
all routes in the final solution have proper depot start and end tasks.
"""

import sys
import os

# Add the necessary paths to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
vrp_example_dir = os.path.join(current_dir, "..", "VRPExample", "heuristicapproach")
algo_dir = os.path.join(vrp_example_dir, "algo")
sys.path.insert(0, vrp_example_dir)
sys.path.insert(0, algo_dir)

def test_depot_task_creation():
    """Test that all routes have proper depot start and end tasks."""
    try:
        # Import necessary modules
        from epdt_data_structures import Order, Vehicle, TaskType, Task
        from first_level import l1_heuristic
        
        print("🧪 Testing depot task creation...")
        
        # Create a simple test scenario with 3 orders and 2 vehicles
        orders = []
        
        # Order 1
        pickup_task_1 = Task(
            id="PICKUP_001",
            location_id="PICKUP_LOC_001",
            task_type=TaskType.PICKUP,
            order_id="ORDER_001",
            lat=44.9100,
            lon=8.2100,
            service_time=15.0,
            demand=10.0,
            volume=5.0,
            earliest_time=480,  # 8:00 AM
            latest_time=600     # 10:00 AM
        )
        
        delivery_task_1 = Task(
            id="DELIVERY_001",
            location_id="DELIVERY_LOC_001",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_001",
            lat=44.9200,
            lon=8.2200,
            service_time=15.0,
            demand=-10.0,  # Negative for delivery
            volume=-5.0,   # Negative for delivery
            earliest_time=660,  # 11:00 AM
            latest_time=780     # 1:00 PM
        )
        
        order_1 = Order(
            id="ORDER_001",
            pickup_tasks=[pickup_task_1],
            delivery_tasks=[delivery_task_1]
        )
        orders.append(order_1)
        
        # Order 2
        pickup_task_2 = Task(
            id="PICKUP_002",
            location_id="PICKUP_LOC_002",
            task_type=TaskType.PICKUP,
            order_id="ORDER_002",
            lat=44.9150,
            lon=8.2150,
            service_time=15.0,
            demand=15.0,
            volume=8.0,
            earliest_time=540,  # 9:00 AM
            latest_time=660     # 11:00 AM
        )
        
        delivery_task_2 = Task(
            id="DELIVERY_002",
            location_id="DELIVERY_LOC_002",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_002",
            lat=44.9250,
            lon=8.2250,
            service_time=15.0,
            demand=-15.0,  # Negative for delivery
            volume=-8.0,   # Negative for delivery
            earliest_time=720,  # 12:00 PM
            latest_time=840     # 2:00 PM
        )
        
        order_2 = Order(
            id="ORDER_002",
            pickup_tasks=[pickup_task_2],
            delivery_tasks=[delivery_task_2]
        )
        orders.append(order_2)
        
        # Create 2 vehicles
        vehicles = [
            Vehicle(
                id="VEHICLE_001",
                depot_id="DEPOT-ASTI",
                weight_capacity=100.0,
                volume_capacity=50.0,
                pallet_capacity=20  # Set pallet capacity to avoid None comparison
            ),
            Vehicle(
                id="VEHICLE_002", 
                depot_id="DEPOT-ASTI",
                weight_capacity=120.0,
                volume_capacity=60.0,
                pallet_capacity=25  # Set pallet capacity to avoid None comparison
            )
        ]
        
        # Set up parameters for the optimizer
        params = {
            'M1': 10,  # Reduced for faster testing
            'M2': 5,   # Reduced for faster testing
            'tabu_tenure': 3,  # Reduced for faster testing
            'exploration_strategy': 'local_search',
            'initialization_method': 'cluster_aware',
            'enable_destroy_and_repair': False,
            'enforce_pickup_first': True,
            'max_iterations': 2  # Very small for testing
        }
        
        print(f"📊 Running l1_heuristic with {len(orders)} orders and {len(vehicles)} vehicles...")
        
        # Run the optimizer
        solution = l1_heuristic(orders, vehicles, params)
        
        print(f"✅ Optimizer completed. Checking {len(solution.routes)} routes...")
        
        # Test: Verify that all routes have proper depot tasks
        routes_checked = 0
        routes_with_tasks = 0
        
        for vehicle_id, route in solution.routes.items():
            print(f"🔍 Checking route for vehicle {vehicle_id}...")
            
            routes_checked += 1
            
            # Skip empty routes (they should not have depot tasks)
            if not hasattr(route, 'tasks') or not route.tasks:
                print(f"   ⚪ Route {vehicle_id} is empty - OK")
                continue
                
            routes_with_tasks += 1
            print(f"   📋 Route {vehicle_id} has {len(route.tasks)} tasks")
            
            # Verify first task is depot start
            first_task = route.tasks[0]
            assert first_task.is_depot_start(), f"First task in route {vehicle_id} is not a depot start: {first_task.task_type}"
            print(f"   ✅ First task is depot start: {first_task.id}")
            
            # Verify last task is depot return  
            last_task = route.tasks[-1]
            assert last_task.is_depot_return(), f"Last task in route {vehicle_id} is not a depot return: {last_task.task_type}"
            print(f"   ✅ Last task is depot return: {last_task.id}")
            
            # Print all tasks for debugging
            print(f"   📝 Task sequence for route {vehicle_id}:")
            for i, task in enumerate(route.tasks):
                print(f"     {i+1}. {task.id} ({task.task_type})")
        
        print(f"\n🎯 Test Results:")
        print(f"   Total routes checked: {routes_checked}")
        print(f"   Routes with tasks: {routes_with_tasks}")
        print(f"   All depot task assertions passed: ✅")
        
        # Additional verification: Check that we have some assigned orders
        assigned_orders = 0
        for vehicle_id, route in solution.routes.items():
            if hasattr(route, 'tasks') and route.tasks:
                # Count non-depot tasks
                for task in route.tasks:
                    if not task.is_depot_start() and not task.is_depot_return():
                        assigned_orders += 1
        
        print(f"   Orders assigned to routes: {assigned_orders}")
        
        if assigned_orders == 0:
            print("⚠️  Warning: No orders were assigned to any routes")
        
        print("\n🎉 TEST PASSED: All routes have proper depot start and end tasks!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_depot_task_creation()
    sys.exit(0 if success else 1)
