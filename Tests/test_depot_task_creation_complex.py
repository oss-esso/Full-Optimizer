"""
Enhanced Targeted Test for Issue 5: Depot Tasks Under Complex Optimization

This test creates a more complex scenario with multiple orders and vehicles,
allowing the optimizer to run longer and test that depot tasks remain intact
through various optimization phases including VND, neighborhood operations,
and potential destroy/repair operations.
"""

import sys
import os

# Add the necessary paths to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
vrp_example_dir = os.path.join(current_dir, "..", "VRPExample", "heuristicapproach")
algo_dir = os.path.join(vrp_example_dir, "algo")
sys.path.insert(0, vrp_example_dir)
sys.path.insert(0, algo_dir)

def test_depot_task_creation_complex():
    """Test that all routes maintain proper depot start and end tasks through complex optimization."""
    try:
        # Import necessary modules
        from epdt_data_structures import Order, Vehicle, TaskType, Task
        from first_level import l1_heuristic
        
        print("🧪 Testing depot task creation under complex optimization...")
        
        # Create a complex test scenario with 6 orders and 4 vehicles
        orders = []
        
        # Order 1 - Early morning delivery
        pickup_task_1 = Task(
            id="PICKUP_001",
            location_id="PICKUP_LOC_001",
            task_type=TaskType.PICKUP,
            order_id="ORDER_001",
            lat=44.9100,
            lon=8.2100,
            service_time=20.0,
            demand=25.0,
            volume=12.0,
            pallets=3,
            earliest_time=420,  # 7:00 AM
            latest_time=540     # 9:00 AM
        )
        
        delivery_task_1 = Task(
            id="DELIVERY_001",
            location_id="DELIVERY_LOC_001",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_001",
            lat=44.9200,
            lon=8.2200,
            service_time=25.0,
            demand=-25.0,  # Negative for delivery
            volume=-12.0,   # Negative for delivery
            pallets=-3,
            earliest_time=600,  # 10:00 AM
            latest_time=720     # 12:00 PM
        )
        
        order_1 = Order(
            id="ORDER_001",
            pickup_tasks=[pickup_task_1],
            delivery_tasks=[delivery_task_1],
            priority=2,
            is_urgent=True
        )
        orders.append(order_1)
        
        # Order 2 - Large order with tight time window
        pickup_task_2 = Task(
            id="PICKUP_002",
            location_id="PICKUP_LOC_002",
            task_type=TaskType.PICKUP,
            order_id="ORDER_002",
            lat=44.9300,
            lon=8.2300,
            service_time=30.0,
            demand=45.0,
            volume=20.0,
            pallets=5,
            earliest_time=480,  # 8:00 AM
            latest_time=600     # 10:00 AM
        )
        
        delivery_task_2 = Task(
            id="DELIVERY_002",
            location_id="DELIVERY_LOC_002",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_002",
            lat=44.9150,
            lon=8.2150,
            service_time=30.0,
            demand=-45.0,
            volume=-20.0,
            pallets=-5,
            earliest_time=660,  # 11:00 AM
            latest_time=780     # 1:00 PM
        )
        
        order_2 = Order(
            id="ORDER_002",
            pickup_tasks=[pickup_task_2],
            delivery_tasks=[delivery_task_2],
            priority=3,
            is_urgent=True
        )
        orders.append(order_2)
        
        # Order 3 - Distant location
        pickup_task_3 = Task(
            id="PICKUP_003",
            location_id="PICKUP_LOC_003",
            task_type=TaskType.PICKUP,
            order_id="ORDER_003",
            lat=44.8800,  # Further south
            lon=8.1800,   # Further west
            service_time=15.0,
            demand=15.0,
            volume=8.0,
            pallets=2,
            earliest_time=540,  # 9:00 AM
            latest_time=720     # 12:00 PM
        )
        
        delivery_task_3 = Task(
            id="DELIVERY_003",
            location_id="DELIVERY_LOC_003",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_003",
            lat=44.9400,  # Further north
            lon=8.2400,   # Further east
            service_time=15.0,
            demand=-15.0,
            volume=-8.0,
            pallets=-2,
            earliest_time=720,  # 12:00 PM
            latest_time=900     # 3:00 PM
        )
        
        order_3 = Order(
            id="ORDER_003",
            pickup_tasks=[pickup_task_3],
            delivery_tasks=[delivery_task_3],
            priority=1
        )
        orders.append(order_3)
        
        # Order 4 - Small order, flexible timing
        pickup_task_4 = Task(
            id="PICKUP_004",
            location_id="PICKUP_LOC_004",
            task_type=TaskType.PICKUP,
            order_id="ORDER_004",
            lat=44.9050,
            lon=8.2050,
            service_time=10.0,
            demand=8.0,
            volume=4.0,
            pallets=1,
            earliest_time=480,  # 8:00 AM
            latest_time=840     # 2:00 PM (very flexible)
        )
        
        delivery_task_4 = Task(
            id="DELIVERY_004",
            location_id="DELIVERY_LOC_004",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_004",
            lat=44.9120,
            lon=8.2120,
            service_time=10.0,
            demand=-8.0,
            volume=-4.0,
            pallets=-1,
            earliest_time=600,  # 10:00 AM
            latest_time=960     # 4:00 PM (very flexible)
        )
        
        order_4 = Order(
            id="ORDER_004",
            pickup_tasks=[pickup_task_4],
            delivery_tasks=[delivery_task_4],
            priority=1
        )
        orders.append(order_4)
        
        # Order 5 - Medium order
        pickup_task_5 = Task(
            id="PICKUP_005",
            location_id="PICKUP_LOC_005",
            task_type=TaskType.PICKUP,
            order_id="ORDER_005",
            lat=44.9250,
            lon=8.2250,
            service_time=20.0,
            demand=30.0,
            volume=15.0,
            pallets=4,
            earliest_time=600,  # 10:00 AM
            latest_time=780     # 1:00 PM
        )
        
        delivery_task_5 = Task(
            id="DELIVERY_005",
            location_id="DELIVERY_LOC_005",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_005",
            lat=44.9350,
            lon=8.2350,
            service_time=20.0,
            demand=-30.0,
            volume=-15.0,
            pallets=-4,
            earliest_time=840,  # 2:00 PM
            latest_time=1020    # 5:00 PM
        )
        
        order_5 = Order(
            id="ORDER_005",
            pickup_tasks=[pickup_task_5],
            delivery_tasks=[delivery_task_5],
            priority=2
        )
        orders.append(order_5)
        
        # Order 6 - Late afternoon order
        pickup_task_6 = Task(
            id="PICKUP_006",
            location_id="PICKUP_LOC_006",
            task_type=TaskType.PICKUP,
            order_id="ORDER_006",
            lat=44.9000,
            lon=8.2000,
            service_time=15.0,
            demand=20.0,
            volume=10.0,
            pallets=2,
            earliest_time=720,  # 12:00 PM
            latest_time=900     # 3:00 PM
        )
        
        delivery_task_6 = Task(
            id="DELIVERY_006",
            location_id="DELIVERY_LOC_006",
            task_type=TaskType.DELIVERY,
            order_id="ORDER_006",
            lat=44.9180,
            lon=8.2180,
            service_time=15.0,
            demand=-20.0,
            volume=-10.0,
            pallets=-2,
            earliest_time=900,  # 3:00 PM
            latest_time=1080    # 6:00 PM
        )
        
        order_6 = Order(
            id="ORDER_006",
            pickup_tasks=[pickup_task_6],
            delivery_tasks=[delivery_task_6],
            priority=1
        )
        orders.append(order_6)
        
        # Create 4 vehicles with different capacities
        vehicles = [
            Vehicle(
                id="VEHICLE_001",
                depot_id="DEPOT-ASTI",
                weight_capacity=60.0,  # Small vehicle
                volume_capacity=30.0,
                pallet_capacity=10,
                vehicle_type="standard"
            ),
            Vehicle(
                id="VEHICLE_002", 
                depot_id="DEPOT-ASTI",
                weight_capacity=100.0,  # Medium vehicle
                volume_capacity=50.0,
                pallet_capacity=20,
                vehicle_type="standard"
            ),
            Vehicle(
                id="VEHICLE_003",
                depot_id="DEPOT-ASTI", 
                weight_capacity=80.0,   # Medium vehicle
                volume_capacity=40.0,
                pallet_capacity=15,
                vehicle_type="standard"
            ),
            Vehicle(
                id="VEHICLE_004",
                depot_id="DEPOT-ASTI",
                weight_capacity=120.0,  # Large vehicle
                volume_capacity=60.0,
                pallet_capacity=25,
                vehicle_type="heavy"
            )
        ]
        
        # Set up parameters for more complex optimization
        params = {
            'M1': 50,   # Allow more non-improving iterations
            'M2': 100,  # Allow more total iterations
            'tabu_tenure': 15,  # Longer tabu tenure
            'exploration_strategy': 'local_search',
            'initialization_method': 'cluster_aware',
            'enable_destroy_and_repair': True,  # Enable for complexity
            'enforce_pickup_first': True,
            'max_iterations': 50,  # More iterations for complexity
            'enable_advanced_neighborhoods': True,  # Enable advanced neighborhoods
            'max_neighbors_per_iteration': 30,  # More neighbors to evaluate
            'max_neighbors_to_evaluate': 25    # More neighbor exploration
        }
        
        print(f"📊 Running complex l1_heuristic with {len(orders)} orders and {len(vehicles)} vehicles...")
        print(f"🔧 Advanced optimization enabled: destroy_repair={params['enable_destroy_and_repair']}, advanced_neighborhoods={params['enable_advanced_neighborhoods']}")
        
        # Run the optimizer with complex scenario
        solution = l1_heuristic(orders, vehicles, params)
        
        print(f"✅ Complex optimizer completed. Checking {len(solution.routes)} routes...")
        
        # Test: Verify that all routes have proper depot tasks
        routes_checked = 0
        routes_with_tasks = 0
        routes_passed = 0
        total_tasks = 0
        
        for vehicle_id, route in solution.routes.items():
            print(f"🔍 Checking route for vehicle {vehicle_id}...")
            
            routes_checked += 1
            
            # Skip empty routes (they should not have depot tasks)
            if not hasattr(route, 'tasks') or not route.tasks:
                print(f"   ⚪ Route {vehicle_id} is empty - OK")
                continue
                
            routes_with_tasks += 1
            total_tasks += len(route.tasks)
            print(f"   📋 Route {vehicle_id} has {len(route.tasks)} tasks")
            
            # Verify first task is depot start
            first_task = route.tasks[0]
            assert first_task.is_depot_start(), f"First task in route {vehicle_id} is not a depot start: {first_task.task_type}"
            print(f"   ✅ First task is depot start: {first_task.id}")
            
            # Verify last task is depot return  
            last_task = route.tasks[-1]
            assert last_task.is_depot_return(), f"Last task in route {vehicle_id} is not a depot return: {last_task.task_type}"
            print(f"   ✅ Last task is depot return: {last_task.id}")
            
            routes_passed += 1
            
            # Print all tasks for debugging
            print(f"   📝 Task sequence for route {vehicle_id}:")
            for i, task in enumerate(route.tasks):
                task_type_str = f"({task.task_type})" if hasattr(task, 'task_type') else ""
                print(f"     {i+1}. {task.id} {task_type_str}")
        
        print(f"\n🎯 Complex Test Results:")
        print(f"   Total routes checked: {routes_checked}")
        print(f"   Routes with tasks: {routes_with_tasks}")
        print(f"   Routes passed depot test: {routes_passed}")
        print(f"   Total tasks across all routes: {total_tasks}")
        print(f"   All depot task assertions passed: ✅")
        
        # Additional verification: Check that we have assigned orders
        assigned_orders = 0
        depot_tasks = 0
        customer_tasks = 0
        
        for vehicle_id, route in solution.routes.items():
            if hasattr(route, 'tasks') and route.tasks:
                # Count task types
                for task in route.tasks:
                    if task.is_depot_start() or task.is_depot_return():
                        depot_tasks += 1
                    else:
                        customer_tasks += 1
                        assigned_orders += 1
        
        print(f"   Customer tasks assigned: {customer_tasks}")
        print(f"   Depot tasks created: {depot_tasks}")
        
        # Check for unassigned orders
        unassigned_count = len(getattr(solution, 'unassigned_orders', set()))
        print(f"   Unassigned orders: {unassigned_count}")
        
        if assigned_orders == 0:
            print("⚠️  Warning: No customer orders were assigned to any routes")
        
        # Verify we have the expected ratio of depot to customer tasks
        expected_depot_tasks = routes_with_tasks * 2  # Start + End for each non-empty route
        if depot_tasks != expected_depot_tasks:
            print(f"⚠️  Warning: Expected {expected_depot_tasks} depot tasks, found {depot_tasks}")
        else:
            print(f"✅ Correct depot task count: {depot_tasks} depot tasks for {routes_with_tasks} routes")
        
        print("\n🎉 COMPLEX TEST PASSED: All routes maintain proper depot start and end tasks through complex optimization!")
        return True
        
    except Exception as e:
        print(f"\n❌ COMPLEX TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_depot_task_creation_complex()
    sys.exit(0 if success else 1)
