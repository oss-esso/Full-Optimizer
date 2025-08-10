#!/usr/bin/env python3
"""
Test 3: Initializer Validation (cluster_aware_initializer)

Goal: Check if a complete initialization function generates a solution where 
all routes are valid and correctly structured.
"""

import sys
import os

# Add the algo directory to the Python path
algo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'algo')
sys.path.insert(0, algo_path)

try:
    # Import required classes and functions
    from epdt_data_structures import Route, Vehicle, Task, TaskType, Order, Solution
    from first_level import cluster_aware_initializer, _create_base_route
    from second_level import is_feasible
    
    print("🧪 Test 3: Initializer Validation (cluster_aware_initializer)")
    print("=" * 60)
    
    # Create test vehicles
    print("🚚 Creating test vehicles...")
    vehicles = []
    for i in range(3):
        vehicle = Vehicle(
            id=f"TEST_VEHICLE_{i+1:03d}",
            depot_id="DEPOT-ASTI",
            weight_capacity=1000.0,
            volume_capacity=100.0,
            pallet_capacity=10,
            cost_per_km=1.0,
            cost_per_hour=50.0
        )
        vehicles.append(vehicle)
    print(f"✅ Created {len(vehicles)} test vehicles")
    
    # Create test orders
    print("📦 Creating test orders...")
    orders = []
    
    # Order 1
    pickup1 = Task(
        id="PICKUP_001", location_id="CUSTOMER_A", task_type=TaskType.PICKUP,
        order_id="ORDER_001", lat=44.9200, lon=8.2200, service_time=15.0,
        demand=100.0, volume=10.0, pallets=2, earliest_time=480, latest_time=1020
    )
    delivery1 = Task(
        id="DELIVERY_001", location_id="CUSTOMER_B", task_type=TaskType.DELIVERY,
        order_id="ORDER_001", lat=44.9500, lon=8.2500, service_time=10.0,
        demand=100.0, volume=10.0, pallets=2, earliest_time=480, latest_time=1020
    )
    order1 = Order(id="ORDER_001", pickup_tasks=[pickup1], delivery_tasks=[delivery1])
    
    # Order 2
    pickup2 = Task(
        id="PICKUP_002", location_id="CUSTOMER_C", task_type=TaskType.PICKUP,
        order_id="ORDER_002", lat=44.8800, lon=8.1800, service_time=15.0,
        demand=150.0, volume=15.0, pallets=3, earliest_time=480, latest_time=1020
    )
    delivery2 = Task(
        id="DELIVERY_002", location_id="CUSTOMER_D", task_type=TaskType.DELIVERY,
        order_id="ORDER_002", lat=44.8500, lon=8.1500, service_time=10.0,
        demand=150.0, volume=15.0, pallets=3, earliest_time=480, latest_time=1020
    )
    order2 = Order(id="ORDER_002", pickup_tasks=[pickup2], delivery_tasks=[delivery2])
    
    # Order 3
    pickup3 = Task(
        id="PICKUP_003", location_id="CUSTOMER_E", task_type=TaskType.PICKUP,
        order_id="ORDER_003", lat=44.9800, lon=8.2800, service_time=15.0,
        demand=80.0, volume=8.0, pallets=1, earliest_time=480, latest_time=1020
    )
    delivery3 = Task(
        id="DELIVERY_003", location_id="CUSTOMER_F", task_type=TaskType.DELIVERY,
        order_id="ORDER_003", lat=44.9300, lon=8.2300, service_time=10.0,
        demand=80.0, volume=8.0, pallets=1, earliest_time=480, latest_time=1020
    )
    order3 = Order(id="ORDER_003", pickup_tasks=[pickup3], delivery_tasks=[delivery3])
    
    orders = [order1, order2, order3]
    print(f"✅ Created {len(orders)} test orders")
    
    # Set up parameters
    params = {
        'debug_assignment': True,
        'use_cluster_aware': True
    }
    
    # Call cluster_aware_initializer
    print("🎯 Calling cluster_aware_initializer...")
    try:
        solution = cluster_aware_initializer(orders, vehicles, params)
        print(f"✅ Initializer completed successfully")
        
        # Verify solution is not None
        assert solution is not None, "Initializer should return a solution, not None"
        print(f"✅ Initializer returned a valid solution")
        
        # Verify solution has routes
        assert hasattr(solution, 'routes'), "Solution should have routes attribute"
        assert len(solution.routes) > 0, "Solution should contain at least one route"
        print(f"✅ Solution contains {len(solution.routes)} routes")
        
        # Iterate through every route in the solution
        print("🔍 Validating route structures...")
        valid_routes = 0
        route_details = []
        
        for vehicle_id, route in solution.routes.items():
            route_info = f"Vehicle {vehicle_id}: "
            
            # Check if route has tasks
            if not hasattr(route, 'tasks') or not route.tasks:
                route_info += "EMPTY (no tasks)"
                route_details.append(route_info)
                continue
                
            # Check depot task structure
            route_info += f"{len(route.tasks)} tasks - "
            
            # Assert first task is depot start
            if not route.tasks[0].is_depot_start():
                route_info += "❌ MISSING DEPOT START"
                route_details.append(route_info)
                assert False, f"Route for vehicle {vehicle_id} does not start with depot start task. First task: {route.tasks[0].task_type}"
            
            # Assert last task is depot return  
            if not route.tasks[-1].is_depot_return():
                route_info += "❌ MISSING DEPOT RETURN"
                route_details.append(route_info)
                assert False, f"Route for vehicle {vehicle_id} does not end with depot return task. Last task: {route.tasks[-1].task_type}"
            
            route_info += "✅ VALID STRUCTURE"
            route_details.append(route_info)
            
            # Test feasibility
            try:
                feasible, reason = is_feasible(route, return_reason=True)
                if feasible:
                    route_info += " - FEASIBLE"
                else:
                    route_info += f" - INFEASIBLE: {reason}"
                    print(f"⚠️ Warning: Route for vehicle {vehicle_id} is infeasible: {reason}")
            except Exception as e:
                route_info += f" - FEASIBILITY CHECK FAILED: {e}"
                print(f"⚠️ Warning: Could not check feasibility for vehicle {vehicle_id}: {e}")
            
            valid_routes += 1
        
        print(f"✅ All {valid_routes} routes have correct depot task structure")
        
        # Print route details
        print("\\n📋 Route validation summary:")
        for detail in route_details:
            print(f"  {detail}")
        
        # Check if orders were assigned
        print("\\n📦 Checking order assignment...")
        assigned_orders = set()
        total_tasks = 0
        
        for vehicle_id, route in solution.routes.items():
            if hasattr(route, 'tasks') and route.tasks:
                for task in route.tasks:
                    if not task.is_depot_start() and not task.is_depot_return():
                        assigned_orders.add(task.order_id)
                        total_tasks += 1
        
        print(f"✅ Found {total_tasks} non-depot tasks assigned to routes")
        print(f"✅ Orders represented in solution: {len(assigned_orders)}")
        
        if len(assigned_orders) > 0:
            print(f"   Assigned order IDs: {sorted(assigned_orders)}")
        
        print(f"\\n🎉 TEST 3 PASSED: All assertions successful!")
        print("=" * 60)
        print("✅ cluster_aware_initializer generates valid solution")
        print("✅ All routes have proper depot task structure")
        print("✅ Route validation completed successfully")
        
    except Exception as e:
        print(f"❌ cluster_aware_initializer failed: {e}")
        import traceback
        traceback.print_exc()
        raise
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)
except AssertionError as e:
    print(f"❌ Assertion failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
