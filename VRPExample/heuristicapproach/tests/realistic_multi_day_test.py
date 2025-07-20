"""
Realistic Multi-Day VRP Test Scenario

This module creates realistic multi-day VRP scenarios that respect European HoS regulations
and demonstrate the advanced HoS simulation working correctly with feasible routes.
"""

import sys
import os
from typing import List, Dict, Any
from dataclasses import dataclass

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)

try:
    from algo.epdt_data_structures import Task, Order, Vehicle, Route, Solution, TaskType
    from algo.second_level import _simulate_hos_advanced, DriverState, calculate_route_days
    print("✅ Successfully imported EPDT data structures and HoS simulation")
except ImportError as e:
    print(f"❌ Failed to import EPDT modules: {e}")
    # Try alternative import paths
    try:
        import epdt_data_structures as epdt
        import second_level as sl
        Task, Order, Vehicle, Route, Solution, TaskType = epdt.Task, epdt.Order, epdt.Vehicle, epdt.Route, epdt.Solution, epdt.TaskType
        _simulate_hos_advanced, DriverState, calculate_route_days = sl._simulate_hos_advanced, sl.DriverState, sl.calculate_route_days
        print("✅ Successfully imported EPDT modules with alternative paths")
    except ImportError as e2:
        print(f"❌ Failed to import EPDT modules with alternative paths: {e2}")
        sys.exit(1)


def create_realistic_multi_day_scenario():
    """
    Create a realistic multi-day VRP scenario that respects HoS regulations.
    
    This scenario demonstrates:
    - Routes that span multiple days with proper rest periods
    - Realistic service and travel times
    - Strategic task distribution to respect daily limits
    
    Returns:
        tuple: Orders and vehicles for the scenario
    """
    print("🏗️  Creating realistic multi-day VRP scenario...")
    
    # Create vehicles with realistic capacities
    vehicles = [
        Vehicle(
            id="V001",
            depot_id="DEPOT_001", 
            weight_capacity=1200,  # 1.2 tons
            volume_capacity=8.0,   # 8 m³
            vehicle_type="standard"
        ),
        Vehicle(
            id="V002", 
            depot_id="DEPOT_001",
            weight_capacity=1500,  # 1.5 tons
            volume_capacity=10.0,  # 10 m³
            vehicle_type="standard"
        ),
        Vehicle(
            id="V003",
            depot_id="DEPOT_001",
            weight_capacity=2000,  # 2 tons
            volume_capacity=12.0,  # 12 m³
            vehicle_type="heavy"
        )
    ]
    
    # Create orders with realistic characteristics for multi-day scenarios
    orders = []
    
    # Day 1 Orders - Local deliveries (2-3 hours total per vehicle)
    orders.extend(_create_daily_orders("DAY1", 1, 3, service_time_range=(10, 20)))
    
    # Day 2 Orders - Regional deliveries (3-4 hours total per vehicle)  
    orders.extend(_create_daily_orders("DAY2", 2, 3, service_time_range=(15, 25)))
    
    # Day 3-4 Orders - Long-haul multi-day (spans 2 days with mandatory rest)
    orders.extend(_create_multi_day_orders("MULTI", 3, 2, service_time_range=(20, 30)))
    
    print(f"✅ Created scenario with {len(orders)} orders and {len(vehicles)} vehicles")
    print(f"   📦 Day 1 orders: 3 (local deliveries)")
    print(f"   📦 Day 2 orders: 3 (regional deliveries)")  
    print(f"   📦 Multi-day orders: 2 (long-haul)")
    
    return orders, vehicles


def _create_daily_orders(prefix: str, day: int, count: int, service_time_range: tuple):
    """Create orders that can be completed within a single day."""
    orders = []
    
    for i in range(count):
        order_id = f"{prefix}_{i+1:03d}"
        
        # Create pickup task  
        pickup_task = Task(
            id=f"{order_id}_PICKUP",
            location_id=f"LOC_{prefix}_P_{i+1:03d}",
            task_type=TaskType.PICKUP,
            order_id=order_id,
            lat=45.0 + (i * 0.01),  # Spread locations realistically
            lon=9.0 + (i * 0.01),
            service_time=service_time_range[0] + (i % (service_time_range[1] - service_time_range[0])),
            demand=50 + (i * 20),   # 50-150 kg
            volume=0.5 + (i * 0.3), # 0.5-2.0 m³
            day=day,
            earliest_time=8 * 60,   # 8:00 AM
            latest_time=18 * 60,    # 6:00 PM
            # Add realistic travel time estimation (this will be used by HoS simulation)
            estimated_travel_time=30 + (i * 10)  # 30-80 minutes travel between locations
        )
        
        # Create delivery task
        delivery_task = Task(
            id=f"{order_id}_DELIVERY", 
            location_id=f"LOC_{prefix}_D_{i+1:03d}",
            task_type=TaskType.DELIVERY,
            order_id=order_id,
            lat=45.1 + (i * 0.01),  # Different location for delivery
            lon=9.1 + (i * 0.01),
            service_time=service_time_range[0] + (i % (service_time_range[1] - service_time_range[0])),
            demand=-(50 + (i * 20)), # Negative for delivery
            volume=-(0.5 + (i * 0.3)),
            day=day,
            earliest_time=10 * 60,   # 10:00 AM
            latest_time=20 * 60      # 8:00 PM
        )
        
        order = Order(
            id=order_id,
            pickup_tasks=[pickup_task],
            delivery_tasks=[delivery_task]
        )
        orders.append(order)
    
    return orders


def _create_multi_day_orders(prefix: str, start_day: int, count: int, service_time_range: tuple):
    """Create orders that span multiple days (long-haul scenarios)."""
    orders = []
    
    for i in range(count):
        order_id = f"{prefix}_{i+1:03d}"
        
        # Pickup on start_day
        pickup_task = Task(
            id=f"{order_id}_PICKUP",
            location_id=f"LOC_{prefix}_P_{i+1:03d}",
            task_type=TaskType.PICKUP,
            order_id=order_id,
            lat=45.0 + (i * 0.05),  # More spread out for long-haul
            lon=9.0 + (i * 0.05),
            service_time=service_time_range[1],  # Longer service times
            demand=100 + (i * 50),   # 100-250 kg
            volume=1.0 + (i * 0.5),  # 1.0-2.5 m³
            day=start_day,
            earliest_time=6 * 60,    # 6:00 AM
            latest_time=16 * 60      # 4:00 PM
        )
        
        # Delivery on start_day + 1 (after mandatory daily rest)
        delivery_task = Task(
            id=f"{order_id}_DELIVERY",
            location_id=f"LOC_{prefix}_D_{i+1:03d}",
            task_type=TaskType.DELIVERY,
            order_id=order_id,
            lat=46.0 + (i * 0.05),  # Far destination (long-haul)
            lon=10.0 + (i * 0.05),
            service_time=service_time_range[1],
            demand=-(100 + (i * 50)),
            volume=-(1.0 + (i * 0.5)),
            day=start_day + 1,       # Next day delivery
            earliest_time=8 * 60,    # 8:00 AM
            latest_time=18 * 60      # 6:00 PM
        )
        
        order = Order(
            id=order_id,
            pickup_tasks=[pickup_task],
            delivery_tasks=[delivery_task]
        )
        orders.append(order)
    
    return orders


def create_realistic_multi_day_solution(orders, vehicles):
    """
    Create a realistic multi-day solution that respects HoS regulations.
    
    This demonstrates proper multi-day route planning with:
    - Daily time limits respected
    - Multi-day routes with mandatory rest periods
    - Strategic task distribution
    """
    print("🎯 Creating realistic multi-day solution...")
    
    # Create a simple solution structure
    class SimpleSolution:
        def __init__(self, vehicles):
            self.routes = {}
            self.assigned_orders = {}
            self.unassigned_orders = []
            for vehicle in vehicles:
                self.routes[vehicle.id] = SimpleRoute(vehicle)
        
        def assign_order(self, order_id, vehicle_id):
            self.assigned_orders[order_id] = vehicle_id
        
        def get_total_vehicles_used(self):
            return len([r for r in self.routes.values() if r.tasks])
    
    class SimpleRoute:
        def __init__(self, vehicle):
            self.vehicle = vehicle
            self.tasks = []
    
        # Fix the import inside the function
        pass
    solution = SimpleSolution(vehicles)
    
    # Strategy: Assign orders to vehicles considering daily limits (max 2-3 orders per vehicle per day)
    vehicle_assignments = {v.id: [] for v in vehicles}
    
    # Day 1 orders: One order per vehicle
    day1_orders = [o for o in orders if o.id.startswith("DAY1")]
    for i, order in enumerate(day1_orders):
        vehicle_id = vehicles[i % len(vehicles)].id
        vehicle_assignments[vehicle_id].append(order)
        solution.assign_order(order.id, vehicle_id)
    
    # Day 2 orders: One order per vehicle
    day2_orders = [o for o in orders if o.id.startswith("DAY2")]
    for i, order in enumerate(day2_orders):
        vehicle_id = vehicles[i % len(vehicles)].id
        vehicle_assignments[vehicle_id].append(order)
        solution.assign_order(order.id, vehicle_id)
    
    # Multi-day orders: One per vehicle for long-haul demonstration
    multi_orders = [o for o in orders if o.id.startswith("MULTI")]
    for i, order in enumerate(multi_orders):
        vehicle_id = vehicles[i % len(vehicles)].id
        vehicle_assignments[vehicle_id].append(order)
        solution.assign_order(order.id, vehicle_id)
    
    # Create routes with proper task sequencing
    for vehicle in vehicles:
        if not vehicle_assignments[vehicle.id]:
            continue
            
        route = solution.routes[vehicle.id]
        assigned_orders = vehicle_assignments[vehicle.id]
        
        # Group tasks by day and type
        tasks_by_day = {}
        for order in assigned_orders:
            for task in order.pickup_tasks + order.delivery_tasks:
                day = getattr(task, 'day', 0)
                if day not in tasks_by_day:
                    tasks_by_day[day] = {'pickups': [], 'deliveries': []}
                
                if task.task_type == TaskType.PICKUP:
                    tasks_by_day[day]['pickups'].append(task)
                else:
                    tasks_by_day[day]['deliveries'].append(task)
        
        # Create chronological task sequence
        for day in sorted(tasks_by_day.keys()):
            # Add pickups for this day
            for pickup_task in tasks_by_day[day]['pickups']:
                route.tasks.append(pickup_task)
            
            # Add deliveries for this day  
            for delivery_task in tasks_by_day[day]['deliveries']:
                route.tasks.append(delivery_task)
        
        print(f"   🚚 {vehicle.id}: {len(route.tasks)} tasks across {len(tasks_by_day)} days")
    
    print(f"✅ Realistic solution created with proper multi-day sequencing")
    return solution


def test_realistic_multi_day_scenario():
    """Test the realistic multi-day scenario with advanced HoS simulation."""
    print("\n" + "="*80)
    print("🧪 TESTING REALISTIC MULTI-DAY VRP SCENARIO")
    print("="*80)
    
    # Create scenario
    orders, vehicles = create_realistic_multi_day_scenario()
    solution = create_realistic_multi_day_solution(orders, vehicles)
    
    # Test each route with advanced HoS simulation
    print(f"\n🔍 Testing route feasibility with advanced HoS simulation:")
    
    feasible_routes = 0
    total_routes = 0
    
    for vehicle_id, route in solution.routes.items():
        if not route.tasks:
            continue
            
        total_routes += 1
        print(f"\n🚚 Testing {vehicle_id}:")
        print(f"   📋 Tasks: {len(route.tasks)}")
        
        # Sort tasks chronologically
        sorted_tasks = sorted(route.tasks, key=lambda t: (getattr(t, 'day', 0), t.id))
        
        # Create initial driver state
        driver_state = DriverState()
        
        # Test feasibility with advanced HoS simulation
        feasible, total_time = _simulate_hos_advanced(route, driver_state, sorted_tasks)
        
        # Also test the new calculate_route_days function
        days_required = calculate_route_days(route)
        
        if feasible:
            feasible_routes += 1
            print(f"   ✅ Route FEASIBLE - Total time: {total_time/60:.1f} hours")
            print(f"   📅 Route duration: {days_required} day(s) (via calculate_route_days)")
            
            # Show task sequence with days
            print(f"   📅 Task sequence:")
            current_day = None
            for task in sorted_tasks:
                task_day = getattr(task, 'day', 0)
                if task_day != current_day:
                    print(f"      📅 Day {task_day}:")
                    current_day = task_day
                
                task_type = "📦" if task.task_type == TaskType.PICKUP else "🏪"
                print(f"         {task_type} {task.location_id} ({task.service_time}min)")
        else:
            print(f"   ❌ Route INFEASIBLE - Exceeds HoS regulations")
            print(f"   📅 Route duration: {days_required} (should be inf for infeasible routes)")
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Feasible routes: {feasible_routes}/{total_routes}")
    print(f"   🎯 Success rate: {(feasible_routes/total_routes*100):.1f}%" if total_routes > 0 else "   🎯 No routes to test")
    
    if feasible_routes == total_routes:
        print(f"   🏆 ALL ROUTES FEASIBLE - Advanced HoS simulation working correctly!")
    else:
        print(f"   ⚠️  Some routes infeasible - May need adjustment for stricter HoS regulations")
    
    return feasible_routes == total_routes


if __name__ == "__main__":
    success = test_realistic_multi_day_scenario()
    if success:
        print(f"\n✅ Realistic multi-day scenario test PASSED")
    else:
        print(f"\n❌ Realistic multi-day scenario test FAILED")
