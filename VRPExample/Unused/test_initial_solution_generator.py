#!/usr/bin/env python3
"""
Test script for Initial Solution Generator
Tests multiple construction heuristics including firefly algorithm
"""
import sys
import os
import logging

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from initial_solution_generator import InitialSolutionGenerator, Customer, Vehicle
from vrp_scenarios import get_all_scenarios

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def convert_vrp_instance_to_generator_format(instance):
    """Convert VRP instance to InitialSolutionGenerator format."""
    
    # Handle both standard VRP and ride pooling scenarios
    if hasattr(instance, 'ride_requests') and instance.ride_requests:
        # For ride pooling, we'll treat pickup locations as customers
        customers = []
        for request in instance.ride_requests:
            pickup_loc = instance.locations[request.pickup_location]
            customer = Customer(
                id=request.pickup_location,
                x=pickup_loc.lat if hasattr(pickup_loc, 'lat') else hash(request.pickup_location) % 100,
                y=pickup_loc.lon if hasattr(pickup_loc, 'lon') else hash(request.pickup_location) % 100,
                demand=request.passengers
            )
            customers.append(customer)
    else:
        # Standard VRP: use customer locations
        customers = []
        depot_locations = [loc_id for loc_id in instance.location_ids if loc_id.startswith("depot")]
        customer_locations = [loc_id for loc_id in instance.location_ids if not loc_id.startswith("depot")]
        
        for loc_id in customer_locations:
            location = instance.locations[loc_id]
            customer = Customer(
                id=loc_id,
                x=location.lat if hasattr(location, 'lat') else hash(loc_id) % 100,
                y=location.lon if hasattr(location, 'lon') else hash(loc_id) % 100,
                demand=getattr(location, 'demand', 1)
            )
            customers.append(customer)
    
    # Convert vehicles
    vehicles = []
    for vehicle_id, vehicle in instance.vehicles.items():
        vehicles.append(Vehicle(
            id=vehicle_id,
            capacity=getattr(vehicle, 'capacity', 100)
        ))
    
    # Get depot location
    depot_locations = [loc_id for loc_id in instance.location_ids if loc_id.startswith("depot")]
    if depot_locations:
        depot_loc = instance.locations[depot_locations[0]]
        depot_x = depot_loc.lat if hasattr(depot_loc, 'lat') else 0.0
        depot_y = depot_loc.lon if hasattr(depot_loc, 'lon') else 0.0
    else:
        depot_x, depot_y = 0.0, 0.0
    
    return customers, vehicles, (depot_x, depot_y)

def test_initial_solution_generator():
    """Test the InitialSolutionGenerator with different VRP scenarios."""
    print("Testing Initial Solution Generator")
    print("=" * 50)
    
    # Get VRP scenarios
    scenarios = get_all_scenarios()
    
    # Test on a few different scenarios
    test_scenarios = ["simple_4_customers", "MODA_small"]
    
    for scenario_name in test_scenarios:
        if scenario_name not in scenarios:
            print(f"Scenario {scenario_name} not found, skipping...")
            continue
            
        print(f"\nTesting {scenario_name}:")
        instance = scenarios[scenario_name]
        
        try:
            # Convert to generator format
            customers, vehicles, depot_location = convert_vrp_instance_to_generator_format(instance)
            
            print(f"  Customers: {len(customers)}")
            print(f"  Vehicles: {len(vehicles)}")
            print(f"  Depot: {depot_location}")
            
            if not customers:
                print("  No customers found, skipping...")
                continue
            
            # Initialize generator
            generator = InitialSolutionGenerator(customers, vehicles, depot_location)
            
            # Test different methods
            methods = [
                ("Nearest Neighbor", lambda: generator.generate_nearest_neighbor_solution(0.1)),
                ("Savings Algorithm", lambda: generator.generate_savings_algorithm_solution(0.1)),
                ("Firefly Algorithm", lambda: generator.generate_firefly_algorithm_solution(10, 0.12)),
                ("Greedy Insertion", lambda: generator.generate_greedy_insertion_solution(True, 0.1)),
            ]
            
            results = {}
            
            for method_name, method_func in methods:
                try:
                    print(f"  Testing {method_name}...")
                    routes = method_func()
                    
                    if routes:
                        total_distance = sum(r.total_distance for r in routes)
                        total_customers = sum(len(r.customers) for r in routes)
                        avg_capacity_utilization = sum(r.total_demand for r in routes) / sum(v.capacity for v in vehicles[:len(routes)])
                        
                        results[method_name] = {
                            'routes': len(routes),
                            'distance': total_distance,
                            'customers_served': total_customers,
                            'capacity_util': avg_capacity_utilization
                        }
                        
                        print(f"    Routes: {len(routes)}")
                        print(f"    Total distance: {total_distance:.2f}")
                        print(f"    Customers served: {total_customers}/{len(customers)}")
                        print(f"    Avg capacity utilization: {avg_capacity_utilization:.2%}")
                        
                        # Show first route details
                        if routes:
                            first_route = routes[0]
                            print(f"    First route: {first_route.vehicle_id} -> {' -> '.join(first_route.customers[:3])}{'...' if len(first_route.customers) > 3 else ''}")
                    else:
                        print(f"    No routes generated")
                        
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
            
            # Compare results
            if results:
                print(f"  \nMethod Comparison:")
                best_distance = min(r['distance'] for r in results.values() if r['distance'] > 0)
                for method, result in results.items():
                    if result['distance'] > 0:
                        ratio = result['distance'] / best_distance
                        print(f"    {method}: {result['distance']:.2f} ({ratio:.2f}x best)")
            
            # Test diverse solutions
            print(f"  \nTesting diverse solution generation...")
            try:
                diverse_solutions = generator.generate_diverse_solutions(3)
                print(f"    Generated {len(diverse_solutions)} diverse solutions")
                
                for i, solution in enumerate(diverse_solutions):
                    total_dist = sum(r.total_distance for r in solution)
                    print(f"      Solution {i+1}: {len(solution)} routes, distance: {total_dist:.2f}")
                    
            except Exception as e:
                print(f"    ERROR in diverse solutions: {str(e)}")
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

def test_firefly_algorithm_details():
    """Test firefly algorithm components in detail."""
    print("\n" + "=" * 50)
    print("Testing Firefly Algorithm Details")
    print("=" * 50)
    
    # Create simple test data
    customers = [
        Customer("c1", 1.0, 1.0, 5),
        Customer("c2", 2.0, 3.0, 3),
        Customer("c3", 4.0, 2.0, 4),
        Customer("c4", 3.0, 4.0, 2),
    ]
    
    vehicles = [
        Vehicle("v1", 8),
        Vehicle("v2", 6),
    ]
    
    depot_location = (0.0, 0.0)
    
    print(f"Test scenario: {len(customers)} customers, {len(vehicles)} vehicles")
    print(f"Customer demands: {[c.demand for c in customers]}")
    print(f"Vehicle capacities: {[v.capacity for v in vehicles]}")
    
    generator = InitialSolutionGenerator(customers, vehicles, depot_location)
    
    # Test firefly creation and decoding
    print(f"\nTesting firefly creation and decoding:")
    from initial_solution_generator import Firefly
    
    firefly = Firefly(len(customers), len(vehicles))
    print(f"  Initial position: {firefly.position}")
    
    routes = firefly.decode_to_routes(customers, vehicles)
    print(f"  Decoded to {len(routes)} routes")
    print(f"  Feasible: {firefly.feasible}")
    print(f"  Total distance: {firefly.total_distance:.2f}")
    print(f"  Brightness: {firefly.brightness:.4f}")
    
    for route in routes:
        print(f"    {route.vehicle_id}: {route.customers} (demand: {route.total_demand}, distance: {route.total_distance:.2f})")
    
    # Test firefly algorithm
    print(f"\nTesting full firefly algorithm:")
    fa_routes = generator.generate_firefly_algorithm_solution(20, 0.15)
    
    if fa_routes:
        total_distance = sum(r.total_distance for r in fa_routes)
        print(f"  Generated {len(fa_routes)} routes with total distance: {total_distance:.2f}")
        
        for route in fa_routes:
            print(f"    {route.vehicle_id}: {len(route.customers)} customers, "
                  f"demand: {route.total_demand}/{vehicles[0].capacity}, "
                  f"distance: {route.total_distance:.2f}")

if __name__ == "__main__":
    test_initial_solution_generator()
    test_firefly_algorithm_details()
