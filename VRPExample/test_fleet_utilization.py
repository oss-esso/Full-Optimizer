#!/usr/bin/env python3
"""
Test fleet utilization - Focus on using more vehicles instead of strict constraints.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FleetUtilizationOptimizer(VRPOptimizerEnhanced):
    """Version focused on fleet utilization rather than strict constraints."""
    
    def _solve_with_constraint_level(self, instance, level: str, time_limit: int):
        """Override to focus on fleet utilization."""
        try:
            # Setup
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            depot_index = 0
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            routing = pywrapcp.RoutingModel(manager)
            location_list = list(instance.locations.values())
            
            constraints_applied = []
            
            # 1. Distance callback
            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                from_loc = location_list[from_node]
                to_loc = location_list[to_node]
                
                # Use Haversine formula for accurate distance calculation
                import math
                lat1, lon1 = math.radians(from_loc.y), math.radians(from_loc.x)
                lat2, lon2 = math.radians(to_loc.y), math.radians(to_loc.x)
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                distance_km = 6371 * 2 * math.asin(math.sqrt(a))  # Earth radius in km
                
                return int(distance_km * 1000)  # Return as scaled integer (meters) for OR-Tools
            
            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            constraints_applied.append("distance")
            
            # 2. FLEET UTILIZATION FOCUSED TIME DIMENSION
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                # Travel time: distance in km converted to minutes
                distance_scaled = distance_callback(from_index, to_index)
                distance_km = distance_scaled / 1000.0
                travel_time = max(1, int(distance_km)) if distance_km > 0 else 0  # Simplified: 1 min per km
                
                # MINIMAL service time to encourage fleet utilization
                service_time = 0
                if level != "distance_only":
                    to_loc = location_list[to_node]
                    base_service_time = getattr(to_loc, 'service_time', 0)
                    # Reduce service time to encourage more vehicles
                    service_time = max(1, base_service_time // 4)  # Use only 1/4 of service time
                
                return int(travel_time + service_time)
            
            time_callback_index = routing.RegisterTransitCallback(time_callback)
            
            # VERY GENEROUS time dimension to allow fleet utilization
            max_route_time = 2000  # Very generous to allow solutions
            routing.AddDimension(
                time_callback_index,
                300,   # Large slack - 5 hours
                max_route_time,
                True,
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')
            constraints_applied.append("time_dimension_generous")
            
            # 3. ENCOURAGE FLEET UTILIZATION
            logger.info(f"🚛 Encouraging fleet utilization across {num_vehicles} vehicles...")
            
            # Add a penalty for not using vehicles (vehicle fixed cost)
            for vehicle_id in range(num_vehicles):
                routing.SetFixedCostOfVehicle(1000, vehicle_id)  # Small penalty for using each vehicle
            
            constraints_applied.append("fleet_utilization_encouragement")
            
            logger.info(f"Applied constraints: {constraints_applied}")
            
            # SOLVE with strategies that encourage fleet utilization
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
            search_parameters.time_limit.FromSeconds(time_limit)
            search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            
            start_time = time.time()
            assignment = routing.SolveWithParameters(search_parameters)
            solve_time = time.time() - start_time
            
            if assignment:
                try:
                    objective_value = assignment.ObjectiveValue()
                except AttributeError:
                    objective_value = routing.GetCost(assignment)
                
                logger.info(f"✅ SOLUTION FOUND!")
                logger.info(f"   Objective: {objective_value}")
                logger.info(f"   Time: {solve_time:.2f}s")
                
                # Extract and analyze routes
                routes_info = self._extract_routes_with_analysis(instance, manager, routing, assignment, time_dimension)
                
                # FLEET UTILIZATION ANALYSIS
                vehicles_used = routes_info['vehicles_used']
                utilization_rate = (vehicles_used / num_vehicles) * 100
                
                logger.info(f"🚛 FLEET UTILIZATION ANALYSIS:")
                logger.info(f"   Vehicles used: {vehicles_used}/{num_vehicles} ({utilization_rate:.1f}%)")
                
                # Analyze distribution
                stops_per_vehicle = []
                for analysis in routes_info['analysis']:
                    stops_per_vehicle.append(analysis['stops'])
                
                if stops_per_vehicle:
                    avg_stops = sum(stops_per_vehicle) / len(stops_per_vehicle)
                    min_stops = min(stops_per_vehicle)
                    max_stops = max(stops_per_vehicle)
                    
                    logger.info(f"   Stops distribution: {min_stops}-{max_stops} (avg: {avg_stops:.1f})")
                    logger.info(f"   Load balance: {max_stops/avg_stops:.2f}x max vs avg")
                
                return {
                    'success': True,
                    'objective_value': objective_value,
                    'routes': routes_info['routes'],
                    'vehicles_used': vehicles_used,
                    'fleet_utilization_rate': utilization_rate,
                    'route_analysis': routes_info['analysis'],
                    'solve_time': solve_time,
                    'constraints_applied': constraints_applied
                }
            else:
                return {
                    'success': False,
                    'error': f'No solution found with {level} constraints',
                    'constraints_applied': constraints_applied
                }
                
        except Exception as e:
            logger.error(f"Error in {level} constraint level: {e}")
            return {'success': False, 'error': str(e)}

def test_fleet_utilization():
    """Test fleet utilization focused approach."""
    logger.info("=" * 80)
    logger.info("TESTING FLEET UTILIZATION APPROACH")
    logger.info("=" * 80)
    
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Loaded scenario: {scenario.name}")
        logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations")
        
        optimizer = FleetUtilizationOptimizer()
        
        logger.info("\n🚀 Starting fleet utilization test...")
        result = optimizer._solve_with_constraint_level(scenario, "distance_only", time_limit=120)
        
        logger.info(f"\n🎯 FLEET UTILIZATION RESULTS:")
        if result['success']:
            logger.info(f"✅ SUCCESS!")
            logger.info(f"   Objective: {result['objective_value']}")
            logger.info(f"   Fleet utilization: {result['fleet_utilization_rate']:.1f}%")
            logger.info(f"   Vehicles used: {result['vehicles_used']}/{len(scenario.vehicles)}")
            
            if result['vehicles_used'] > 10:  # Good utilization
                logger.info("🎉 EXCELLENT: Using many vehicles - good distribution!")
            elif result['vehicles_used'] > 5:
                logger.info("👍 GOOD: Using multiple vehicles")
            else:
                logger.warning(f"⚠️ POOR: Only using {result['vehicles_used']} vehicles - need better distribution")
                
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fleet_utilization()
    
    # Calculate theoretical workload distribution
    total_locations = len(scenario.locations) - 2  # Exclude depots
    num_vehicles = len(scenario.vehicles)
    locations_per_vehicle = total_locations / num_vehicles
    
    logger.info(f"\n🧮 THEORETICAL DISTRIBUTION:")
    logger.info(f"   📍 Locations per vehicle (even split): {locations_per_vehicle:.1f}")
    logger.info(f"   📦 Requests per vehicle (even split): {len(scenario.ride_requests) / num_vehicles:.1f}")
    
    # Analyze vehicle capacities and time limits
    vehicle_capacities = []
    vehicle_time_limits = []
    
    for vid, vehicle in scenario.vehicles.items():
        capacity = vehicle.capacity
        max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                   getattr(vehicle, 'max_time', None) or 600)
        vehicle_capacities.append(capacity)
        vehicle_time_limits.append(max_time)
    
    total_capacity = sum(vehicle_capacities)
    total_demand = sum(req.passengers for req in scenario.ride_requests)
    
    logger.info(f"\n💼 CAPACITY ANALYSIS:")
    logger.info(f"   📊 Total fleet capacity: {total_capacity}kg")
    logger.info(f"   📊 Total demand: {total_demand}kg")
    logger.info(f"   📊 Capacity utilization: {total_demand/total_capacity*100:.1f}%")
    logger.info(f"   📊 Vehicle capacities: {min(vehicle_capacities)}-{max(vehicle_capacities)}kg")
    
    logger.info(f"\n⏰ TIME ANALYSIS:")
    logger.info(f"   📊 Vehicle time limits: {min(vehicle_time_limits)}-{max(vehicle_time_limits)}min")
    
    # Test with different strategies
    logger.info(f"\n🧪 TESTING DIFFERENT STRATEGIES:")
    
    # Strategy 1: Minimize vehicles (current default)
    logger.info(f"\n1️⃣ STRATEGY 1: Minimize vehicles (current)")
    test_strategy(scenario, "minimize_vehicles")
    
    # Strategy 2: Force vehicle distribution
    logger.info(f"\n2️⃣ STRATEGY 2: Force vehicle distribution")
    test_strategy(scenario, "distribute_vehicles")

def test_strategy(scenario, strategy_name):
    """Test a specific strategy."""
    
    class FleetUtilizationOptimizer(VRPOptimizerEnhanced):
        def _solve_with_constraint_level(self, instance, level, time_limit):
            logger.info(f"   🎯 Testing {strategy_name} with {level} constraints")
            
            # Only test Level 1 for speed
            if level != "distance_only":
                return {'success': False, 'error': 'Skipping non-Level 1 for fleet analysis'}
            
            # Call parent method first
            result = super()._solve_with_constraint_level(instance, level, time_limit)
            
            if result['success']:
                vehicles_used = result['vehicles_used']
                total_vehicles = len(instance.vehicles)
                utilization = vehicles_used / total_vehicles * 100
                
                logger.info(f"   📊 Vehicles used: {vehicles_used}/{total_vehicles} ({utilization:.1f}%)")
                
                # Analyze load distribution
                if 'route_analysis' in result:
                    stops_per_vehicle = [analysis['stops'] for analysis in result['route_analysis']]
                    loads_per_vehicle = [analysis['max_load_reached'] for analysis in result['route_analysis']]
                    
                    logger.info(f"   📊 Stops per vehicle: {stops_per_vehicle}")
                    logger.info(f"   📊 Max loads: {loads_per_vehicle}")
                    
                    if stops_per_vehicle:
                        logger.info(f"   📊 Stop distribution: min={min(stops_per_vehicle)}, max={max(stops_per_vehicle)}, avg={sum(stops_per_vehicle)/len(stops_per_vehicle):.1f}")
            
            return result
    
    optimizer = FleetUtilizationOptimizer()
    
    try:
        result = optimizer.solve(scenario, time_limit_seconds=60)  # Quick test
        
        if not result['success']:
            logger.warning(f"   ❌ {strategy_name} failed: {result.get('error', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"   ❌ {strategy_name} crashed: {e}")

if __name__ == "__main__":
    analyze_fleet_utilization()
