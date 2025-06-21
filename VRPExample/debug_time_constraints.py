#!/usr/bin/env python3
"""
Debug script to understand why Level 1 is failing and investigate time constraint issues.
"""

import sys
import os
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

def debug_time_constraints():
    """Debug why time constraints are making the problem infeasible."""
    logger.info("=" * 80)
    logger.info("DEBUGGING TIME CONSTRAINTS - WHY IS LEVEL 1 FAILING?")
    logger.info("=" * 80)
    
    # Get the scenario
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_first']
        logger.info(f"✅ Loaded scenario: {scenario.name}")
        logger.info(f"   📊 Scale: {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations")
    except Exception as e:
        logger.error(f"❌ Failed to load scenario: {e}")
        return
    
    # Analyze time feasibility
    logger.info("\n🔍 ANALYZING TIME FEASIBILITY:")
    
    # Calculate minimum time needed
    vehicle_times = []
    service_times = []
    
    for vehicle in scenario.vehicles.values():
        max_time = (getattr(vehicle, 'max_total_work_time', None) or 
                   getattr(vehicle, 'max_time', None) or 
                   getattr(vehicle, 'max_work_time', None) or 600)
        vehicle_times.append(int(max_time))
    
    for location in scenario.locations.values():
        service_time = getattr(location, 'service_time', 0)
        if service_time > 0:
            service_times.append(service_time)
    
    min_vehicle_time = min(vehicle_times)
    max_vehicle_time = max(vehicle_times)
    total_service_time = sum(service_times)
    avg_service_time = sum(service_times) / len(service_times) if service_times else 0
    
    logger.info(f"   🚛 Vehicle time limits: {min_vehicle_time}-{max_vehicle_time}min")
    logger.info(f"   ⏱️  Total service time for all locations: {total_service_time}min")
    logger.info(f"   ⏱️  Average service time per location: {avg_service_time:.1f}min")
    logger.info(f"   📊 Locations with service time: {len(service_times)}/{len(scenario.locations)}")
    
    # Calculate theoretical minimum time needed
    # Assume we visit all locations with one vehicle
    locations_per_vehicle = len(scenario.locations) / len(scenario.vehicles)
    service_time_per_vehicle = total_service_time / len(scenario.vehicles)
    
    # Rough estimate of travel time (assume average 1 hour between locations)
    estimated_travel_time_per_vehicle = locations_per_vehicle * 30  # 30 min average travel between locations
    
    estimated_total_time_per_vehicle = service_time_per_vehicle + estimated_travel_time_per_vehicle
    
    logger.info(f"\n📊 THEORETICAL ANALYSIS:")
    logger.info(f"   📍 Avg locations per vehicle: {locations_per_vehicle:.1f}")
    logger.info(f"   ⏱️  Avg service time per vehicle: {service_time_per_vehicle:.1f}min")
    logger.info(f"   🚗 Estimated travel time per vehicle: {estimated_travel_time_per_vehicle:.1f}min")
    logger.info(f"   📊 Estimated total time per vehicle: {estimated_total_time_per_vehicle:.1f}min")
    logger.info(f"   📊 Minimum vehicle limit: {min_vehicle_time}min")
    
    if estimated_total_time_per_vehicle > min_vehicle_time:
        logger.error(f"   ❌ PROBLEM: Estimated time ({estimated_total_time_per_vehicle:.1f}min) > min vehicle limit ({min_vehicle_time}min)")
        logger.error(f"   ❌ The problem may be fundamentally infeasible with current time constraints!")
        
        # Suggest solutions
        logger.info(f"\n💡 POTENTIAL SOLUTIONS:")
        logger.info(f"   1. Increase vehicle time limits to at least {estimated_total_time_per_vehicle:.0f}min")
        logger.info(f"   2. Reduce service times (currently avg {avg_service_time:.1f}min)")
        logger.info(f"   3. Add more vehicles to distribute the load")
        logger.info(f"   4. Allow vehicles to make multiple trips")
    else:
        logger.info(f"   ✅ Theoretical feasibility looks OK")
    
    # Try a very relaxed version to see if it's solvable at all
    logger.info(f"\n🔧 TESTING VERY RELAXED VERSION:")
    
    class RelaxedOptimizer(VRPOptimizerEnhanced):
        def solve(self, instance, time_limit_seconds=300):
            """Test with very relaxed constraints."""
            if not ORTOOLS_AVAILABLE:
                return {'success': False, 'error': 'OR-Tools not available'}
            
            # Setup basic OR-Tools problem
            num_locations = len(instance.locations)
            num_vehicles = len(instance.vehicles)
            depot_index = 0
            
            manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
            routing = pywrapcp.RoutingModel(manager)
            location_list = list(instance.locations.values())
            
            # Simple distance callback only (no service time)
            def simple_distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                
                from_loc = location_list[from_node]
                to_loc = location_list[to_node]
                
                # Simple Euclidean distance in degrees * 1000 for scaling
                dx = to_loc.x - from_loc.x
                dy = to_loc.y - from_loc.y
                distance = (dx*dx + dy*dy)**0.5
                
                return int(distance * 1000)
            
            transit_callback_index = routing.RegisterTransitCallback(simple_distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
            
            # NO TIME CONSTRAINTS - just pure distance optimization
            logger.info("   🔧 Testing with NO time constraints, only distance...")
            
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            search_parameters.time_limit.FromSeconds(30)
            
            assignment = routing.SolveWithParameters(search_parameters)
            
            if assignment:
                vehicles_used = 0
                for vehicle_id in range(routing.vehicles()):
                    index = routing.Start(vehicle_id)
                    route_length = 0
                    while not routing.IsEnd(index):
                        route_length += 1
                        index = assignment.Value(routing.NextVar(index))
                    if route_length > 1:
                        vehicles_used += 1
                
                logger.info(f"   ✅ SUCCESS with no constraints: {vehicles_used} vehicles used")
                return {'success': True, 'vehicles_used': vehicles_used, 'message': 'No constraints'}
            else:
                logger.error(f"   ❌ FAILED even with no constraints!")
                return {'success': False, 'error': 'Failed even without constraints'}
    
    # Test the relaxed version
    relaxed_optimizer = RelaxedOptimizer()
    relaxed_result = relaxed_optimizer.solve(scenario, time_limit_seconds=30)
    
    if relaxed_result['success']:
        logger.info(f"   ✅ Basic routing works: {relaxed_result['vehicles_used']} vehicles")
        logger.info(f"   💡 The issue is likely with time constraint enforcement")
    else:
        logger.error(f"   ❌ Even basic routing fails: {relaxed_result['error']}")
        logger.error(f"   💡 The problem may have fundamental issues")

if __name__ == "__main__":
    debug_time_constraints()
