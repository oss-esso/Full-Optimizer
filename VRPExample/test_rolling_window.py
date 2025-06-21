"""
Test the new rolling window optimizer with MODA scenarios.
This script validates that the rolling 10-hour window logic allows 
feasible solutions for realistic trucking scenarios.
"""

import os
import sys
import logging

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios import create_moda_small_scenario, create_moda_first_scenario
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rolling_window_moda_scenarios():
    """Test rolling window optimizer with both MODA scenarios."""
    
    logger.info("=" * 60)
    logger.info("Testing Rolling Window Optimizer with MODA Scenarios")
    logger.info("=" * 60)    
    scenarios = [
        ("MODA_small", create_moda_small_scenario),
        ("MODA_first", create_moda_first_scenario)
    ]
    
    results = {}
    
    for scenario_name, scenario_func in scenarios:
        logger.info(f"\n--- Testing {scenario_name} ---")
        
        try:
            # Generate scenario
            logger.info(f"Generating {scenario_name} scenario...")
            instance = scenario_func()
            
            if not instance:
                logger.error(f"Failed to generate {scenario_name} scenario")
                continue
            
            # Log scenario details
            num_vehicles = len(instance.vehicles)
            num_requests = len(instance.ride_requests) if instance.ride_requests else 0
            num_locations = len(instance.location_ids)
            
            logger.info(f"Scenario details:")
            logger.info(f"  - Vehicles: {num_vehicles}")
            logger.info(f"  - Ride requests: {num_requests}")
            logger.info(f"  - Total locations: {num_locations}")
            
            # Check vehicle time constraints
            for vehicle_id, vehicle in instance.vehicles.items():
                max_time = getattr(vehicle, 'max_total_work_time', None)
                if max_time:
                    logger.info(f"  - Vehicle {vehicle_id}: max {max_time} minutes ({max_time/60:.1f} hours)")
            
            # Check time window spans in ride requests
            if instance.ride_requests:
                time_spans = []
                for request in instance.ride_requests:
                    pickup_location = instance.locations.get(request.pickup_location)
                    if pickup_location and hasattr(pickup_location, 'time_window'):
                        start_time, end_time = pickup_location.time_window
                        span = end_time - start_time
                        time_spans.append(span)
                
                if time_spans:
                    avg_span = sum(time_spans) / len(time_spans)
                    max_span = max(time_spans)
                    logger.info(f"  - Time window spans: avg {avg_span:.1f} min, max {max_span:.1f} min")
            
            # Test with rolling window optimizer
            logger.info(f"Optimizing {scenario_name} with rolling window approach...")
            optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
            result = optimizer.optimize_with_rolling_window()
            
            # Log results
            logger.info(f"Optimization result:")
            logger.info(f"  - Status: {result.status}")
            logger.info(f"  - Runtime: {result.runtime:.2f} ms")
            
            if result.status == "optimal":
                metrics = result.metrics
                logger.info(f"  - Total distance: {metrics.get('total_distance', 0):.2f}")
                logger.info(f"  - Vehicles used: {metrics.get('vehicles_used', 0)}")
                logger.info(f"  - Locations served: {metrics.get('total_locations_served', 0)}")
                logger.info(f"  - Average route length: {metrics.get('average_route_length', 0):.2f}")
                
                if 'rolling_window' in metrics:
                    logger.info(f"  - Rolling window: {metrics['rolling_window']}")
                if 'max_route_duration_minutes' in metrics:
                    max_duration = metrics['max_route_duration_minutes']
                    if max_duration:
                        logger.info(f"  - Max route duration: {max_duration} minutes ({max_duration/60:.1f} hours)")
                
                # Analyze routes
                active_routes = {k: v for k, v in result.routes.items() if len(v) > 2}
                logger.info(f"  - Active routes: {len(active_routes)}")
                
                for vehicle_id, route in active_routes.items():
                    route_length = len(route) - 2  # Exclude start/end depot
                    logger.info(f"    - Vehicle {vehicle_id}: {route_length} stops")
                    
                    # Calculate route distance
                    route_distance = 0.0
                    for i in range(len(route) - 1):
                        from_loc = route[i]
                        to_loc = route[i + 1]
                        distance = instance.get_distance(from_loc, to_loc)
                        route_distance += distance
                    logger.info(f"      Distance: {route_distance:.2f}")
            else:
                logger.warning(f"  - Optimization failed: {result.metrics}")
            
            results[scenario_name] = result
            
        except Exception as e:
            logger.error(f"Error testing {scenario_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n" + "=" * 60)
    logger.info("ROLLING WINDOW TEST SUMMARY")
    logger.info("=" * 60)
    
    for scenario_name, result in results.items():
        status = result.status if result else "failed"
        logger.info(f"{scenario_name}: {status}")
        
        if result and result.status == "optimal":
            vehicles_used = result.metrics.get('vehicles_used', 0)
            total_distance = result.metrics.get('total_distance', 0)
            logger.info(f"  -> {vehicles_used} vehicles, {total_distance:.2f} total distance")
    
    return results

def compare_time_constraints():
    """Compare the time constraint approaches between scenarios."""
    
    logger.info(f"\n" + "=" * 60)
    logger.info("TIME CONSTRAINT ANALYSIS")
    logger.info("=" * 60)    
    scenarios = [
        ("MODA_small", create_moda_small_scenario),
        ("MODA_first", create_moda_first_scenario)
    ]
    
    for scenario_name, scenario_func in scenarios:
        logger.info(f"\n--- {scenario_name} Time Analysis ---")
        
        try:
            instance = scenario_func()
            
            # Analyze vehicle constraints
            logger.info("Vehicle time constraints:")
            for vehicle_id, vehicle in instance.vehicles.items():
                max_time = getattr(vehicle, 'max_total_work_time', None)
                if max_time:
                    logger.info(f"  - {vehicle_id}: {max_time} minutes ({max_time/60:.1f} hours)")
                else:
                    logger.info(f"  - {vehicle_id}: No explicit time constraint (default 10 hours)")
            
            # Analyze location time windows
            if instance.ride_requests:
                logger.info("Location time windows:")
                earliest_time = float('inf')
                latest_time = float('-inf')
                
                for request in instance.ride_requests:
                    pickup_location = instance.locations.get(request.pickup_location)
                    dropoff_location = instance.locations.get(request.dropoff_location)
                    
                    for location_id, location in [(request.pickup_location, pickup_location), 
                                                  (request.dropoff_location, dropoff_location)]:
                        if location and hasattr(location, 'time_window'):
                            start_time, end_time = location.time_window
                            earliest_time = min(earliest_time, start_time)
                            latest_time = max(latest_time, end_time)
                            logger.info(f"  - {location_id}: [{start_time:.0f}, {end_time:.0f}] minutes")
                
                if earliest_time != float('inf'):
                    total_span = latest_time - earliest_time
                    logger.info(f"Overall time span: {total_span:.0f} minutes ({total_span/60:.1f} hours)")
                    logger.info(f"From {earliest_time:.0f} to {latest_time:.0f} minutes")
                    
                    # Compare with vehicle constraints
                    vehicle_max_times = []
                    for vehicle in instance.vehicles.values():
                        max_time = getattr(vehicle, 'max_total_work_time', 600)  # Default 10 hours
                        vehicle_max_times.append(max_time)
                    
                    max_vehicle_time = max(vehicle_max_times) if vehicle_max_times else 600
                    
                    logger.info(f"Max vehicle route duration: {max_vehicle_time} minutes ({max_vehicle_time/60:.1f} hours)")
                    
                    if total_span > max_vehicle_time:
                        logger.warning(f"⚠️  Time span ({total_span:.0f} min) > Max route duration ({max_vehicle_time} min)")
                        logger.info("🔄 Rolling window approach allows vehicles to start at different times")
                        logger.info("   Each vehicle can handle any 10-hour window within the overall span")
                    else:
                        logger.info("✅ Time span fits within vehicle route duration limits")
                        
        except Exception as e:
            logger.error(f"Error analyzing {scenario_name}: {str(e)}")

if __name__ == "__main__":
    print("Testing Rolling Window VRP Optimizer")
    print("=====================================")
    
    # Test the scenarios
    results = test_rolling_window_moda_scenarios()
    
    # Analyze time constraints
    compare_time_constraints()
    
    print("\nTest completed!")
