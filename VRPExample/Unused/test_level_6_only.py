#!/usr/bin/env python3
"""
Test script to specifically test Level 6 driver breaks with REALISTIC TRAVEL TIMES
Focus on debugging service time and constraint enforcement issues
"""

import logging
import sys
import os
from vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from vrp_scenarios import create_moda_small_scenario
from hybrid_travel_calculator import hybrid_calculator

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_level_6_service_time_focused():
    """Test Level 6 with focus on proper service time enforcement."""
    logger.info("🧪 TESTING LEVEL 6 - SERVICE TIME & DRIVER BREAKS FOCUSED")
    logger.info("=" * 80)
    
    # Load scenario
    scenario = create_moda_small_scenario()
    logger.info(f"✅ Loaded scenario: {len(scenario.locations)} locations, {len(scenario.vehicles)} vehicles")
      # Print service time details for debugging
    logger.info("🔍 SERVICE TIME DEBUGGING:")
    total_expected_service_time = test_service_time_configuration()
    
    if total_expected_service_time == 0:
        logger.warning("⚠️  No service times found in scenario! This might be the issue.")
    else:
        logger.info(f"📊 Total expected service time across all locations: {total_expected_service_time}min")
    
    # Initialize optimizer
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 6 with specific configuration
    logger.info("🎯 Testing Level 6: Full constraints + Driver breaks + Service time enforcement")
    logger.info("🔧 Configuration:")
    logger.info("   - Distance matrix: Hybrid (Haversine + OSRM correction)")
    logger.info("   - Time windows: Enforced")
    logger.info("   - Capacity: Enforced") 
    logger.info("   - Pickup-delivery: Enforced")
    logger.info("   - Service times: ENFORCED (debugging focus)")
    logger.info("   - Driver breaks: Added")
    
    try:        # Use the internal method to test Level 6 specifically
        result = optimizer._solve_with_constraint_level(scenario, "full", 120)
        
        if result['success']:
            logger.info("✅ Level 6 SUCCESS!")
            logger.info(f"   🎯 Objective: {result['objective_value']}")
            logger.info(f"   🚛 Vehicles used: {result['vehicles_used']}/{len(scenario.vehicles)}")
            logger.info(f"   ⏱️  Solve time: {result['solve_time']:.2f}s")
            logger.info(f"   🔧 Constraints: {result.get('constraints_applied', 'N/A')}")
              # Detailed route analysis with service time focus
            if 'route_analysis' in result:
                logger.info("")
                logger.info("📊 DETAILED ROUTE ANALYSIS (Service Time Focus):")
                analysis = result['route_analysis']
                
                total_actual_service = 0
                total_driving = 0
                
                # Handle the case where analysis might be a list or have different structure
                if isinstance(analysis, list):
                    logger.info("   Route analysis is a list - extracting vehicle information...")
                    vehicles_data = analysis
                elif isinstance(analysis, dict) and 'vehicles' in analysis:
                    vehicles_data = analysis['vehicles']
                else:
                    logger.warning(f"   Unknown route analysis structure: {type(analysis)}")
                    vehicles_data = []
                
                for vehicle_info in vehicles_data:
                    if isinstance(vehicle_info, dict):
                        vid = vehicle_info.get('vehicle_id', 'N/A')
                        stops = vehicle_info.get('stops', 0)
                        total_time = vehicle_info.get('total_time', 0)
                        driving_time = vehicle_info.get('driving_time', 0)
                        service_time = vehicle_info.get('service_time', 0)
                        distance = vehicle_info.get('distance', 0)
                        
                        total_actual_service += service_time
                        total_driving += driving_time
                        
                        logger.info(f"   🚛 Vehicle {vid}: {stops} stops, {total_time}min total")
                        logger.info(f"      🚗 Driving: {driving_time}min, 🏢 Service: {service_time}min")
                        logger.info(f"      📏 Distance: {distance:.2f}km")
                        
                        if service_time == 0 and stops > 2:  # More than just depot visits
                            logger.warning(f"      ⚠️  ISSUE: Vehicle {vid} has {stops} stops but 0 service time!")
                    else:
                        logger.info(f"   Vehicle info: {vehicle_info}")
                
                logger.info("")
                logger.info("📈 SUMMARY STATISTICS:")
                logger.info(f"   🚗 Total realistic driving time: {total_driving} minutes")
                logger.info(f"   🏢 Total actual service time: {total_actual_service} minutes")
                logger.info(f"   🏢 Expected service time: {total_expected_service_time} minutes")
                logger.info(f"   ⏱️  Total working time: {total_driving + total_actual_service} minutes")
                
                if total_actual_service != total_expected_service_time:
                    logger.error(f"   ❌ SERVICE TIME MISMATCH!")
                    logger.error(f"      Expected: {total_expected_service_time}min, Actual: {total_actual_service}min")
                    logger.error(f"      Difference: {total_expected_service_time - total_actual_service}min")
                else:
                    logger.info(f"   ✅ Service time correctly accounted for!")
                
        else:
            logger.error("❌ Level 6 FAILED!")
            logger.error(f"   Error: {result.get('error', 'Unknown')}")
            logger.error(f"   Details: {result.get('details', 'No additional details')}")
            
    except Exception as e:
        logger.error(f"❌ Level 6 EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

def test_service_time_configuration():
    """Test to understand how service times are configured in the scenario."""
    logger.info("")
    logger.info("🔍 SERVICE TIME CONFIGURATION ANALYSIS")
    logger.info("=" * 60)
    
    scenario = create_moda_small_scenario()
    
    # Check service time configuration - locations are stored in instance.locations dict
    total_expected_service_time = 0
    for location_name, location_obj in scenario.locations.items():
        service_time = getattr(location_obj, 'service_time', 0)
        if service_time > 0:
            logger.info(f"  Location {location_name}: {service_time}min service time")
            total_expected_service_time += service_time
        else:
            logger.info(f"  Location {location_name}: No service time (depot or 0)")
    
    logger.info(f"📊 Total expected service time: {total_expected_service_time}min")
    logger.info(f"📊 Pickup locations (15min each): {len([loc for loc in scenario.locations.keys() if 'pickup' in loc])}")
    logger.info(f"📊 Dropoff locations (10min each): {len([loc for loc in scenario.locations.keys() if 'dropoff' in loc])}")
    
    return total_expected_service_time

def test_constraint_levels_progression():
    """Test different constraint levels to isolate where the issue occurs."""
    logger.info("")
    logger.info("🔍 CONSTRAINT LEVELS PROGRESSION TEST")
    logger.info("=" * 60)
    
    scenario = create_moda_small_scenario()
    optimizer = VRPOptimizerEnhanced()
    
    # Test levels in progression
    levels_to_test = [
        ("distance", "Level 1: Distance only"),
        ("time_dimension", "Level 2: Distance + Time"),
        ("time_windows", "Level 3: + Time windows"),
        ("capacity", "Level 4: + Capacity"),
        ("pickup_delivery", "Level 5: + Pickup-delivery"),
        ("full", "Level 6: + Driver breaks")
    ]
    
    for level, description in levels_to_test:
        logger.info(f"🧪 Testing {description}")
        try:
            result = optimizer._solve_with_constraint_level(scenario, level, 120)
            if result['success']:
                logger.info(f"   ✅ {level.upper()} SUCCESS - {result['vehicles_used']} vehicles used")
            else:
                logger.error(f"   ❌ {level.upper()} FAILED - {result.get('error', 'Unknown error')}")
                # If this level fails, no point testing higher levels
                logger.error(f"   🛑 Stopping progression test - issue found at {level}")
                break
        except Exception as e:
            logger.error(f"   ❌ {level.upper()} EXCEPTION: {e}")
            break

def test_simplified_depot_requests():
    """Test with only a few depot pickup requests to isolate the issue."""
    logger.info("")
    logger.info("🔍 SIMPLIFIED DEPOT REQUESTS TEST")
    logger.info("=" * 60)
    
    scenario = create_moda_small_scenario()
    
    # Keep only 2 depot requests and 2 field requests
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"][:2]
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"][:2]
    
    scenario.ride_requests = depot_requests + field_requests
    
    logger.info(f"Reduced scenario to {len(scenario.ride_requests)} requests:")
    for req in scenario.ride_requests:
        logger.info(f"  {req.id}: {req.pickup_location} → {req.dropoff_location} ({req.passengers}kg)")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 5 first
    logger.info("🧪 Testing simplified scenario - Level 5 (pickup-delivery)")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "pickup_delivery", 120)
        if result['success']:
            logger.info(f"   ✅ Level 5 SUCCESS - {result['vehicles_used']} vehicles used")
            
            # Now test Level 6
            logger.info("🧪 Testing simplified scenario - Level 6 (full)")
            result = optimizer._solve_with_constraint_level(scenario, "full", 120)
            if result['success']:
                logger.info(f"   ✅ Level 6 SUCCESS - {result['vehicles_used']} vehicles used")
            else:
                logger.error(f"   ❌ Level 6 FAILED - {result.get('error', 'Unknown error')}")
        else:
            logger.error(f"   ❌ Level 5 FAILED - {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"   ❌ EXCEPTION: {e}")

def test_only_field_requests():
    """Test with only field pickup requests (no depot pickups)."""
    logger.info("")
    logger.info("🔍 FIELD REQUESTS ONLY TEST")
    logger.info("=" * 60)
    
    scenario = create_moda_small_scenario()
    
    # Keep only field requests (remove all depot pickups)
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"]
    scenario.ride_requests = field_requests
    
    logger.info(f"Field-only scenario with {len(scenario.ride_requests)} requests:")
    for req in scenario.ride_requests[:3]:  # Show first 3
        logger.info(f"  {req.id}: {req.pickup_location} → {req.dropoff_location} ({req.passengers}kg)")
    logger.info(f"  ... and {len(scenario.ride_requests)-3} more")
    
    optimizer = VRPOptimizerEnhanced()
    
    # Test Level 6 directly
    logger.info("🧪 Testing field-only scenario - Level 6 (full)")
    try:
        result = optimizer._solve_with_constraint_level(scenario, "full", 120)
        if result['success']:
            logger.info(f"   ✅ Level 6 SUCCESS - {result['vehicles_used']} vehicles used")
        else:
            logger.error(f"   ❌ Level 6 FAILED - {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"   ❌ EXCEPTION: {e}")

def test_capacity_feasibility():
    """Test if the depot pickup weights are feasible given vehicle capacities."""
    logger.info("")
    logger.info("🔍 CAPACITY FEASIBILITY ANALYSIS")
    logger.info("=" * 60)
    
    scenario = create_moda_small_scenario()
    
    # Analyze vehicle capacities
    vehicle_capacities = [v.capacity for v in scenario.vehicles.values()]
    logger.info(f"Vehicle capacities: {vehicle_capacities}")
    logger.info(f"Total fleet capacity: {sum(vehicle_capacities)} kg")
    
    # Analyze depot requests
    depot_requests = [req for req in scenario.ride_requests if req.pickup_location == "depot_1"]
    field_requests = [req for req in scenario.ride_requests if req.pickup_location != "depot_1"]
    
    logger.info(f"\n📦 DEPOT PICKUP REQUESTS ({len(depot_requests)}):")
    depot_weights = []
    for req in depot_requests:
        weight = req.passengers
        depot_weights.append(weight)
        # Check which vehicles can handle this weight
        capable_vehicles = [i for i, cap in enumerate(vehicle_capacities) if cap >= weight]
        logger.info(f"  {req.id}: {weight}kg → {req.dropoff_location}")
        logger.info(f"    Can fit in vehicles: {capable_vehicles} (capacities: {[vehicle_capacities[i] for i in capable_vehicles]})")
        
        if not capable_vehicles:
            logger.error(f"    ❌ NO VEHICLE CAN HANDLE {weight}kg!")
    
    logger.info(f"\n📦 FIELD PICKUP REQUESTS ({len(field_requests)}):")
    field_weights = []
    for req in field_requests:
        weight = req.passengers
        field_weights.append(weight)
        capable_vehicles = [i for i, cap in enumerate(vehicle_capacities) if cap >= weight]
        logger.info(f"  {req.id}: {weight}kg from {req.pickup_location} → {req.dropoff_location}")
        logger.info(f"    Can fit in vehicles: {capable_vehicles}")
    
    # Check if total cargo exceeds fleet capacity
    total_depot_weight = sum(depot_weights)
    total_field_weight = sum(field_weights)
    total_cargo = total_depot_weight + total_field_weight
    fleet_capacity = sum(vehicle_capacities)
    
    logger.info(f"\n📊 CAPACITY SUMMARY:")
    logger.info(f"  Total depot cargo: {total_depot_weight} kg")
    logger.info(f"  Total field cargo: {total_field_weight} kg")
    logger.info(f"  Total cargo: {total_cargo} kg")
    logger.info(f"  Fleet capacity: {fleet_capacity} kg")
    logger.info(f"  Utilization: {total_cargo/fleet_capacity*100:.1f}%")
    
    if total_cargo > fleet_capacity:
        logger.error(f"  ❌ OVERLOADED: Cargo exceeds fleet capacity by {total_cargo - fleet_capacity}kg!")
    else:
        logger.info(f"  ✅ Fleet has sufficient capacity ({fleet_capacity - total_cargo}kg spare)")
    
    # Check for problematic weight combinations
    logger.info(f"\n🔍 PROBLEMATIC WEIGHTS:")
    large_depot_requests = [w for w in depot_weights if w > min(vehicle_capacities)]
    if large_depot_requests:
        logger.warning(f"  Large depot requests that only fit in heavy vehicles: {large_depot_requests}")
        logger.warning(f"  Heavy vehicles available: {len([c for c in vehicle_capacities if c >= 20000])}")
    
    # Check if multiple depot requests can be combined
    logger.info(f"\n🚚 VEHICLE LOADING ANALYSIS:")
    for i, capacity in enumerate(vehicle_capacities):
        logger.info(f"  Vehicle {i+1} ({capacity}kg capacity):")
        
        # Check which depot requests this vehicle could handle individually
        possible_depot = [w for w in depot_weights if w <= capacity]
        possible_field = [w for w in field_weights if w <= capacity]
        
        logger.info(f"    Could handle {len(possible_depot)} depot requests individually")
        logger.info(f"    Could handle {len(possible_field)} field requests individually")
        
        # Check combinations
        if len(possible_depot) >= 2:
            # Try combinations of 2 depot requests
            import itertools
            feasible_pairs = 0
            for pair in itertools.combinations(possible_depot, 2):
                if sum(pair) <= capacity:
                    feasible_pairs += 1
            logger.info(f"    Could handle {feasible_pairs} pairs of depot requests")

if __name__ == "__main__":
    test_service_time_configuration()
    print()
    
    # Check capacity feasibility first (your hypothesis!)
    test_capacity_feasibility()
    print()
    
    # Run diagnostic tests to isolate the issue
    test_constraint_levels_progression()
    print()
    test_simplified_depot_requests()
    print()
    test_only_field_requests()
    print()
    
    # Original Level 6 test
    test_level_6_service_time_focused()
