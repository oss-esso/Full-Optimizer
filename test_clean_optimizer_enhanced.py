#!/usr/bin/env python3
"""
Test script for the enhanced Clean VRP Optimizer with comprehensive constraints check
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from VRPExample.vrp_optimizer_multiday import CleanVRPOptimizer

def test_enhanced_clean_optimizer():
    """Test the enhanced clean optimizer with comprehensive constraints check."""
    print("🧪 Testing Enhanced Clean VRP Optimizer")
    print("=" * 60)
    
    # Create a simple test scenario
    vehicle_list = [
        {"id": "vehicle_1", "capacity": 100, "start_location": "depot", "end_location": "depot", "max_time": 480},
        {"id": "vehicle_2", "capacity": 150, "start_location": "depot", "end_location": "depot", "max_time": 480},
    ]

    location_list = [
        {"id": "depot", "x": 0, "y": 0, "demand": 0, "time_window": (0, 1440), "service_time": 0, "address": "Main Depot"},
        {"id": "pickup_1", "x": 10, "y": 10, "demand": 0, "time_window": (60, 300), "service_time": 15, "address": "Pickup Location 1"},
        {"id": "dropoff_1", "x": 20, "y": 20, "demand": 0, "time_window": (120, 400), "service_time": 15, "address": "Dropoff Location 1"},
        {"id": "pickup_2", "x": -10, "y": 15, "demand": 0, "time_window": (90, 350), "service_time": 15, "address": "Pickup Location 2"},
        {"id": "dropoff_2", "x": -20, "y": 25, "demand": 0, "time_window": (150, 450), "service_time": 15, "address": "Dropoff Location 2"},
    ]
    
    # Create ride requests
    class RideRequest:
        def __init__(self, id, pickup_location, dropoff_location, passengers):
            self.id = id
            self.pickup_location = pickup_location
            self.dropoff_location = dropoff_location
            self.passengers = passengers
    
    ride_requests = [
        RideRequest("req_1", "pickup_1", "dropoff_1", 50),
        RideRequest("req_2", "pickup_2", "dropoff_2", 75),
    ]
    
    optimizer = CleanVRPOptimizer(
        vehicles=vehicle_list,
        locations=location_list
    )
    optimizer.ride_requests = ride_requests
    
    # Test different constraint levels
    levels = ["none", "capacity", "pickup_delivery", "time_windows", "full"]
    
    for level in levels:
        print(f"\n{'='*20} TESTING LEVEL: {level.upper()} {'='*20}")
        
        try:
            # Test both with and without hybrid calculator
            for use_hybrid in [False, True]:
                hybrid_label = "WITH HYBRID CALC" if use_hybrid else "WITHOUT HYBRID CALC"
                print(f"\n--- {level.upper()} {hybrid_label} ---")
                
                result, status, applied_constraints = optimizer.solve(
                    constraint_level=level, 
                    verbose=False, 
                    use_hybrid_calculator=use_hybrid
                )
                
                if result:
                    print(f"✅ SUCCESS - {level} {hybrid_label}")
                    print(f"   Status: {status}")
                    print(f"   Applied constraints: {applied_constraints}")
                    print(f"   Objective value: {result['objective_value']}")
                    print(f"   Total distance: {result['total_distance']:.1f} km")
                    
                    # Show validation results if available
                    if 'validation_results' in result:
                        print(f"   Validation results: {result['validation_results']}")
                    
                    # Show first few routes
                    for vehicle_id, route_data in list(result['routes'].items())[:1]:
                        route = route_data['route']
                        print(f"   {vehicle_id}: {len(route)} stops, {route_data['distance']:.1f}km, {route_data['time']}min")
                        
                else:
                    print(f"❌ FAILED - {level} {hybrid_label}")
                    print(f"   Status: {status}")
                    print(f"   Applied constraints: {applied_constraints}")
                
        except Exception as e:
            print(f"❌ ERROR with {level}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_clean_optimizer()