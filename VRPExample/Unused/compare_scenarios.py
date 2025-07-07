#!/usr/bin/env python3
"""
Compare old working scenarios with current scenarios to identify the root cause of infeasibility.
"""

import os
import sys
import json
from typing import Dict, Any
import numpy as np
import pandas as pd

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import both scenario modules
import vrp_scenarios
import vrp_scenarios_old
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_data_models import VRPObjective, VRPInstance

def compare_scenario_structures(old_scenario: VRPInstance, new_scenario: VRPInstance) -> Dict[str, Any]:
    """Compare the structure of old and new scenarios to identify key differences."""
    differences = {}
    
    # Compare basic properties
    differences["name_match"] = old_scenario.name == new_scenario.name
    differences["location_count"] = {
        "old": len(old_scenario.locations),
        "new": len(new_scenario.locations),
        "diff": len(new_scenario.locations) - len(old_scenario.locations)
    }
    differences["vehicle_count"] = {
        "old": len(old_scenario.vehicles),
        "new": len(new_scenario.vehicles),
        "diff": len(new_scenario.vehicles) - len(old_scenario.vehicles)
    }
    differences["ride_request_count"] = {
        "old": len(old_scenario.ride_requests) if old_scenario.ride_requests else 0,
        "new": len(new_scenario.ride_requests) if new_scenario.ride_requests else 0,
        "diff": (len(new_scenario.ride_requests) if new_scenario.ride_requests else 0) - 
               (len(old_scenario.ride_requests) if old_scenario.ride_requests else 0)
    }
    
    # Check vehicle properties
    old_vehicle = next(iter(old_scenario.vehicles.values()))
    new_vehicle = next(iter(new_scenario.vehicles.values()))
    
    differences["vehicle_capacity"] = {
        "old": getattr(old_vehicle, "capacity", None),
        "new": getattr(new_vehicle, "capacity", None),
        "match": getattr(old_vehicle, "capacity", None) == getattr(new_vehicle, "capacity", None)
    }
    differences["vehicle_max_time"] = {
        "old": getattr(old_vehicle, "max_time", None),
        "new": getattr(new_vehicle, "max_time", None),
        "match": getattr(old_vehicle, "max_time", None) == getattr(new_vehicle, "max_time", None)
    }
    
    # Compare time window implementations
    old_tw_count = sum(1 for loc in old_scenario.locations.values() 
                     if hasattr(loc, 'time_window_start') and loc.time_window_start is not None)
    new_tw_count = sum(1 for loc in new_scenario.locations.values() 
                     if hasattr(loc, 'time_window_start') and loc.time_window_start is not None)
    
    differences["time_windows"] = {
        "old_count": old_tw_count,
        "new_count": new_tw_count,
        "old_percentage": old_tw_count / len(old_scenario.locations) if len(old_scenario.locations) > 0 else 0,
        "new_percentage": new_tw_count / len(new_scenario.locations) if len(new_scenario.locations) > 0 else 0
    }
      # Check for fractional cargo weights in ride requests
    if old_scenario.ride_requests and new_scenario.ride_requests:
        old_fractional = any(req.passengers != int(req.passengers) for req in old_scenario.ride_requests)
        new_fractional = any(req.passengers != int(req.passengers) for req in new_scenario.ride_requests)
        
        old_cargo_weights = [req.passengers for req in old_scenario.ride_requests]
        new_cargo_weights = [req.passengers for req in new_scenario.ride_requests]
        
        differences["cargo_weights"] = {
            "old_has_fractional": old_fractional,
            "new_has_fractional": new_fractional,
            "old_min": min(old_cargo_weights) if old_cargo_weights else None,
            "old_max": max(old_cargo_weights) if old_cargo_weights else None,
            "old_avg": sum(old_cargo_weights) / len(old_cargo_weights) if old_cargo_weights else None,
            "new_min": min(new_cargo_weights) if new_cargo_weights else None,
            "new_max": max(new_cargo_weights) if new_cargo_weights else None,
            "new_avg": sum(new_cargo_weights) / len(new_cargo_weights) if new_cargo_weights else None
        }
    
    return differences

def analyze_time_windows(scenario: VRPInstance):
    """Analyze time window implementation in the given scenario."""
    time_windows = []
    
    for loc_id, location in scenario.locations.items():
        if hasattr(location, 'time_window_start') and location.time_window_start is not None:
            start = location.time_window_start
            end = location.time_window_end if hasattr(location, 'time_window_end') else None
            window_size = end - start if end is not None else None;
            
            time_windows.append({
                'location_id': loc_id,
                'start': start,
                'end': end,
                'window_size': window_size
            })
    
    if time_windows:
        df = pd.DataFrame(time_windows)
        stats = {
            'count': len(df),
            'start_min': int(df['start'].min()),
            'start_max': int(df['start'].max()),
            'start_mean': float(df['start'].mean()),
            'end_min': int(df['end'].min()),
            'end_max': int(df['end'].max()),
            'end_mean': float(df['end'].mean()),
            'size_min': int(df['window_size'].min()),
            'size_max': int(df['window_size'].max()), 
            'size_mean': float(df['window_size'].mean())
        }
        return stats
    else:
        return {'count': 0}

def test_scenario_solvability(scenario: VRPInstance, scenario_name: str):
    """Test if a scenario is solvable using OR-Tools."""
    print(f"\nTesting solvability of {scenario_name}...")
    
    # Try to solve with OR-Tools
    optimizer = VRPQuantumOptimizer(scenario, VRPObjective.MINIMIZE_DISTANCE)
    result = optimizer.optimize_with_ortools()
    
    is_solved = result.status == "optimal" and result.objective_value > 0
    vehicles_used = len([r for r in result.routes.values() if len(r) > 2]) if result.routes else 0
    
    solvability_info = {
        'solved': is_solved,
        'status': result.status,
        'vehicles_used': vehicles_used,
        'objective_value': result.objective_value,
        'runtime': result.runtime
    }
    
    # Print solver outcome
    print(f"  Result: {'✅ SOLVED' if is_solved else '❌ NOT SOLVED'}")
    print(f"  Status: {result.status}")
    print(f"  Vehicles used: {vehicles_used}")
    print(f"  Objective value: {result.objective_value}")
    print(f"  Runtime: {result.runtime:.2f}ms")
    
    return solvability_info

def analyze_scenario_differences():
    """Main function to analyze differences between old and new scenarios."""
    print("Comparing old and new MODA scenarios to identify root cause of infeasibility")
    print("="*80)
    
    # Load scenarios
    old_scenarios = vrp_scenarios_old.get_all_scenarios()
    new_scenarios = vrp_scenarios.get_all_scenarios()
    
    # Print available scenarios
    print(f"Old scenarios: {list(old_scenarios.keys())}")
    print(f"New scenarios: {list(new_scenarios.keys())}")
    
    # Get MODA scenarios from both
    old_moda_small = old_scenarios.get('MODA_small')
    new_moda_small = new_scenarios.get('MODA_small')
    
    old_moda_first = old_scenarios.get('MODA_first')
    new_moda_first = new_scenarios.get('MODA_first')
    
    # Compare MODA_small scenarios
    if old_moda_small and new_moda_small:
        print("\n--- MODA_small Comparison ---")
        differences = compare_scenario_structures(old_moda_small, new_moda_small)
        print(json.dumps(differences, indent=2))
        
        print("\nTime Window Analysis (Old MODA_small):")
        old_tw_stats = analyze_time_windows(old_moda_small)
        print(json.dumps(old_tw_stats, indent=2))
        
        print("\nTime Window Analysis (New MODA_small):")
        new_tw_stats = analyze_time_windows(new_moda_small)
        print(json.dumps(new_tw_stats, indent=2))
        
        # Test solvability
        old_solvability = test_scenario_solvability(old_moda_small, "Old MODA_small")
        new_solvability = test_scenario_solvability(new_moda_small, "New MODA_small")
        
        # Summary
        print("\nMODA_small Solvability Summary:")
        print(f"  Old scenario: {'✅ SOLVABLE' if old_solvability['solved'] else '❌ UNSOLVABLE'}")
        print(f"  New scenario: {'✅ SOLVABLE' if new_solvability['solved'] else '❌ UNSOLVABLE'}")
    
    # Compare MODA_first scenarios
    if old_moda_first and new_moda_first:
        print("\n--- MODA_first Comparison ---")
        differences = compare_scenario_structures(old_moda_first, new_moda_first)
        print(json.dumps(differences, indent=2))
        
        print("\nTime Window Analysis (Old MODA_first):")
        old_tw_stats = analyze_time_windows(old_moda_first)
        print(json.dumps(old_tw_stats, indent=2))
        
        print("\nTime Window Analysis (New MODA_first):")
        new_tw_stats = analyze_time_windows(new_moda_first)
        print(json.dumps(new_tw_stats, indent=2))
        
        # Test solvability
        old_solvability = test_scenario_solvability(old_moda_first, "Old MODA_first")
        new_solvability = test_scenario_solvability(new_moda_first, "New MODA_first")
        
        # Summary
        print("\nMODA_first Solvability Summary:")
        print(f"  Old scenario: {'✅ SOLVABLE' if old_solvability['solved'] else '❌ UNSOLVABLE'}")
        print(f"  New scenario: {'✅ SOLVABLE' if new_solvability['solved'] else '❌ UNSOLVABLE'}")
      # Identify potential issues based on differences
    print("\n--- Potential Issues Identified ---")
    
    if old_moda_small and new_moda_small:
        if old_solvability['solved'] and not new_solvability['solved']:
            print("🚨 MODA_small: Working in old version but not in new version")
            
            # Time window issues
            old_tw = analyze_time_windows(old_moda_small)
            new_tw = analyze_time_windows(new_moda_small)
            
            if new_tw['count'] > old_tw['count']:
                print("⚠️ Time windows added or modified in new version")
                
                if new_tw['count'] > 0:
                    if new_tw['size_min'] <= 0:
                        print("🔴 ERROR: Some time windows have zero or negative duration")
                        print("   This makes locations impossible to visit")
              # Check for unreachable time windows (e.g., pickup after dropoff)
            if new_moda_small.ride_requests:
                for req in new_moda_small.ride_requests:
                    pickup_loc = new_moda_small.locations.get(req.pickup_location)
                    dropoff_loc = new_moda_small.locations.get(req.dropoff_location)
                    
                    if (pickup_loc and dropoff_loc and 
                        hasattr(pickup_loc, 'time_window_end') and hasattr(dropoff_loc, 'time_window_start') and
                        pickup_loc.time_window_end and dropoff_loc.time_window_start and
                        pickup_loc.time_window_end > dropoff_loc.time_window_start):
                        print(f"🔴 ERROR: Pickup time window ends after dropoff time window starts for request {req.id}")            # Cargo weight issues (fractional is OK, just note it)
            if hasattr(differences, 'cargo_weights') and differences['cargo_weights']['new_has_fractional']:
                print("ℹ️ New scenario has fractional cargo weights (this is normal and supported)")
    
    if old_moda_first and new_moda_first:
        # Get solvability info for MODA_first (reuse old_solvability and new_solvability variables from earlier)
        old_solvability_first = test_scenario_solvability(old_moda_first, "Old MODA_first")
        new_solvability_first = test_scenario_solvability(new_moda_first, "New MODA_first")
        
        if old_solvability_first['solved'] and not new_solvability_first['solved']:
            print("🚨 MODA_first: Working in old version but not in new version")
            
            # Time window issues
            old_tw_first = analyze_time_windows(old_moda_first)
            new_tw_first = analyze_time_windows(new_moda_first)
            
            if new_tw_first['count'] > old_tw_first['count']:
                print("⚠️ Time windows added or modified in new version")
                
                if new_tw_first['count'] > 0:
                    if new_tw_first['size_min'] <= 0:
                        print("🔴 ERROR: Some time windows have zero or negative duration")
                        print("   This makes locations impossible to visit")
            
            # Check for vehicle constraints vs scenario size
            total_requests = len(new_moda_first.ride_requests) if new_moda_first.ride_requests else 0
            total_vehicles = len(new_moda_first.vehicles) if new_moda_first.vehicles else 0
            
            if total_vehicles == 0:
                print("🔴 ERROR: No vehicles available")
            elif total_requests > 0:
                print(f"ℹ️ Scenario has {total_requests} requests and {total_vehicles} vehicles")
                print("   In VRP, vehicles can serve multiple passengers per day via multiple trips")
                  # Check for unreachable requests (e.g., pickup after dropoff time windows)
                if new_moda_first.ride_requests:
                    for req in new_moda_first.ride_requests:
                        pickup_loc = new_moda_first.locations.get(req.pickup_location)
                        dropoff_loc = new_moda_first.locations.get(req.dropoff_location)
                        
                        if (pickup_loc and dropoff_loc and 
                            hasattr(pickup_loc, 'time_window_end') and hasattr(dropoff_loc, 'time_window_start') and
                            pickup_loc.time_window_end and dropoff_loc.time_window_start and
                            pickup_loc.time_window_end > dropoff_loc.time_window_start):                            print(f"🔴 ERROR: Pickup time window ends after dropoff time window starts for request {req.id}")
    
    print("\n--- Recommended Actions ---")
    print("1. Check time window implementations - ensure all time windows have positive duration")
    print("2. Verify that pickup time windows don't end after dropoff time windows start")
    print("3. Ensure OR-Tools properly handles fractional cargo weights in capacity constraints")
    print("4. Review time window constraints in OR-Tools formulation")
    print("5. Consider if vehicle time limits (10h = 600min) match real-world trucking constraints")

if __name__ == "__main__":
    analyze_scenario_differences()
        