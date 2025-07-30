#!/usr/bin/env python3
"""
Quick test to check HoS data availability for violated routes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scenario_loader import create_scenario_from_excel
from algo.l1_heuristic import EPDTMegaSolver
from algo.driver_assignment_enhanced import print_assignment_summary, assign_drivers_to_routes_enhanced
from src.driver_loader import load_drivers_from_excel_enhanced

def test_hos_data():
    print("=== Quick HoS Data Test ===")
    
    # Load scenario
    excel_file = "../src/furgoni2.xlsx"
    scenario = create_scenario_from_excel(excel_file)
    print(f"✅ Loaded scenario with {len(scenario.orders)} orders, {len(scenario.vehicles)} vehicles")
    
    # Run solver
    solver = EPDTMegaSolver(scenario)
    solution = solver.solve()
    print(f"✅ Solver completed with {len(solution.routes)} routes")
    
    # Check HoS data on routes before driver assignment
    print("\n=== Route HoS Data Status BEFORE Driver Assignment ===")
    for i, route in enumerate(solution.routes[:5]):  # Check first 5 routes
        has_hos = hasattr(route, 'hos_daily_summary') and route.hos_daily_summary
        print(f"Route {i+1} ({route.vehicle.id}): HoS data = {has_hos}")
        if has_hos:
            print(f"  Days: {list(route.hos_daily_summary.keys())}")
    
    # Load drivers and assign them
    drivers = load_drivers_from_excel_enhanced(excel_file)
    print(f"✅ Loaded {len(drivers)} drivers")
    
    assigned_routes = assign_drivers_to_routes_enhanced(solution.routes, drivers)
    print(f"✅ Assigned drivers to {len(assigned_routes)} routes")
    
    # Check HoS data on routes after driver assignment
    print("\n=== Route HoS Data Status AFTER Driver Assignment ===")
    for i, route in enumerate(assigned_routes[:5]):  # Check first 5 routes
        has_hos = hasattr(route, 'hos_daily_summary') and route.hos_daily_summary
        print(f"Route {i+1} ({route.vehicle.id}): HoS data = {has_hos}")
        if has_hos:
            print(f"  Days: {list(route.hos_daily_summary.keys())}")
    
    print("\n=== Driver Assignment Summary ===")
    print_assignment_summary(assigned_routes, drivers)

if __name__ == "__main__":
    test_hos_data()
