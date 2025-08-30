#!/usr/bin/env python3
"""
Targeted Sequence Test for Orders 5, 6, 8

This test analyzes all possible task sequences for orders 5, 6, and 8
using OSRM routing, testing with both small and large vehicles,
and with/without HoS breaks.
"""

import sys
import os
sys.path.append('..')

from itertools import permutations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time

# Import necessary modules
from comprehensive_integration_test import (
    create_scenario_from_excel, get_order_requirements, 
    _format_time_hhmm, _format_date_from_minutes,
    calculate_travel_time_with_counter
)

from algo.hos_simulation import calculate_travel_time_between_tasks


@dataclass
class SequenceResult:
    """Results for a specific task sequence"""
    sequence: List[Dict]
    total_time: float
    total_distance: float
    vehicle_type: str
    with_breaks: bool
    violation_count: int
    latest_arrival: float
    details: List[Dict]


@dataclass 
class OrderAnalysis:
    """Analysis results for a specific order"""
    order_id: int
    task_count: int
    best_sequence: Optional[SequenceResult]
    worst_sequence: Optional[SequenceResult]


def get_test_vehicles():
    """Get representative small and large vehicles for testing"""
    # Small vehicle (Furgone)
    small_vehicle = type('Vehicle', (), {
        'id': 'TEST_SMALL_GW895CW',
        'vehicle_type': 'furgone',
        'weight_capacity': 1500,
        'volume_capacity': 12.0,
        'pallet_capacity': 8,
        'average_speed': 90,  # Fast furgone
        'capabilities': 'STANDARD'
    })()
    
    # Large vehicle (Heavy truck)
    large_vehicle = type('Vehicle', (), {
        'id': 'TEST_LARGE_XA359KW', 
        'vehicle_type': 'heavy',
        'weight_capacity': 2800,
        'volume_capacity': 25.0,
        'pallet_capacity': 21,
        'average_speed': 70,  # Slower heavy truck
        'capabilities': 'HEAVY,LOADER'
    })()
    
    return [small_vehicle, large_vehicle]


def extract_order_tasks(order):
    """Extract all tasks for an order"""
    tasks = []
    
    # Add pickup tasks
    for pickup in order.pickup_tasks:
        # Handle location - could be location_id or actual location object
        location = getattr(pickup, 'location', None)
        if location is None:
            location_id = getattr(pickup, 'location_id', None)
            # Create a dummy location object if needed
            location = type('Location', (), {
                'name': f"Location_{pickup.id}",
                'latitude': 44.9 + (hash(str(pickup.id)) % 10) * 0.01,  # Dummy coordinates
                'longitude': 8.2 + (hash(str(pickup.id)) % 10) * 0.01,
                'address': str(location_id) if location_id else f"Address {pickup.id}"
            })() if location_id else None
        
        tasks.append({
            'id': f"PICKUP_{order.id}_{pickup.id}",
            'type': 'pickup',
            'order_id': order.id,
            'task': pickup,
            'location': location,
            'earliest_time': getattr(pickup, 'earliest_time', None),
            'latest_time': getattr(pickup, 'latest_time', None),
            'service_time': getattr(pickup, 'service_time', 15.0),  # Reduced service time
        })
    
    # Add delivery tasks  
    for delivery in order.delivery_tasks:
        # Handle location - could be location_id or actual location object
        location = getattr(delivery, 'location', None)
        if location is None:
            location_id = getattr(delivery, 'location_id', None)
            # Create a dummy location object if needed
            location = type('Location', (), {
                'name': f"Location_{delivery.id}",
                'latitude': 44.9 + (hash(str(delivery.id)) % 10) * 0.01,  # Dummy coordinates
                'longitude': 8.2 + (hash(str(delivery.id)) % 10) * 0.01,
                'address': str(location_id) if location_id else f"Address {delivery.id}"
            })() if location_id else None
        
        tasks.append({
            'id': f"DELIVERY_{order.id}_{delivery.id}",
            'type': 'delivery',
            'order_id': order.id,
            'task': delivery,
            'location': location,
            'earliest_time': getattr(delivery, 'earliest_time', None),
            'latest_time': getattr(delivery, 'latest_time', None),
            'service_time': getattr(delivery, 'service_time', 10.0),  # Reduced service time
        })
    
    return tasks


def calculate_sequence_time_osrm(tasks, vehicle, depot_location, with_breaks=True):
    """Calculate total time for a task sequence using OSRM"""
    if not tasks:
        return 0.0, 0.0, []
    
    total_time = 0.0
    total_distance = 0.0
    sequence_details = []
    current_location = depot_location
    current_time = 480.0  # Start at 8:00 AM
    driving_time_since_break = 0.0
    
    # Process each task
    for i, task in enumerate(tasks):
        # Calculate travel time from current location
        try:
            # Create a mock task for the current location
            if hasattr(current_location, 'location'):
                current_task = type('Task', (), {'location': current_location})()
            else:
                current_task = type('Task', (), {'location': current_location})()
            
            # Create actual task object for the destination
            dest_task = type('Task', (), {'location': task['location']})()
            
            travel_time = calculate_travel_time_between_tasks(current_task, dest_task, vehicle)
            travel_distance = 25.0  # Estimated distance (OSRM doesn't return distance in this function)
        except Exception as e:
            print(f"Travel time calculation error: {e}, using fallback")
            travel_time = 30.0  # Fallback
            travel_distance = 25.0
        
        # Apply HoS breaks if needed
        break_time = 0.0
        if with_breaks and travel_time > 0:
            driving_time_since_break += travel_time
            if driving_time_since_break >= 270:  # 4.5 hours
                break_time = 45.0  # 45 minute break
                driving_time_since_break = 0.0
        
        # Calculate arrival time
        arrival_time = current_time + travel_time + break_time
        
        # Check time window violation
        latest_allowed = task.get('latest_time', None)
        violation = 0
        if latest_allowed and arrival_time > latest_allowed:
            violation = arrival_time - latest_allowed
        
        # Service time
        service_time = task['service_time']
        departure_time = arrival_time + service_time
        
        # Record details
        sequence_details.append({
            'task_id': task['id'],
            'task_type': task['type'],
            'travel_time': travel_time,
            'break_time': break_time,
            'arrival_time': arrival_time,
            'service_time': service_time,
            'departure_time': departure_time,
            'time_window_violation': violation,
            'location': getattr(task['location'], 'name', 'Unknown')
        })
        
        # Update for next iteration
        current_time = departure_time
        current_location = task['location']
        total_time += travel_time + break_time + service_time
        total_distance += travel_distance
    
    # Return trip to depot
    try:
        if hasattr(current_location, 'location'):
            current_task = type('Task', (), {'location': current_location})()
        else:
            current_task = type('Task', (), {'location': current_location})()
        
        depot_task = type('Task', (), {'location': depot_location})()
        return_travel = calculate_travel_time_between_tasks(current_task, depot_task, vehicle)
        return_distance = 25.0
    except Exception as e:
        return_travel = 30.0
        return_distance = 25.0
    
    total_time += return_travel
    total_distance += return_distance
    
    return total_time, total_distance, sequence_details


def analyze_order_sequences(order, vehicle, depot_location, with_breaks=True):
    """Analyze all possible sequences for an order"""
    print(f"\n🔍 Analyzing Order {order.id} with {vehicle.id} ({'with' if with_breaks else 'without'} breaks)")
    
    # Extract tasks
    tasks = extract_order_tasks(order)
    if not tasks:
        print(f"⚠️ No tasks found for Order {order.id}")
        return None
    
    print(f"   📋 Tasks: {len(tasks)} ({[t['type'] for t in tasks]})")
    
    # Generate all possible sequences
    sequences = list(permutations(tasks))
    print(f"   🔄 Testing {len(sequences)} possible sequences...")
    
    best_result = None
    worst_result = None
    results = []
    
    for seq_num, sequence in enumerate(sequences):
        total_time, total_distance, details = calculate_sequence_time_osrm(
            sequence, vehicle, depot_location, with_breaks
        )
        
        # Count violations
        violation_count = sum(1 for d in details if d['time_window_violation'] > 0)
        latest_arrival = max((d['arrival_time'] for d in details), default=0)
        
        result = SequenceResult(
            sequence=list(sequence),
            total_time=total_time,
            total_distance=total_distance,
            vehicle_type=vehicle.id,
            with_breaks=with_breaks,
            violation_count=violation_count,
            latest_arrival=latest_arrival,
            details=details
        )
        
        results.append(result)
        
        # Track best/worst
        if best_result is None or total_time < best_result.total_time:
            best_result = result
        if worst_result is None or total_time > worst_result.total_time:
            worst_result = result
    
    print(f"   ✅ Analysis complete: Best={_format_time_hhmm(best_result.total_time)}, Worst={_format_time_hhmm(worst_result.total_time)}")
    
    return OrderAnalysis(
        order_id=order.id,
        task_count=len(tasks),
        best_sequence=best_result,
        worst_sequence=worst_result
    )


def print_sequence_details(result: SequenceResult):
    """Print detailed breakdown of a sequence"""
    print(f"\n📊 Sequence Details ({result.vehicle_type}, {'with' if result.with_breaks else 'without'} breaks)")
    print(f"   Total Time: {_format_time_hhmm(result.total_time)}")
    print(f"   Total Distance: {result.total_distance:.1f}km")
    print(f"   Time Window Violations: {result.violation_count}")
    print(f"   Latest Arrival: {_format_time_hhmm(result.latest_arrival % 1440)} ({_format_date_from_minutes(result.latest_arrival)})")
    
    print(f"\n   Task Sequence:")
    current_time = 480.0  # 8:00 AM start
    
    for i, detail in enumerate(result.details):
        arrival_str = _format_time_hhmm(detail['arrival_time'] % 1440)
        departure_str = _format_time_hhmm(detail['departure_time'] % 1440)
        
        violation_str = ""
        if detail['time_window_violation'] > 0:
            violation_str = f" ⚠️ LATE by {_format_time_hhmm(detail['time_window_violation'])}"
        
        print(f"   {i+1:2d}. {detail['task_id']:20s} | {detail['location']:20s} | Arrive: {arrival_str} | Depart: {departure_str}{violation_str}")
    
    print()


def main():
    """Main test function"""
    print("🎯 TARGETED SEQUENCE TEST - Orders 5, 6, 8")
    print("="*80)
    
    # Load scenario
    try:
        scenario_file = "../src/furgoni_con_prova.xlsx"
        orders, vehicles, depots = create_scenario_from_excel(scenario_file)
        print(f"✅ Loaded scenario: {len(orders)} orders, {len(vehicles)} vehicles")
    except Exception as e:
        print(f"❌ Failed to load scenario: {e}")
        return
    
    # Get depot location
    depot_location = depots[0] if depots else None
    if not depot_location:
        # Create a dummy depot location (ASTI coordinates)
        depot_location = type('Location', (), {
            'name': 'DEPOT-ASTI',
            'latitude': 44.9009,
            'longitude': 8.2065,
            'address': 'Asti Depot'
        })()
        print("⚠️ No depot found, using default ASTI coordinates")
    
    # Get test vehicles
    test_vehicles = get_test_vehicles()
    print(f"🚛 Test vehicles: {[v.id for v in test_vehicles]}")
    
    # Find target orders
    print(f"📋 Available orders: {[order.id for order in orders[:10]]}...")  # Show first 10
    
    target_orders = []
    for order in orders:
        if str(order.id) in ['5', '6', '8']:
            target_orders.append(order)
    
    if not target_orders:
        print(f"❌ Target orders 5, 6, 8 not found. Available orders: {[order.id for order in orders]}")
        return
    
    print(f"🎯 Target orders found: {[o.id for o in target_orders]}")
    
    # Run analysis for each combination
    all_results = {}
    
    for order in target_orders:
        print(f"\n{'='*60}")
        print(f"ORDER {order.id} ANALYSIS")
        print(f"{'='*60}")
        
        order_results = {}
        
        for vehicle in test_vehicles:
            for with_breaks in [True, False]:
                key = f"{vehicle.id}_{'breaks' if with_breaks else 'no_breaks'}"
                
                analysis = analyze_order_sequences(order, vehicle, depot_location, with_breaks)
                if analysis:
                    order_results[key] = analysis
                    
                    # Print summary
                    print(f"\n📈 SUMMARY for {vehicle.id} ({'with' if with_breaks else 'without'} breaks):")
                    print(f"   Best sequence: {_format_time_hhmm(analysis.best_sequence.total_time)} ({analysis.best_sequence.violation_count} violations)")
                    print(f"   Worst sequence: {_format_time_hhmm(analysis.worst_sequence.total_time)} ({analysis.worst_sequence.violation_count} violations)")
        
        all_results[order.id] = order_results
        
        # Print best sequence details for this order
        if order_results:
            best_overall = min(order_results.values(), key=lambda x: x.best_sequence.total_time)
            print(f"\n🏆 BEST OVERALL SEQUENCE for Order {order.id}:")
            print_sequence_details(best_overall.best_sequence)
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON - GOOGLE MAPS vs ALGORITHM")
    print(f"{'='*80}")
    
    google_times = {'5': "3:41", '6': "4:26", '8': "4:48"}
    
    for order_id in ['5', '6', '8']:
        if order_id in all_results:
            print(f"\nOrder {order_id}:")
            print(f"   🗺️ Google Maps: {google_times[order_id]} (314-430km)")
            
            order_data = all_results[order_id]
            for key, analysis in order_data.items():
                vehicle_type = "Small" if "SMALL" in key else "Large"
                breaks_str = "with breaks" if "breaks" in key and "no_breaks" not in key else "no breaks"
                
                best_time = _format_time_hhmm(analysis.best_sequence.total_time)
                violations = analysis.best_sequence.violation_count
                violation_str = f" ({violations} violations)" if violations > 0 else " (feasible)"
                
                print(f"   🤖 Algorithm ({vehicle_type}, {breaks_str}): {best_time}{violation_str}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
