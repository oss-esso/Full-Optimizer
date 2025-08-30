#!/usr/bin/env python3
"""
Targeted Sequence Test for Orders 5, 6, and 8

This test analyzes all possible route sequences for the problematic orders
using OSRM routing with proper vehicle configurations:
- Small vehicles (furgoni): No mandatory breaks, faster speeds
- Large vehicles (heavy): Mandatory HoS breaks, slower speeds

Goal: Find optimal sequences and understand why current algorithm fails.
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
from algo.route_provider import get_route_provider, calculate_travel_time_between_tasks


@dataclass
class SequenceResult:
    """Results for a specific task sequence"""
    sequence: List[str]
    total_time: float
    vehicle_type: str
    has_breaks: bool
    time_window_violations: int
    latest_arrival: float
    sequence_details: List[Dict]
    osrm_used: bool


@dataclass
class VehicleTestResult:
    """Results for all sequences tested with one vehicle"""
    vehicle_id: str
    vehicle_type: str
    has_mandatory_breaks: bool
    best_sequence: Optional[SequenceResult]
    worst_sequence: Optional[SequenceResult]
    total_sequences_tested: int
    feasible_sequences: int


def get_test_vehicles():
    """Get properly configured small and large vehicles for testing"""
    
    # Small vehicle (Furgone) - NO mandatory breaks, fast speed
    small_vehicle = type('Vehicle', (), {
        'id': 'TEST_FURGONE_GW895CW',
        'vehicle_type': 'furgone',
        'weight_capacity': 1500,
        'volume_capacity': 12.0,
        'pallet_capacity': 8,
        'average_speed': 90,  # Fast furgone speed
        'capabilities': 'STANDARD',
        'mandatory_breaks': False,  # Small vehicles don't need mandatory breaks
        'max_driving_time': 600,  # 10 hours without mandatory breaks
    })()
    
    # Large vehicle (Heavy truck) - MANDATORY breaks, slower speed
    large_vehicle = type('Vehicle', (), {
        'id': 'TEST_HEAVY_XA359KW', 
        'vehicle_type': 'heavy',
        'weight_capacity': 2800,
        'volume_capacity': 25.0,
        'pallet_capacity': 21,
        'average_speed': 70,  # Slower heavy truck speed
        'capabilities': 'HEAVY,LOADER',
        'mandatory_breaks': True,  # Heavy vehicles MUST take breaks
        'max_driving_time': 270,  # 4.5 hours max continuous driving
    })()
    
    print(f"🚛 Vehicle configurations:")
    print(f"   Small: {small_vehicle.id} - {small_vehicle.average_speed}km/h, breaks={small_vehicle.mandatory_breaks}")
    print(f"   Large: {large_vehicle.id} - {large_vehicle.average_speed}km/h, breaks={large_vehicle.mandatory_breaks}")
    
    return [small_vehicle, large_vehicle]


def extract_order_tasks(order):
    """Extract all tasks for an order with proper location handling"""
    tasks = []
    
    print(f"      DEBUG: Extracting tasks for Order {order.id}")
    
    # Add pickup tasks
    for pickup in order.pickup_tasks:
        print(f"         Pickup task attributes: {dir(pickup)}")
        print(f"         Pickup task vars: {vars(pickup)}")
        
        # Handle location - get coordinates from the task
        location = getattr(pickup, 'location', None)
        print(f"         Pickup location: {location}")
        
        if location is None:
            # Use geocoded coordinates if available
            lat = getattr(pickup, 'latitude', None)
            lon = getattr(pickup, 'longitude', None)
            print(f"         Pickup lat/lon: {lat}, {lon}")
            
            if lat and lon:
                location = type('Location', (), {
                    'name': f"Pickup_{pickup.id}",
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'lat': float(lat),  # Add both attributes
                    'lng': float(lon),
                    'address': getattr(pickup, 'location_id', f"Pickup {pickup.id}")
                })()
            else:
                # Fallback with dummy coordinates near Asti
                base_lat = 44.9009
                base_lon = 8.2065
                hash_offset = (hash(str(pickup.id)) % 100) * 0.01
                location = type('Location', (), {
                    'name': f"Pickup_{pickup.id}",
                    'latitude': base_lat + hash_offset,
                    'longitude': base_lon + hash_offset, 
                    'lat': base_lat + hash_offset,  # Add both attributes
                    'lng': base_lon + hash_offset,
                    'address': getattr(pickup, 'location_id', f"Pickup {pickup.id}")
                })()
        else:
            # Make sure existing location has required attributes
            if not hasattr(location, 'lat'):
                location.lat = getattr(location, 'latitude', 44.9009)
            if not hasattr(location, 'lng'):
                location.lng = getattr(location, 'longitude', 8.2065)
        
        print(f"         Final pickup location: {vars(location)}")
        
        tasks.append({
            'id': f"PICKUP_{order.id}_{pickup.id}",
            'type': 'pickup',
            'order_id': order.id,
            'task': pickup,
            'location': location,
            'earliest_time': getattr(pickup, 'earliest_time', None),
            'latest_time': getattr(pickup, 'latest_time', None),
            'service_time': getattr(pickup, 'service_time', 15.0),  # Realistic pickup time
        })
        
        # Only debug first task to avoid spam
        break
    
    return tasks[:1]  # Return only first task for debugging


def calculate_sequence_time_with_proper_osrm(tasks, vehicle, depot_location):
    """Calculate total time for a task sequence using PROPER OSRM with vehicle-specific settings"""
    if not tasks:
        return 0.0, [], False
    
    total_time = 0.0
    sequence_details = []
    current_location = depot_location
    current_time = 480.0  # Start at 8:00 AM
    driving_time_since_break = 0.0
    osrm_used = False
    
    # Vehicle-specific settings
    has_mandatory_breaks = getattr(vehicle, 'mandatory_breaks', False)
    max_driving_time = getattr(vehicle, 'max_driving_time', 600)  # Default 10 hours
    
    print(f"      Vehicle {vehicle.id}: speed={vehicle.average_speed}km/h, breaks={has_mandatory_breaks}")
    
    # Process each task
    for i, task in enumerate(tasks):
        # Calculate travel time from current location using OSRM
        try:
            # Create proper task objects for OSRM
            current_task = type('Task', (), {'location': current_location})()
            dest_task = type('Task', (), {'location': task['location']})()
            
            # Use OSRM with vehicle-specific routing
            travel_time = calculate_travel_time_between_tasks(current_task, dest_task, vehicle)
            osrm_used = True
            
            print(f"         OSRM: {current_location.name} → {task['location'].name} = {travel_time:.1f} min")
            
        except Exception as e:
            print(f"         OSRM FAILED: {e}, using fallback")
            # Fallback calculation based on vehicle speed
            distance_km = 25.0  # Default distance
            speed_kmh = getattr(vehicle, 'average_speed', 80)
            travel_time = (distance_km / speed_kmh) * 60.0  # Convert to minutes
            osrm_used = False
        
        # Apply HoS breaks if this is a heavy vehicle with mandatory breaks
        break_time = 0.0
        if has_mandatory_breaks and travel_time > 0:
            driving_time_since_break += travel_time
            if driving_time_since_break >= max_driving_time:
                break_time = 45.0  # 45 minute mandatory break
                driving_time_since_break = 0.0
                print(f"         BREAK: 45 min mandatory break required")
        
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
            'location': task['location'].name,
            'osrm_travel_time': travel_time if osrm_used else None
        })
        
        # Update for next iteration
        current_time = departure_time
        current_location = task['location']
        total_time += travel_time + break_time + service_time
    
    # Return trip to depot
    try:
        current_task = type('Task', (), {'location': current_location})()
        depot_task = type('Task', (), {'location': depot_location})()
        return_travel = calculate_travel_time_between_tasks(current_task, depot_task, vehicle)
        print(f"         OSRM Return: {current_location.name} → {depot_location.name} = {return_travel:.1f} min")
    except Exception as e:
        print(f"         Return OSRM FAILED: {e}")
        distance_km = 25.0
        speed_kmh = getattr(vehicle, 'average_speed', 80)
        return_travel = (distance_km / speed_kmh) * 60.0
    
    total_time += return_travel
    
    return total_time, sequence_details, osrm_used


def analyze_order_sequences(order, vehicle, depot_location):
    """Analyze all possible sequences for an order with one vehicle configuration"""
    print(f"\n🔍 Analyzing Order {order.id} with {vehicle.id}")
    
    # Extract tasks
    tasks = extract_order_tasks(order)
    if not tasks:
        print(f"⚠️ No tasks found for Order {order.id}")
        return None
    
    print(f"   📋 Tasks: {len(tasks)} ({[t['type'] for t in tasks]})")
    
    # Vehicle configuration
    has_breaks = getattr(vehicle, 'mandatory_breaks', False)
    vehicle_type = "Heavy (with breaks)" if has_breaks else "Light (no breaks)"
    print(f"   🚛 Vehicle: {vehicle_type}, Speed: {vehicle.average_speed}km/h")
    
    # Generate all possible sequences
    sequences = list(permutations(tasks))
    print(f"   🔄 Testing {len(sequences)} possible sequences...")
    
    best_result = None
    worst_result = None
    feasible_count = 0
    
    for seq_num, sequence in enumerate(sequences, 1):
        print(f"      Sequence {seq_num}/{len(sequences)}: {' → '.join([t['type'][:1].upper() for t in sequence])}")
        
        total_time, details, osrm_used = calculate_sequence_time_with_proper_osrm(
            sequence, vehicle, depot_location
        )
        
        # Count violations
        violation_count = sum(1 for d in details if d['time_window_violation'] > 0)
        latest_arrival = max((d['arrival_time'] for d in details), default=0)
        
        if violation_count == 0:
            feasible_count += 1
        
        result = SequenceResult(
            sequence=[t['id'] for t in sequence],
            total_time=total_time,
            vehicle_type=vehicle.id,
            has_breaks=has_breaks,
            time_window_violations=violation_count,
            latest_arrival=latest_arrival,
            sequence_details=details,
            osrm_used=osrm_used
        )
        
        # Track best/worst
        if best_result is None or total_time < best_result.total_time:
            best_result = result
        if worst_result is None or total_time > worst_result.total_time:
            worst_result = result
        
        print(f"         Result: {_format_time_hhmm(total_time)}, {violation_count} violations, OSRM={osrm_used}")
    
    print(f"   ✅ Analysis complete: Best={_format_time_hhmm(best_result.total_time)}, Worst={_format_time_hhmm(worst_result.total_time)}")
    print(f"   📊 Feasible sequences: {feasible_count}/{len(sequences)} ({feasible_count/len(sequences)*100:.1f}%)")
    
    return VehicleTestResult(
        vehicle_id=vehicle.id,
        vehicle_type=vehicle_type,
        has_mandatory_breaks=has_breaks,
        best_sequence=best_result,
        worst_sequence=worst_result,
        total_sequences_tested=len(sequences),
        feasible_sequences=feasible_count
    )


def print_sequence_details(result: SequenceResult):
    """Print detailed breakdown of a sequence"""
    print(f"\n📊 Sequence Details ({result.vehicle_type}, {'with' if result.has_breaks else 'without'} mandatory breaks)")
    print(f"   Total Time: {_format_time_hhmm(result.total_time)}")
    print(f"   Time Window Violations: {result.time_window_violations}")
    print(f"   Latest Arrival: {_format_time_hhmm(result.latest_arrival % 1440)}")
    print(f"   OSRM Used: {'✅' if result.osrm_used else '❌ (fallback)'}")
    
    print(f"\n   Task Sequence:")
    
    for i, detail in enumerate(result.sequence_details):
        arrival_str = _format_time_hhmm(detail['arrival_time'] % 1440)
        departure_str = _format_time_hhmm(detail['departure_time'] % 1440)
        
        travel_info = ""
        if detail.get('osrm_travel_time'):
            travel_info = f" (OSRM: {detail['osrm_travel_time']:.1f}min)"
        
        violation_str = ""
        if detail['time_window_violation'] > 0:
            violation_str = f" ⚠️ LATE by {_format_time_hhmm(detail['time_window_violation'])}"
        
        break_str = ""
        if detail['break_time'] > 0:
            break_str = f" + {detail['break_time']:.0f}min break"
        
        print(f"   {i+1:2d}. {detail['task_id']:25s} | Arrive: {arrival_str} | Depart: {departure_str}{travel_info}{break_str}{violation_str}")
    
    print()


def main():
    """Main test function"""
    print("🎯 TARGETED SEQUENCE TEST - Orders 5, 6, 8")
    print("="*80)
    print("Testing with PROPER vehicle configurations:")
    print("• Small vehicles (furgoni): Fast speed, no mandatory breaks")
    print("• Large vehicles (heavy): Slower speed, mandatory HoS breaks")
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
    
    # Get properly configured test vehicles (only 2 configurations)
    test_vehicles = get_test_vehicles()
    
    # Find target orders
    print(f"\n📋 Available orders: {[order.id for order in orders[:10]]}...")
    
    target_orders = []
    for order in orders:
        if str(order.id) in ['5', '6', '8']:
            target_orders.append(order)
    
    if not target_orders:
        print(f"❌ Target orders 5, 6, 8 not found. Available orders: {[order.id for order in orders]}")
        return
    
    print(f"🎯 Target orders found: {[o.id for o in target_orders]}")
    
    # Run analysis for each combination (only 2 vehicle types now)
    all_results = {}
    
    for order in target_orders:
        print(f"\n{'='*60}")
        print(f"ORDER {order.id} ANALYSIS")
        print(f"{'='*60}")
        
        order_results = {}
        
        for vehicle in test_vehicles:
            analysis = analyze_order_sequences(order, vehicle, depot_location)
            if analysis:
                order_results[vehicle.id] = analysis
                
                # Print summary
                print(f"\n📈 SUMMARY for {analysis.vehicle_type}:")
                print(f"   Best sequence: {_format_time_hhmm(analysis.best_sequence.total_time)} ({analysis.best_sequence.time_window_violations} violations)")
                print(f"   Worst sequence: {_format_time_hhmm(analysis.worst_sequence.total_time)} ({analysis.worst_sequence.time_window_violations} violations)")
                print(f"   Feasible sequences: {analysis.feasible_sequences}/{analysis.total_sequences_tested}")
        
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
            for vehicle_id, analysis in order_data.items():
                best_time = _format_time_hhmm(analysis.best_sequence.total_time)
                violations = analysis.best_sequence.time_window_violations
                violation_str = f" ({violations} violations)" if violations > 0 else " (feasible)"
                osrm_str = "✅ OSRM" if analysis.best_sequence.osrm_used else "❌ Fallback"
                
                print(f"   🤖 Algorithm ({analysis.vehicle_type}): {best_time}{violation_str} [{osrm_str}]")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
