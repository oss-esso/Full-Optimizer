#!/usr/bin/env python3
"""
Targeted Sequence Test for Orders 5, 6, and 8

This test         # Create location objec        # Create location object with correct attributes for OSRM
        location = type('Location', (), {
            'name': f"Delivery_{delivery.id}",
            'latitude': delivery.lat,
            'longitude': delivery.lon,
            'lat': delivery.lat,  # OSRM needs this
            'lng': delivery.lon,  # OSRM needs this
            'address': delivery.location_id
        })()rrect attributes for OSRM
        location = type('Location', (), {
            'name': f"Pickup_{pickup.id}",
            'latitude': pickup.lat,
            'longitude': pickup.lon,
            'lat': pickup.lat,  # OSRM needs this
            'lng': pickup.lon,  # OSRM needs this
            'address': pickup.location_id
        })()all possible route sequences for the problematic orders
using OSRM routing with both small and large vehicles, with and without HoS breaks.
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
    total_distance: float
    with_breaks_time: float
    without_breaks_time: float
    feasible_with_breaks: bool
    feasible_without_breaks: bool
    time_window_violations: List[str]
    latest_arrival: float
    sequence_details: List[Dict]


@dataclass
class VehicleTestResult:
    """Results for a vehicle type"""
    vehicle_id: str
    vehicle_type: str
    best_sequence: Optional[SequenceResult]
    worst_sequence: Optional[SequenceResult]
    total_sequences: int
    feasible_sequences: int


def get_test_vehicles():
    """Get representative small and large vehicles for testing - ONLY 2 configurations"""
    # Small vehicle (Furgone) - NO mandatory breaks
    small_vehicle = type('Vehicle', (), {
        'id': 'TEST_FURGONE_GW895CW',
        'vehicle_type': 'furgone',
        'weight_capacity': 1500,
        'volume_capacity': 12.0,
        'pallet_capacity': 8,
        'average_speed': 90,  # Fast furgone
        'capabilities': 'STANDARD',
        'mandatory_breaks': False
    })()
    
    # Large vehicle (Heavy truck) - MANDATORY breaks
    large_vehicle = type('Vehicle', (), {
        'id': 'TEST_HEAVY_XA359KW', 
        'vehicle_type': 'heavy',
        'weight_capacity': 2800,
        'volume_capacity': 25.0,
        'pallet_capacity': 21,
        'average_speed': 70,  # Slower heavy truck
        'capabilities': 'HEAVY,LOADER',
        'mandatory_breaks': True
    })()
    
    return [small_vehicle, large_vehicle]


def extract_order_tasks(order):
    """Extract all tasks for an order"""
    tasks = []
    
    # Add pickup tasks
    for pickup in order.pickup_tasks:
        # Create location object with correct attributes for OSRM
        location = type('Location', (), {
            'name': f"Pickup_{pickup.id}",
            'latitude': pickup.lat,
            'longitude': pickup.lon,
            'lat': pickup.lat,  # OSRM needs this
            'lng': pickup.lon,  # OSRM needs this
            'address': pickup.location_id
        })()
        
        tasks.append({
            'id': f"PICKUP_{order.id}_{pickup.id}",
            'type': 'pickup',
            'order_id': order.id,
            'task': pickup,
            'location': location,
            'earliest_time': getattr(pickup, 'earliest_time', None),
            'latest_time': getattr(pickup, 'latest_time', None),
            'service_time': pickup.service_time,
        })
    
    # Add delivery tasks  
    for delivery in order.delivery_tasks:
        # Create location object with correct attributes for OSRM
        location = type('Location', (), {
            'name': f"Delivery_{delivery.id}",
            'latitude': delivery.lat,
            'longitude': delivery.lon,
            'lat': delivery.lat,  # OSRM needs this
            'lng': delivery.lon,  # OSRM needs this
            'address': delivery.location_id
        })()
        
        tasks.append({
            'id': f"DELIVERY_{order.id}_{delivery.id}",
            'type': 'delivery',
            'order_id': order.id,
            'task': delivery,
            'location': location,
            'earliest_time': getattr(delivery, 'earliest_time', None),
            'latest_time': getattr(delivery, 'latest_time', None),
            'service_time': delivery.service_time,
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
            # Create proper task objects for OSRM with all required attributes
            current_task = type('Task', (), {
                'location': current_location,
                'lat': current_location.lat,
                'lon': current_location.lng
            })()
            dest_task = type('Task', (), {
                'location': task['location'],
                'lat': task['location'].lat,
                'lon': task['location'].lng
            })()
            
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
                break_time = 45.0  # 45 min break
                driving_time_since_break = 0.0
        
        # Update time and position
        current_time += travel_time + break_time
        current_location = task['location']
        
        # Check time window arrival
        earliest = task['earliest_time']
        latest = task['latest_time']
        
        # Wait if arriving early
        waiting_time = 0.0
        if earliest and current_time < earliest:
            waiting_time = earliest - current_time
            current_time = earliest
        
        # Add service time
        service_time = task['service_time']
        current_time += service_time
        
        # Record details
        detail = {
            'task_id': task['id'],
            'travel_time': travel_time,
            'travel_distance': travel_distance,
            'break_time': break_time,
            'waiting_time': waiting_time,
            'service_time': service_time,
            'arrival_time': current_time - service_time,
            'completion_time': current_time,
            'earliest_allowed': earliest,
            'latest_allowed': latest,
            'on_time': latest is None or (current_time - service_time) <= latest
        }
        sequence_details.append(detail)
        
        total_time += travel_time + break_time + waiting_time + service_time
        total_distance += travel_distance
    
    # Return to depot
    try:
        # Create task for current location with OSRM attributes
        current_task = type('Task', (), {
            'location': current_location,
            'lat': current_location.lat,
            'lon': current_location.lng
        })()
        depot_task = type('Task', (), {
            'location': depot_location,
            'lat': depot_location.lat,
            'lon': depot_location.lng
        })()
        
        return_time = calculate_travel_time_between_tasks(current_task, depot_task, vehicle)
        return_distance = 25.0  # Estimated
    except:
        return_time = 30.0
        return_distance = 25.0
    
    # Apply final break if needed
    final_break = 0.0
    if with_breaks and return_time > 0:
        driving_time_since_break += return_time
        if driving_time_since_break >= 270:
            final_break = 45.0
    
    total_time += return_time + final_break
    total_distance += return_distance
    
    return total_time, total_distance, sequence_details


def analyze_order_sequences(order, vehicle, depot_location):
    """Analyze all possible sequences for an order with a specific vehicle"""
    tasks = extract_order_tasks(order)
    if not tasks:
        return None
    
    print(f"   📋 Tasks: {len(tasks)} ({[t['type'] for t in tasks]})")
    
    # Vehicle configuration
    has_breaks = getattr(vehicle, 'mandatory_breaks', False)
    
    # Generate all possible sequences
    sequences = list(permutations(tasks))
    print(f"   🔄 Testing {len(sequences)} sequences...")
    
    best_time = float('inf')
    best_sequence = None
    worst_time = 0
    worst_sequence = None
    feasible_count = 0
    
    for sequence in sequences:
        total_time, total_distance, details = calculate_sequence_time_osrm(sequence, vehicle, depot_location, has_breaks)
        
        # Check feasibility (time windows)
        violations = 0
        for detail in details:
            if not detail['on_time']:
                violations += 1
        
        if violations == 0:
            feasible_count += 1
        
        # Track best/worst
        if total_time < best_time:
            best_time = total_time
            best_sequence = SequenceResult(
                sequence=[t['id'] for t in sequence],
                total_time=total_time,
                total_distance=total_distance,
                with_breaks_time=total_time,
                without_breaks_time=total_time,
                feasible_with_breaks=(violations == 0),
                feasible_without_breaks=(violations == 0),
                time_window_violations=[],
                latest_arrival=max((d['arrival_time'] for d in details), default=0),
                sequence_details=details
            )
        
        if total_time > worst_time:
            worst_time = total_time
            worst_sequence = SequenceResult(
                sequence=[t['id'] for t in sequence],
                total_time=total_time,
                total_distance=total_distance,
                with_breaks_time=total_time,
                without_breaks_time=total_time,
                feasible_with_breaks=(violations == 0),
                feasible_without_breaks=(violations == 0),
                time_window_violations=[],
                latest_arrival=max((d['arrival_time'] for d in details), default=0),
                sequence_details=details
            )
    
    vehicle_type = "Small (no breaks)" if not has_breaks else "Large (with breaks)"
    result = VehicleTestResult(
        vehicle_id=vehicle.id,
        vehicle_type=vehicle_type,
        best_sequence=best_sequence,
        worst_sequence=worst_sequence,
        total_sequences=len(sequences),
        feasible_sequences=feasible_count
    )
    
    print(f"   ✅ Best: {_format_time_hhmm(best_time)}, Feasible: {feasible_count}/{len(sequences)}")
    
    return result


def test_all_sequences_for_order(order, vehicles, depot_location):
    """Test all possible sequences for an order with different vehicles"""
    print(f"\n{'='*80}")
    print(f"TESTING ORDER {order.id} - ALL POSSIBLE SEQUENCES")
    print(f"{'='*80}")
    
    # Get order requirements
    weight, volume, pallets = get_order_requirements(order)
    print(f"Order Requirements: {weight:.1f}kg, {volume:.2f}m³, {pallets:.0f} pallets")
    
    # Extract tasks
    tasks = extract_order_tasks(order)
    print(f"Total Tasks: {len(tasks)} ({len(order.pickup_tasks)} pickups, {len(order.delivery_tasks)} deliveries)")
    
    # Generate all possible sequences (pickup before delivery constraint)
    valid_sequences = []
    
    # For orders with multiple pickups/deliveries, we need to ensure pickups come before deliveries
    pickup_tasks = [t for t in tasks if t['type'] == 'pickup']
    delivery_tasks = [t for t in tasks if t['type'] == 'delivery']
    
    # Generate all pickup permutations
    for pickup_perm in permutations(pickup_tasks):
        # Generate all delivery permutations
        for delivery_perm in permutations(delivery_tasks):
            # Combine: all pickups first, then all deliveries
            full_sequence = list(pickup_perm) + list(delivery_perm)
            valid_sequences.append(full_sequence)
            
            # Also test interleaved sequences (pickup-delivery-pickup-delivery)
            if len(pickup_tasks) == len(delivery_tasks):
                interleaved = []
                for p, d in zip(pickup_perm, delivery_perm):
                    interleaved.extend([p, d])
                valid_sequences.append(interleaved)
    
    # Remove duplicates
    unique_sequences = []
    seen_sequences = set()
    for seq in valid_sequences:
        seq_signature = tuple(t['id'] for t in seq)
        if seq_signature not in seen_sequences:
            seen_sequences.add(seq_signature)
            unique_sequences.append(seq)
    
    print(f"Testing {len(unique_sequences)} unique valid sequences")
    
    # Test each vehicle
    vehicle_results = []
    
    for vehicle in vehicles:
        print(f"\n{'-'*60}")
        print(f"VEHICLE: {vehicle.id} ({vehicle.vehicle_type})")
        print(f"Capacity: {vehicle.weight_capacity}kg, {vehicle.volume_capacity}m³, {vehicle.pallet_capacity} pallets")
        print(f"Speed: {vehicle.average_speed}km/h")
        print(f"{'-'*60}")
        
        # Check basic capacity compatibility
        capacity_ok = (weight <= vehicle.weight_capacity and 
                      volume <= vehicle.volume_capacity and 
                      pallets <= vehicle.pallet_capacity)
        
        if not capacity_ok:
            print(f"❌ CAPACITY INCOMPATIBLE: Requires {weight:.1f}kg, {volume:.2f}m³, {pallets:.0f}pal")
            continue
        
        print(f"✅ CAPACITY COMPATIBLE")
        
        sequence_results = []
        
        # Test each sequence
        for seq_num, sequence in enumerate(unique_sequences, 1):
            sequence_name = " → ".join([t['id'].replace(f"_{order.id}_", "_") for t in sequence])
            
            # Test with breaks
            time_with_breaks, dist_with_breaks, details_with_breaks = calculate_sequence_time_osrm(
                sequence, vehicle, depot_location, with_breaks=True
            )
            
            # Test without breaks
            time_without_breaks, dist_without_breaks, details_without_breaks = calculate_sequence_time_osrm(
                sequence, vehicle, depot_location, with_breaks=False
            )
            
            # Check feasibility
            violations = []
            latest_arrival = 0.0
            
            for detail in details_with_breaks:
                if detail['latest_allowed'] and not detail['on_time']:
                    violation = f"{detail['task_id']}: arrives {detail['arrival_time']:.1f} > {detail['latest_allowed']:.1f}"
                    violations.append(violation)
                
                if detail['latest_allowed']:
                    latest_arrival = max(latest_arrival, detail['arrival_time'])
            
            feasible_with_breaks = len(violations) == 0
            feasible_without_breaks = True  # Check this separately if needed
            
            result = SequenceResult(
                sequence=[t['id'] for t in sequence],
                total_time=time_with_breaks,
                total_distance=dist_with_breaks,
                with_breaks_time=time_with_breaks,
                without_breaks_time=time_without_breaks,
                feasible_with_breaks=feasible_with_breaks,
                feasible_without_breaks=feasible_without_breaks,
                time_window_violations=violations,
                latest_arrival=latest_arrival,
                sequence_details=details_with_breaks
            )
            
            sequence_results.append(result)
            
            # Print sequence summary
            status = "✅ FEASIBLE" if feasible_with_breaks else "❌ VIOLATES"
            time_savings = time_with_breaks - time_without_breaks
            
            print(f"  {seq_num:2d}. {sequence_name}")
            print(f"      With breaks: {_format_time_hhmm(time_with_breaks)} | "
                  f"Without: {_format_time_hhmm(time_without_breaks)} | "
                  f"Savings: {_format_time_hhmm(time_savings)} | {status}")
            
            if violations:
                print(f"      Violations: {'; '.join(violations[:2])}...")
        
        # Find best and worst sequences
        feasible_sequences = [r for r in sequence_results if r.feasible_with_breaks]
        
        best_sequence = min(feasible_sequences, key=lambda x: x.total_time) if feasible_sequences else None
        worst_sequence = max(sequence_results, key=lambda x: x.total_time)
        
        vehicle_result = VehicleTestResult(
            vehicle_id=vehicle.id,
            vehicle_type=vehicle.vehicle_type,
            best_sequence=best_sequence,
            worst_sequence=worst_sequence,
            total_sequences=len(sequence_results),
            feasible_sequences=len(feasible_sequences)
        )
        
        vehicle_results.append(vehicle_result)
        
        # Print summary for this vehicle
        print(f"\n  SUMMARY for {vehicle.id}:")
        print(f"    Total sequences tested: {len(sequence_results)}")
        print(f"    Feasible sequences: {len(feasible_sequences)}")
        
        if best_sequence:
            print(f"    ✅ BEST: {_format_time_hhmm(best_sequence.total_time)} - {' → '.join(best_sequence.sequence)}")
        else:
            print(f"    ❌ NO FEASIBLE SEQUENCES FOUND")
            
        print(f"    ⚠️  WORST: {_format_time_hhmm(worst_sequence.total_time)} - {' → '.join(worst_sequence.sequence)}")
    
    return vehicle_results


def print_detailed_analysis(order_id, vehicle_results):
    """Print detailed analysis of the best sequences"""
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS - ORDER {order_id}")
    print(f"{'='*80}")
    
    for vehicle_result in vehicle_results:
        if not vehicle_result.best_sequence:
            continue
            
        print(f"\n🚀 OPTIMAL SEQUENCE for {vehicle_result.vehicle_id} ({vehicle_result.vehicle_type}):")
        print(f"   Total Time: {_format_time_hhmm(vehicle_result.best_sequence.total_time)}")
        print(f"   Total Distance: {vehicle_result.best_sequence.total_distance:.1f}km")
        print(f"   Break Time Savings: {_format_time_hhmm(vehicle_result.best_sequence.with_breaks_time - vehicle_result.best_sequence.without_breaks_time)}")
        
        print(f"\n   DETAILED TIMELINE:")
        current_time = 480.0  # 8:00 AM start
        
        for detail in vehicle_result.best_sequence.sequence_details:
            current_time += detail['travel_time'] + detail['break_time'] + detail['waiting_time']
            
            time_str = _format_time_hhmm(current_time % 1440)
            status = "✅" if detail['on_time'] else "❌"
            
            print(f"     {detail['task_id']}: Arrive {time_str} | "
                  f"Travel: {_format_time_hhmm(detail['travel_time'])} | "
                  f"Service: {_format_time_hhmm(detail['service_time'])} | {status}")
            
            if detail['break_time'] > 0:
                print(f"       → Break: {_format_time_hhmm(detail['break_time'])}")
            if detail['waiting_time'] > 0:
                print(f"       → Wait: {_format_time_hhmm(detail['waiting_time'])}")
                
            current_time += detail['service_time']


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
        # Create a dummy depot location (ASTI coordinates) with proper attributes
        depot_location = type('Location', (), {
            'name': 'DEPOT-ASTI',
            'latitude': 44.9009,
            'longitude': 8.2065,
            'lat': 44.9009,  # OSRM needs this
            'lng': 8.2065,   # OSRM needs this
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
    
    # Run analysis for each order with ONLY 2 vehicle configurations
    all_results = {}
    
    for order in target_orders:
        print(f"\n{'='*60}")
        print(f"ORDER {order.id} ANALYSIS")
        print(f"{'='*60}")
        
        order_results = []
        
        for vehicle in test_vehicles:
            vehicle_type = "Small (no breaks)" if not getattr(vehicle, 'mandatory_breaks', False) else "Large (with breaks)"
            print(f"\n🔍 Testing {vehicle.id} ({vehicle_type})")
            
            analysis = analyze_order_sequences(order, vehicle, depot_location)
            if analysis:
                order_results.append(analysis)
        
        all_results[order.id] = order_results
        
        # Print summary for this order
        if order_results:
            best_overall = min(order_results, key=lambda x: x.best_sequence.total_time if x.best_sequence else float('inf'))
            print(f"\n🏆 BEST for Order {order.id}: {best_overall.vehicle_id} - {_format_time_hhmm(best_overall.best_sequence.total_time)}")
        print(f"\n🔄 Testing Order {order.id}...")
        vehicle_results = test_all_sequences_for_order(order, test_vehicles, depot_location)
        all_results[order.id] = vehicle_results
        
        # Print detailed analysis
        print_detailed_analysis(order.id, vehicle_results)
    
    # Print final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON - ALL ORDERS")
    print(f"{'='*80}")
    
    for order_id, vehicle_results in all_results.items():
        print(f"\nOrder {order_id}:")
        for vehicle_result in vehicle_results:
            if vehicle_result.best_sequence:
                print(f"  {vehicle_result.vehicle_id} ({vehicle_result.vehicle_type}): "
                      f"{_format_time_hhmm(vehicle_result.best_sequence.total_time)} ✅")
            else:
                print(f"  {vehicle_result.vehicle_id} ({vehicle_result.vehicle_type}): NO SOLUTION ❌")
    
    print(f"\n🎯 Test completed! Check results above for optimal sequences.")


if __name__ == "__main__":
    main()
