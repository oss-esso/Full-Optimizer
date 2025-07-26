"""
Focused Debugging Test for Order Assignment Failures

This script implements TODO2.md Step 19.3 - a focused debugging test that:
1. Loads the full scenario (furgoni.xlsx)
2. Identifies specific unassigned orders from comprehensive test output
3. Runs simplified heuristic attempts to insert only specific orders
4. Uses detailed logging to trace exactly why insertion fails

This helps isolate the exact constraint or logic causing assignment issues.
"""

import sys
import os

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')
utils_dir = os.path.join(heuristic_root, 'utils')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)
sys.path.insert(0, utils_dir)

from scenario_creator import create_scenario_from_excel
from second_level import l2_heuristic, is_feasible
from epdt_data_structures import Route

print("=== FOCUSED ORDER ASSIGNMENT DEBUGGING ===")

def debug_single_order_assignment():
    """
    Debug why specific orders fail to be assigned by testing them against all vehicles.
    """
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Loading scenario from: {excel_file}")
    
    try:
        orders, vehicles = create_scenario_from_excel(excel_file)
        print(f"✅ Loaded {len(orders)} orders and {len(vehicles)} vehicles")
        
        # Known unassigned orders from comprehensive integration test output
        # These are orders that consistently fail to be assigned
        failing_order_ids = [
            "ORDER_FRADDY_SPA_MILANO_27",
            "ORDER_IL_PIZZICAGNOLO_10", 
            "ORDER_FRADDY_SPA_CHIAVARI_28",
            "ORDER_COLUMBIA__SERRAVALLE_0",
            "ORDER_FIAT_V._I_SPA_STAB_B_21",
            "ORDER_FRANCO_RIBALTA_NOSTR_31"
        ]
        
        print(f"\n🎯 DEBUGGING {len(failing_order_ids)} CONSISTENTLY FAILING ORDERS:")
        
        for order_idx, order_id in enumerate(failing_order_ids[:3]):  # Debug first 3 for detailed analysis
            # Find the order
            target_order = None
            for order in orders:
                if order.id == order_id:
                    target_order = order
                    break
            
            if not target_order:
                print(f"\n❌ Order {order_id} not found in scenario")
                continue
            
            print(f"\n{'='*100}")
            print(f"🔍 DEBUGGING ORDER {order_idx + 1}: {order_id}")
            print(f"{'='*100}")
            
            # Analyze order characteristics
            print(f"📊 ORDER PROFILE:")
            print(f"   • Pickup tasks: {len(target_order.pickup_tasks)}")
            print(f"   • Delivery tasks: {len(target_order.delivery_tasks)}")
            
            total_weight = 0.0
            total_volume = 0.0
            total_pallets = 0
            
            all_tasks = target_order.pickup_tasks + target_order.delivery_tasks
            print(f"   • Total tasks: {len(all_tasks)}")
            
            for i, task in enumerate(all_tasks):
                print(f"     Task {i+1}: {task.task_type.value}")
                print(f"       - Weight: {task.demand:.1f} kg")
                print(f"       - Volume: {task.volume:.2f} m³")
                print(f"       - Pallets: {task.pallets}")
                print(f"       - Location: {task.location_id}")
                print(f"       - Coordinates: ({task.lat:.4f}, {task.lon:.4f})")
                
                total_weight += task.demand
                total_volume += task.volume
                total_pallets += task.pallets
            
            print(f"   • Net weight: {total_weight:.1f} kg")
            print(f"   • Net volume: {total_volume:.2f} m³")
            print(f"   • Net pallets: {total_pallets}")
            
            # Test insertion with multiple vehicles
            print(f"\n🧪 TESTING INSERTION WITH ALL VEHICLES:")
            
            success_count = 0
            test_results = []
            
            for vehicle_idx, vehicle in enumerate(vehicles[:15]):  # Test first 15 vehicles for performance
                print(f"\n   Vehicle {vehicle_idx + 1}: {vehicle.id}")
                print(f"      Capacity - Weight: {vehicle.weight_capacity:.1f} kg")
                print(f"      Capacity - Volume: {vehicle.volume_capacity:.2f} m³")
                print(f"      Capacity - Pallets: {vehicle.pallet_capacity}")
                
                # Basic capacity pre-check
                weight_fits = vehicle.weight_capacity >= abs(total_weight)
                volume_fits = vehicle.volume_capacity >= abs(total_volume)
                pallets_fit = vehicle.pallet_capacity >= abs(total_pallets) if vehicle.pallet_capacity else True
                
                print(f"      Pre-check - Weight: {'✅' if weight_fits else '❌'}")
                print(f"      Pre-check - Volume: {'✅' if volume_fits else '❌'}")
                print(f"      Pre-check - Pallets: {'✅' if pallets_fit else '❌'}")
                
                if not (weight_fits and volume_fits and pallets_fit):
                    print(f"      ⚠️  Pre-check failed - skipping detailed test")
                    test_results.append((vehicle.id, False, "Pre-check failed: insufficient capacity"))
                    continue
                
                # Create empty route for this vehicle
                empty_route = Route(vehicle=vehicle, tasks=[])
                
                # Test L2 heuristic insertion
                print(f"      🔍 Testing L2 insertion...")
                try:
                    result_route = l2_heuristic(empty_route, target_order, debug_assignment=True)
                    
                    if result_route:
                        print(f"      ✅ L2 insertion: SUCCESS")
                        
                        # Test final route feasibility with detailed reason
                        print(f"      🔍 Testing route feasibility...")
                        feasible, reason = is_feasible(result_route, debug_feasibility=True, return_reason=True)
                        
                        if feasible:
                            print(f"      ✅ Route feasibility: SUCCESS")
                            success_count += 1
                            test_results.append((vehicle.id, True, "Successfully assigned"))
                            
                            # Print final route structure
                            print(f"      📋 Final Route Structure ({len(result_route.tasks)} tasks):")
                            for j, task in enumerate(result_route.tasks):
                                task_location = "DEPOT" if "DEPOT" in task.location_id else "CUSTOMER"
                                print(f"         {j+1}. {task.task_type.value} at {task_location} ({task.demand:.1f}kg, {task.volume:.2f}m³)")
                            
                            break  # Stop testing once we find a working vehicle
                        else:
                            print(f"      ❌ Route feasibility: FAILED")
                            print(f"         Reason: {reason}")
                            test_results.append((vehicle.id, False, f"Feasibility failed: {reason}"))
                    else:
                        print(f"      ❌ L2 insertion: FAILED")
                        test_results.append((vehicle.id, False, "L2 insertion failed"))
                        
                except Exception as e:
                    print(f"      💥 Exception during insertion: {e}")
                    test_results.append((vehicle.id, False, f"Exception: {str(e)}"))
            
            # Summary for this order
            print(f"\n📊 SUMMARY FOR {order_id}:")
            print(f"   • Vehicles tested: {len(test_results)}")
            print(f"   • Successful assignments: {success_count}")
            print(f"   • Success rate: {(success_count/len(test_results)*100):.1f}%")
            
            if success_count == 0:
                print(f"   ❌ CRITICAL: No vehicles can handle this order!")
                print(f"   🔍 Common failure reasons:")
                
                failure_reasons = {}
                for _, success, reason in test_results:
                    if not success:
                        failure_type = reason.split(':')[0] if ':' in reason else reason
                        failure_reasons[failure_type] = failure_reasons.get(failure_type, 0) + 1
                
                for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                    print(f"      • {reason}: {count} vehicles")
            
            print(f"\n" + "-"*100)
        
        if len(failing_order_ids) > 3:
            print(f"\n📝 Note: Analyzed first 3 orders. Run with more orders by modifying the slice [:3]")
        
        print(f"\n🎯 ANALYSIS COMPLETE")
        print(f"This detailed debugging helps identify the exact constraints causing assignment failures.")
        print(f"Use the failure reasons to guide constraint refinements in algo/second_level.py.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_single_order_assignment()
