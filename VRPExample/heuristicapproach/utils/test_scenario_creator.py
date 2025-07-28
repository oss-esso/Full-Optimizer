"""
Test version of scenario creator without geocoding for quick testing
"""
import pandas as pd
import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple

# Add the algo directory to the path
current_dir = Path(__file__).parent
algo_dir = current_dir.parent / "algo"
sys.path.append(str(algo_dir))

from epdt_data_structures import Order, Task, Vehicle, TaskType

def create_test_scenario(file_path: str) -> Tuple[List[Order], List[Vehicle]]:
    """Test version without geocoding"""
    print(f"Reading scenario from {file_path}...")
    
    # Read Excel sheets
    consegne_df = pd.read_excel(file_path, sheet_name="CONSEGNE")
    veicoli_df = pd.read_excel(file_path, sheet_name="VEICOLI")
    
    print(f"Loaded {len(consegne_df)} delivery locations and {len(veicoli_df)} vehicles")
    
    # Create vehicles
    vehicles = []
    for idx, row in veicoli_df.iterrows():
        try:
            vehicle_id = str(row['NUMBER PLATE']).strip()
            weight_capacity = float(row.get('MAX LOAD KG', 3500.0))
            volume_capacity = float(row.get('MAX LOAD VOLUME M^3', 25.0))
            
            vehicle_type_str = str(row.get('TYPE OF VEHICLE', 'FURGONE')).strip().upper()
            vehicle_type = 'heavy' if vehicle_type_str == 'CAMION' else 'standard'
            
            lifo_str = str(row.get('LAST IN FIRST OUT', 'NO')).strip().upper()
            lifo_required = lifo_str in ['YES', 'SI', 'TRUE']
            
            vehicle = Vehicle(
                id=vehicle_id,
                depot_id='DEPOT-ASTI',
                weight_capacity=weight_capacity,
                volume_capacity=volume_capacity,
                cost_per_km=1.5 if vehicle_type == 'heavy' else 1.0,
                fixed_cost=100.0 if vehicle_type == 'heavy' else 50.0,
                vehicle_type=vehicle_type,
                lifo_required=lifo_required
            )
            vehicles.append(vehicle)
            print(f"Created vehicle: {vehicle_id} ({vehicle_type}) - {weight_capacity}kg/{volume_capacity:.1f}m³")
            
        except Exception as e:
            print(f"Error creating vehicle from row {idx}: {e}")
    
    # Create orders (using fake coordinates for testing)
    orders = []
    for idx, row in consegne_df.iterrows():
        try:
            company_name = str(row['NAME']).strip()
            task_id = f"TASK_{company_name.replace(' ', '_')[:20]}_{idx}"
            order_id = f"ORDER_{company_name.replace(' ', '_')[:20]}_{idx}"
            
            # Parse task type
            task_type_str = str(row.get('DELIVERY OR PICKUP', 'DELIVERY')).strip().upper()
            task_type = TaskType.PICKUP if 'PICKUP' in task_type_str else TaskType.DELIVERY
            
            load_kg = float(row.get('LOAD KG', 0.0)) if pd.notna(row.get('LOAD KG')) else 0.0
            load_volume = float(row.get('LOAD VOLUME M^3', 0.0)) if pd.notna(row.get('LOAD VOLUME M^3')) else 0.0
            service_time = float(row.get('SERVICE TIME', 15.0)) if pd.notna(row.get('SERVICE TIME')) else 15.0
            
            # Set demand (positive for pickup, negative for delivery)
            demand = load_kg if task_type == TaskType.PICKUP else -load_kg
            volume = load_volume if task_type == TaskType.PICKUP else -load_volume
            
            # Use fake coordinates for testing (Padova area)
            lat = 45.4064 + (idx * 0.01)  # Spread around Padova
            lon = 11.8768 + (idx * 0.01)
            
            address = str(row['ADDRESS']).strip() if pd.notna(row['ADDRESS']) else "Unknown Address"
            
            task = Task(
                id=task_id,
                location_id=address,
                task_type=task_type,
                order_id=order_id,
                lat=lat,
                lon=lon,
                service_time=service_time,
                demand=demand,
                volume=volume,
                priority=1
            )
            
            # Create single-task order
            if task.is_pickup():
                pickup_tasks = [task]
                delivery_tasks = []
            else:
                pickup_tasks = []
                delivery_tasks = [task]
            
            order = Order(
                id=order_id,
                pickup_tasks=pickup_tasks,
                delivery_tasks=delivery_tasks,
                priority=1,
                is_mandatory=True
            )
            
            orders.append(order)
            print(f"Created order: {company_name[:30]} ({task_type.value}) - {abs(load_kg)}kg/{abs(load_volume):.1f}m³")
            
        except Exception as e:
            print(f"Error creating order from row {idx}: {e}")
    
    print(f"\nScenario created: {len(orders)} orders, {len(vehicles)} vehicles")
    return orders, vehicles

def main():
    file_path = os.path.join('..', 'src', 'furgoni.xlsx')
    
    try:
        orders, vehicles = create_test_scenario(file_path)
        
        # Basic validation
        total_demand = sum(abs(task.demand) for order in orders for task in order.get_all_tasks())
        total_capacity = sum(vehicle.weight_capacity for vehicle in vehicles)
        
        print(f"\nValidation:")
        print(f"Total demand: {total_demand:.1f} kg")
        print(f"Total vehicle capacity: {total_capacity:.1f} kg")
        print(f"Capacity utilization: {(total_demand/total_capacity)*100:.1f}%")
        
        # Show some examples
        print(f"\nExample vehicles:")
        for i, vehicle in enumerate(vehicles[:3]):
            print(f"  {vehicle.id}: {vehicle.vehicle_type}, {vehicle.weight_capacity}kg, LIFO: {vehicle.lifo_required}")
        
        print(f"\nExample orders:")
        for i, order in enumerate(orders[:3]):
            task = order.get_all_tasks()[0]
            print(f"  {order.id}: {task.task_type.value}, {abs(task.demand)}kg at {task.location_id[:50]}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
