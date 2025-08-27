"""
Debug Vehicle Speed and HoS Investigation
Test to verify that vehicle types are properly configured with correct speeds and HoS rules.
"""

import sys
import os

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
src_dir = os.path.join(heuristic_root, 'src')
algo_dir = os.path.join(heuristic_root, 'algo')
utils_dir = os.path.join(heuristic_root, 'utils')

sys.path.insert(0, heuristic_root)
sys.path.insert(0, src_dir)
sys.path.insert(0, algo_dir)
sys.path.insert(0, utils_dir)

def debug_vehicle_configuration():
    """Debug the vehicle configuration for speed and HoS rules."""
    print("DEBUGGING VEHICLE CONFIGURATION")
    print("="*50)
    
    try:
        from scenario_creator import create_scenario_from_excel
        
        # Load the scenario to get actual vehicle objects
        print("Loading scenario...")
        orders, vehicles, depot = create_scenario_from_excel(
            "d:/Projects/OQI_Project/Full Optimizer/VRPExample/heuristicapproach/src/furgoni3_2.xlsx"
        )
        
        print(f"Loaded {len(vehicles)} vehicles")
        
        # Analyze vehicle configuration
        furgoni = []
        camion = []
        unknown = []
        
        for vehicle in vehicles[:10]:  # Check first 10 vehicles
            vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
            avg_speed = getattr(vehicle, 'average_speed', 'not set')
            
            print(f"\nVehicle: {vehicle.id}")
            print(f"  Type: {vehicle_type}")
            print(f"  Average Speed: {avg_speed}")
            print(f"  Weight Capacity: {getattr(vehicle, 'weight_capacity', 'unknown')}")
            print(f"  Pallet Capacity: {getattr(vehicle, 'pallet_capacity', 'unknown')}")
            
            # Try to determine if it's furgone or camion based on attributes
            if hasattr(vehicle, 'vehicle_type'):
                if 'furg' in str(vehicle.vehicle_type).lower():
                    furgoni.append(vehicle)
                elif 'cam' in str(vehicle.vehicle_type).lower() or 'heavy' in str(vehicle.vehicle_type).lower():
                    camion.append(vehicle)
                else:
                    unknown.append(vehicle)
            else:
                unknown.append(vehicle)
        
        print(f"\nVEHICLE TYPE SUMMARY:")
        print(f"  Furgoni (light): {len(furgoni)}")
        print(f"  Camion (heavy): {len(camion)}")
        print(f"  Unknown type: {len(unknown)}")
        
        # Test HoS simulation for different vehicle types
        if furgoni and camion:
            print(f"\nTESTING FURGONE HoS SIMULATION:")
            test_hos_for_vehicle(furgoni[0])
            
            print(f"\nTESTING CAMION HoS SIMULATION:")
            test_hos_for_vehicle(camion[0])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_hos_for_vehicle(vehicle):
    """Test HoS simulation for a specific vehicle."""
    print(f"  Vehicle: {vehicle.id}")
    print(f"  Type: {getattr(vehicle, 'vehicle_type', 'unknown')}")
    print(f"  Speed: {getattr(vehicle, 'average_speed', 60.0)} km/h")
    
    # Test travel time calculation
    try:
        from hos_simulation import calculate_travel_time_between_tasks
        
        # Create dummy tasks for testing
        class DummyTask:
            def __init__(self, lat, lon, task_id):
                self.lat = lat
                self.lon = lon
                self.id = task_id
        
        task1 = DummyTask(44.9778, 8.5452, "depot")  # Asti
        task2 = DummyTask(45.0703, 7.6869, "task1")  # Turin
        
        travel_time = calculate_travel_time_between_tasks(task1, task2, vehicle)
        distance_km = 47.0  # Approximate distance Asti-Turin
        speed_kmh = distance_km / (travel_time / 60.0) if travel_time > 0 else 0
        
        print(f"  Travel time Asti->Turin: {travel_time:.1f} min")
        print(f"  Calculated speed: {speed_kmh:.1f} km/h")
        
    except Exception as e:
        print(f"  Error testing travel time: {e}")

if __name__ == "__main__":
    debug_vehicle_configuration()
