"""
Quick test of scenario creator without geocoding
"""
import pandas as pd
import sys
import os
from pathlib import Path

# Add the algo directory to the path to import data structures
current_dir = Path(__file__).parent
algo_dir = current_dir.parent / "algo"
sys.path.append(str(algo_dir))

from epdt_data_structures import Order, Task, Vehicle, TaskType

def test_excel_reading():
    file_path = os.path.join('..', 'src', 'furgoni.xlsx')
    
    try:
        # Read Excel sheets
        consegne_df = pd.read_excel(file_path, sheet_name="CONSEGNE")
        veicoli_df = pd.read_excel(file_path, sheet_name="VEICOLI")
        
        print(f"CONSEGNE sheet: {len(consegne_df)} rows")
        print("Sample CONSEGNE data:")
        print(consegne_df.head(2))
        print()
        
        print(f"VEICOLI sheet: {len(veicoli_df)} rows")
        print("Sample VEICOLI data:")
        print(veicoli_df.head(2))
        print()
        
        # Test creating a vehicle without geocoding
        if len(veicoli_df) > 0:
            first_vehicle_row = veicoli_df.iloc[0]
            print("Testing vehicle creation:")
            print(f"NUMBER PLATE: {first_vehicle_row['NUMBER PLATE']}")
            print(f"TYPE OF VEHICLE: {first_vehicle_row['TYPE OF VEHICLE']}")
            print(f"MAX LOAD KG: {first_vehicle_row['MAX LOAD KG']}")
            print(f"LIFO: {first_vehicle_row.get('LAST IN FIRST OUT', 'N/A')}")
            
        # Test task type parsing
        if len(consegne_df) > 0:
            first_consegne_row = consegne_df.iloc[0]
            print("\nTesting task creation:")
            print(f"NAME: {first_consegne_row['NAME']}")
            print(f"ADDRESS: {first_consegne_row['ADDRESS']}")
            print(f"DELIVERY OR PICKUP: {first_consegne_row.get('DELIVERY OR PICKUP', 'N/A')}")
            print(f"LOAD KG: {first_consegne_row.get('LOAD KG', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_excel_reading()
