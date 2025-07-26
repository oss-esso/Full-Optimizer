"""
Excel Data Investigation Script

This script examines the Excel file directly and traces the conversion process
to understand why orders are ending up with 0 demand when they shouldn't.
"""

import sys
import os
import pandas as pd

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

print("=== EXCEL DATA INVESTIGATION ===")

def examine_excel_data():
    """Examine the raw Excel data to understand the demand values."""
    excel_file = os.path.join(src_dir, 'furgoni.xlsx')
    
    if not os.path.exists(excel_file):
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"📂 Examining Excel file: {excel_file}")
    
    try:
        # Read the Excel sheets
        print(f"\n📊 EXAMINING EXCEL SHEETS:")
        
        # Check available sheets
        xl_file = pd.ExcelFile(excel_file)
        print(f"   Available sheets: {xl_file.sheet_names}")
        
        # Read the orders sheet (CONSEGNE)
        orders_df = pd.read_excel(excel_file, sheet_name="CONSEGNE")
        print(f"\n📦 CONSEGNE SHEET ANALYSIS:")
        print(f"   • Rows: {len(orders_df)}")
        print(f"   • Columns: {list(orders_df.columns)}")
        
        # Show first few rows
        print(f"\n   First 5 rows:")
        print(orders_df.head().to_string())
        
        # Check specific columns that might contain demand/weight data
        demand_columns = []
        for col in orders_df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['weight', 'kg', 'demand', 'peso', 'quantit', 'volume', 'pallet']):
                demand_columns.append(col)
        
        print(f"\n   🔍 POTENTIAL DEMAND COLUMNS:")
        for col in demand_columns:
            print(f"     • {col}")
            print(f"       - Data type: {orders_df[col].dtype}")
            print(f"       - Sample values: {orders_df[col].dropna().head(3).tolist()}")
            print(f"       - Null count: {orders_df[col].isnull().sum()}")
            print(f"       - Min: {orders_df[col].min() if pd.api.types.is_numeric_dtype(orders_df[col]) else 'N/A'}")
            print(f"       - Max: {orders_df[col].max() if pd.api.types.is_numeric_dtype(orders_df[col]) else 'N/A'}")
        
        # Look for specific problematic orders
        problematic_order_names = [
            "DIPHARMA FRANCIS SRL",
            "GEODIS", 
            "COLUMBIA  SERRAVALLE",
            "AGRIACRO TECH SRL",
            "FIAT V. I SPA STAB B"
        ]
        
        print(f"\n🎯 SEARCHING FOR PROBLEMATIC ORDERS:")
        for order_name in problematic_order_names:
            # Search in all text columns
            matches = []
            for col in orders_df.columns:
                if orders_df[col].dtype == 'object':  # Text columns
                    mask = orders_df[col].astype(str).str.contains(order_name, case=False, na=False)
                    if mask.any():
                        matches.extend(orders_df[mask].index.tolist())
            
            if matches:
                print(f"\n   Found '{order_name}' in rows: {matches}")
                for row_idx in matches[:2]:  # Show first 2 matches
                    print(f"   Row {row_idx}:")
                    row_data = orders_df.iloc[row_idx]
                    for col in demand_columns:
                        print(f"     {col}: {row_data[col]}")
        
        # Read vehicles sheet
        vehicles_df = pd.read_excel(excel_file, sheet_name="VEICOLI")
        print(f"\n🚛 VEICOLI SHEET ANALYSIS:")
        print(f"   • Rows: {len(vehicles_df)}")
        print(f"   • Columns: {list(vehicles_df.columns)}")
        
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        import traceback
        traceback.print_exc()

def trace_conversion_process():
    """Trace how the Excel data gets converted to EPDT orders."""
    print(f"\n🔄 TRACING CONVERSION PROCESS:")
    
    try:
        from scenario_creator import create_scenario_from_excel
        
        excel_file = os.path.join(src_dir, 'furgoni.xlsx')
        
        print(f"   Calling create_scenario_from_excel...")
        orders, vehicles = create_scenario_from_excel(excel_file)
        
        print(f"\n📊 CONVERSION RESULTS:")
        print(f"   • Orders created: {len(orders)}")
        print(f"   • Vehicles created: {len(vehicles)}")
        
        # Examine problematic orders
        problematic_order_ids = [
            "ORDER_DIPHARMA_FRANCIS_SRL_39",
            "ORDER_GEODIS_18", 
            "ORDER_COLUMBIA__SERRAVALLE_0",
            "ORDER_AGRIACRO_TECH_SRL_19"
        ]
        
        print(f"\n🔍 EXAMINING CONVERTED PROBLEMATIC ORDERS:")
        for order_id in problematic_order_ids:
            order = None
            for o in orders:
                if o.id == order_id:
                    order = o
                    break
            
            if order:
                print(f"\n   ORDER: {order.id}")
                print(f"   • Pickup tasks: {len(order.pickup_tasks)}")
                print(f"   • Delivery tasks: {len(order.delivery_tasks)}")
                
                for i, task in enumerate(order.pickup_tasks + order.delivery_tasks):
                    print(f"   • Task {i+1} ({task.task_type.value}):")
                    print(f"     - Weight demand: {getattr(task, 'demand', 'NOT_FOUND')}")
                    print(f"     - Volume demand: {getattr(task, 'volume', 'NOT_FOUND')}")
                    print(f"     - Pallet demand: {getattr(task, 'pallets', 'NOT_FOUND')}")
                    print(f"     - Location: {getattr(task, 'location_id', 'NOT_FOUND')}")
                    
                    # Check if task has coordinate data
                    if hasattr(task, 'lat') and hasattr(task, 'lon'):
                        print(f"     - Coordinates: ({task.lat}, {task.lon})")
                    else:
                        print(f"     - Coordinates: NOT_FOUND")
            else:
                print(f"\n   ORDER {order_id}: NOT FOUND IN CONVERTED DATA")
    
    except Exception as e:
        print(f"❌ Error in conversion tracing: {e}")
        import traceback
        traceback.print_exc()

def main():
    examine_excel_data()
    trace_conversion_process()

if __name__ == "__main__":
    main()
