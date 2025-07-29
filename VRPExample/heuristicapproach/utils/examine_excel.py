"""
Simple script to examine the furgoni.xlsx file structure
"""
import pandas as pd
import sys
import os

# Add path to find the Excel file
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def examine_excel():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'furgoni2.xlsx')
    
    try:
        # Read all sheets
        df_dict = pd.read_excel(file_path, sheet_name=None)
        
        print("=== FURGONI.XLSX STRUCTURE ===\n")
        print(f"Available sheets: {list(df_dict.keys())}\n")
        
        for sheet_name, df in df_dict.items():
            print(f"=== SHEET: {sheet_name} ===")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print("\nFirst 3 rows:")
            print(df.head(3))
            print("\n" + "="*50 + "\n")
            
    except Exception as e:
        print(f"Error reading Excel file: {e}")

if __name__ == "__main__":
    examine_excel()
