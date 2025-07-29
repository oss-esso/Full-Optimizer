#!/usr/bin/env python3
"""
Analyze the AUTISTI sheet structure and update the implementation to handle it properly.
"""

import pandas as pd
from pathlib import Path

def analyze_autisti_sheet():
    """Analyze the AUTISTI sheet structure in detail."""
    excel_path = "src/furgoni2.xlsx"
    
    print("🔍 Analyzing AUTISTI sheet structure...")
    
    try:
        df = pd.read_excel(excel_path, sheet_name="AUTISTI")
        print(f"📊 AUTISTI sheet has {len(df)} rows and {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show sample data
        print(f"\n📋 Sample data (first 5 rows):")
        print(df.head().to_string())
        
        # Analyze data types and patterns
        print(f"\n📊 Data Analysis:")
        for col in df.columns:
            if col.startswith('Unnamed'):
                continue
            non_null_count = df[col].notna().sum()
            unique_count = df[col].nunique()
            print(f"   {col}: {non_null_count}/{len(df)} non-null, {unique_count} unique values")
            
            # Show sample values
            sample_values = df[col].dropna().head(3).tolist()
            print(f"      Sample values: {sample_values}")
        
    except Exception as e:
        print(f"❌ Error analyzing AUTISTI sheet: {e}")

def update_implementation_for_autisti():
    """Update scenario_creator.py to properly handle AUTISTI sheet."""
    
    print(f"\n🔧 The implementation needs to be updated to handle:")
    print(f"   1. AUTISTI sheet (instead of DRIVERS)")
    print(f"   2. Different column names in AUTISTI")
    print(f"   3. Driver-Vehicle mapping via NUMBER PLATE")
    
    # The AUTISTI sheet maps drivers to vehicles via NUMBER PLATE
    # This is different from the proposed independent DRIVERS sheet
    
    print(f"\n💡 Proposed updates:")
    print(f"   - Add AUTISTI as alternative to DRIVERS sheet")
    print(f"   - Map DRIVER -> DRIVER_NAME") 
    print(f"   - Map LICENSE -> capability inference")
    print(f"   - Map COST PER HOUR -> COST_PER_HOUR")
    print(f"   - Use NUMBER PLATE to link drivers to vehicles")

if __name__ == "__main__":
    analyze_autisti_sheet()
    update_implementation_for_autisti()
