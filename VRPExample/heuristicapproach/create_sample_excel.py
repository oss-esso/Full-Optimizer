#!/usr/bin/env python3
"""
Create a sample Excel file with the new three-sheet format for testing.
"""

import pandas as pd
from pathlib import Path

def create_sample_excel():
    """Create a sample Excel file with CONSEGNE, VEICOLI, and DRIVERS sheets."""
    
    # Sample CONSEGNE data (new format)
    consegne_data = [
        {
            'ORDER_ID': 'ORD-101',
            'COMPANY_NAME': 'Customer A',
            'STREET': 'Via Roma',
            'HOUSE_NUMBER': '10',
            'CITY': 'Torino',
            'PROVINCE': 'TO',
            'POSTAL_CODE': '10121',
            'COUNTRY': 'Italy',
            'EARLIEST_DAY': 1,
            'LATEST_DAY': 1,
            'TIME_WINDOW_START': '09:00',
            'TIME_WINDOW_END': '18:00',
            'SERVICE_TIME': 20,
            'DELIVERY_OR_PICKUP': 'DELIVERY',
            'LOAD_KG': 150.5,
            'LOAD_VOLUME_M3': 1.5,
            'PALLETS': 2,
            'REQUIRED_CAPABILITIES': 'LOW_TEMP, LOADER'
        },
        {
            'ORDER_ID': 'ORD-102',
            'COMPANY_NAME': 'Customer B',
            'STREET': 'Corso Francia',
            'HOUSE_NUMBER': '25',
            'CITY': 'Milano',
            'PROVINCE': 'MI',
            'POSTAL_CODE': '20100',
            'COUNTRY': 'Italy',
            'EARLIEST_DAY': 1,
            'LATEST_DAY': 2,
            'TIME_WINDOW_START': '10:00',
            'TIME_WINDOW_END': '16:00',
            'SERVICE_TIME': 15,
            'DELIVERY_OR_PICKUP': 'PICKUP',
            'LOAD_KG': 75.0,
            'LOAD_VOLUME_M3': 0.8,
            'PALLETS': 1,
            'REQUIRED_CAPABILITIES': 'ADR_CERTIFIED'
        }
    ]
    
    # Sample VEICOLI data (new format)
    veicoli_data = [
        {
            'NUMBER PLATE': 'AB123CD',
            'TYPE OF VEHICLE': 'Van',
            'MAX LOAD KG': 3500,
            'PALLET': 8,
            'MAX LOAD VOLUME M^3': 15,
            'COST_PER_KM': 0.55,
            'FIXED_COST': 50,
            'CAPABILITIES': 'LOW_TEMP, LOADER',
            'REGULATIONS': 'YES'
        },
        {
            'NUMBER PLATE': 'EF456GH',
            'TYPE OF VEHICLE': 'Truck',
            'MAX LOAD KG': 10000,
            'PALLET': 20,
            'MAX LOAD VOLUME M^3': 40,
            'COST_PER_KM': 0.75,
            'FIXED_COST': 100,
            'CAPABILITIES': 'ADR_CERTIFIED, LOADER',
            'REGULATIONS': 'YES'
        }
    ]
    
    # Sample DRIVERS data (new format)
    drivers_data = [
        {
            'DRIVER_ID': 'DRV-01',
            'DRIVER_NAME': 'Mario Rossi',
            'COST_PER_HOUR': 28.50,
            'MAX_SHIFT_HOURS': 13,
            'MAX_DRIVING_HOURS': 9,
            'CAPABILITIES': 'ADR_CERTIFIED'
        },
        {
            'DRIVER_ID': 'DRV-02',
            'DRIVER_NAME': 'Luigi Bianchi',
            'COST_PER_HOUR': 25.00,
            'MAX_SHIFT_HOURS': 12,
            'MAX_DRIVING_HOURS': 8,
            'CAPABILITIES': 'FORKLIFT_LICENSE'
        }
    ]
    
    # Create DataFrames
    consegne_df = pd.DataFrame(consegne_data)
    veicoli_df = pd.DataFrame(veicoli_data)
    drivers_df = pd.DataFrame(drivers_data)
    
    # Save to Excel file
    output_path = Path("sample_new_format.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        consegne_df.to_excel(writer, sheet_name='CONSEGNE', index=False)
        veicoli_df.to_excel(writer, sheet_name='VEICOLI', index=False)
        drivers_df.to_excel(writer, sheet_name='DRIVERS', index=False)
    
    print(f"✅ Created sample Excel file: {output_path}")
    print(f"   📋 CONSEGNE: {len(consegne_df)} orders")
    print(f"   🚚 VEICOLI: {len(veicoli_df)} vehicles")
    print(f"   👨‍💼 DRIVERS: {len(drivers_df)} drivers")
    
    return output_path

if __name__ == "__main__":
    create_sample_excel()
