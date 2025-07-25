"""
Sample Excel File Creator for Testing Scenario Creator

This script creates a sample Excel file with the expected structure
for testing the scenario_creator.py functionality.
"""

import pandas as pd
from pathlib import Path

def create_sample_excel():
    """Create a sample Excel file for testing scenario creation."""
    
    # Sample orders data (consegne sheet)
    consegne_data = [
        {
            'Order ID': 'ORD-001',
            'Task ID': 'T1-P1',
            'Task Type': 'PICKUP',
            'Address': 'Via Roma 1',
            'City': 'Padova',
            'Postal Code': '35121',
            'Country': 'IT',
            'Weight (kg)': 150,
            'Volume (m³)': 1.2,
            'TW Start (HH:MM)': '09:00',
            'TW End (HH:MM)': '11:00',
            'Service Time (min)': 20,
            'Priority': 'urgent',
            'CUSTOMER_ID': 'CUST-001',
            'ARTICLE_ID': 'ART-001'
        },
        {
            'Order ID': 'ORD-001',
            'Task ID': 'T1-D1',
            'Task Type': 'DELIVERY',
            'Address': 'Via Milano 2',
            'City': 'Verona',
            'Postal Code': '37121',
            'Country': 'IT',
            'Weight (kg)': -150,
            'Volume (m³)': -1.2,
            'TW Start (HH:MM)': '14:00',
            'TW End (HH:MM)': '17:00',
            'Service Time (min)': 15,
            'Priority': 'urgent',
            'CUSTOMER_ID': 'CUST-001',
            'ARTICLE_ID': 'ART-001'
        },
        {
            'Order ID': 'ORD-002',
            'Task ID': 'T2-P1',
            'Task Type': 'PICKUP',
            'Address': 'Corso del Popolo 10',
            'City': 'Padova',
            'Postal Code': '35131',
            'Country': 'IT',
            'Weight (kg)': 200,
            'Volume (m³)': 2.0,
            'TW Start (HH:MM)': '10:00',
            'TW End (HH:MM)': '12:00',
            'Service Time (min)': 25,
            'Priority': 'mandatory',
            'CUSTOMER_ID': 'CUST-002',
            'ARTICLE_ID': 'ART-002'
        },
        {
            'Order ID': 'ORD-002',
            'Task ID': 'T2-D1',
            'Task Type': 'DELIVERY',
            'Address': 'Piazza delle Erbe',
            'City': 'Verona',
            'Postal Code': '37121',
            'Country': 'IT',
            'Weight (kg)': -200,
            'Volume (m³)': -2.0,
            'TW Start (HH:MM)': '15:00',
            'TW End (HH:MM)': '18:00',
            'Service Time (min)': 20,
            'Priority': 'mandatory',
            'CUSTOMER_ID': 'CUST-002',
            'ARTICLE_ID': 'ART-002'
        },
        {
            'Order ID': 'ORD-003',
            'Task ID': 'T3-P1',
            'Task Type': 'PICKUP',
            'Address': 'Via Venezia 15',
            'City': 'Padova',
            'Postal Code': '35122',
            'Country': 'IT',
            'Weight (kg)': 75,
            'Volume (m³)': 0.8,
            'TW Start (HH:MM)': '08:30',
            'TW End (HH:MM)': '10:30',
            'Service Time (min)': 15,
            'Priority': 'normal',
            'CUSTOMER_ID': 'CUST-003',
            'ARTICLE_ID': 'ART-003'
        },
        {
            'Order ID': 'ORD-003',
            'Task ID': 'T3-D1',
            'Task Type': 'DELIVERY',
            'Address': 'Corso Porta Nuova 5',
            'City': 'Verona',
            'Postal Code': '37122',
            'Country': 'IT',
            'Weight (kg)': -75,
            'Volume (m³)': -0.8,
            'TW Start (HH:MM)': '13:00',
            'TW End (HH:MM)': '16:00',
            'Service Time (min)': 12,
            'Priority': 'normal',
            'CUSTOMER_ID': 'CUST-003',
            'ARTICLE_ID': 'ART-003'
        }
    ]
    
    # Sample vehicles data
    vehicles_data = [
        {
            'Vehicle ID': 'V001',
            'Depot ID': 'DEPOT-PD',
            'Weight Capacity (kg)': 3500,
            'Volume Capacity (m³)': 25,
            'Cost per km': 1.2,
            'Fixed Cost': 50,
            'Vehicle Type': 'standard',
            'LIFO Required': False,
            'MAX_ORDERS': 10
        },
        {
            'Vehicle ID': 'V002',
            'Depot ID': 'DEPOT-PD',
            'Weight Capacity (kg)': 7500,
            'Volume Capacity (m³)': 40,
            'Cost per km': 1.8,
            'Fixed Cost': 100,
            'Vehicle Type': 'heavy',
            'LIFO Required': True,
            'MAX_ORDERS': 20
        },
        {
            'Vehicle ID': 'V003',
            'Depot ID': 'DEPOT-PD',
            'Weight Capacity (kg)': 1500,
            'Volume Capacity (m³)': 15,
            'Cost per km': 0.8,
            'Fixed Cost': 30,
            'Vehicle Type': 'car',
            'LIFO Required': False,
            'MAX_ORDERS': 5
        }
    ]
    
    # Create DataFrames
    consegne_df = pd.DataFrame(consegne_data)
    vehicles_df = pd.DataFrame(vehicles_data)
    
    # Save to Excel file
    output_path = Path(__file__).parent / "sample_scenario.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        consegne_df.to_excel(writer, sheet_name='consegne', index=False)
        vehicles_df.to_excel(writer, sheet_name='vehicles', index=False)
    
    print(f"Sample Excel file created: {output_path}")
    return output_path

if __name__ == "__main__":
    create_sample_excel()
