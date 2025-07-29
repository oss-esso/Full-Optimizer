"""
Create a sample Excel file that matches TODO #23 specifications for testing
"""

import pandas as pd
from openpyxl import Workbook
import os

def create_todo23_sample_excel():
    """Create sample Excel file matching TODO #23 specification"""
    
    # CONSEGNE sheet - with ORDER column for grouping
    consegne_data = [
        # Order 1: Pickup and delivery at different locations
        {
            'ORDER': 'ORD001',
            'COMPANY': 'Company A',
            'STREET': 'Via Roma',
            'HOUSE NUMBER': '10',
            'CITY': 'Milano',
            'PROVINCE': 'MI',
            'POSTAL CODE': '20121',
            'COUNTRY': 'Italy',
            'EARLIEST DAY': 1,
            'LATEST DAY': 1,
            'TIME WINDOW START': '09:00:00',
            'TIME WINDOW END': '12:00:00',
            'SERVICE TIME': 30.0,
            'TASK': 'PICKUP',
            'LOAD KG': 100.0,
            'LOAD VOLUME M^3': 2.0,
            'PALLETS': 2,
            'LOW_TEMP': 'NO',
            'LOADER': 'YES',
            'HANGERS': 'NO'
        },
        {
            'ORDER': 'ORD001',
            'COMPANY': 'Company B',
            'STREET': 'Via Torino',
            'HOUSE NUMBER': '25',
            'CITY': 'Milano',
            'PROVINCE': 'MI',
            'POSTAL CODE': '20123',
            'COUNTRY': 'Italy',
            'EARLIEST DAY': 1,
            'LATEST DAY': 1,
            'TIME WINDOW START': '14:00:00',
            'TIME WINDOW END': '17:00:00',
            'SERVICE TIME': 20.0,
            'TASK': 'DELIVERY',
            'LOAD KG': 100.0,  # Will be made negative by the parser
            'LOAD VOLUME M^3': 2.0,  # Will be made negative by the parser
            'PALLETS': 2,  # Will be made negative by the parser
            'LOW_TEMP': 'NO',
            'LOADER': 'YES',
            'HANGERS': 'NO'
        },
        # Order 2: Multiple pickups, one delivery
        {
            'ORDER': 'ORD002',
            'COMPANY': 'Supplier X',
            'STREET': 'Via Napoleone',
            'HOUSE NUMBER': '15',
            'CITY': 'Roma',
            'PROVINCE': 'RM',
            'POSTAL CODE': '00186',
            'COUNTRY': 'Italy',
            'EARLIEST DAY': 2,
            'LATEST DAY': 2,
            'TIME WINDOW START': '08:00:00',
            'TIME WINDOW END': '11:00:00',
            'SERVICE TIME': 25.0,
            'TASK': 'PICKUP',
            'LOAD KG': 50.0,
            'LOAD VOLUME M^3': 1.0,
            'PALLETS': 1,
            'LOW_TEMP': 'YES',
            'LOADER': 'NO',
            'HANGERS': 'NO'
        },
        {
            'ORDER': 'ORD002',
            'COMPANY': 'Supplier Y',
            'STREET': 'Via del Corso',
            'HOUSE NUMBER': '100',
            'CITY': 'Roma',
            'PROVINCE': 'RM',
            'POSTAL CODE': '00187',
            'COUNTRY': 'Italy',
            'EARLIEST DAY': 2,
            'LATEST DAY': 2,
            'TIME WINDOW START': '09:30:00',
            'TIME WINDOW END': '12:00:00',
            'SERVICE TIME': 15.0,
            'TASK': 'PICKUP',
            'LOAD KG': 30.0,
            'LOAD VOLUME M^3': 0.5,
            'PALLETS': 1,
            'LOW_TEMP': 'YES',
            'LOADER': 'NO',
            'HANGERS': 'NO'
        },
        {
            'ORDER': 'ORD002',
            'COMPANY': 'Customer Z',
            'STREET': 'Via Veneto',
            'HOUSE NUMBER': '50',
            'CITY': 'Roma',
            'PROVINCE': 'RM',
            'POSTAL CODE': '00187',
            'COUNTRY': 'Italy',
            'EARLIEST DAY': 2,
            'LATEST DAY': 2,
            'TIME WINDOW START': '15:00:00',
            'TIME WINDOW END': '18:00:00',
            'SERVICE TIME': 30.0,
            'TASK': 'DELIVERY',
            'LOAD KG': 80.0,  # Total of both pickups (will be made negative)
            'LOAD VOLUME M^3': 1.5,  # Total of both pickups (will be made negative)
            'PALLETS': 2,  # Total of both pickups (will be made negative)
            'LOW_TEMP': 'YES',
            'LOADER': 'NO',
            'HANGERS': 'NO'
        }
    ]
    
    # VEICOLI sheet
    veicoli_data = [
        {
            'NUMBER PLATE': 'AB123CD',
            'TYPE OF VEHICLE': 'Van',
            'MAX LOAD KG': 3500,
            'MAX LOAD VOLUME M^3': 15.0,
            'PALLET': 8,
            'COST PER KM': 0.55,
            'FIXED COST': 50.0,
            'LOW_TEMP': 'YES',
            'LOADER': 'YES',
            'HANGERS': 'NO',
            'REGULATIONS': 'NO',
            'LAST IN FIRST OUT': 'YES'
        },
        {
            'NUMBER PLATE': 'EF456GH',
            'TYPE OF VEHICLE': 'Truck',
            'MAX LOAD KG': 12000,
            'MAX LOAD VOLUME M^3': 40.0,
            'PALLET': 24,
            'COST PER KM': 0.85,
            'FIXED COST': 80.0,
            'LOW_TEMP': 'NO',
            'LOADER': 'YES',
            'HANGERS': 'YES',
            'REGULATIONS': 'YES',
            'LAST IN FIRST OUT': 'NO'
        }
    ]
    
    # AUTISTI sheet
    autisti_data = [
        {
            'LICENSE PLATE': 'AB123CD',
            'DRIVER': 'Mario Rossi',
            'LICENSE': 'B',
            'COST PER HOUR': 18.50
        },
        {
            'LICENSE PLATE': 'EF456GH', 
            'DRIVER': 'Giuseppe Verdi',
            'LICENSE': 'CE',
            'COST PER HOUR': 22.00
        }
    ]
    
    # Create Excel file
    output_path = "sample_todo23_scenario.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame(consegne_data).to_excel(writer, sheet_name='CONSEGNE', index=False)
        pd.DataFrame(veicoli_data).to_excel(writer, sheet_name='VEICOLI', index=False)
        pd.DataFrame(autisti_data).to_excel(writer, sheet_name='AUTISTI', index=False)
    
    print(f"Created sample Excel file: {output_path}")
    print("\nCONSEGNE sheet structure:")
    print(f"- {len(consegne_data)} tasks")
    print(f"- Orders: {set(row['ORDER'] for row in consegne_data)}")
    print(f"- Task types: {[row['TASK'] for row in consegne_data]}")
    
    print("\nVEICOLI sheet structure:")
    print(f"- {len(veicoli_data)} vehicles")
    
    print("\nAUTISTI sheet structure:")
    print(f"- {len(autisti_data)} drivers")
    
    return output_path


if __name__ == "__main__":
    create_todo23_sample_excel()
