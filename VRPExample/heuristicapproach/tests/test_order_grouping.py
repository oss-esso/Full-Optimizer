import unittest
import os
import sys
from pathlib import Path
import pandas as pd

# Add necessary paths
current_dir = Path(__file__).parent
heuristic_root = current_dir.parent
utils_dir = heuristic_root / 'utils'
sys.path.insert(0, str(heuristic_root))
sys.path.insert(0, str(utils_dir))

from scenario_creator import create_scenario_from_excel

class TestOrderGrouping(unittest.TestCase):

    def setUp(self):
        """Create a sample Excel file for testing."""
        self.test_excel_path = current_dir / 'sample_multi_task_order.xlsx'
        
        # Create CONSEGNE sheet
        consegne_data = {
            'ORDER_ID': ['ORDER_1', 'ORDER_1', 'ORDER_2'],
            'COMPANY_NAME': ['Company A', 'Company B', 'Company C'],
            'STREET': ['Via Roma', 'Via Milano', 'Via Napoli'],
            'HOUSE NUMBER': [1, 2, 3],
            'CITY': ['Torino', 'Milano', 'Napoli'],
            'PROVINCE': ['TO', 'MI', 'NA'],
            'POSTAL CODE': [10121, 20121, 80100],
            'COUNTRY': ['ITALY', 'ITALY', 'ITALY'],
            'DELIVERY OR PICKUP': ['PICKUP', 'DELIVERY', 'PICKUP'],
            'LOAD KG': [100, 100, 50],
            'LOAD VOLUME M^3': [1.0, 1.0, 0.5]
        }
        consegne_df = pd.DataFrame(consegne_data)

        # Create VEICOLI sheet
        veicoli_data = {
            'NUMBER PLATE': ['AB123CD'],
            'MAX LOAD KG': [1000],
            'MAX LOAD VOLUME M^3': [10],
            'PALLET': [2]
        }
        veicoli_df = pd.DataFrame(veicoli_data)

        # Create DRIVERS sheet
        drivers_data = {
            'DRIVER_ID': ['DRV_1'],
            'DRIVER_NAME': ['John Doe']
        }
        drivers_df = pd.DataFrame(drivers_data)

        with pd.ExcelWriter(self.test_excel_path) as writer:
            consegne_df.to_excel(writer, sheet_name='CONSEGNE', index=False)
            veicoli_df.to_excel(writer, sheet_name='VEICOLI', index=False)
            drivers_df.to_excel(writer, sheet_name='DRIVERS', index=False)

    def tearDown(self):
        """Remove the sample Excel file after the test."""
        try:
            if os.path.exists(self.test_excel_path):
                os.remove(self.test_excel_path)
        except PermissionError:
            # File might be locked, ignore for test purposes
            pass

    def test_orders_are_grouped_correctly(self):
        """Verify that tasks are grouped into orders by ORDER_ID."""
        orders, _, _ = create_scenario_from_excel(str(self.test_excel_path))

        # We expect 2 orders to be created from the 3 rows
        self.assertEqual(len(orders), 2, "Should create 2 orders from 3 rows with shared ORDER_ID")

        # Find ORDER_1
        order_1 = next((o for o in orders if o.id == 'ORDER_1'), None)
        self.assertIsNotNone(order_1, "ORDER_1 should exist")

        # ORDER_1 should have 1 pickup task and 1 delivery task (NO depot tasks)
        self.assertEqual(len(order_1.pickup_tasks), 1, "ORDER_1 should have one pickup task")
        self.assertEqual(len(order_1.delivery_tasks), 1, "ORDER_1 should have one delivery task")

        # Find ORDER_2
        order_2 = next((o for o in orders if o.id == 'ORDER_2'), None)
        self.assertIsNotNone(order_2, "ORDER_2 should exist")
        
        # ORDER_2 should have 1 pickup and 0 delivery tasks
        self.assertEqual(len(order_2.pickup_tasks), 1, "ORDER_2 should have one pickup task")
        self.assertEqual(len(order_2.delivery_tasks), 0, "ORDER_2 should have zero delivery tasks")

if __name__ == '__main__':
    unittest.main()
