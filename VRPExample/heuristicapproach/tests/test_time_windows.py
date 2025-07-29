import unittest
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import time as datetime_time

# Add necessary paths
current_dir = Path(__file__).parent
heuristic_root = current_dir.parent
utils_dir = heuristic_root / 'utils'
sys.path.insert(0, str(heuristic_root))
sys.path.insert(0, str(utils_dir))

from scenario_creator import create_scenario_from_excel

class TestTimeWindowParsing(unittest.TestCase):

    def setUp(self):
        """Create a sample Excel file for testing time windows."""
        self.test_excel_path = current_dir / 'sample_time_window_test.xlsx'
        
        consegne_data = {
            'ORDER_ID': ['TW_ORDER_1', 'TW_ORDER_2'],
            'COMPANY_NAME': ['Time Test 1', 'Time Test 2'],
            'STREET': ['Via Prova', 'Via Test'],
            'HOUSE NUMBER': [1, 2],
            'CITY': ['Asti', 'Asti'],
            'PROVINCE': ['AT', 'AT'],
            'POSTAL CODE': [14100, 14100],
            'COUNTRY': ['ITALY', 'ITALY'],
            'DELIVERY OR PICKUP': ['DELIVERY', 'DELIVERY'],
            'LOAD KG': [10, 10],
            'LOAD VOLUME M^3': [1, 1],
            'EARLIEST DAY': [2, 1],
            'LATEST DAY': [3, 1],
            'TIME WINDOW START': [datetime_time(9, 0), datetime_time(14, 30)],
            'TIME WINDOW END': [datetime_time(17, 0), datetime_time(18, 0)]
        }
        consegne_df = pd.DataFrame(consegne_data)

        veicoli_data = {'NUMBER PLATE': ['TW123CD']}
        veicoli_df = pd.DataFrame(veicoli_data)

        with pd.ExcelWriter(self.test_excel_path) as writer:
            consegne_df.to_excel(writer, sheet_name='CONSEGNE', index=False)
            veicoli_df.to_excel(writer, sheet_name='VEICOLI', index=False)

    def tearDown(self):
        try:
            if os.path.exists(self.test_excel_path):
                os.remove(self.test_excel_path)
        except PermissionError:
            # File might be locked, ignore for test purposes
            pass

    def test_multi_day_time_window_parsing(self):
        """Verify that multi-day time windows are parsed into absolute minutes."""
        orders, _, _ = create_scenario_from_excel(str(self.test_excel_path))

        self.assertEqual(len(orders), 2)

        # Test Order 1 (Day 2, 09:00 to Day 3, 17:00)
        order1 = next((o for o in orders if o.id == 'TW_ORDER_1'), None)
        self.assertIsNotNone(order1)
        task1 = order1.delivery_tasks[0]
        # Expected earliest: (2-1) * 1440 + 9*60 = 1440 + 540 = 1980
        self.assertEqual(task1.earliest_time, 1980)
        # Expected latest: (3-1) * 1440 + 17*60 = 2880 + 1020 = 3900
        self.assertEqual(task1.latest_time, 3900)

        # Test Order 2 (Day 1, 14:30 to Day 1, 18:00)
        order2 = next((o for o in orders if o.id == 'TW_ORDER_2'), None)
        self.assertIsNotNone(order2)
        task2 = order2.delivery_tasks[0]
        # Expected earliest: (1-1) * 1440 + 14*60 + 30 = 870
        self.assertEqual(task2.earliest_time, 870)
        # Expected latest: (1-1) * 1440 + 18*60 = 1080
        self.assertEqual(task2.latest_time, 1080)

if __name__ == '__main__':
    unittest.main()
