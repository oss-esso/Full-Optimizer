#!/usr/bin/env python3
"""Quick test of safe_parse_value function"""

import pandas as pd
import sys
import os
from pathlib import Path

# Add the utils directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from scenario_creator import safe_parse_value

def test_safe_parse():
    # Test data with comma decimal
    test_data = {
        'test_float': '1,54',
        'test_int': '42',
        'test_bool': 'YES',
        'test_na': None
    }
    
    row = pd.Series(test_data)
    
    print("Testing safe_parse_value function:")
    print(f"Float '1,54': {safe_parse_value(row, 'test_float', 0.0, float)}")
    print(f"Int '42': {safe_parse_value(row, 'test_int', 0, int)}")
    print(f"Bool 'YES': {safe_parse_value(row, 'test_bool', False, bool)}")
    print(f"NA value: {safe_parse_value(row, 'test_na', 'default', str)}")

if __name__ == "__main__":
    test_safe_parse()
