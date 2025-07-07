"""
Test script for OSM route calculator with detailed logging
"""
import sys
import os
sys.path.append('.')

# Import the OSM calculator from the main file
with open('vrp_optimizer_clean copy.py', 'r', encoding='utf-8') as f:
    code = f.read()
    
# Execute everything except the main section
main_split = code.split('if __name__ == "__main__"')
exec(main_split[0])

# Now run the test
if __name__ == "__main__":
    test_osm_calculator()
