#!/usr/bin/env python3
"""
Simple wrapper to run the SA test with error handling.
"""

import sys
import traceback

def main():
    try:
        # Import and run the test
        sys.path.append('.')
        from Tests.simple_test_sa import main as test_main
        
        print("Starting SA test with PuLP comparison...")
        result = test_main()
        print(f"Test completed with result: {result}")
        return result
        
    except Exception as e:
        print(f"Error running test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
