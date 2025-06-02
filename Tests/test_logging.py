#!/usr/bin/env python3
"""
Test script to verify logging works.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test logging."""
    logger.info("Starting test...")
    print("Print statement test")
    logger.info("Testing complete")
    return True

if __name__ == "__main__":
    print("Script starting...")
    success = main()
    print(f"Script completed with success: {success}")
    sys.exit(0 if success else 1)
