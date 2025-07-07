#!/usr/bin/env python3
"""
Debug script to understand how locations are stored and accessed in the VRP scenario
"""

import logging
from vrp_scenarios import create_moda_small_scenario

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_location_access():
    """Debug how locations are stored and accessed."""
    logger.info("🔍 DEBUGGING LOCATION ACCESS")
    logger.info("=" * 50)
    
    scenario = create_moda_small_scenario()
    
    logger.info(f"📊 Scenario.locations type: {type(scenario.locations)}")
    logger.info(f"📊 Number of locations: {len(scenario.locations)}")
    
    # Check what's in locations dict
    logger.info("🔍 Checking instance.locations dict:")
    for key, value in list(scenario.locations.items())[:3]:  # First 3 items
        logger.info(f"  Key: {key} (type: {type(key)})")
        logger.info(f"  Value: {value} (type: {type(value)})")
        if hasattr(value, 'service_time'):
            logger.info(f"    service_time: {value.service_time}")
        if hasattr(value, '__dict__'):
            logger.info(f"    attributes: {list(value.__dict__.keys())}")
        logger.info("")
    
    # Check location_list creation like the optimizer does
    logger.info("🔍 Creating location_list like optimizer does:")
    location_list = list(scenario.locations.values())
    logger.info(f"location_list type: {type(location_list)}")
    logger.info(f"location_list length: {len(location_list)}")
    
    for i, loc in enumerate(location_list[:3]):  # First 3 locations
        logger.info(f"  location_list[{i}]: {loc} (type: {type(loc)})")
        if hasattr(loc, 'service_time'):
            logger.info(f"    service_time: {loc.service_time}")
        else:
            logger.info(f"    NO service_time attribute!")
        if hasattr(loc, 'id'):
            logger.info(f"    id: {loc.id}")
        else:
            logger.info(f"    NO id attribute!")
        logger.info("")
    
    # Check location_ids list
    logger.info("🔍 Checking instance.location_ids:")
    logger.info(f"location_ids: {scenario.location_ids[:5]}...")  # First 5
    
    # Simulate the time callback logic
    logger.info("🔍 Simulating time callback logic:")
    if len(location_list) > 1:
        to_node = 2  # Third location (pickup_1)
        to_loc = location_list[to_node]
        logger.info(f"Accessing location_list[{to_node}]: {to_loc}")
        
        if hasattr(to_loc, 'service_time') and to_loc.service_time is not None:
            service_time = int(to_loc.service_time)
            logger.info(f"✅ Found service_time: {service_time}")
        else:
            logger.warning("❌ No service_time found - would use default logic")
            if hasattr(to_loc, 'id'):
                if 'depot' in to_loc.id.lower():
                    service_time = 5
                else:
                    service_time = 15
                logger.info(f"⚠️  Using default service_time: {service_time}")
            else:
                logger.error("❌ Location has no id attribute either!")

if __name__ == "__main__":
    debug_location_access()
