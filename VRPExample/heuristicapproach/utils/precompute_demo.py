#!/usr/bin/env python3
"""
OSRM Route Pre-computation CLI Tool - Working Version

This is a simplified but functional version of the pre-computation tool that works 
with manually specified coordinates. The infrastructure is complete and can be 
extended to work with any scenario data format.

Usage:
    python precompute_demo.py --dry-run
    python precompute_demo.py --preset asti_area
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))

def demo_osrm_precomputation():
    """Demo the OSRM pre-computation functionality with sample locations."""
    print("🚀 OSRM Pre-computation Demo")
    print("=" * 50)
    
    try:
        from osrm_utils import (init_route_cache_db, query_osrm_and_cache, 
                               get_cache_stats, is_route_cached)
        from precompute_routes import LocationInfo
        
        # Sample locations around Asti, Italy (similar to our VRP scenarios)
        sample_locations = [
            LocationInfo("depot", 44.8985, 8.2061, "depot"),               # Asti depot
            LocationInfo("torino", 45.0703, 7.6869, "delivery"),           # Torino
            LocationInfo("milano", 45.4642, 9.1900, "delivery"),           # Milano  
            LocationInfo("genova", 44.4056, 8.9463, "delivery"),           # Genova
            LocationInfo("alessandria", 44.9132, 8.6140, "delivery"),      # Alessandria
            LocationInfo("pickup_1", 44.9200, 8.3000, "pickup"),           # Near Asti pickup
            LocationInfo("pickup_2", 44.8500, 8.1500, "pickup"),           # Near Asti pickup
        ]
        
        print(f"📍 Demo locations: {len(sample_locations)}")
        for loc in sample_locations:
            print(f"   - {loc.id} ({loc.location_type}): {loc.lat:.4f}, {loc.lon:.4f}")
        
        total_pairs = len(sample_locations) * (len(sample_locations) - 1)
        print(f"🔗 Total route pairs to pre-compute: {total_pairs}")
        
        # Initialize database
        db_path = str(Path(__file__).parent.parent / "demo_osrm_cache.db")
        init_route_cache_db(db_path)
        print(f"📁 Demo database: {db_path}")
        
        # Check current cache status
        stats = get_cache_stats(db_path)
        print(f"💾 Current cache: {stats['total_routes']} routes")
        
        # Demo: Pre-compute a few sample routes (not all pairs to avoid OSRM abuse)
        sample_pairs = [
            (sample_locations[0], sample_locations[1]),  # depot -> torino
            (sample_locations[0], sample_locations[2]),  # depot -> milano
            (sample_locations[5], sample_locations[0]),  # pickup_1 -> depot
        ]
        
        print(f"\n🔄 Demo: Pre-computing {len(sample_pairs)} sample routes...")
        print("⚠️  Note: Using public OSRM server - please be respectful with requests")
        
        osrm_url = "https://router.project-osrm.org"
        success_count = 0
        
        for i, (start_loc, end_loc) in enumerate(sample_pairs):
            print(f"\n📡 Route {i+1}/{len(sample_pairs)}: {start_loc.id} -> {end_loc.id}")
            
            # Check if already cached
            if is_route_cached(db_path, start_loc.id, end_loc.id):
                print("💾 Already cached, skipping")
                success_count += 1
                continue
            
            # Query OSRM and cache
            route_data = query_osrm_and_cache(
                osrm_url=osrm_url,
                db_path=db_path,
                start_coords=(start_loc.lon, start_loc.lat),
                end_coords=(end_loc.lon, end_loc.lat),
                start_node_id=start_loc.id,
                end_node_id=end_loc.id
            )
            
            if route_data:
                print(f"✅ Success: {route_data['distance_km']:.1f} km, {route_data['duration_minutes']:.1f} min")
                success_count += 1
            else:
                print("❌ Failed")
            
            # Rate limiting for public OSRM
            time.sleep(1.0)
        
        # Final stats
        final_stats = get_cache_stats(db_path)
        print(f"\n📊 Final Results:")
        print(f"✅ Successfully pre-computed: {success_count}/{len(sample_pairs)} routes")
        print(f"💾 Total cache size: {final_stats['total_routes']} routes")
        print(f"💾 Database location: {db_path}")
        
        print(f"\n✨ Pre-computation demo completed!")
        print(f"🔧 Next steps:")
        print(f"   1. Integrate with your VRP scenario loader")
        print(f"   2. Set up local OSRM server for production use")  
        print(f"   3. Run full pre-computation: python precompute_routes.py --scenario furgoni")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure osrm_utils.py is in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

def main():
    """Main function with CLI arguments."""
    parser = argparse.ArgumentParser(description="OSRM Pre-computation Demo")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making OSRM calls')
    parser.add_argument('--preset', choices=['asti_area'], default='asti_area',
                       help='Use predefined location set')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🏃 Dry run mode - showing demo without OSRM calls")
        print("This would pre-compute routes for sample locations around Asti, Italy")
        print("Run without --dry-run to actually test OSRM integration")
        return
    
    demo_osrm_precomputation()

if __name__ == '__main__':
    main()
