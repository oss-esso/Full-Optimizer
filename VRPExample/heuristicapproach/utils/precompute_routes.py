#!/usr/bin/env python3
"""
OSRM Route Pre-computation Script

This script pre-populates the moda_routes.db cache with OSRM data for all unique 
location pairs in a given scenario. This turns expensive online OSRM calls into 
a one-time offline pre-computation step, drastically reducing the runtime of the 
main heuristic solver.

Usage:
    python precompute_routes.py --scenario furgoni
    python precompute_routes.py --excel-file ../src/furgoni.xlsx
    python precompute_routes.py --scenario furgoni --osrm-url http://localhost:5000

Features:
- Extracts all unique location coordinates from scenarios
- Generates all possible pairs of locations (N*(N-1) combinations)
- Pre-populates OSRM route cache with distance, duration, and road composition
- Supports multiple scenario formats (scenario functions, Excel files)
- Progress tracking and error handling
- Configurable OSRM server URL
- Batch processing with rate limiting
"""

import argparse
import sys
import os
import time
import sqlite3
import itertools
from pathlib import Path
from typing import List, Tuple, Set, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'algo'))
sys.path.append(str(Path(__file__).parent.parent.parent))  # For VRP models
sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # For root level

@dataclass
class LocationInfo:
    """Information about a unique location in the scenario."""
    id: str
    lat: float
    lon: float
    location_type: str  # 'depot', 'pickup', 'delivery', 'depot_bay', 'pickup_bay'

class RoutePrecomputer:
    """Pre-computes and caches OSRM routes for all location pairs in a scenario."""
    
    def __init__(self, osrm_url: str = "https://router.project-osrm.org", 
                 db_path: str = None, batch_size: int = 50, 
                 rate_limit_delay: float = 0.1):
        """
        Initialize the route pre-computer.
        
        Args:
            osrm_url: OSRM server URL
            db_path: Path to SQLite cache database
            batch_size: Number of requests to process in each batch
            rate_limit_delay: Delay between requests (seconds)
        """
        self.osrm_url = osrm_url
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        
        # Set default database path
        if db_path is None:
            self.db_path = str(Path(__file__).parent.parent / "moda_routes.db")
        else:
            self.db_path = db_path
            
        self.locations: List[LocationInfo] = []
        self.total_pairs = 0
        self.processed_pairs = 0
        self.failed_pairs = 0
        self.cache_hits = 0
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize the SQLite database with the route cache schema."""
        from osrm_utils import init_route_cache_db
        init_route_cache_db(self.db_path)
        print(f"📁 Database initialized at: {self.db_path}")
        
    def load_scenario_from_function(self, scenario_name: str) -> List[LocationInfo]:
        """Load scenario using scenario creation functions."""
        try:
            from moda_scenarios import create_furgoni_scenario
            
            if scenario_name.lower() == 'furgoni':
                print(f"🔄 Loading scenario: {scenario_name}")
                vrp_instance = create_furgoni_scenario()
                return self._extract_locations_from_vrp_instance(vrp_instance)
            else:
                raise ValueError(f"Unknown scenario: {scenario_name}")
                
        except ImportError as e:
            print(f"❌ Error importing scenario functions: {e}")
            sys.exit(1)
            
    def load_scenario_from_excel(self, excel_path: str) -> List[LocationInfo]:
        """Load scenario from Excel file."""
        try:
            print(f"📁 Loading scenario from Excel file: {excel_path}")
            
            # Import the scenario creator
            import sys
            from pathlib import Path
            
            # Add parent directories for imports
            current_dir = Path(__file__).parent
            src_dir = current_dir.parent / "src"
            sys.path.insert(0, str(src_dir))
            
            from scenario_creator import create_scenario_from_excel
            
            # Create the scenario
            orders, vehicles, drivers = create_scenario_from_excel(excel_path)
            print(f"✅ Loaded scenario with {len(orders)} orders, {len(vehicles)} vehicles, and {len(drivers)} drivers")
            
            # Extract unique locations
            locations = []
            seen_coords = set()
            location_id_counter = 0
            
            # Add depot (assuming it's consistent across all scenarios)
            depot_lat, depot_lon = 44.89149000, 8.20317000  # DEPOT_BAY_ASTI coordinates
            depot_key = (depot_lat, depot_lon)
            if depot_key not in seen_coords:
                locations.append(LocationInfo(
                    id="DEPOT_BAY_ASTI",
                    lat=depot_lat,
                    lon=depot_lon,
                    location_type="depot"
                ))
                seen_coords.add(depot_key)
            
            # Extract order locations
            for order in orders:
                if hasattr(order, 'tasks'):
                    for task in order.tasks:
                        if hasattr(task, 'location') and hasattr(task.location, 'latitude') and hasattr(task.location, 'longitude'):
                            lat = float(task.location.latitude)
                            lon = float(task.location.longitude)
                            coord_key = (lat, lon)
                            
                            if coord_key not in seen_coords:
                                # Use location name if available, otherwise generate ID
                                if hasattr(task.location, 'name') and task.location.name:
                                    location_id = task.location.name
                                else:
                                    location_id = f"location_{location_id_counter}"
                                    location_id_counter += 1
                                
                                # Determine location type based on task
                                location_type = "pickup" if hasattr(task, 'task_type') and 'pickup' in str(task.task_type).lower() else "delivery"
                                
                                locations.append(LocationInfo(
                                    id=location_id,
                                    lat=lat,
                                    lon=lon,
                                    location_type=location_type
                                ))
                                seen_coords.add(coord_key)
            
            print(f"✅ Extracted {len(locations)} unique locations")
            self.locations = locations
            return locations
            
        except Exception as e:
            print(f"❌ Error loading scenario from Excel: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def _extract_locations_from_vrp_instance(self, vrp_instance) -> List[LocationInfo]:
        """Extract all unique locations from a VRP instance."""
        locations = []
        seen_coords = set()
        
        print(f"🔍 Debug: VRP instance attributes: {dir(vrp_instance)}")
        
        # Extract depot locations
        if hasattr(vrp_instance, 'depot') and vrp_instance.depot:
            depot = vrp_instance.depot
            print(f"🔍 Debug: Depot attributes: {dir(depot)}")
            if hasattr(depot, 'lat') and hasattr(depot, 'lon') and depot.lat is not None:
                coord_key = (depot.lat, depot.lon)
                if coord_key not in seen_coords:
                    locations.append(LocationInfo(
                        id=depot.id,
                        lat=depot.lat,
                        lon=depot.lon,
                        location_type='depot'
                    ))
                    seen_coords.add(coord_key)
                    print(f"✅ Added depot: {depot.id} at {depot.lat}, {depot.lon}")
        
        # Extract locations from all locations list
        if hasattr(vrp_instance, 'locations'):
            print(f"🔍 Debug: Found {len(vrp_instance.locations)} locations")
            
            # Debug first few locations
            for i in range(min(3, len(vrp_instance.locations))):
                location = vrp_instance.locations[i]
                print(f"🔍 Debug: Location {i} attributes: {dir(location)}")
                print(f"🔍 Debug: Location {i} id: {getattr(location, 'id', 'NO_ID')}")
                print(f"🔍 Debug: Location {i} lat: {getattr(location, 'lat', 'NO_LAT')}")
                print(f"🔍 Debug: Location {i} lon: {getattr(location, 'lon', 'NO_LON')}")
                
            for location in vrp_instance.locations:
                if hasattr(location, 'lat') and hasattr(location, 'lon') and location.lat is not None and location.lon is not None:
                    coord_key = (location.lat, location.lon)
                    if coord_key not in seen_coords:
                        # Determine location type based on ID or other attributes
                        location_type = 'location'
                        if hasattr(location, 'id') and location.id:
                            if 'depot' in location.id.lower():
                                location_type = 'depot' 
                            elif 'pickup' in location.id.lower():
                                location_type = 'pickup'
                            elif 'bay' in location.id.lower():
                                location_type = 'depot_bay'
                            elif 'delivery' in location.id.lower() or any(x in location.id.lower() for x in ['_it', '_fr', '_de', '_se', '_ro', '_ge', '_at', '_bg', '_lo', '_an', '_vi']):
                                location_type = 'delivery'
                                
                        locations.append(LocationInfo(
                            id=location.id,
                            lat=location.lat,
                            lon=location.lon,
                            location_type=location_type
                        ))
                        seen_coords.add(coord_key)
        
        print(f"📍 Extracted {len(locations)} unique locations from scenario")
        return locations
        
    def _is_route_cached(self, start_id: str, end_id: str) -> bool:
        """Check if route is already cached in database."""
        from osrm_utils import is_route_cached
        return is_route_cached(self.db_path, start_id, end_id)
        
    def _query_osrm_and_cache(self, start_loc: LocationInfo, end_loc: LocationInfo) -> bool:
        """Query OSRM for a route and cache the results."""
        try:
            # Import OSRM utilities
            from osrm_utils import query_osrm_and_cache
            
            # Query OSRM and cache the results
            route_data = query_osrm_and_cache(
                osrm_url=self.osrm_url,
                db_path=self.db_path,
                start_coords=(start_loc.lon, start_loc.lat),
                end_coords=(end_loc.lon, end_loc.lat),
                start_node_id=start_loc.id,
                end_node_id=end_loc.id
            )
            
            return route_data is not None
            
        except Exception as e:
            print(f"❌ OSRM query failed for {start_loc.id} -> {end_loc.id}: {e}")
            return False
            
    def precompute_all_routes(self):
        """Pre-compute routes for all unique location pairs."""
        if not self.locations:
            print("❌ No locations loaded. Load a scenario first.")
            return
            
        # Generate all unique pairs (including both directions)
        location_pairs = []
        for start_loc in self.locations:
            for end_loc in self.locations:
                if start_loc.id != end_loc.id:  # Skip self-loops
                    location_pairs.append((start_loc, end_loc))
                    
        self.total_pairs = len(location_pairs)
        print(f"🔄 Pre-computing routes for {self.total_pairs} location pairs...")
        print(f"📡 OSRM server: {self.osrm_url}")
        print(f"💾 Cache database: {self.db_path}")
        
        start_time = time.time()
        
        # Process pairs in batches
        for i in range(0, len(location_pairs), self.batch_size):
            batch = location_pairs[i:i + self.batch_size]
            self._process_batch(batch, i + 1)
            
            # Progress update
            progress_pct = (i + len(batch)) / self.total_pairs * 100
            elapsed_time = time.time() - start_time
            eta = elapsed_time / ((i + len(batch)) / self.total_pairs) - elapsed_time if i > 0 else 0
            
            print(f"📊 Progress: {progress_pct:.1f}% | "
                  f"Processed: {self.processed_pairs} | "
                  f"Failed: {self.failed_pairs} | "
                  f"Cache hits: {self.cache_hits} | "
                  f"ETA: {eta:.0f}s")
                  
        total_time = time.time() - start_time
        success_rate = (self.processed_pairs / self.total_pairs) * 100 if self.total_pairs > 0 else 0
        
        print(f"\n✅ Pre-computation completed!")
        print(f"📊 Total pairs: {self.total_pairs}")
        print(f"✅ Successfully processed: {self.processed_pairs}")
        print(f"❌ Failed: {self.failed_pairs}")
        print(f"💾 Cache hits: {self.cache_hits}")
        print(f"📈 Success rate: {success_rate:.1f}%")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"⚡ Average time per pair: {total_time/self.total_pairs:.2f}s")
        
    def _process_batch(self, batch: List[Tuple[LocationInfo, LocationInfo]], batch_num: int):
        """Process a batch of location pairs."""
        for start_loc, end_loc in batch:
            # Check if already cached
            if self._is_route_cached(start_loc.id, end_loc.id):
                self.cache_hits += 1
                self.processed_pairs += 1
                continue
                
            # Query OSRM and cache
            if self._query_osrm_and_cache(start_loc, end_loc):
                self.processed_pairs += 1
            else:
                self.failed_pairs += 1
                
            # Rate limiting
            time.sleep(self.rate_limit_delay)

def main():
    """Main function to handle command line arguments and run pre-computation."""
    parser = argparse.ArgumentParser(
        description="Pre-compute OSRM routes for VRP scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python precompute_routes.py --scenario furgoni
  python precompute_routes.py --excel-file ../src/furgoni.xlsx
  python precompute_routes.py --scenario furgoni --osrm-url http://localhost:5000
  python precompute_routes.py --scenario furgoni --batch-size 100 --rate-limit 0.05
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--scenario', type=str, 
                           help='Scenario name (e.g., "furgoni")')
    input_group.add_argument('--excel-file', type=str,
                           help='Path to Excel scenario file')
    
    # Configuration options
    parser.add_argument('--osrm-url', type=str,
                       default='https://router.project-osrm.org',
                       help='OSRM server URL (default: public OSRM)')
    parser.add_argument('--db-path', type=str,
                       help='Path to SQLite cache database (default: ../moda_routes.db)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Number of requests per batch (default: 50)')
    parser.add_argument('--rate-limit', type=float, default=0.1,
                       help='Delay between requests in seconds (default: 0.1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be pre-computed without actually doing it')
    
    args = parser.parse_args()
    
    print("🚀 OSRM Route Pre-computation Tool")
    print("=" * 50)
    
    # Initialize pre-computer
    precomputer = RoutePrecomputer(
        osrm_url=args.osrm_url,
        db_path=args.db_path,
        batch_size=args.batch_size,
        rate_limit_delay=args.rate_limit
    )
    
    # Load scenario
    try:
        if args.scenario:
            precomputer.locations = precomputer.load_scenario_from_function(args.scenario)
        elif args.excel_file:
            precomputer.locations = precomputer.load_scenario_from_excel(args.excel_file)
            
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        sys.exit(1)
    
    if not precomputer.locations:
        print("❌ No locations found in scenario")
        sys.exit(1)
        
    # Show summary
    print(f"📍 Locations found: {len(precomputer.locations)}")
    location_types = {}
    for loc in precomputer.locations:
        location_types[loc.location_type] = location_types.get(loc.location_type, 0) + 1
    
    for loc_type, count in location_types.items():
        print(f"   - {loc_type}: {count}")
        
    total_pairs = len(precomputer.locations) * (len(precomputer.locations) - 1)
    print(f"🔗 Total route pairs to pre-compute: {total_pairs:,}")
    
    if args.dry_run:
        print("🏃 Dry run mode - not actually pre-computing routes")
        return
        
    # Confirm before proceeding
    try:
        confirm = input(f"\n📡 Proceed with pre-computation using {args.osrm_url}? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("❌ Pre-computation cancelled")
            return
    except KeyboardInterrupt:
        print("\n❌ Pre-computation cancelled")
        return
        
    # Run pre-computation
    try:
        precomputer.precompute_all_routes()
    except KeyboardInterrupt:
        print("\n⚠️  Pre-computation interrupted by user")
        print(f"📊 Partial results: {precomputer.processed_pairs}/{precomputer.total_pairs} pairs processed")
    except Exception as e:
        print(f"\n❌ Pre-computation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
