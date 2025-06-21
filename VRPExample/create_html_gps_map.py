#!/usr/bin/env python3
"""
Create HTML GPS map for MODA Small scenario (Asti-centered) using enhanced optimizer.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from VRPExample.vrp_scenarios import get_all_scenarios
from VRPExample.vrp_optimizer_enhanced_full import VRPOptimizerEnhanced
from VRPExample.vrp_map_visualization import create_all_map_visualizations
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_html_gps_map():
    """Create HTML GPS map for MODA Small scenario centered around Asti."""
    print("=" * 70)
    print("HTML GPS MAP GENERATION - MODA SMALL (ASTI-CENTERED)")
    print("=" * 70)
    
    # Load MODA_small scenario (already Asti-centered)
    try:
        scenarios = get_all_scenarios()
        scenario = scenarios['MODA_small']
        
        print(f"✅ Loaded scenario: {scenario.name}")
        print(f"   📊 {len(scenario.vehicles)} vehicles, {len(scenario.locations)} locations, {len(scenario.ride_requests)} requests")
        
        # Show some location details to confirm Asti centering
        print(f"\n📍 SCENARIO DETAILS:")
        depot_locations = [loc for loc_id, loc in scenario.locations.items() if 'depot' in loc_id.lower()]
        for depot in depot_locations:
            print(f"   Depot: {depot.id} at ({depot.y:.4f}, {depot.x:.4f})")
        
        sample_locations = list(scenario.locations.items())[:5]
        print(f"   Sample locations:")
        for loc_id, loc in sample_locations:
            print(f"     {loc_id}: ({loc.y:.4f}, {loc.x:.4f})")
            
    except Exception as e:
        print(f"❌ Failed to load scenario: {e}")
        return
    
    # Solve with enhanced optimizer
    print(f"\n🚀 Running enhanced VRP optimization...")
    optimizer = VRPOptimizerEnhanced()
    
    try:
        result = optimizer.solve(scenario, time_limit_seconds=120)
        
        if result['success']:
            print(f"✅ OPTIMIZATION SUCCESS!")
            print(f"   Strategy: {result.get('strategy_used', 'Unknown')}")
            print(f"   Vehicles used: {result['vehicles_used']}/{len(scenario.vehicles)}")
            print(f"   Solve time: {result['solve_time']:.2f}s")
            print(f"   Objective: {result['objective_value']}")
            
            # Show route details
            if 'routes' in result:
                print(f"\n📊 ROUTE SUMMARY:")
                for vehicle_id, route in result['routes'].items():
                    print(f"   {vehicle_id}: {len(route)} stops")
                    print(f"     Route: {' → '.join(route[:3])}...{' → '.join(route[-2:])}")
            
        else:
            print(f"❌ OPTIMIZATION FAILED: {result.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create HTML GPS maps
    print(f"\n🗺️  CREATING HTML GPS MAPS...")
    
    try:
        # Check dependencies
        print("📦 Checking map visualization dependencies...")
        
        try:
            import folium
            print("✅ folium available")
        except ImportError:
            print("❌ folium not available - cannot create HTML maps")
            return
            
        try:
            import requests
            print("✅ requests available")
        except ImportError:
            print("⚠️  requests not available - will use basic routing")
        
        # Create maps directory
        maps_dir = "maps"
        os.makedirs(maps_dir, exist_ok=True)
        
        # Generate all map visualizations
        from VRPExample.vrp_data_models import VRPResult
          # Convert our result to VRPResult format for the visualizer
        vrp_result = VRPResult(
            status="success",  # Required status parameter
            routes=result['routes'],
            objective_value=result['objective_value'],
            runtime=result['solve_time'] * 1000,  # Convert to ms
            metrics={
                'total_distance': sum(analysis.get('total_distance', 0) for analysis in result.get('route_analysis', [])),
                'vehicles_used': result['vehicles_used']
            }
        )
        
        print(f"📁 Creating map files in '{maps_dir}' directory...")
        
        # Create all map visualizations
        map_files = create_all_map_visualizations(
            instance=scenario,
            result=vrp_result,
            results_dir=maps_dir,
            scenario_name="MODA_small_Asti"
        )
        
        if map_files:
            print(f"✅ Successfully created {len(map_files)} map files:")
            for map_file in map_files:
                print(f"   📄 {map_file}")
                
            # Find and highlight the main HTML map
            html_files = [f for f in map_files if f.endswith('.html')]
            if html_files:
                main_html = html_files[0]
                full_path = os.path.abspath(main_html)
                print(f"\n🌐 MAIN HTML MAP: {main_html}")
                print(f"   Full path: {full_path}")
                print(f"   Open this file in your web browser to view the interactive map!")
                
                # Try to open the map automatically
                try:
                    import webbrowser
                    webbrowser.open(f"file://{full_path}")
                    print(f"🚀 Attempting to open map in default browser...")
                except:
                    print(f"💡 Please manually open the HTML file in your browser")
            
        else:
            print(f"⚠️  No map files were created")
            
    except Exception as e:
        print(f"❌ Error creating GPS maps: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: create simple matplotlib map
        print(f"\n📊 Creating fallback matplotlib map...")
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Plot locations
            for loc_id, location in scenario.locations.items():
                if 'depot' in loc_id.lower():
                    plt.scatter(location.x, location.y, c='red', s=200, marker='s', label='Depot')
                elif 'pickup' in loc_id.lower():
                    plt.scatter(location.x, location.y, c='green', s=80, marker='^', alpha=0.7)
                elif 'dropoff' in loc_id.lower():
                    plt.scatter(location.x, location.y, c='blue', s=80, marker='v', alpha=0.7)
            
            # Plot routes if available
            if 'routes' in result:
                colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
                for i, (vehicle_id, route) in enumerate(result['routes'].items()):
                    color = colors[i % len(colors)]
                    route_coords = []
                    for loc_id in route:
                        if loc_id in scenario.locations:
                            location = scenario.locations[loc_id]
                            route_coords.append((location.x, location.y))
                    
                    if len(route_coords) > 1:
                        for j in range(len(route_coords) - 1):
                            x1, y1 = route_coords[j]
                            x2, y2 = route_coords[j + 1]
                            plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
            
            plt.title(f'MODA Small Solution - Asti Area')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            fallback_file = 'moda_small_asti_fallback.png'
            plt.savefig(fallback_file, dpi=300, bbox_inches='tight')
            print(f"📁 Fallback map saved as: {fallback_file}")
            plt.close()
            
        except Exception as fallback_error:
            print(f"❌ Even fallback map creation failed: {fallback_error}")
    
    print(f"\n✅ HTML GPS map generation complete!")

if __name__ == "__main__":
    create_html_gps_map()
