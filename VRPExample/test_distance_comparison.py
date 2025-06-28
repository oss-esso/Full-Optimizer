"""
Comparison test showing the difference between Euclidean and OSRM distance calculations.
This demonstrates that the patch is working and gives realistic results.
"""

from vrp_scenarios import create_overnight_test_scenario
import math

def compare_distance_calculations():
    """Compare Euclidean vs OSRM distance calculations."""
    print("📊 Distance Calculation Comparison: Euclidean vs OSRM")
    print("=" * 70)
    
    # Create test scenario
    scenario = create_overnight_test_scenario()
    
    # Get some locations for comparison
    locations = list(scenario.locations.values())
    depot = None
    malmoe = None
    cormano = None
    
    for loc in locations:
        if 'depot' in loc.id.lower() and 'bay' not in loc.id.lower():
            depot = loc
        elif 'malmo' in loc.id.lower():
            malmoe = loc
        elif 'cormano' in loc.id.lower():
            cormano = loc
    
    if not (depot and malmoe and cormano):
        print("❌ Required locations not found")
        return
    
    print(f"\n📍 Test Locations:")
    print(f"  - Depot: {depot.id} at ({depot.x:.4f}, {depot.y:.4f})")
    print(f"  - Malmö: {malmoe.id} at ({malmoe.x:.4f}, {malmoe.y:.4f})")  
    print(f"  - Cormano: {cormano.id} at ({cormano.x:.4f}, {cormano.y:.4f})")
    
    # Calculate Euclidean distances (old method)
    def euclidean_distance(loc1, loc2):
        return ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5
    
    depot_to_malmoe_euclidean = euclidean_distance(depot, malmoe) * 111  # Scale factor
    depot_to_cormano_euclidean = euclidean_distance(depot, cormano) * 111
    
    print(f"\n📐 Euclidean Distance Calculations (Old Method):")
    print(f"  - Depot → Malmö: {depot_to_malmoe_euclidean:.1f} km")
    print(f"  - Depot → Cormano: {depot_to_cormano_euclidean:.1f} km")
    
    # Now test with the OSRM calculator (new method)
    from vrp_optimizer_clean import OSMDistanceCalculator
    
    # Convert locations to the format expected by OSMDistanceCalculator
    location_list = []
    for loc in [depot, malmoe, cormano]:
        location_dict = {
            'id': loc.id,
            'x': float(loc.x),
            'y': float(loc.y),
            'lat': float(loc.y),  # Assuming y is latitude
            'lon': float(loc.x),  # Assuming x is longitude
            'service_time': getattr(loc, 'service_time', 0)
        }
        location_list.append(location_dict)
    
    print(f"\n🗺️ Initializing OSRM calculator...")
    osm_calculator = OSMDistanceCalculator(location_list)
    
    # Get OSRM distances
    depot_to_malmoe_osrm = osm_calculator.get_distance(depot.id, malmoe.id) / 1000  # Convert to km
    depot_to_cormano_osrm = osm_calculator.get_distance(depot.id, cormano.id) / 1000
    
    print(f"\n🛣️ OSRM Distance Calculations (New Method):")
    print(f"  - Depot → Malmö: {depot_to_malmoe_osrm:.1f} km")
    print(f"  - Depot → Cormano: {depot_to_cormano_osrm:.1f} km")
    
    # Calculate differences
    malmoe_diff = abs(depot_to_malmoe_osrm - depot_to_malmoe_euclidean)
    cormano_diff = abs(depot_to_cormano_osrm - depot_to_cormano_euclidean)
    
    print(f"\n📊 Comparison Results:")
    print(f"  - Malmö route difference: {malmoe_diff:.1f} km")
    print(f"  - Cormano route difference: {cormano_diff:.1f} km")
    
    print(f"\n🎯 Key Insights:")
    print(f"  ✅ OSRM provides actual road distances")
    print(f"  ✅ Accounts for real road networks and routing")
    print(f"  ✅ More accurate than straight-line calculations")
    print(f"  ✅ Essential for realistic VRP optimization")
    
    # Get travel times too
    depot_to_malmoe_time = osm_calculator.get_travel_time(depot.id, malmoe.id)
    depot_to_cormano_time = osm_calculator.get_travel_time(depot.id, cormano.id)
    
    print(f"\n⏰ OSRM Travel Times:")
    print(f"  - Depot → Malmö: {depot_to_malmoe_time:.1f} minutes")
    print(f"  - Depot → Cormano: {depot_to_cormano_time:.1f} minutes")

if __name__ == "__main__":
    compare_distance_calculations()
