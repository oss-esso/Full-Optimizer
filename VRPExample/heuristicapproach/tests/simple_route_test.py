"""
Simple test for route provider debugging
"""
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
algo_dir = os.path.join(current_dir, '..', 'algo')
sys.path.insert(0, algo_dir)

from route_provider import RouteProvider

def simple_test():
    print("Testing basic RouteProvider...")
    provider = RouteProvider(db_path="simple_test.db")
    
    milan_coords = (9.18951, 45.46427)
    rome_coords = (12.49637, 41.90278)
    
    try:
        result = provider.get_route_details("milan", "rome", milan_coords, rome_coords)
        print(f"Result type: {type(result)}")
        if result:
            print(f"Distance: {result.get('distance_km')}")
            print(f"Duration: {result.get('base_duration_minutes')}")
            print(f"Composition: {result.get('road_composition')}")
        else:
            print("No result returned")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    if os.path.exists("simple_test.db"):
        os.remove("simple_test.db")

if __name__ == "__main__":
    simple_test()
