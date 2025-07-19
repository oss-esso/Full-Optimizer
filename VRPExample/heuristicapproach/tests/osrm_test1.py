import requests
import json

# Base URL for the public OSRM demo server
OSRM_BASE_URL = "http://router.project-osrm.org"

def make_request(endpoint, coordinates, params=None, is_json=True):
    """
    Helper function to make a request to the OSRM API.

    :param endpoint: The API service endpoint (e.g., 'route', 'nearest').
    :param coordinates: A string of semicolon-separated lon,lat pairs.
    :param params: A dictionary of additional query parameters.
    :param is_json: Whether to expect a JSON response.
    :return: The JSON response or raw content from the API, or None on error.
    """
    if params is None:
        params = {}
    
    url = f"{OSRM_BASE_URL}/{endpoint}/v1/driving/{coordinates}"
    
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        
        if is_json:
            data = response.json()
            if data.get('code') == 'Ok':
                print(f"Successfully called '{endpoint}' service.")
                return data
            else:
                print(f"OSRM API error for '{endpoint}': {data.get('message', 'No message')}")
                return None
        else:
            # For non-JSON (like FlatBuffers), return raw content
            return response.content

    except requests.exceptions.RequestException as e:
        print(f"Network error calling OSRM API for '{endpoint}': {e}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response from '{endpoint}'.")
        return None

# 1. Route Service
def get_route(start_coords, end_coords, alternatives=False, steps=False, annotations=False, overview='simplified'):
    """
    Demonstrates the 'route' service with various options.
    """
    print("\n--- Testing Route Service ---")
    coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    params = {
        'alternatives': str(alternatives).lower(),
        'steps': str(steps).lower(),
        'annotations': str(annotations).lower(),
        'overview': overview,
        'geometries': 'geojson' if overview != 'false' else ''
    }
    
    data = make_request('route', coords_str, params)
    if data:
        print(f"Found {len(data.get('routes', []))} route(s).")

# 2. Nearest Service
def get_nearest(coords):
    """
    Demonstrates the 'nearest' service.
    """
    print("\n--- Testing Nearest Service ---")
    coords_str = f"{coords[0]},{coords[1]}"
    data = make_request('nearest', coords_str, {'number': 3})
    if data:
        print(f"Found {len(data.get('waypoints', []))} nearest waypoints.")

# 3. Table Service
def get_table(points, annotations='duration'):
    """
    Demonstrates the 'table' service.
    """
    print(f"\n--- Testing Table Service (Annotation: {annotations}) ---")
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in points])
    data = make_request('table', coords_str, {'annotations': annotations})
    if data:
        print(f"Successfully retrieved {annotations} matrix.")


# 5. Trip Service
def get_trip(points):
    """
    Demonstrates the 'trip' service.
    """
    print("\n--- Testing Trip Service ---")
    coords_str = ";".join([f"{lon},{lat}" for lon, lat in points])
    data = make_request('trip', coords_str)
    if data:
        print(f"Found {len(data.get('trips', []))} trip solution(s).")

# 6. Tile Service
def get_tile_url(lon, lat, zoom):
    """
    Demonstrates constructing a 'tile' service URL.
    """
    print("\n--- Testing Tile Service ---")
    import math
    def deg2num(lat_deg, lon_deg, zoom):
      lat_rad = math.radians(lat_deg)
      n = 2.0 ** zoom
      xtile = int((lon_deg + 180.0) / 360.0 * n)
      ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
      return (xtile, ytile)
    x, y = deg2num(lat, lon, zoom)
    url = f"{OSRM_BASE_URL}/tile/v1/driving/{zoom}/{x}/{y}.mvt"
    print(f"Generated vector tile URL: {url}")

# 7. Detailed Result Object Parsing
def parse_route_result_objects(start_coords, end_coords):
    """
    Demonstrates parsing the detailed result objects from a route response.
    """
    print("\n--- 7. Parsing Detailed Result Objects ---")
    coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    params = {'steps': 'true', 'annotations': 'true', 'overview': 'full', 'geometries': 'geojson'}
    data = make_request('route', coords_str, params)
    if not data: return

    print("\n[Waypoint Objects]")
    for i, wp in enumerate(data.get('waypoints', [])):
        print(f"  Waypoint {i}: name='{wp.get('name')}', location={wp.get('location')}, hint='{wp.get('hint', '')[:10]}...')")

    print("\n[Route Object]")
    route = data['routes'][0]
    print(f"  Route: distance={route['distance']}m, duration={route['duration']}s, weight_name='{route['weight_name']}'")

    print("\n  [RouteLeg Object]")
    leg = route['legs'][0]
    print(f"    Leg: summary='{leg.get('summary')}', distance={leg['distance']}m, duration={leg['duration']}s, steps={len(leg.get('steps',[]))}")

    print("\n    [Annotation Object]")
    if 'annotation' in leg:
        ann = leg['annotation']
        # Displaying the first 5 values as an example
        print(f"      Annotation (showing first 5 values):")
        print(f"        Durations: {ann.get('duration',[])[:5]}")
        print(f"        Distances: {ann.get('distance',[])[:5]}")
        print(f"        Speeds (km/h): {[s * 3.6 for s in ann.get('speed', [])[:5]]}") # Convert m/s to km/h
        if 'nodes' in ann:
            print(f"        OSM Node IDs included: True (count: {len(ann.get('nodes',[]))})")

    print("\n    [RouteStep Object (First Step)]")
    step = leg['steps'][0]
    # The 'intersections' array often contains road classification
    road_class = 'unknown'
    if 'intersections' in step and len(step['intersections']) > 0:
        if 'classes' in step['intersections'][0]:
            road_class = step['intersections'][0]['classes'][0]

    print(f"      Step: name='{step.get('name')}', mode='{step.get('mode')}', distance={step['distance']}m")
    print(f"      -> Inferred Road Type: {road_class}")
    
    print("\n        [StepManeuver Object]")
    maneuver = step.get('maneuver', {})
    print(f"          Maneuver: type='{maneuver.get('type')}', modifier='{maneuver.get('modifier')}', location={maneuver.get('location')}")

    print("\n        [Lane Object (from first intersection)]")
    intersection = step.get('intersections', [{}])[0]
    if 'lanes' in intersection:
        print(f"          Intersection at {intersection.get('location')} has {len(intersection.get('lanes',[]))} lanes.")
        lane = intersection['lanes'][0]
        print(f"            Lane 0: valid={lane.get('valid')}, indications={lane.get('indications')}")
    else:
        print("          No lane data in this intersection.")

# 8. FlatBuffers Format

if __name__ == "__main__":
    milan = (9.18951, 45.46427)
    rome = (12.49637, 41.90278)
    florence = (11.25581, 43.76956)
    bologna = (11.34262, 44.49489)
    off_road_rome = (12.49, 41.9)
    # A realistic, short trace for matching within Rome, e.g., around the Colosseum
    noisy_trace = [
        (12.4923, 41.8902), (12.4930, 41.8915), 
        (12.4945, 41.8925), (12.4960, 41.8930)
    ]
    locations = [milan, florence, bologna, rome]

    # --- Service Demonstrations ---
    get_route(milan, rome, alternatives=True, steps=True)
    get_nearest(off_road_rome)
    get_table(locations, annotations='distance')
    get_trip(locations)
    get_tile_url(lon=12.49637, lat=41.90278, zoom=14)

    # --- Result Object & Format Demonstrations ---
    parse_route_result_objects(milan, florence) # A longer route is fine for JSON





