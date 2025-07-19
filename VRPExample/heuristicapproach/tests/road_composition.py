import requests
from collections import defaultdict
import json

# --- Data Setup ---
# Based on the OSRM car.lua profile, simplified and structured as a Python dict.
# Speeds are in km/h.
CAR_SPEEDS = {
    'motorway': 100,
    'trunk': 85,
    'primary': 75,
    'secondary': 65,
    'tertiary': 50,
    'unclassified': 30,
    'residential': 30,
    'service': 20,
    'motorway_link': 50,
    'trunk_link': 45,
    'primary_link': 35,
    'secondary_link': 30,
    'tertiary_link': 25,
    'living_street': 15,
}

# Speed ratios for different truck types compared to a car.
# A value of 0.8 means the truck travels at 80% of the car's speed on that road type.
TRUCK_SPEEDS = {
    'standard': {
        'motorway': 130,      # Light trucks: 80 km/h, Cars: ~130 km/h
        'trunk': 100,         # Light trucks: 70 km/h, Cars: ~100 km/h
        'primary': 90,        # Light trucks: 60 km/h, Cars: ~90 km/h
        'secondary': 70,      # Light trucks: 50 km/h, Cars: ~70 km/h
        'tertiary': 60,       # Light trucks: 45 km/h, Cars: ~60 km/h
        'residential': 40,    # Light trucks: 30 km/h, Cars: ~50 km/h
        'service': 25,        # Light trucks: 25 km/h, Cars: ~30 km/h
        'unclassified': 30,
        'motorway_link': 50,
        'trunk_link': 45,
        'primary_link': 35,
        'secondary_link': 30,
        'tertiary_link': 25,
        'living_street': 15,
    },
    'heavy': {
        'motorway': 90,      # Heavy trucks: 70 km/h, Cars: ~130 km/h
        'trunk': 80,         # Heavy trucks: 60 km/h, Cars: ~100 km/h
        'primary': 70,        # Heavy trucks: 50 km/h, Cars: ~90 km/h
        'secondary': 70,      # Heavy trucks: 40 km/h, Cars: ~70 km/h
        'tertiary': 60,       # Heavy trucks: 35 km/h, Cars: ~60 km/h
        'residential': 25,    # Heavy trucks: 25 km/h, Cars: ~50 km/h
        'service': 15,        # Heavy trucks: 20 km/h, Cars: ~30 km/h
        'unclassified': 30,
        'motorway_link': 50,
        'trunk_link': 45,
        'primary_link': 35,
        'secondary_link': 30,
        'tertiary_link': 25,
        'living_street': 15,
        }
}

# Country-specific speed limit exceptions (in km/h).
# NOTE: This data is for reference and is typically used by the OSRM backend during
# map data processing. It cannot be used directly for API response analysis, as the
# API does not provide the country code for each road segment.
COUNTRY_SPECIFIC_SPEEDS = {
    "at:rural": 100,
    "at:trunk": 100,
    "be:motorway": 120,
    "de:living_street": 7,
    "de:rural": 100,
    "gb:nsl_single": 96.5,   # 60 mph
    "gb:nsl_dual": 112.6,  # 70 mph
    "gb:motorway": 112.6,  # 70 mph
    # ... and so on for other countries
}

def _infer_road_type_from_speed(speed_kmh):
    """Finds the closest matching road type from our CAR_SPEEDS profile."""
    return min(CAR_SPEEDS, key=lambda road: abs(CAR_SPEEDS[road] - speed_kmh))

def get_route_composition_and_estimated_times(start_coords, end_coords, vehicle_type='car'):
    """
    Infers road composition and estimates travel time using a weighted average speed ratio.
    """
    print(f"\n--- Analyzing Route for: {vehicle_type.replace('_', ' ').title()} ---")
    osrm_url = "http://router.project-osrm.org"
    coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    url = f"{osrm_url}/route/v1/driving/{coords_str}"
    params = {'annotations': 'true'}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get('code') != 'Ok' or not data['routes']:
            print(f"OSRM could not find a route: {data.get('code')}")
            return None

        road_distances = defaultdict(float)
        total_car_duration_seconds = data['routes'][0]['duration']
        total_distance_m = data['routes'][0]['distance']

        if total_distance_m == 0:
            return {}

        annotation = data['routes'][0]['legs'][0]['annotation']
        distances = annotation['distance']
        durations = annotation['duration']

        for i in range(len(distances)):
            if durations[i] > 0:
                speed_kmh = (distances[i] / durations[i]) * 3.6
                road_type = _infer_road_type_from_speed(speed_kmh)
                road_distances[road_type] += distances[i]

        print("Inferred Road Composition:")
        composition_percent = {rt: (dist / total_distance_m) * 100 for rt, dist in road_distances.items()}
        for rt, perc in sorted(composition_percent.items(), key=lambda item: item[1], reverse=True):
            print(f"- {rt.capitalize():<15}: {perc:.2f}%")

        car_minutes = total_car_duration_seconds / 60.0
        print(f"\nEstimated Car Time: {car_minutes:.2f} minutes")

        if vehicle_type != 'car':
            vehicle_speeds = TRUCK_SPEEDS.get(vehicle_type)
            if not vehicle_speeds:
                print(f"Warning: Unknown vehicle type '{vehicle_type}'.")
                return

            # Calculate truck time using: sum_over_all_compositions car_time * composition_percent * (car_speed/truck_speed)
            vehicle_minutes = 0
            for rt, perc in composition_percent.items():
                car_speed = CAR_SPEEDS.get(rt)  # fallback to unclassified speed
                truck_speed = vehicle_speeds.get(rt)
                
                if truck_speed > 0:
                    segment_time_adjustment = (perc / 100) * (car_speed / truck_speed)
                    vehicle_minutes += car_minutes * segment_time_adjustment
                else:
                    print(f"Warning: Invalid truck speed for road type '{rt}'.")
                    return

            print(f"Estimated {vehicle_type.replace('_', ' ').title()} Time: {vehicle_minutes:.2f} minutes")
            print(f"Difference: {vehicle_minutes - car_minutes:+.2f} minutes")

        return {
            'composition': composition_percent,
            'car_time_minutes': car_minutes
        }

    except requests.exceptions.RequestException as e:
        print(f"Error calling OSRM: {e}")
        return None

if __name__ == "__main__":
    milan = (9.18951, 45.46427)
    rome = (12.49637, 41.90278)
    
    get_route_composition_and_estimated_times(milan, rome, vehicle_type='car')
    get_route_composition_and_estimated_times(milan, rome, vehicle_type='standard')
    get_route_composition_and_estimated_times(milan, rome, vehicle_type='heavy')


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
def parse_route_result_objects(start_coords, end_coords):
    """
    Demonstrates parsing the detailed result objects from a route response.
    """
    print("\n--- 7. Parsing Detailed Result Objects ---")
    coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
    params = {'steps': 'true', 'annotations': 'true', 'overview': 'full', 'geometries': 'geojson'}
    data = make_request('route', coords_str, params)
    if not data: return

    route = data['routes'][0]
    leg = route['legs'][0]
    print(f"    Leg: summary='{leg.get('summary')}', distance={leg['distance']}m, duration={leg['duration']}s, steps={len(leg.get('steps',[]))}")

    if 'annotation' in leg:
        ann = leg['annotation']
        # Collect speed data for plotting
        speeds = []
        for i in range(len(ann.get('distance', []))):
            #print(f"    Step {i+1}:")
            #print(f"        Distance: {ann['distance'][i]}m")
            #print(f"        Duration: {ann['duration'][i]}s")
            if 'speed' in ann:
                speed_kmh = ann['speed'][i] * 3.6  # Convert m/s to km/h
                speeds.append(speed_kmh)
                #print(f"        Speed: {speed_kmh:.2f} km/h")
            
        # Create and display a plot of speed vs step
        #if 'speed' in ann and speeds:
            #import matplotlib.pyplot as plt
            #plt.figure(figsize=(10, 5))
            #plt.plot(range(1, len(speeds) + 1), speeds, marker='o')
            #plt.title('Speed vs Step')
            #plt.xlabel('Step Number')
            #plt.ylabel('Speed (km/h)')
            #plt.grid(True)
            #plt.tight_layout()
            #plt.show()
        if 'nodes' in ann:
            print(f"        OSM Node IDs included: True (count: {len(ann.get('nodes',[]))})")

    


parse_route_result_objects(milan, rome)