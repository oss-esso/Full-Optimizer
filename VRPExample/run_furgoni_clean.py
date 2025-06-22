import logging
from vrp_scenarios import create_furgoni_scenario
from vrp_optimizer_clean import CleanVRPOptimizer

def main():
    logging.basicConfig(level=logging.INFO)
    print("🚚 Running Furgoni scenario with CleanVRPOptimizer")
    scenario = create_furgoni_scenario()

    # Prepare vehicles and locations as dicts
    vehicles = [{
        'id': v.id,
        'capacity': v.capacity,
        'start_location': v.depot_id,
        'end_location': v.depot_id,
        'max_time': getattr(v, 'max_time', 9 * 60)
    } for v in scenario.vehicles.values()]
    locations = [vars(l) for l in scenario.locations.values()]

    # Diagnostic: Print full distance matrix
    print("\nFull distance matrix:")
    loc_ids = list(scenario.locations.keys())
    matrix = scenario.distance_matrix
    header = '     ' + '  '.join([f'{i:2d}' for i in range(len(loc_ids))])
    print(header)
    for i in range(len(loc_ids)):
        row = matrix[i][:len(loc_ids)]
        print(f"{i:2d} {[f'{d:6.1f}' for d in row]}")
    print("Location index to ID mapping:")
    for i, loc_id in enumerate(loc_ids):
        print(f"  {i}: {loc_id}")

    optimizer = CleanVRPOptimizer(vehicles=vehicles, locations=locations)
    optimizer.ride_requests = scenario.ride_requests

    # Print ride requests and which are skipped due to conflicts (EXACT optimizer logic)
    print("\nRide requests and conflict analysis (EXACT optimizer logic):")
    ride_requests = scenario.ride_requests
    def get_dropoff(r):
        return getattr(r, 'dropoff', getattr(r, 'dropoff_id', None))
    def get_pickup(r):
        return getattr(r, 'pickup', getattr(r, 'pickup_id', None))
    def get_demand(r):
        return getattr(r, 'passengers', getattr(r, 'demand', getattr(r, 'weight', 0)))

    used_pairs = set()
    skipped_pairs = []
    kept_requests = []
    excluded_pairs = []  # For summary
    # 1. Add all depot→delivery pairs
    for idx, r in enumerate(ride_requests):
        if get_pickup(r) == 'depot' and get_dropoff(r) != 'depot':
            used_pairs.add((get_pickup(r), get_dropoff(r)))
            kept_requests.append(r)
    # 2. For pickup→depot, only allow the first, skip the rest
    depot_pickups = [r for r in ride_requests if get_dropoff(r) == 'depot' and get_pickup(r) != 'depot']
    if depot_pickups:
        first = depot_pickups[0]
        used_pairs.add((get_pickup(first), get_dropoff(first)))
        kept_requests.append(first)
        for r in depot_pickups[1:]:
            skipped_pairs.append((get_pickup(r), get_dropoff(r), get_demand(r)))
            excluded_pairs.append({'pickup': get_pickup(r), 'dropoff': get_dropoff(r), 'weight': get_demand(r), 'index': ride_requests.index(r)})
    # 3. For all other unique pairs (not depot→delivery or pickup→depot), keep only the largest demand per dropoff
    # Build dropoff->requests dict for non-depot dropoffs
    other_requests = [r for r in ride_requests if get_pickup(r) != 'depot' and get_dropoff(r) != 'depot']
    dropoff_dict = {}
    for r in other_requests:
        dropoff = get_dropoff(r)
        if dropoff not in dropoff_dict:
            dropoff_dict[dropoff] = []
        dropoff_dict[dropoff].append(r)
    for dropoff, requests in dropoff_dict.items():
        if len(requests) > 1:
            requests_sorted = sorted(requests, key=get_demand, reverse=True)
            kept = requests_sorted[0]
            if (get_pickup(kept), get_dropoff(kept)) not in used_pairs:
                used_pairs.add((get_pickup(kept), get_dropoff(kept)))
                kept_requests.append(kept)
            for r in requests_sorted[1:]:
                skipped_pairs.append((get_pickup(r), get_dropoff(r), get_demand(r)))
                excluded_pairs.append({'pickup': get_pickup(r), 'dropoff': get_dropoff(r), 'weight': get_demand(r), 'index': ride_requests.index(r)})
        elif requests:
            r = requests[0]
            if (get_pickup(r), get_dropoff(r)) not in used_pairs:
                used_pairs.add((get_pickup(r), get_dropoff(r)))
                kept_requests.append(r)

    print(f"Total ride requests: {len(ride_requests)}")
    print(f"Used pickup-delivery pairs: {len(used_pairs)}")
    print(f"Skipped pairs due to conflicts: {len(skipped_pairs)}")
    if skipped_pairs:
        print("Skipped ride requests:")
        for pickup, dropoff, weight in skipped_pairs:
            print(f"  {pickup} → {dropoff} (weight: {weight}kg)")
    print("Used pickup-delivery pairs:")
    for r in kept_requests:
        print(f"  {get_pickup(r)} → {get_dropoff(r)} (weight: {get_demand(r)}kg)")
    if excluded_pairs:
        print("\nSummary of all excluded pickup-delivery pairs:")
        for pair in excluded_pairs:
            print(f"  Excluded: {pair['pickup']} → {pair['dropoff']} (weight: {pair['weight']}kg, index: {pair['index']})")

    result, status, constraints = optimizer.solve(constraint_level="pickup_delivery", verbose=True)
    print("\n=== Furgoni Scenario Result ===")
    print(f"Status: {status}")
    print(f"Constraints applied: {constraints}")
    if result:
        print(f"Objective value: {result['objective_value']}")
        print(f"Total distance: {result['total_distance']:.1f} km")
        for vehicle_id, route_data in result['routes'].items():
            route = route_data['route']
            print(f"Vehicle {vehicle_id}: {len(route)} stops, distance: {route_data['distance']:.1f} km")
    else:
        print("❌ No solution found.")

    # Optional: plot the solution
    try:
        optimizer.plot_solution(result, title="Furgoni VRP Solution")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
