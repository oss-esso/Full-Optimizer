import numpy as np
from typing import List, Dict, Tuple
from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest

class VRPScenarioGenerator:
    """Generates various VRP scenarios for testing."""
    
    @staticmethod
    def create_small_delivery_scenario() -> VRPInstance:
        """Create a small delivery scenario for testing."""
        instance = VRPInstance("Small Delivery Problem")
        
        # Add depot
        instance.add_location(Location("depot", 50, 50, demand=0))
        
        # Add customers
        instance.add_location(Location("customer_1", 20, 30, demand=3))
        instance.add_location(Location("customer_2", 80, 20, demand=2))
        instance.add_location(Location("customer_3", 70, 80, demand=4))
        instance.add_location(Location("customer_4", 30, 70, demand=1))
        
        # Add vehicles
        instance.add_vehicle(Vehicle("vehicle_1", capacity=10, depot_id="depot"))
        instance.add_vehicle(Vehicle("vehicle_2", capacity=10, depot_id="depot"))
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        
        return instance
    
    @staticmethod
    def create_vrppd_scenario() -> VRPInstance:
        """Create a VRPPD scenario based on the paper."""
        instance = VRPInstance("Vehicle Routing Problem with Pickup and Delivery")
        
        # Add depot (vehicle starting point)
        instance.add_location(Location("depot_1", 0, 0))
        instance.add_location(Location("depot_2", 10, 0))
        
        # Add pickup and dropoff locations
        pickup_dropoff_locations = [
            ("pickup_1", 2, 3),
            ("dropoff_1", 8, 2),
            ("pickup_2", 1, 5),
            ("dropoff_2", 7, 6),
            ("pickup_3", 3, 1),
            ("dropoff_3", 9, 4),
            ("pickup_4", 4, 7),
            ("dropoff_4", 6, 1),
        ]
        
        for loc_id, x, y in pickup_dropoff_locations:
            instance.add_location(Location(loc_id, x, y))
        
        # Add vehicles
        instance.add_vehicle(Vehicle("vehicle_1", capacity=4, depot_id="depot_1"))
        instance.add_vehicle(Vehicle("vehicle_2", capacity=4, depot_id="depot_2"))
        
        # Add ride requests
        ride_requests = [
            RideRequest("request_1", "pickup_1", "dropoff_1", passengers=2),
            RideRequest("request_2", "pickup_2", "dropoff_2", passengers=1),
            RideRequest("request_3", "pickup_3", "dropoff_3", passengers=3),
            RideRequest("request_4", "pickup_4", "dropoff_4", passengers=1),
        ]
        
        for request in ride_requests:
            instance.add_ride_request(request)
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        return instance
    
    @staticmethod
    def create_medium_delivery_scenario() -> VRPInstance:
        """Create a medium-sized delivery scenario."""
        instance = VRPInstance("Medium Delivery Problem")
        
        # Add depot
        instance.add_location(Location("depot", 100, 100, demand=0))
        
        # Add customers in a more complex pattern
        customers = [
            ("customer_1", 20, 20, 5),
            ("customer_2", 180, 30, 3),
            ("customer_3", 170, 170, 7),
            ("customer_4", 30, 180, 4),
            ("customer_5", 60, 60, 2),
            ("customer_6", 140, 140, 6),
            ("customer_7", 80, 160, 3),
            ("customer_8", 160, 80, 5),
            ("customer_9", 40, 120, 4),
            ("customer_10", 120, 40, 2),
        ]
        
        for name, x, y, demand in customers:
            instance.add_location(Location(name, x, y, demand=demand))
        
        # Add vehicles
        for i in range(3):
            instance.add_vehicle(Vehicle(f"vehicle_{i+1}", capacity=15, depot_id="depot"))
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        
        return instance
    
    @staticmethod
    def create_time_window_scenario() -> VRPInstance:
        """Create a VRPTW scenario with time windows."""
        instance = VRPInstance("Time Window VRP")
        
        # Add depot
        instance.add_location(Location("depot", 0, 0, time_window_start=0, time_window_end=480))  # 8 hours
        
        # Add customers with time windows
        customers_data = [
            ("customer_1", 3, 4, 5, 60, 120),   # service between 1-2 hours
            ("customer_2", -2, 3, 3, 120, 180), # service between 2-3 hours
            ("customer_3", 4, -1, 7, 180, 240), # service between 3-4 hours
            ("customer_4", -1, -3, 4, 240, 300), # service between 4-5 hours
            ("customer_5", 2, 2, 2, 300, 360),  # service between 5-6 hours
        ]
        
        for loc_id, x, y, demand, tw_start, tw_end in customers_data:
            instance.add_location(Location(
                loc_id, x, y, demand, 
                time_window_start=tw_start, 
                time_window_end=tw_end,
                service_time=15  # 15 minutes service time
            ))
        
        # Add vehicles
        instance.add_vehicle(Vehicle("vehicle_1", capacity=20, depot_id="depot", max_time=480))
        instance.add_vehicle(Vehicle("vehicle_2", capacity=15, depot_id="depot", max_time=480))
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        return instance
    
    @staticmethod
    def create_multi_depot_scenario() -> VRPInstance:
        """Create a multi-depot VRP scenario."""
        instance = VRPInstance("Multi-Depot VRP")
        
        # Add depots
        instance.add_location(Location("depot_A", 0, 0))
        instance.add_location(Location("depot_B", 8, 8))
        
        # Add customers
        customers_data = [
            ("customer_1", 1, 2, 4),
            ("customer_2", 2, 1, 3),
            ("customer_3", 3, 3, 5),
            ("customer_4", 6, 7, 2),
            ("customer_5", 7, 6, 6),
            ("customer_6", 9, 9, 3),
            ("customer_7", 4, 5, 4),
            ("customer_8", 5, 4, 2),
        ]
        
        for loc_id, x, y, demand in customers_data:
            instance.add_location(Location(loc_id, x, y, demand))
        
        # Add vehicles to different depots
        instance.add_vehicle(Vehicle("vehicle_A1", capacity=15, depot_id="depot_A"))
        instance.add_vehicle(Vehicle("vehicle_A2", capacity=12, depot_id="depot_A"))
        instance.add_vehicle(Vehicle("vehicle_B1", capacity=18, depot_id="depot_B"))
        instance.add_vehicle(Vehicle("vehicle_B2", capacity=10, depot_id="depot_B"))
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        return instance
    
    @staticmethod
    def create_large_delivery_scenario() -> VRPInstance:
        """Create a larger delivery scenario for performance testing."""
        instance = VRPInstance("Large Delivery Problem")
        
        # Add depot
        instance.add_location(Location("depot", 150, 150, demand=0))
        
        # Generate customers in clusters to simulate real-world patterns
        import random
        random.seed(42)  # For reproducibility
        
        # Cluster centers
        cluster_centers = [(80, 80), (220, 80), (220, 220), (80, 220), (150, 150)]
        customers = []
        
        customer_id = 1
        for center_x, center_y in cluster_centers:
            # Add 4-6 customers around each cluster center
            for _ in range(random.randint(4, 6)):
                x = center_x + random.randint(-30, 30)
                y = center_y + random.randint(-30, 30)
                demand = random.randint(1, 8)
                customers.append((f"customer_{customer_id}", x, y, demand))
                customer_id += 1
        
        for name, x, y, demand in customers:
            instance.add_location(Location(name, x, y, demand=demand))
        
        # Add vehicles
        for i in range(5):
            instance.add_vehicle(Vehicle(f"vehicle_{i+1}", capacity=20, depot_id="depot"))
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        
        return instance
    
    @staticmethod
    def create_benchmark_cvrp_scenario() -> VRPInstance:
        """Create a scenario based on standard CVRP benchmarks."""
        instance = VRPInstance("Benchmark CVRP")
        
        # Based on simplified Christofides benchmark instance
        # Depot at center
        instance.add_location(Location("depot", 50, 50, demand=0))
        
        # Customer locations and demands (simplified version)
        customers = [
            ("customer_1", 37, 52, 7),
            ("customer_2", 49, 49, 30),
            ("customer_3", 52, 64, 16),
            ("customer_4", 20, 26, 9),
            ("customer_5", 40, 30, 21),
            ("customer_6", 21, 47, 15),
            ("customer_7", 17, 63, 19),
            ("customer_8", 31, 62, 23),
            ("customer_9", 52, 33, 11),
            ("customer_10", 51, 21, 5),
            ("customer_11", 42, 41, 19),
            ("customer_12", 31, 32, 29),
            ("customer_13", 5, 25, 23),
            ("customer_14", 12, 42, 21),
            ("customer_15", 36, 16, 10),
        ]
        
        for name, x, y, demand in customers:
            instance.add_location(Location(name, x, y, demand=demand))
        
        # Add vehicles with capacity from benchmark
        for i in range(4):
            instance.add_vehicle(Vehicle(f"vehicle_{i+1}", capacity=50, depot_id="depot"))
        
        # Calculate distance matrix using Manhattan distance 
        instance.calculate_distance_matrix(distance_method="manhattan")
        
        return instance

def get_all_scenarios() -> Dict[str, VRPInstance]:
    """Get all available VRP scenarios - MODA scenarios and synthetic test scenarios only."""
    # Core synthetic test scenarios for algorithm validation
    scenarios = {
        "small_delivery": VRPScenarioGenerator.create_small_delivery_scenario(),
        "vrppd": VRPScenarioGenerator.create_vrppd_scenario(),
        "medium_delivery": VRPScenarioGenerator.create_medium_delivery_scenario(),
        "time_window": VRPScenarioGenerator.create_time_window_scenario(),
        "multi_depot": VRPScenarioGenerator.create_multi_depot_scenario(),
    }
    
    # Add the large MODA_first scenario for realistic testing
    try:
        print("Creating MODA_first large VRPPD scenario in Northern Italy...")
        moda_scenario = create_moda_first_scenario()
        scenarios["MODA_first"] = moda_scenario
        print("+ Added MODA_first scenario (60 vehicles, 200 locations)")
    except Exception as e:
        print(f"! Error creating MODA_first scenario: {str(e)}")
    
    # Add the MODA_small scenario for focused testing
    try:
        print("Creating MODA_small VRPPD scenario in Northern Italy...")
        moda_scenario = create_moda_small_scenario()
        scenarios["MODA_small"] = moda_scenario
        print("+ Added MODA_small scenario (5 vehicles, 20 locations)")
    except Exception as e:
        print(f"! Error creating MODA_small scenario: {str(e)}")
    
    # NOTE: Offline/OSM realistic scenarios are disabled to focus on MODA scenarios
    # This reduces testing complexity and focuses on the core VRPPD use cases
    
    # Summary of realistic scenarios
    realistic_count = sum(1 for name, instance in scenarios.items() 
                         if hasattr(instance, 'is_realistic') and instance.is_realistic)
    
    print(f"\n+ Scenarios loaded: {len(scenarios)} total")
    print(f"  - Synthetic test scenarios: {len(scenarios) - realistic_count}")
    print(f"  - MODA realistic scenarios: {realistic_count}")
    print("  - Note: Other realistic/offline scenarios disabled for focused testing")
    
    return scenarios

def create_moda_small_scenario() -> VRPInstance:
    """Create MODA_small VRPPD scenario with 5 vehicles and 20 locations in Northern Italy."""
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("MODA_small Northern Italy VRPPD")
    instance.is_realistic = True
    
    # Use Northern Italy (Milan area) as the base for GPS coordinates
    # Milan metropolitan area bounds
    MIN_LAT, MAX_LAT = 45.3, 45.7
    MIN_LON, MAX_LON = 8.8, 9.4
    
    # Create depot locations in Northern Italy cities
    depot_locations = [
        ("depot_1", 9.1896, 45.4642, "Milano Centrale Hub"),  # Milan Central
        ("depot_2", 9.0700, 45.4773, "Milano Porta Garibaldi Hub"),  # Porta Garibaldi
    ]
    
    # Add depot locations
    for depot_id, lon, lat, address in depot_locations:
        depot = Location(depot_id, lon, lat, demand=0)
        depot.address = address
        depot.lat = lat
        depot.lon = lon
        instance.add_location(depot)
    
    # Generate 20 pickup/dropoff locations distributed across Northern Italy
    pickup_locations = []
    dropoff_locations = []
    
    # Generate locations in clusters to simulate realistic ride patterns
    cluster_centers = [
        (45.4642, 9.1896, "Milano Centro"),  # Milan Center
        (45.4773, 9.0700, "Porta Garibaldi"),  # Porta Garibaldi
        (45.4408, 9.2297, "Lambrate"),  # Lambrate
        (45.5017, 9.1586, "Isola"),  # Isola district
        (45.4384, 9.1699, "Navigli"),  # Navigli area
    ]
    
    location_counter = 1
    
    for cluster_idx, (center_lat, center_lon, area_name) in enumerate(cluster_centers):
        # Generate 2 pickup/dropoff pairs per cluster (total 10 pairs = 20 locations)
        for i in range(2):
            # Pickup location (scattered around cluster center)
            pickup_lat = center_lat + random.uniform(-0.01, 0.01)
            pickup_lon = center_lon + random.uniform(-0.01, 0.01)
            pickup_id = f"pickup_{location_counter}"
            
            pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0)
            pickup.address = f"Pickup {location_counter} - {area_name}"
            pickup.lat = pickup_lat
            pickup.lon = pickup_lon
            instance.add_location(pickup)
            pickup_locations.append(pickup_id)
            
            # Corresponding dropoff location (different area but realistic)
            # Choose a random different cluster for dropoff
            target_cluster = random.choice([c for c in cluster_centers if c != (center_lat, center_lon, area_name)])
            dropoff_lat = target_cluster[0] + random.uniform(-0.008, 0.008)
            dropoff_lon = target_cluster[1] + random.uniform(-0.008, 0.008)
            dropoff_id = f"dropoff_{location_counter}"
            
            dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0)
            dropoff.address = f"Dropoff {location_counter} - {target_cluster[2]}"
            dropoff.lat = dropoff_lat
            dropoff.lon = dropoff_lon
            instance.add_location(dropoff)
            dropoff_locations.append(dropoff_id)
            
            location_counter += 1
    
    # Add 5 vehicles distributed across depots
    vehicles_per_depot = 5 // len(depot_locations)
    remaining_vehicles = 5 % len(depot_locations)
    
    vehicle_id = 1
    for depot_idx, (depot_id, _, _, _) in enumerate(depot_locations):
        vehicle_count = vehicles_per_depot
        if depot_idx < remaining_vehicles:
            vehicle_count += 1
            
        for i in range(vehicle_count):
            vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=4, depot_id=depot_id)
            instance.add_vehicle(vehicle)
            vehicle_id += 1
    
    # Create 10 ride requests (20 locations = 10 pickup/dropoff pairs)
    for i in range(10):
        pickup_id = pickup_locations[i]
        dropoff_id = dropoff_locations[i]
        passengers = random.randint(1, 4)
        
        request = RideRequest(f"request_{i+1}", pickup_id, dropoff_id, passengers=passengers)
        instance.add_ride_request(request)
    
    # Calculate distance matrix using Manhattan distance for consistency
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print(f"Created MODA_small scenario in Northern Italy:")
    print(f"  - {len(instance.locations)} total locations")
    print(f"  - {len(depot_locations)} depots")
    print(f"  - {len(pickup_locations)} pickup locations")
    print(f"  - {len(dropoff_locations)} dropoff locations")
    print(f"  - {len(instance.vehicles)} vehicles")
    print(f"  - {len(instance.ride_requests)} ride requests")
    print(f"  - Location: Northern Italy (Milan metropolitan area)")
    
    return instance

def create_moda_first_scenario() -> VRPInstance:
    """Create MODA_first large VRPPD scenario with 60 vehicles and 200 locations in Northern Italy."""
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("MODA_first Northern Italy Large VRPPD")
    instance.is_realistic = True
    
    # Create depot locations in Northern Italy (Asti and Milan)
    depot_locations = [
        ("depot_asti", 8.2062, 44.9009, "Asti Transportation Hub - Piazza Alfieri 1"),
        ("depot_milan", 9.1896, 45.4642, "Milano Centrale Hub - Piazza Duca d'Aosta 1"),
    ]
    
    # Add depot locations
    for depot_id, lon, lat, address in depot_locations:
        depot = Location(depot_id, lon, lat, demand=0)
        depot.address = address
        depot.lat = lat
        depot.lon = lon
        instance.add_location(depot)
    
    # Generate 200 pickup/dropoff locations distributed across Northern Italy
    pickup_locations = []
    dropoff_locations = []
    
    # Northern Italy cities and areas (avoiding lakes and mountains)
    # Format: (lat, lon, city/area name, sample street)
    northern_italy_areas = [
        # Milan metropolitan area
        (45.4642, 9.1896, "Milano Centro", "Via Dante"),
        (45.4773, 9.0700, "Milano Porta Garibaldi", "Corso Como"),
        (45.4384, 9.1699, "Milano Navigli", "Naviglio Grande"),
        (45.5017, 9.1586, "Milano Isola", "Via Paolo Sarpi"),
        (45.4408, 9.2297, "Milano Lambrate", "Via Rombon"),
        (45.5149, 9.2167, "Sesto San Giovanni", "Via Matteotti"),
        (45.4180, 9.1307, "Milano Porta Romana", "Corso di Porta Romana"),
        
        # Asti and surroundings
        (44.9009, 8.2062, "Asti Centro", "Corso Alfieri"),
        (44.8950, 8.1980, "Asti San Paolo", "Via Casale"),
        (44.9100, 8.2200, "Asti Madonna del Portone", "Via Torino"),
        
        # Other Northern Italy cities
        (45.0703, 7.6869, "Torino Centro", "Via Roma"),
        (45.0522, 7.5150, "Torino Mirafiori", "Corso Unione Sovietica"),
        (45.1128, 7.7421, "Torino Barriera di Milano", "Corso Giulio Cesare"),
        
        # Novara area
        (45.4469, 8.6169, "Novara Centro", "Corso Italia"),
        (45.4333, 8.6000, "Novara Ovest", "Via Biella"),
        
        # Alessandria area
        (44.9133, 8.6167, "Alessandria Centro", "Via Milano"),
        (44.9050, 8.6300, "Alessandria Pista", "Via Casale"),
        
        # Bergamo area
        (45.6947, 9.6700, "Bergamo Città Alta", "Via Gombito"),
        (45.6983, 9.6773, "Bergamo Città Bassa", "Via XX Settembre"),
        
        # Como area
        (45.8081, 9.0852, "Como Centro", "Via Vittorio Emanuele"),
        
        # Brescia area
        (45.5416, 10.2118, "Brescia Centro", "Via dei Musei"),
        
        # Pavia area
        (45.1847, 9.1582, "Pavia Centro", "Corso Strada Nuova"),
        
        # Varese area
        (45.8206, 8.8250, "Varese Centro", "Corso Matteotti"),
        
        # Vercelli area
        (45.3206, 8.4219, "Vercelli Centro", "Corso Libertà"),
        
        # Cuneo area
        (44.3841, 7.5420, "Cuneo Centro", "Via Roma"),
    ]
    
    location_counter = 1
    
    # Generate locations based on Northern Italy areas (ensuring land coordinates)
    areas_per_pair = len(northern_italy_areas)
    pairs_per_area = max(1, 100 // areas_per_pair)  # Distribute 100 pairs across areas
    
    for area_idx, (center_lat, center_lon, area_name, sample_street) in enumerate(northern_italy_areas):
        # Generate pickup/dropoff pairs for this area
        pairs_to_generate = min(pairs_per_area, 100 - len(pickup_locations))
        if pairs_to_generate <= 0:
            break
            
        # Smaller spread to ensure we stay within city bounds
        spread = 0.015  # approximately 1.5 km
        
        for i in range(pairs_to_generate):
            # Use smaller offsets to stay within city area
            pickup_lat = center_lat + random.uniform(-spread, spread)
            pickup_lon = center_lon + random.uniform(-spread, spread)
            
            pickup_id = f"pickup_{location_counter}"
            pickup_addr = f"{100+i*10} {sample_street}, {area_name}"
            
            pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0)
            pickup.address = pickup_addr
            pickup.lat = pickup_lat
            pickup.lon = pickup_lon
            instance.add_location(pickup)
            pickup_locations.append(pickup_id)
            
            # Corresponding dropoff location (different area for realistic trips)
            # Choose a random different area for dropoff
            target_areas = [a for a in northern_italy_areas if a != (center_lat, center_lon, area_name, sample_street)]
            if target_areas:
                target = random.choice(target_areas)
                dropoff_lat = target[0] + random.uniform(-spread, spread)
                dropoff_lon = target[1] + random.uniform(-spread, spread)
                dropoff_area = target[2]
                dropoff_street = target[3]
            else:
                # Fallback to same area but different location
                dropoff_lat = center_lat + random.uniform(-spread, spread)
                dropoff_lon = center_lon + random.uniform(-spread, spread)
                dropoff_area = area_name
                dropoff_street = sample_street
            
            dropoff_id = f"dropoff_{location_counter}"
            dropoff_addr = f"{200+i*10} {dropoff_street}, {dropoff_area}"
            
            dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0)
            dropoff.address = dropoff_addr
            dropoff.lat = dropoff_lat
            dropoff.lon = dropoff_lon
            instance.add_location(dropoff)
            dropoff_locations.append(dropoff_id)
            
            location_counter += 1
    
    # Add additional locations to reach exactly 100 pairs if needed
    while len(pickup_locations) < 100:
        # Pick random area for any remaining locations
        area = random.choice(northern_italy_areas)
        center_lat, center_lon, area_name, sample_street = area
        spread = 0.015
        
        pickup_lat = center_lat + random.uniform(-spread, spread)
        pickup_lon = center_lon + random.uniform(-spread, spread)
        
        pickup_id = f"pickup_{location_counter}"
        pickup_addr = f"{300+location_counter} {sample_street}, {area_name}"
        
        pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0)
        pickup.address = pickup_addr
        pickup.lat = pickup_lat
        pickup.lon = pickup_lon
        instance.add_location(pickup)
        pickup_locations.append(pickup_id)
        
        # Choose a different area for dropoff
        target_areas = [a for a in northern_italy_areas if a != area]
        if target_areas:
            target = random.choice(target_areas)
            dropoff_lat = target[0] + random.uniform(-spread, spread)
            dropoff_lon = target[1] + random.uniform(-spread, spread)
            dropoff_area = target[2]
            dropoff_street = target[3]
        else:
            dropoff_lat = center_lat + random.uniform(-spread, spread)
            dropoff_lon = center_lon + random.uniform(-spread, spread)
            dropoff_area = area_name
            dropoff_street = sample_street
        
        dropoff_id = f"dropoff_{location_counter}"
        dropoff_addr = f"{400+location_counter} {dropoff_street}, {dropoff_area}"
        
        dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0)
        dropoff.address = dropoff_addr
        dropoff.lat = dropoff_lat
        dropoff.lon = dropoff_lon
        instance.add_location(dropoff)
        dropoff_locations.append(dropoff_id)
        
        location_counter += 1
    
    # Add 60 vehicles distributed across the two depots
    vehicles_per_depot = 60 // len(depot_locations)
    remaining_vehicles = 60 % len(depot_locations)
    
    vehicle_id = 1
    for depot_idx, (depot_id, _, _, _) in enumerate(depot_locations):
        vehicle_count = vehicles_per_depot
        if depot_idx < remaining_vehicles:
            vehicle_count += 1
            
        for i in range(vehicle_count):
            vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=4, depot_id=depot_id)
            instance.add_vehicle(vehicle)
            vehicle_id += 1
    
    # Create 100 ride requests (200 locations = 100 pickup/dropoff pairs)
    for i in range(100):
        pickup_id = pickup_locations[i]
        dropoff_id = dropoff_locations[i]
        passengers = random.randint(1, 4)
        
        request = RideRequest(f"request_{i+1}", pickup_id, dropoff_id, passengers=passengers)
        instance.add_ride_request(request)
    
    # Calculate distance matrix using Manhattan distance for consistency
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print(f"Created MODA_first scenario in Northern Italy:")
    print(f"  - {len(instance.locations)} total locations")
    print(f"  - {len(depot_locations)} depots (Asti & Milan)")
    print(f"  - {len(pickup_locations)} pickup locations")
    print(f"  - {len(dropoff_locations)} dropoff locations")
    print(f"  - {len(instance.vehicles)} vehicles")
    print(f"  - {len(instance.ride_requests)} ride requests")
    print(f"  - Coverage: Northern Italy cities and metropolitan areas")
    print(f"  - Depots: Asti and Milan transportation hubs")
    
    return instance
