import numpy as np
from typing import List, Dict, Tuple
from vrp_data_models import VRPInstance, Location, Vehicle, RideRequest

# Default truck speed profiles for different vehicle types
DEFAULT_TRUCK_SPEED_RATIOS = {
    'standard': {
        'motorway': 80 / 130,      # Light trucks: 80 km/h, Cars: ~130 km/h
        'trunk': 70 / 100,         # Light trucks: 70 km/h, Cars: ~100 km/h
        'primary': 60 / 90,        # Light trucks: 60 km/h, Cars: ~90 km/h
        'secondary': 50 / 70,      # Light trucks: 50 km/h, Cars: ~70 km/h
        'tertiary': 45 / 60,       # Light trucks: 45 km/h, Cars: ~60 km/h
        'residential': 30 / 50,    # Light trucks: 30 km/h, Cars: ~50 km/h
        'service': 25 / 30,        # Light trucks: 25 km/h, Cars: ~30 km/h
        'default': 0.80            # Light trucks ~20% slower than cars
    },
    'heavy': {
        'motorway': 70 / 130,      # Heavy trucks: 70 km/h, Cars: ~130 km/h
        'trunk': 60 / 100,         # Heavy trucks: 60 km/h, Cars: ~100 km/h
        'primary': 50 / 90,        # Heavy trucks: 50 km/h, Cars: ~90 km/h
        'secondary': 40 / 70,      # Heavy trucks: 40 km/h, Cars: ~70 km/h
        'tertiary': 35 / 60,       # Heavy trucks: 35 km/h, Cars: ~60 km/h
        'residential': 25 / 50,    # Heavy trucks: 25 km/h, Cars: ~50 km/h
        'service': 20 / 30,        # Heavy trucks: 20 km/h, Cars: ~30 km/h
        'default': 0.65            # Heavy trucks ~35% slower than cars
    }
}

def calculate_cargo_volume(weight_kg: float, cargo_density_kg_m3: float = 200.0) -> float:
    """
    Calculate cargo volume based on weight and estimated density.
    
    Args:
        weight_kg: Cargo weight in kilograms
        cargo_density_kg_m3: Cargo density in kg/m³ (default: 200 kg/m³ for mixed cargo)
    
    Returns:
        Volume in cubic meters
    
    Note:
        Typical cargo densities:
        - Dense cargo (machinery, metals): 500-800 kg/m³
        - General cargo (packages, boxes): 150-300 kg/m³
        - Light cargo (textiles, electronics): 50-150 kg/m³
        - Mixed cargo (default): ~200 kg/m³
    """
    return weight_kg / cargo_density_kg_m3

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
          # Add vehicles with trucking capacity (4000 kg)
        instance.add_vehicle(Vehicle("vehicle_1", capacity=4000, depot_id="depot_1"))
        instance.add_vehicle(Vehicle("vehicle_2", capacity=4000, depot_id="depot_2"))
        
        # Add ride requests with cargo loads
        ride_requests = [
            RideRequest("request_1", "pickup_1", "dropoff_1", passengers=1500),  # 1.5 tons
            RideRequest("request_2", "pickup_2", "dropoff_2", passengers=800),   # 800 kg
            RideRequest("request_3", "pickup_3", "dropoff_3", passengers=2000),  # 2 tons
            RideRequest("request_4", "pickup_4", "dropoff_4", passengers=1200),  # 1.2 tons
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
        print("Creating MODA_small VRPPD scenario in Asti, Northern Italy...")
        moda_scenario = create_moda_small_scenario()
        scenarios["MODA_small"] = moda_scenario
        print("+ Added MODA_small scenario (5 vehicles, 20 locations)")
    except Exception as e:
        print(f"! Error creating MODA_small scenario: {str(e)}")
    
    # Add the overnight test scenario for overnight/multi-day testing
    try:
        print("Creating overnight test scenario...")
        overnight_scenario = create_overnight_test_scenario()
        scenarios["overnight_test"] = overnight_scenario
        print("+ Added overnight test scenario (2 vehicles, 1 close + 1 far destination)")
    except Exception as e:
        print(f"! Error creating overnight test scenario: {str(e)}")
    
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
    """Create MODA_small VRPPD scenario with 5 vehicles and 20 locations in Asti, Northern Italy.
    Implements trucking company logic: drivers work max 10 hours but can start at any time,
    and locations have realistic time windows throughout the day.
    """
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("MODA_small Asti Northern Italy VRPPD")
    instance.is_realistic = True
      # Use Northern Italy (Asti area) as the base for GPS coordinates
    # Asti metropolitan area bounds
    MIN_LAT, MAX_LAT = 44.8, 45.1
    MIN_LON, MAX_LON = 8.1, 8.4
      # Create single depot location in Asti
    depot_locations = [
        ("depot_1", 8.2065, 44.8987, "Asti Transportation Hub"),  # Asti Center
    ]
      # Add depot locations with 24/7 availability
    # Depots should always be open for vehicles to start and return
    for depot_id, lon, lat, address in depot_locations:
        # Depots are always open (24/7 operation)
        start_time = 0      # Open from start of day
        end_time = 1440     # Open until end of day (24 hours)
        
        depot = Location(depot_id, lon, lat, demand=0,
                        time_window_start=start_time, time_window_end=end_time, service_time=5)
        depot.address = address
        depot.lat = lat
        depot.lon = lon
        instance.add_location(depot)
    
    # Generate 20 pickup/dropoff locations distributed across Northern Italy
    pickup_locations = []
    dropoff_locations = []
      # Generate locations in clusters to simulate realistic ride patterns around Asti
    cluster_centers = [
        (44.8987, 8.2065, "Asti Centro"),  # Asti Center
        (44.9100, 8.1900, "Asti Nord"),  # North Asti
        (44.8850, 8.2200, "Asti Sud"),  # South Asti
        (44.9050, 8.2300, "Asti Est"),  # East Asti
        (44.8900, 8.1800, "Asti Ovest"),  # West Asti
    ]
    
    location_counter = 1
    
    for cluster_idx, (center_lat, center_lon, area_name) in enumerate(cluster_centers):
        # Generate 2 pickup/dropoff pairs per cluster (total 10 pairs = 20 locations)
        for i in range(2):
            # Pickup location (scattered around cluster center)
            pickup_lat = center_lat + random.uniform(-0.01, 0.01)
            pickup_lon = center_lon + random.uniform(-0.01, 0.01)
            pickup_id = f"pickup_{location_counter}"
            
            # Set pickup time windows: variable length (2-10 hours), can start anytime during day
            pickup_start = random.randint(0, 600)  # Start between 0-10 hours (0-600 min)
            pickup_window_length = random.randint(120, 600)  # 2-10 hours (120-600 min)
            pickup_end = min(pickup_start + pickup_window_length, 1440)  # Cap at end of day
            
            pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0,
                            time_window_start=pickup_start, time_window_end=pickup_end, service_time=15)
            pickup.address = f"Pickup {location_counter} - {area_name}"
            pickup.lat = pickup_lat
            pickup.lon = pickup_lon
            instance.add_location(pickup)
            pickup_locations.append(pickup_id)
              # Corresponding dropoff location (different area but realistic)
            # Choose a random different cluster for dropoff
            target_cluster = random.choice([c for c in cluster_centers if c != (center_lat, center_lon, area_name)])
            dropoff_lat = target_cluster[0] + random.uniform(-0.008, 0.008)
            dropoff_lat = target_cluster[0] + random.uniform(-0.008, 0.008)
            dropoff_lon = target_cluster[1] + random.uniform(-0.008, 0.008)
            dropoff_id = f"dropoff_{location_counter}"
            
            # Set dropoff time windows: variable length (2-8 hours), ensuring feasible sequence
            # Dropoff must start after pickup can potentially end (pickup_end + travel time)
            min_travel_time = 30  # Minimum 30 minutes travel + service time
            min_dropoff_start = max(0, pickup_end + min_travel_time)  # Must start after pickup window ends
            max_dropoff_start = min(1200, min_dropoff_start + 300)  # Allow up to 5 hours flexibility
            if min_dropoff_start >= max_dropoff_start:
                dropoff_start = min_dropoff_start
            else:
                dropoff_start = random.randint(min_dropoff_start, max_dropoff_start)
            dropoff_window_length = random.randint(120, 480)  # 2-8 hours (120-480 min)
            dropoff_end = min(dropoff_start + dropoff_window_length, 1440)  # Cap at end of day
            
            dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0,
                             time_window_start=dropoff_start, time_window_end=dropoff_end, service_time=10)
            dropoff.address = f"Dropoff {location_counter} - {target_cluster[2]}"
            dropoff.lat = dropoff_lat
            dropoff.lon = dropoff_lon
            instance.add_location(dropoff)
            dropoff_locations.append(dropoff_id)
            
            location_counter += 1
      # Add service areas for driver breaks
    try:
        from service_areas_db import service_areas_db
          # Get relevant service areas for Asti area
        relevant_areas = []
        for area_id, area in service_areas_db.service_areas.items():
            # Check if service area is within Asti area bounds
            if (MIN_LAT <= area.lat <= MAX_LAT and MIN_LON <= area.lon <= MAX_LON):
                relevant_areas.append(area_id)
        
        # Add service areas to instance
        for area_id in relevant_areas:
            from service_areas_db import service_area_to_location
            instance.locations[area_id] = service_area_to_location(service_areas_db.service_areas[area_id])
        
        print(f"  Added {len(relevant_areas)} service areas for driver breaks")
        
    except ImportError:
        print(f"  Service areas database not available")    # Add mixed fleet vehicles (5 total: 3 x 4-ton, 2 x 24-ton all from single depot)
    # No pre-loading - let optimizer decide how to load from depot
    total_vehicles = 5
    standard_trucks = 3  # 4-ton capacity
    heavy_trucks = 2     # 24-ton capacity
    
    vehicle_id = 1
    heavy_vehicle_count = 0
    
    # All vehicles assigned to the single depot
    depot_id = depot_locations[0][0]  # depot_1
    
    for i in range(total_vehicles):
        # Distribute heavy trucks: first 2 vehicles are heavy trucks
        if heavy_vehicle_count < heavy_trucks:
            # Heavy truck (24-ton) with driver regulations and truck speed profile
            capacity = 24000
            volume_capacity = 67.0  # 67 m³ for 24t trucks
            cost_per_km = 1.80  # €1.80/km for heavy trucks (higher fuel, maintenance, tolls)
            vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=capacity, volume_capacity=volume_capacity, 
                            depot_id=depot_id, max_time=540, cost_per_km=cost_per_km)
            vehicle.vehicle_type = "heavy"
            vehicle.max_driving_time = 270.0  # 4.5 hours in minutes
            vehicle.required_break_time = 45.0  # 45-minute break
            vehicle.max_total_work_time = 540.0  # 9 hours total work time (EU regulations)
            vehicle.use_truck_speeds = True  # Enable truck speed adjustments for OSM routing
            vehicle.truck_speed_ratios = DEFAULT_TRUCK_SPEED_RATIOS['heavy'].copy()  # Heavy truck speed profile
            heavy_vehicle_count += 1
        else:
            # Standard truck (4-ton) with truck speed profile
            capacity = 4000
            volume_capacity = 24.0  # 24 m³ for 4t trucks
            cost_per_km = 0.95  # €0.95/km for standard trucks (lower operating costs)
            vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=capacity, volume_capacity=volume_capacity, 
                            depot_id=depot_id, max_time=600, cost_per_km=cost_per_km)
            vehicle.vehicle_type = "standard"
            vehicle.max_driving_time = 600.0  # 10 hours (no break requirement)
            vehicle.required_break_time = 0.0
            vehicle.max_total_work_time = 600.0  # 10 hours total work time
            vehicle.use_truck_speeds = True  # Enable truck speed adjustments for OSM routing
            vehicle.truck_speed_ratios = DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()  # Standard truck speed profile
        
        instance.add_vehicle(vehicle)
        vehicle_id += 1    # Create cargo transport requests:
    # 70% from depot (virtual pickups at depot) + 30% from field pickup locations
    total_fleet_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    depot_cargo_weight = int(total_fleet_capacity * 0.7)  # 42,000kg
    field_cargo_weight = int(total_fleet_capacity * 0.3)  # 18,000kg
    
    # Create virtual depot pickup requests (70% of capacity)
    # These represent cargo already at depot with various destinations
    depot_requests = []
    request_id = 1
      # Distribute depot cargo across different dropoff locations
    remaining_depot_cargo = depot_cargo_weight
    # Ensure we don't create more depot requests than available dropoff locations
    depot_request_count = min(12, len(dropoff_locations))  # Max 10 depot requests to match dropoffs
    
    for i in range(depot_request_count):
        # Use unique dropoff location for each depot request
        dropoff_id = dropoff_locations[i]  # Unique assignment, no cycling
        
        # Generate cargo weight for this depot request
        if i == depot_request_count - 1:  # Last request gets remaining weight
            cargo_weight = remaining_depot_cargo
        else:
            # Random weight between 2,000-5,000kg per depot shipment
            avg_weight = remaining_depot_cargo // (depot_request_count - i)            
            variation = int(avg_weight * 0.4)  # ±40% variation
            cargo_weight = random.randint(max(1000, avg_weight - variation), 
                                        min(avg_weight + variation, remaining_depot_cargo))
            cargo_weight = max(1000, min(cargo_weight, remaining_depot_cargo))
        
        remaining_depot_cargo -= cargo_weight        # Create virtual pickup bay location at depot for this request
        # This ensures each depot request has its own pickup location for capacity calculations
        pickup_bay_id = f"depot_bay_{request_id}"
        depot_location = instance.locations[depot_id]
        pickup_bay = Location(
            pickup_bay_id, 
            depot_location.x,  # longitude 
            depot_location.y,  # latitude
            demand=0,
            service_time=5  # Minimal loading time at depot bay
        )
        instance.add_location(pickup_bay)
        
        # Create virtual pickup request from depot bay
        cargo_volume = calculate_cargo_volume(cargo_weight)  # Calculate volume based on weight
        depot_request = RideRequest(f"depot_request_{request_id}", pickup_bay_id, dropoff_id, passengers=cargo_weight, volume=cargo_volume)
        instance.add_ride_request(depot_request)
        depot_requests.append(depot_request)
        request_id += 1
        
        if remaining_depot_cargo <= 0:
            break
    
    # Create field pickup requests (30% of capacity) 
    field_requests = []
    remaining_field_cargo = field_cargo_weight
    field_request_count = len(pickup_locations)  # Use all 10 pickup locations
    
    for i in range(field_request_count):
        pickup_id = pickup_locations[i]
        dropoff_id = dropoff_locations[i]
        
        # Generate field pickup weights that total to 30% of fleet capacity
        if i == field_request_count - 1:  # Last request gets remaining weight
            cargo_weight = remaining_field_cargo
        else:
            # Random weight between 1,000-2,500kg to total around 18,000kg
            avg_weight = remaining_field_cargo // (field_request_count - i)
            variation = int(avg_weight * 0.4)  # ±40% variation
            cargo_weight = random.randint(max(500, avg_weight - variation), 
                                        min(avg_weight + variation, remaining_field_cargo))
            cargo_weight = max(500, min(cargo_weight, remaining_field_cargo))
        
        remaining_field_cargo -= cargo_weight
        
        cargo_volume = calculate_cargo_volume(cargo_weight)  # Calculate volume based on weight
        field_request = RideRequest(f"field_request_{request_id}", pickup_id, dropoff_id, passengers=cargo_weight, volume=cargo_volume)
        instance.add_ride_request(field_request)
        field_requests.append(field_request)
        request_id += 1
    
    # Calculate actual totals
    total_depot_cargo = sum(r.passengers for r in depot_requests)
    total_field_cargo = sum(r.passengers for r in field_requests)
    total_cargo_weight = total_depot_cargo + total_field_cargo    # Calculate total fleet capacity for reporting
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    print(f"MODA_small capacity distribution (depot virtual pickups):")
    print(f"  - Total fleet capacity: {total_capacity} kg")
    print(f"  - Virtual depot pickups (70%): {total_depot_cargo} kg")
    print(f"  - Field pickups (30%): {total_field_cargo} kg")
    print(f"  - Total cargo: {total_cargo_weight} kg = {total_cargo_weight/total_capacity*100:.1f}% of fleet capacity")
    print(f"Fleet composition: {len([v for v in instance.vehicles.values() if v.vehicle_type == 'standard'])} x 4t trucks + {len([v for v in instance.vehicles.values() if v.vehicle_type == 'heavy'])} x 24t trucks")
    print(f"All vehicles start from single depot: {depot_id}")
    
    # Show virtual depot requests strategy
    print(f"\n📦 Virtual Depot Pickup Strategy:")
    print(f"  {len(depot_requests)} depot → dropoff requests: {total_depot_cargo:,}kg")
    print(f"  {len(field_requests)} field pickup → dropoff requests: {total_field_cargo:,}kg")
    print(f"  💡 Optimizer will decide optimal loading and routing strategy")    # Calculate distance matrix using Manhattan distance for consistency
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    print(f"Created MODA_small scenario in Northern Italy:")
    print(f"  - {len(instance.locations)} total locations")
    print(f"  - {len(depot_locations)} depot (single depot)")
    print(f"  - {len(pickup_locations)} field pickup locations")
    print(f"  - {len(dropoff_locations)} dropoff locations")
    print(f"  - {len(instance.vehicles)} vehicles")
    print(f"  - {len(instance.ride_requests)} ride requests ({len(depot_requests)} depot + {len(field_requests)} field)")
    print(f"  - Scenario: Virtual depot pickups (70%) + Field pickups (30%)")
    
    return instance

def create_moda_first_scenario() -> VRPInstance:
    """Create MODA_first large VRPPD scenario with 60 vehicles and 200 locations in Northern Italy.
    Implements trucking company logic: drivers work max 10 hours but can start at any time,
    and locations have realistic time windows throughout the day.
    """
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("MODA_first Northern Italy Large VRPPD")
    instance.is_realistic = True
    
    # Create depot locations in Northern Italy (Asti and Milan)
    depot_locations = [
        ("depot_asti", 8.2062, 44.9009, "Asti Transportation Hub - Piazza Alfieri 1"),
        ("depot_milan", 9.1896, 45.4642, "Milano Centrale Hub - Piazza Duca d'Aosta 1"),
    ]
      # Add depot locations with 24/7 availability
    # Depots should always be open for vehicles to start and return
    for depot_id, lon, lat, address in depot_locations:
        # Depots are always open (24/7 operation)
        start_time = 0      # Open from start of day
        end_time = 1440     # Open until end of day (24 hours)
        
        depot = Location(depot_id, lon, lat, demand=0,
                        time_window_start=start_time, time_window_end=end_time, service_time=5)
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
            
            # Set pickup time windows: variable length (2-10 hours), can start anytime during day
            pickup_start = random.randint(0, 600)  # Start between 0-10 hours (0-600 min)
            pickup_window_length = random.randint(120, 600)  # 2-10 hours (120-600 min)
            pickup_end = min(pickup_start + pickup_window_length, 1440)  # Cap at end of day
            
            pickup = Location(pickup_id, pickup_lon, pickup_lat, demand=0,
                            time_window_start=pickup_start, time_window_end=pickup_end, service_time=15)
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
                # Fallback to same area but different location                dropoff_lat = center_lat + random.uniform(-spread, spread)
                dropoff_lon = center_lon + random.uniform(-spread, spread)
                dropoff_area = area_name
                dropoff_street = sample_street
            dropoff_id = f"dropoff_{location_counter}"
            dropoff_addr = f"{200+i*10} {dropoff_street}, {dropoff_area}"
            
            # Set dropoff time windows: variable length (2-8 hours), ensuring feasible sequence
            # Dropoff must start after pickup can potentially end (pickup_end + travel time)
            min_travel_time = 30  # Minimum 30 minutes travel + service time
            min_dropoff_start = max(0, pickup_end + min_travel_time)  # Must start after pickup window ends
            max_dropoff_start = min(1200, min_dropoff_start + 300)  # Allow up to 5 hours flexibility
            if min_dropoff_start >= max_dropoff_start:
                dropoff_start = min_dropoff_start
            else:
                dropoff_start = random.randint(min_dropoff_start, max_dropoff_start)
            dropoff_window_length = random.randint(120, 480)  # 2-8 hours (120-480 min)
            dropoff_end = min(dropoff_start + dropoff_window_length, 1440)  # Cap at end of day
            
            dropoff = Location(dropoff_id, dropoff_lon, dropoff_lat, demand=0,
                             time_window_start=dropoff_start, time_window_end=dropoff_end, service_time=10)
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
    
    # Add service areas for driver breaks
    try:
        from service_areas_db import service_areas_db, service_area_to_location
        
        # Get all service areas in Northern Italy for large scenario
        relevant_areas = []
        for area_id, area in service_areas_db.service_areas.items():
            # Include all service areas across Northern Italy for large scenario
            if (44.0 <= area.lat <= 46.5 and 7.0 <= area.lon <= 12.0):
                relevant_areas.append(area_id)
        
        # Add service areas to instance
        for area_id in relevant_areas:
            instance.locations[area_id] = service_area_to_location(service_areas_db.service_areas[area_id])
        
        print(f"  Added {len(relevant_areas)} service areas for driver breaks across Northern Italy")
        
    except ImportError:
        print(f"  Service areas database not available")
    
      # Add mixed fleet vehicles (60 total: 40 x 4-ton, 20 x 24-ton distributed across depots)
    total_vehicles = 60
    standard_trucks = 40  # 4-ton capacity
    heavy_trucks = 20     # 24-ton capacity
    
    vehicles_per_depot = total_vehicles // len(depot_locations)
    remaining_vehicles = total_vehicles % len(depot_locations)
    
    vehicle_id = 1
    heavy_vehicle_count = 0
    
    for depot_idx, (depot_id, _, _, _) in enumerate(depot_locations):
        vehicle_count = vehicles_per_depot
        if depot_idx < remaining_vehicles:
            vehicle_count += 1
            
        for i in range(vehicle_count):
            # Distribute heavy trucks proportionally
            if heavy_vehicle_count < heavy_trucks and (vehicle_id - 1) % 3 == 0:
                # Every 3rd vehicle is a heavy truck (24-ton) with truck speed profile
                cost_per_km = 1.80  # €1.80/km for heavy trucks
                vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=24000, volume_capacity=67.0, 
                                depot_id=depot_id, max_time=540, cost_per_km=cost_per_km)
                vehicle.vehicle_type = "heavy"
                vehicle.max_driving_time = 270.0  # 4.5 hours in minutes
                vehicle.required_break_time = 45.0  # 45-minute break
                vehicle.max_total_work_time = 540.0  # 9 hours total work time
                vehicle.use_truck_speeds = True  # Enable truck speed adjustments for OSM routing  
                vehicle.truck_speed_ratios = DEFAULT_TRUCK_SPEED_RATIOS['heavy'].copy()  # Heavy truck speed profile
                heavy_vehicle_count += 1
            else:
                # Standard truck (4-ton) with truck speed profile
                cost_per_km = 0.95  # €0.95/km for standard trucks
                vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=4000, volume_capacity=24.0, 
                                depot_id=depot_id, max_time=600, cost_per_km=cost_per_km)
                vehicle.vehicle_type = "standard"
                vehicle.max_driving_time = 600.0  # 10 hours (no break requirement)
                vehicle.required_break_time = 0.0
                vehicle.max_total_work_time = 600.0  # 10 hours total work time
                vehicle.use_truck_speeds = True  # Enable truck speed adjustments for OSM routing
                vehicle.truck_speed_ratios = DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()  # Standard truck speed profile
            
            instance.add_vehicle(vehicle)
            vehicle_id += 1
      # Create 100 cargo transport requests with realistic weights
    total_cargo_weight = 0
    for i in range(100):
        pickup_id = pickup_locations[i]
        dropoff_id = dropoff_locations[i]
        
        # Generate realistic cargo weights (200kg to 8000kg per shipment for large scenario)
        cargo_weight = random.randint(200, 8000)
        cargo_volume = calculate_cargo_volume(cargo_weight)  # Calculate volume based on weight
        total_cargo_weight += cargo_weight
        
        request = RideRequest(f"request_{i+1}", pickup_id, dropoff_id, passengers=cargo_weight, volume=cargo_volume)
        instance.add_ride_request(request)
      # Calculate total fleet capacity for reporting
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    print(f"MODA_first total demand: {total_cargo_weight} kg, total fleet capacity: {total_capacity} kg")
    print(f"Note: Trucks can do multiple pickup/dropoff cycles, so demand can exceed total capacity")
    print(f"Fleet composition: {len([v for v in instance.vehicles.values() if v.vehicle_type == 'standard'])} x 4t trucks + {len([v for v in instance.vehicles.values() if v.vehicle_type == 'heavy'])} x 24t trucks")
    
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

def create_moda_inverted_scenario() -> VRPInstance:
    """Create a copy of MODA_small with the same locations and requests, but vehicles in inverted order."""
    instance = create_moda_small_scenario()
    # Invert the vehicle order (simulate a different order in the scenario definition)
    vehicle_items = list(instance.vehicles.items())
    inverted = list(reversed(vehicle_items))
    instance.vehicles = {k: v for k, v in inverted}
    return instance






def create_furgoni_scenario() -> VRPInstance:
    """Create VRP scenario from the furgoni Excel data with calculated weights and time windows.
    Fixed to match MODA_small format for solvability, now with unique depot bays for all pickup-to-depot requests."""
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("Furgoni Delivery Scenario")
    instance.is_realistic = True  # Added missing attribute
    
    # Create depot location (Milan coordinates) - matching MODA format
    depot = Location("depot", 9.1896, 45.4642, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=5)
    depot.address = "Asti Distribution Center"
    depot.lat = 44.5404  # Added missing lat/lon attributes
    depot.lon = 8.1407
    instance.add_location(depot)
    


    # Vehicle data with calculated capacities - matching MODA format with max_time
    vehicles_data = {
        "FURGONE 1": 3500,
        "FURGONE 2": 3500, 
        "FURGONE 3": 3500,
        "CAMION 16 T.": 16000,
        "FURGONE 6": 3500,
        "CAMION 7,5 T.": 7500,
        "FURGONE 8": 3500,
        "FURGONE 9": 3500,
        "FURGONE 10": 3500,
        "FURGONE 11": 3500,
    }
    
    # Add vehicles to the scenario with proper attributes
    for vehicle_name, capacity in vehicles_data.items():
        vehicle_id = vehicle_name.lower().replace(" ", "_").replace(",", "").replace(".", "")
        # Set max_time based on vehicle type (matching MODA logic)
        if "CAMION" in vehicle_name:
            max_time =  900 # Heavy vehicles: 9 hours (EU regulations)
            vehicle_type = "heavy"
            max_driving_time = 450 # 4.5 hours
            required_break_time = 45.0
            max_total_work_time = 900
        else:
            max_time = 900  # Light vehicles: 10 hours
            vehicle_type = "standard"
            max_driving_time = 900.0
            required_break_time = 0.0
            max_total_work_time = 900.0
            
        vehicle = Vehicle(vehicle_id, capacity=capacity, depot_id="depot", max_time=max_time)
        vehicle.vehicle_type = vehicle_type
        vehicle.max_driving_time = max_driving_time
        vehicle.required_break_time = required_break_time
        vehicle.max_total_work_time = max_total_work_time
        instance.add_vehicle(vehicle)
    
    # Delivery and pickup addresses with time windows from Excel
    # Format: (location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup)
    locations_data = [
        # FURGONE 1 (1 delivery - 750kg total)
        ("schonaich_de", "gentile gusto carl Zeiss Str. 4 71101 Schonaich DE", 9.0000, 48.6667, 0, 1440, 30, False),
        ("malmoe_sweden", "malmoe sweden menarini diag.slr", 13.0038, 55.6050, 0, 1440, 30, False),
        # FURGONE 2 (4 deliveries - 831kg each for 3.3T total)
        ("salvazzano_pd", "Via Pelosa 20 35030 Salvazzano Dentro PD", 11.7833, 45.3833, 0, 1440, 20, False),
        ("badia_polesine_ro", "Via L. Da Vinci 537 45021 Badia Polesine RO", 11.5000, 45.1167, 0, 1440, 20, False),
        ("badia_polesine_ro_2", "Via L. Da Vinci 537 45021 Badia Polesine RO", 11.5000, 45.1167, 0, 1440, 20, False),
        ("marostica_vi", "via Milano 1 36063 Marostica VI", 11.6500, 45.7500, 0, 1440, 20, False),
        
        # FURGONE 3 (1 delivery - 1500kg total)
        ("st_martin_crau", "Rue Gay Lussac 3 St. Martin de Crau 13310", 4.8000, 43.6333, 0, 1440, 30, False),
        
        # CAMION 16 T. (2 deliveries)
        ("sant_olcese_ge", "Piazza Guglielmo Marconi 40 16010 Sant'Olcese GE", 8.9333, 44.5167, 0, 1440, 45, False),
        ("casarza_ligure_ge", "Via Tangoni 30/32 I-16030 Casarza Ligure GE", 9.4667, 44.2833, 0, 1440, 45, False),
        
        # FURGONE 5 (2 deliveries to same location - 750kg each for 1.5T total)
        ("quinto_do_anjo", "Quinta De Marquesa 1CCI Portoga 10218 29050-557 Quinto Do Anjo", -8.9000, 38.5667, 0, 1440, 30, False),
        ("quinto_do_anjo_2", "Quinta De Marquesa 1CCI Portoga 10218 29050-557 Quinto Do Anjo", -8.9000, 38.5667, 0, 1440, 30, False),
        # FURGONE 6 (1 delivery - 1500kg total)
        ("paris_fr", "Rue de L'Abbaye 10 PARIGI FR", 2.3522, 48.8566, 0, 1440, 30, False),
        
        # CAMION 7.5 T. (1 delivery - 2755kg)
        ("chiva_spagna", "Poligono I La Pamilla 196 46370 Chiva Spagna", -0.7167, 39.4667, 0, 1440, 45, False),
        
        # FURGONE 8 (1 delivery + 3 pickups with time constraints)
        ("cerro_al_lambro_mi", "Via Autosole 7 20070 Cerro Al Lambro MI", 9.3000, 45.3333, 0, 1440, 20, False),
        ("cormano_mi", "via dell'Artigianato 1 20032 Cormano MI", 9.1667, 45.5333, 0, 1440, 20, True),
        ("sicor_caronno", "Sicor Caronno", 8.9000, 45.6000, 0, 840, 15, True),  # Pickup before 14:00
        ("ferno_va", "superstrada per Malpensa uscita Cargocity Torre D 5 piano 21010 Ferno VA", 8.7167, 45.6167, 0, 510, 15, True),  # Pickup before 8:30
        
        # FURGONE 9 (4 deliveries + 1 pickup)
        ("imperia_1", "via Filippo Airenti 2 18100 Imperia IM", 8.0333, 43.8833, 0, 1440, 20, False),
        ("imperia_2", "via Nazionale 356/3 18100 Imperia IM", 8.0333, 43.8833, 0, 1440, 20, False),
        ("villar_cuneo", "via I Maggio 43 Cuneo 12020 Villar San Costanzo", 7.5333, 44.3833, 0, 1440, 20, False),
        ("villar_cuneo_2", "via I Maggio 43 Cuneo 12020 Villar San Costanzo", 7.5333, 44.3833, 0, 1440, 20, False),
        ("castellalfero_at", "Via Statale 25/A Castell'Alfero 14033 AT", 8.2167, 44.9667, 0, 1440, 15, True),
        
        # FURGONE 10 (1 delivery + 1 pickup)
        ("osimo_an", "via Francesco Crispi 2 60027 Osimo AN", 13.4833, 43.4833, 0, 1440, 20, False),
        ("castelfidardo", "via Jesina 27/P 60022 Castelfidardo", 13.5500, 43.4667, 0, 1440, 15, True),
        
        # FURGONE 11 (2 deliveries + 1 pickup)
        ("somaglia_lo", "strada Provinciale 223 26867 Località Cantonale Somaglia LO", 9.6667, 45.1667, 0, 1440, 20, False),
        ("capriate_bg", "via Bergamo 61/63 24042 Capriate San Gervasio BG", 9.5333, 45.6000, 0, 1440, 20, False),
        ("cazzano_bg", "via Cavalier Pietro Radici 19 24026 Cazzano Sant'andrea BG", 9.8000, 45.7500, 0, 1440, 15, True),
    ]
    
    # Add all locations with proper attributes (matching MODA format)
    for location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup in locations_data:
        location = Location(location_id, lon, lat, demand=0,
                           time_window_start=tw_start, time_window_end=tw_end, 
                           service_time=service_time)
        location.address = address
        location.lat = lat  # Added missing lat/lon attributes
        location.lon = lon
        location.is_pickup = is_pickup
        instance.add_location(location)
    
    # Create delivery/pickup requests using MODA depot bay pattern
    # Separate delivery cargo (depot bays → destinations) from pickup cargo (pickups → depot bays)
    
    # Delivery cargo data: create depot bays for each delivery
    delivery_cargo = [
        # FURGONE 1
        ("schonaich_de", 750),
        
        # FURGONE 2
        ("salvazzano_pd", 831), 
        ("badia_polesine_ro", 750), 
        ("badia_polesine_ro_2", 810), 
        ("marostica_vi", 950),
        
        # FURGONE 3
        ("st_martin_crau", 1500),
        
        # CAMION 16 T.
        ("sant_olcese_ge", 4300), 
        ("casarza_ligure_ge", 4300),
        
        # FURGONE 6
        ("paris_fr", 1500),
        
        # CAMION 7.5 T.
        ("chiva_spagna", 2700),
        
        # FURGONE 5 deliveries (2 to same location)
        ("quinto_do_anjo", 750),
        ("quinto_do_anjo_2", 750),


        # FURGONE 8 delivery
        ("cerro_al_lambro_mi", 800),
        
        # FURGONE 9 deliveries
        ("imperia_1", 600),
        ("imperia_2", 700),
        ("villar_cuneo", 550),
        ("villar_cuneo_2", 665),
        
        # FURGONE 10 delivery
        ("osimo_an", 750),
        
        # FURGONE 11 deliveries
        ("somaglia_lo", 1100),
        ("capriate_bg", 900),
    ]
    
    # Pickup cargo data: direct pickup → depot requests
    pickup_cargo = [
        # FURGONE 8 pickups
        ("cormano_mi", 700),
        ("sicor_caronno", 750),
        ("ferno_va", 820),
        
        # FURGONE 9 pickup
        ("castellalfero_at", 650),
        
        # FURGONE 10 pickup
        ("castelfidardo", 550),
        
        # FURGONE 11 pickup
        ("cazzano_bg", 1200),
    ]
    
    # Create depot bays for delivery cargo (following MODA pattern)
    depot_requests = []
    request_id = 1
    depot_location = instance.locations["depot"]

    for dropoff_id, cargo_weight in delivery_cargo:
        # Create virtual pickup bay at depot for this delivery
        pickup_bay_id = f"depot_bay_{request_id}"
        pickup_bay = Location(
            pickup_bay_id, 
            depot_location.x,  # Same longitude as main depot
            depot_location.y,  # Same latitude as main depot
            demand=0,
            service_time=5     # Minimal loading time at depot bay
        )
        pickup_bay.lat = depot_location.lat
        pickup_bay.lon = depot_location.lon
        pickup_bay.address = f"Depot Loading Bay {request_id}"
        instance.add_location(pickup_bay)
        # Create request from depot bay to dropoff location
        depot_request = RideRequest(f"depot_request_{request_id}", pickup_bay_id, dropoff_id, passengers=cargo_weight)
        instance.add_ride_request(depot_request)
        depot_requests.append(depot_request)
        request_id += 1

    # Create pickup requests (pickup location → unique depot bay)
    pickup_requests = []
    pickup_bay_counter = 1
    for pickup_id, cargo_weight in pickup_cargo:
        # Create a unique depot bay for each pickup
        pickup_bay_id = f"pickup_bay_{pickup_bay_counter}"
        pickup_bay = Location(
            pickup_bay_id,
            depot_location.x,
            depot_location.y,
            demand=0,
            service_time=5
        )
        pickup_bay.lat = depot_location.lat
        pickup_bay.lon = depot_location.lon
        pickup_bay.address = f"Depot Pickup Bay {pickup_bay_counter}"
        instance.add_location(pickup_bay)
        # Create request from pickup location to unique depot bay
        pickup_request = RideRequest(f"pickup_request_{request_id}", pickup_id, pickup_bay_id, passengers=cargo_weight)
        instance.add_ride_request(pickup_request)
        pickup_requests.append(pickup_request)
        request_id += 1
        pickup_bay_counter += 1
    
    # Calculate distance matrix using same method as MODA
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    # Add summary statistics (matching MODA format)
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    total_delivery_weight = sum(r.passengers for r in depot_requests)  # Changed from delivery_requests
    total_pickup_weight = sum(r.passengers for r in pickup_requests)
    total_cargo_weight = total_delivery_weight + total_pickup_weight
    
    print(f"Created Furgoni scenario with depot bay pattern (unique bays for all pickups and deliveries):")
    print(f"  - Total locations: {len(instance.locations)}")
    print(f"  - Main depot: 1")
    print(f"  - Depot bays: {len(depot_requests)} (for deliveries)")
    print(f"  - Pickup bays: {len(pickup_requests)} (for pickups)")
    print(f"  - Delivery destinations: {len([l for l in instance.locations.values() if not getattr(l, 'is_pickup', False) and l.id != 'depot' and not l.id.startswith('depot_bay') and not l.id.startswith('pickup_bay')])}")
    print(f"  - Field pickup locations: {len([l for l in instance.locations.values() if getattr(l, 'is_pickup', False)])}")
    print(f"  - Vehicles: {len(instance.vehicles)}")
    print(f"  - Ride requests: {len(instance.ride_requests)} ({len(depot_requests)} depot→delivery + {len(pickup_requests)} pickup→depot)")
    print(f"Fleet composition: {len([v for v in instance.vehicles.values() if v.vehicle_type == 'standard'])} light vehicles + {len([v for v in instance.vehicles.values() if v.vehicle_type == 'heavy'])} heavy vehicles")
    
    print(f"\nCapacity utilization:")
    print(f"  - Total fleet capacity: {total_capacity:,} kg")
    print(f"  - Depot cargo (via bays): {total_delivery_weight:,} kg")
    print(f"  - Field pickup cargo: {total_pickup_weight:,} kg")
    print(f"  - Total cargo: {total_cargo_weight:,} kg = {total_cargo_weight/total_capacity*100:.1f}% of fleet capacity")
    
    print(f"\n📦 Depot Bay Strategy (matching MODA_small, now for all pickups):")
    print(f"  - {len(depot_requests)} depot bays at same coordinates as main depot (for deliveries)")
    print(f"  - {len(pickup_requests)} unique pickup bays at depot (for pickups)")
    print(f"  - Each bay serves 1 request (delivery or pickup)")
    print(f"  - 💡 Optimizer decides which bays to visit per vehicle")
    
    return instance

def create_overnight_test_scenario() -> VRPInstance:
    """
    Create a simplified overnight test scenario with close and far destinations.
    
    This scenario is designed to test overnight/multi-day routing features:
    - 2 vehicles with 8-hour (480 minute) daily driving limits
    - 1 close delivery (Cormano, ~150km from depot)
    - 1 far delivery (Malmö, Sweden, ~1500km from depot)  
    - The far delivery exceeds daily driving limits and needs overnight handling
    
    Returns:
        VRPInstance: Configured scenario for overnight feature testing
    """
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("Overnight Test Scenario")
    instance.is_realistic = True  # Added missing attribute
    
    # === LOCATIONS ===
    
    # 1. Main depot (Asti, Italy) - matching furgoni format
    depot = Location("depot", 8.2065, 44.8987, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=5)
    depot.address = "Asti Distribution Center, Italy"
    depot.lat = 44.8987  # Added missing lat/lon attributes
    depot.lon = 8.2065
    instance.add_location(depot)
    
    # === VEHICLES ===
    
    # Vehicle data with calculated capacities - matching furgoni format with max_time
    vehicles_data = {
        "FURGONE_1": 3500,  # Standard truck
        "FURGONE_2": 3500,  # Standard truck
    }
    
    # Add vehicles to the scenario with proper attributes (matching furgoni format)
    for vehicle_name, capacity in vehicles_data.items():
        vehicle_type = 'standard' if capacity <= 5000 else 'heavy'
        max_driving_time = 480  # 8 hours daily limit for overnight testing
        
        vehicle = Vehicle(
            vehicle_name, 
            capacity=capacity, 
            depot_id="depot",
            max_time=max_driving_time,
            vehicle_type=vehicle_type
        )
        instance.add_vehicle(vehicle)
    
    # === DELIVERY LOCATIONS ===
    
    # Delivery locations with time windows - matching furgoni format
    # Format: (location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup)
    locations_data = [
        ("cormano_mi", "via dell'Artigianato 1, 20032 Cormano MI, Italy", 9.1667, 45.5333, 480, 840, 30, False),  # Close destination
        ("malmo_sweden", "Malmö, Sweden - Menarini Diagnostics", 13.0038, 55.6050, 0, 1440, 30, False),  # Far destination - commented out for testing
    ]
    
    # Add all locations with proper attributes (matching furgoni format)
    for location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup in locations_data:
        location = Location(location_id, lon, lat, demand=0,
                           time_window_start=tw_start, time_window_end=tw_end, 
                           service_time=service_time)
        location.address = address
        location.lat = lat  # Added missing lat/lon attributes
        location.lon = lon
        location.is_pickup = is_pickup
        instance.add_location(location)
    
    # === DELIVERY CARGO REQUESTS ===
    
    # Create delivery/pickup requests using furgoni depot bay pattern
    # Delivery cargo data: create depot bays for each delivery
    delivery_cargo = [
        # FURGONE_1 - Close delivery (feasible within 8h)
        ("cormano_mi", 700),
        
        # FURGONE_2 - Far delivery (exceeds 8h limit - needs overnight)
         ("malmo_sweden", 1500),  # Temporarily commented out for testing
    ]
    
    # Create depot bays for delivery cargo (following furgoni/MODA pattern)
    depot_requests = []
    request_id = 1
    depot_location = instance.locations["depot"]

    for dropoff_id, cargo_weight in delivery_cargo:
        # Create virtual pickup bay at depot for this delivery
        pickup_bay_id = f"depot_bay_{request_id}"
        pickup_bay = Location(
            pickup_bay_id, 
            depot_location.x,
            depot_location.y,
            demand=0,
            time_window_start=0,
            time_window_end=1440,
            service_time=5
        )
        pickup_bay.lat = depot_location.lat
        pickup_bay.lon = depot_location.lon
        pickup_bay.address = f"Depot Loading Bay {request_id}"
        instance.add_location(pickup_bay)
        
        # Create request from depot bay to dropoff location
        depot_request = RideRequest(f"delivery_request_{request_id}", pickup_bay_id, dropoff_id, 
                                   passengers=cargo_weight, 
                                   volume=calculate_cargo_volume(cargo_weight),
                                   earliest_pickup=0,
                                   latest_dropoff=1440)
        instance.add_ride_request(depot_request)
        depot_requests.append(depot_request)
        request_id += 1

    # No pickup requests in this simplified overnight test scenario
    pickup_requests = []
    
    # === CALCULATE DISTANCES ===
    
    # Calculate distance matrix using same method as furgoni
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    # === SCENARIO SUMMARY ===
    
    # Add summary statistics (matching furgoni format)
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    total_delivery_weight = sum(r.passengers for r in depot_requests)
    total_pickup_weight = sum(r.passengers for r in pickup_requests)
    total_cargo_weight = total_delivery_weight + total_pickup_weight
    
    print(f"Created Overnight Test scenario with depot bay pattern:")
    print(f"  - Total locations: {len(instance.locations)}")
    print(f"  - Main depot: 1")
    print(f"  - Depot bays: {len(depot_requests)} (for deliveries)")
    print(f"  - Pickup bays: {len(pickup_requests)} (for pickups)")
    print(f"  - Delivery destinations: {len([l for l in instance.locations.values() if not getattr(l, 'is_pickup', False) and l.id != 'depot' and not l.id.startswith('depot_bay') and not l.id.startswith('pickup_bay')])}")
    print(f"  - Field pickup locations: {len([l for l in instance.locations.values() if getattr(l, 'is_pickup', False)])}")
    print(f"  - Vehicles: {len(instance.vehicles)}")
    print(f"  - Ride requests: {len(instance.ride_requests)} ({len(depot_requests)} depot→delivery + {len(pickup_requests)} pickup→depot)")
    print(f"Fleet composition: {len([v for v in instance.vehicles.values() if v.vehicle_type == 'standard'])} light vehicles + {len([v for v in instance.vehicles.values() if v.vehicle_type == 'heavy'])} heavy vehicles")
    
    print(f"\nCapacity utilization:")
    print(f"  - Total fleet capacity: {total_capacity:,} kg")
    print(f"  - Depot cargo (via bays): {total_delivery_weight:,} kg")
    print(f"  - Field pickup cargo: {total_pickup_weight:,} kg")
    print(f"  - Total cargo: {total_cargo_weight:,} kg = {total_cargo_weight/total_capacity*100:.1f}% of fleet capacity")
    
    print(f"\n📦 Depot Bay Strategy (matching furgoni/MODA pattern):")
    print(f"  - {len(depot_requests)} depot bays at same coordinates as main depot (for deliveries)")
    print(f"  - {len(pickup_requests)} unique pickup bays at depot (for pickups)")
    print(f"  - Each bay serves 1 request (delivery or pickup)")
    print(f"  - � Optimizer decides which bays to visit per vehicle")
    
    print(f"\n🌙 OVERNIGHT FEATURE TEST SPECIFICS:")
    print(f"  - ⏰ Daily driving limit: 8 hours (480 minutes) per vehicle")
    print(f"  - ✅ Close delivery: Depot → Cormano (~150km, ~3h travel) - FEASIBLE")
    print(f"  - ❌ Far delivery: Depot → Malmö (~1500km, ~15-20h travel) - EXCEEDS LIMIT")
    print(f"  - � Expected: Close delivery succeeds, far delivery needs overnight planning")
    
    # Print estimated travel times for verification
    if hasattr(instance, 'distance_matrix') and instance.distance_matrix is not None:
        print(f"\n📊 Distance verification:")
        location_names = list(instance.locations.keys())
        depot_idx = location_names.index("depot")
        if "cormano_mi" in location_names:
            cormano_idx = location_names.index("cormano_mi")
            distance_cormano = instance.distance_matrix[depot_idx][cormano_idx]
            estimated_time_cormano = distance_cormano / 50 * 60  # minutes at 50km/h
            print(f"  - Depot → Cormano: {distance_cormano:.1f} km (~{estimated_time_cormano:.0f} min)")
        if "malmo_sweden" in location_names:
            malmo_idx = location_names.index("malmo_sweden")
            distance_malmo = instance.distance_matrix[depot_idx][malmo_idx]
            estimated_time_malmo = distance_malmo / 50 * 60  # minutes at 50km/h
            print(f"  - Depot → Malmö: {distance_malmo:.1f} km (~{estimated_time_malmo:.0f} min)")
    
    return instance


def create_cormano_scenario() -> VRPInstance:
    """
    Create a simplified overnight test scenario with close and far destinations.
    
    This scenario is designed to test overnight/multi-day routing features:
    - 2 vehicles with 8-hour (480 minute) daily driving limits
    - 1 close delivery (Cormano, ~150km from depot)
    - 1 far delivery (Malmö, Sweden, ~1500km from depot)  
    - The far delivery exceeds daily driving limits and needs overnight handling
    
    Returns:
        VRPInstance: Configured scenario for overnight feature testing
    """
    import random
    random.seed(42)  # For reproducibility
    
    instance = VRPInstance("Overnight Test Scenario")
    instance.is_realistic = True  # Added missing attribute
    
    # === LOCATIONS ===
    
    # 1. Main depot (Asti, Italy) - matching furgoni format
    depot = Location("depot", 8.2065, 44.8987, demand=0, 
                    time_window_start=0, time_window_end=1440, service_time=5)
    depot.address = "Asti Distribution Center, Italy"
    depot.lat = 44.8987  # Added missing lat/lon attributes
    depot.lon = 8.2065
    instance.add_location(depot)
    
    # === VEHICLES ===
    
    # Vehicle data with calculated capacities - matching furgoni format with max_time
    vehicles_data = {
        "FURGONE_1": 3500,  # Standard truck
        "FURGONE_2": 3500,  # Standard truck
    }
    
    # Add vehicles to the scenario with proper attributes (matching furgoni format)
    for vehicle_name, capacity in vehicles_data.items():
        vehicle_type = 'standard' if capacity <= 5000 else 'heavy'
        max_driving_time = 480  # 8 hours daily limit for overnight testing
        
        vehicle = Vehicle(
            vehicle_name, 
            capacity=capacity, 
            depot_id="depot",
            max_time=max_driving_time,
            vehicle_type=vehicle_type
        )
        instance.add_vehicle(vehicle)
    
    # === DELIVERY LOCATIONS ===
    
    # Delivery locations with time windows - matching furgoni format
    # Format: (location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup)
    locations_data = [
        ("cormano_mi", "via dell'Artigianato 1, 20032 Cormano MI, Italy", 9.1667, 45.5333, 0, 1440, 30, False),  # Close destination
        # ("malmo_sweden", "Malmö, Sweden - Menarini Diagnostics", 13.0038, 55.6050, 0, 1440, 30, False),  # Far destination - commented out for testing
    ]
    
    # Add all locations with proper attributes (matching furgoni format)
    for location_id, address, lon, lat, tw_start, tw_end, service_time, is_pickup in locations_data:
        location = Location(location_id, lon, lat, demand=0,
                           time_window_start=tw_start, time_window_end=tw_end, 
                           service_time=service_time)
        location.address = address
        location.lat = lat  # Added missing lat/lon attributes
        location.lon = lon
        location.is_pickup = is_pickup
        instance.add_location(location)
    
    # === DELIVERY CARGO REQUESTS ===
    
    # Create delivery/pickup requests using furgoni depot bay pattern
    # Delivery cargo data: create depot bays for each delivery
    delivery_cargo = [
        # FURGONE_1 - Close delivery (feasible within 8h)
        ("cormano_mi", 700),
        
        # FURGONE_2 - Far delivery (exceeds 8h limit - needs overnight)
        # ("malmo_sweden", 1500),  # Temporarily commented out for testing
    ]
    
    # Create depot bays for delivery cargo (following furgoni/MODA pattern)
    depot_requests = []
    request_id = 1
    depot_location = instance.locations["depot"]

    for dropoff_id, cargo_weight in delivery_cargo:
        # Create virtual pickup bay at depot for this delivery
        pickup_bay_id = f"depot_bay_{request_id}"
        pickup_bay = Location(
            pickup_bay_id, 
            depot_location.x,
            depot_location.y,
            demand=0,
            time_window_start=0,
            time_window_end=1440,
            service_time=5
        )
        pickup_bay.lat = depot_location.lat
        pickup_bay.lon = depot_location.lon
        pickup_bay.address = f"Depot Loading Bay {request_id}"
        instance.add_location(pickup_bay)
        
        # Create request from depot bay to dropoff location
        depot_request = RideRequest(f"delivery_request_{request_id}", pickup_bay_id, dropoff_id, 
                                   passengers=cargo_weight, 
                                   volume=calculate_cargo_volume(cargo_weight),
                                   earliest_pickup=0,
                                   latest_dropoff=1440)
        instance.add_ride_request(depot_request)
        depot_requests.append(depot_request)
        request_id += 1

    # No pickup requests in this simplified overnight test scenario
    pickup_requests = []
    
    # === CALCULATE DISTANCES ===
    
    # Calculate distance matrix using same method as furgoni
    instance.calculate_distance_matrix(distance_method="manhattan")
    
    # === SCENARIO SUMMARY ===
    
    # Add summary statistics (matching furgoni format)
    total_capacity = sum(vehicle.capacity for vehicle in instance.vehicles.values())
    total_delivery_weight = sum(r.passengers for r in depot_requests)
    total_pickup_weight = sum(r.passengers for r in pickup_requests)
    total_cargo_weight = total_delivery_weight + total_pickup_weight
    
    print(f"Created Overnight Test scenario with depot bay pattern:")
    print(f"  - Total locations: {len(instance.locations)}")
    print(f"  - Main depot: 1")
    print(f"  - Depot bays: {len(depot_requests)} (for deliveries)")
    print(f"  - Pickup bays: {len(pickup_requests)} (for pickups)")
    print(f"  - Delivery destinations: {len([l for l in instance.locations.values() if not getattr(l, 'is_pickup', False) and l.id != 'depot' and not l.id.startswith('depot_bay') and not l.id.startswith('pickup_bay')])}")
    print(f"  - Field pickup locations: {len([l for l in instance.locations.values() if getattr(l, 'is_pickup', False)])}")
    print(f"  - Vehicles: {len(instance.vehicles)}")
    print(f"  - Ride requests: {len(instance.ride_requests)} ({len(depot_requests)} depot→delivery + {len(pickup_requests)} pickup→depot)")
    print(f"Fleet composition: {len([v for v in instance.vehicles.values() if v.vehicle_type == 'standard'])} light vehicles + {len([v for v in instance.vehicles.values() if v.vehicle_type == 'heavy'])} heavy vehicles")
    
    print(f"\nCapacity utilization:")
    print(f"  - Total fleet capacity: {total_capacity:,} kg")
    print(f"  - Depot cargo (via bays): {total_delivery_weight:,} kg")
    print(f"  - Field pickup cargo: {total_pickup_weight:,} kg")
    print(f"  - Total cargo: {total_cargo_weight:,} kg = {total_cargo_weight/total_capacity*100:.1f}% of fleet capacity")
    
    print(f"\n📦 Depot Bay Strategy (matching furgoni/MODA pattern):")
    print(f"  - {len(depot_requests)} depot bays at same coordinates as main depot (for deliveries)")
    print(f"  - {len(pickup_requests)} unique pickup bays at depot (for pickups)")
    print(f"  - Each bay serves 1 request (delivery or pickup)")
    print(f"  - � Optimizer decides which bays to visit per vehicle")
    
    print(f"\n🌙 OVERNIGHT FEATURE TEST SPECIFICS:")
    print(f"  - ⏰ Daily driving limit: 8 hours (480 minutes) per vehicle")
    print(f"  - ✅ Close delivery: Depot → Cormano (~150km, ~3h travel) - FEASIBLE")
    print(f"  - ❌ Far delivery: Depot → Malmö (~1500km, ~15-20h travel) - EXCEEDS LIMIT")
    print(f"  - � Expected: Close delivery succeeds, far delivery needs overnight planning")
    
    # Print estimated travel times for verification
    if hasattr(instance, 'distance_matrix') and instance.distance_matrix is not None:
        print(f"\n📊 Distance verification:")
        location_names = list(instance.locations.keys())
        depot_idx = location_names.index("depot")
        if "cormano_mi" in location_names:
            cormano_idx = location_names.index("cormano_mi")
            distance_cormano = instance.distance_matrix[depot_idx][cormano_idx]
            estimated_time_cormano = distance_cormano / 50 * 60  # minutes at 50km/h
            print(f"  - Depot → Cormano: {distance_cormano:.1f} km (~{estimated_time_cormano:.0f} min)")
        if "malmo_sweden" in location_names:
            malmo_idx = location_names.index("malmo_sweden")
            distance_malmo = instance.distance_matrix[depot_idx][malmo_idx]
            estimated_time_malmo = distance_malmo / 50 * 60  # minutes at 50km/h
            print(f"  - Depot → Malmö: {distance_malmo:.1f} km (~{estimated_time_malmo:.0f} min)")
    
    return instance