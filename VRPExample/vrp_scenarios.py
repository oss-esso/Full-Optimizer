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
            # Heavy truck (24-ton) with driver regulations
            capacity = 24000
            vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=capacity, depot_id=depot_id, max_time=540)
            vehicle.vehicle_type = "heavy"
            vehicle.max_driving_time = 270.0  # 4.5 hours in minutes
            vehicle.required_break_time = 45.0  # 45-minute break
            vehicle.max_total_work_time = 540.0  # 9 hours total work time (EU regulations)
            heavy_vehicle_count += 1
        else:
            # Standard truck (4-ton)
            capacity = 4000
            vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=capacity, depot_id=depot_id, max_time=600)
            vehicle.vehicle_type = "standard"
            vehicle.max_driving_time = 600.0  # 10 hours (no break requirement)
            vehicle.required_break_time = 0.0
            vehicle.max_total_work_time = 600.0  # 10 hours total work time
        
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
        depot_request = RideRequest(f"depot_request_{request_id}", pickup_bay_id, dropoff_id, passengers=cargo_weight)
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
        
        field_request = RideRequest(f"field_request_{request_id}", pickup_id, dropoff_id, passengers=cargo_weight)
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
                # Every 3rd vehicle is a heavy truck (24-ton)
                vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=24000, depot_id=depot_id, max_time=540)
                vehicle.vehicle_type = "heavy"
                vehicle.max_driving_time = 270.0  # 4.5 hours in minutes
                vehicle.required_break_time = 45.0  # 45-minute break
                vehicle.max_total_work_time = 540.0  # 9 hours total work time
                heavy_vehicle_count += 1
            else:
                # Standard truck (4-ton)
                vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=4000, depot_id=depot_id, max_time=600)
                vehicle.vehicle_type = "standard"
                vehicle.max_driving_time = 600.0  # 10 hours (no break requirement)
                vehicle.required_break_time = 0.0
                vehicle.max_total_work_time = 600.0  # 10 hours total work time
            
            instance.add_vehicle(vehicle)
            vehicle_id += 1
      # Create 100 cargo transport requests with realistic weights
    total_cargo_weight = 0
    for i in range(100):
        pickup_id = pickup_locations[i]
        dropoff_id = dropoff_locations[i]
        
        # Generate realistic cargo weights (200kg to 8000kg per shipment for large scenario)
        cargo_weight = random.randint(200, 8000)
        total_cargo_weight += cargo_weight
        
        request = RideRequest(f"request_{i+1}", pickup_id, dropoff_id, passengers=cargo_weight)
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
