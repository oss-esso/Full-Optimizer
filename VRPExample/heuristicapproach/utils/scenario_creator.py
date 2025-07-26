"""
Scenario Creator for EPDT Algorithm

This module reads company-specific Excel files and converts them into valid
scenarios for the EPDT (Enhanced Parallel Diversified Tabu) algorithm.

The script bridges the gap between real-world operational data and the input
requirements of the EPDT solver, including automated address geocoding.

Author: Generated following the scenario_creation_guide.md
"""

import json
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
import logging
from pathlib import Path
import sys
import os

# Add the algo directory to the path to import data structures
current_dir = Path(__file__).parent
algo_dir = current_dir.parent / "algo"
sys.path.append(str(algo_dir))

from epdt_data_structures import Order, Task, Vehicle, TaskType
from geopy.geocoders import Photon
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Configuration constants
GEOCODE_CACHE_FILE = "geocode_cache.json"
DEFAULT_SERVICE_TIME = 15.0  # Default service time in minutes
DEFAULT_DEPOT_COORDS = (44.9009, 8.2057)  # Asti, Italy - Via del Lavoro 38

# Column mapping for CONSEGNE sheet
CONSEGNE_COLUMN_MAPPING = {
    'NAME': 'company_name',
    'ADDRESS': 'address', 
    'POSTAL CODE': 'postal_code',
    'COUNTRY': 'country',
    'TIME WINDOW DATE (24=0)': 'time_window_date',
    'TIME WINDOW HOURS': 'time_window_hours',
    'SERVICE TIME': 'service_time',
    'DELIVERY OR PICKUP': 'task_type',
    'LOAD KG': 'load_kg',
    'LOAD VOLUME M^3': 'load_volume',
    'PALLET': 'pallets',
    'LOW TEMP': 'low_temp',
    'LOADER': 'loader'
}

# Column mapping for VEICOLI sheet
VEICOLI_COLUMN_MAPPING = {
    'NUMBER PLATE': 'vehicle_id',
    'TYPE OF VEHICLE': 'vehicle_type',
    'MAX LOAD KG': 'max_load_kg',
    'PALLET': 'pallet_capacity',
    'MAX LOAD VOLUME M^3': 'max_load_volume',
    'LOADER': 'loader',
    'LOW TEMP': 'low_temp',
    'LAST IN FIRST OUT': 'lifo_required',
    'HANGERS': 'hangers',
    'REGULATIONS': 'regulations'
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScenarioCreationError(Exception):
    """Custom exception for scenario creation errors."""
    pass


def safe_parse_value(row: pd.Series, column: str, default_value, value_type=None, boolean_values=None):
    """
    Safely parse a value from a DataFrame row with type conversion and error handling.
    
    Args:
        row: Pandas series containing the data
        column: Column name to parse
        default_value: Default value if parsing fails
        value_type: Type to convert to (int, float, str, bool)
        boolean_values: List of values that should be considered True for boolean conversion
        
    Returns:
        Parsed value or default_value if parsing fails
    """
    if column not in row or pd.isna(row[column]):
        return default_value
    
    try:
        value = row[column]
        
        # Handle boolean conversion
        if value_type == bool:
            if boolean_values is None:
                boolean_values = ['YES', 'TRUE', '1', 'SI', 'Y']
            return str(value).strip().upper() in boolean_values
        
        # Handle numeric conversion with comma-decimal format
        if value_type in [int, float]:
            # Handle European decimal format (comma instead of dot)
            if isinstance(value, str):
                value = value.replace(',', '.')
            return value_type(value)
        
        # Handle string conversion
        if value_type == str:
            return str(value).strip()
        
        # No type conversion requested
        return value
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not parse {column} value '{row[column]}': {e}")
        return default_value


def load_geocode_cache() -> Dict[str, Dict[str, float]]:
    """
    Load the geocoding cache from disk.
    
    Returns:
        Dict containing cached address -> coordinates mappings
    """
    try:
        cache_path = Path(GEOCODE_CACHE_FILE)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load geocode cache: {e}")
    return {}


def save_geocode_cache(cache: Dict[str, Dict[str, float]]) -> None:
    """
    Save the geocoding cache to disk.
    
    Args:
        cache: Dictionary containing address -> coordinates mappings
    """
    try:
        with open(GEOCODE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Geocoding cache saved with {len(cache)} entries")
    except Exception as e:
        logger.error(f"Failed to save geocode cache: {e}")


def get_coordinates(address: str, cache: Dict[str, Dict[str, float]]) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude coordinates for an address.
    
    Uses caching to avoid repeated API calls for the same address.
    Enhanced with retry logic and better timeout handling.
    
    Args:
        address: Full address string to geocode
        cache: Geocoding cache dictionary
        
    Returns:
        Tuple of (latitude, longitude) or None if geocoding failed
    """
    # Check cache first
    if address in cache:
        coords = cache[address]
        return (coords["lat"], coords["lon"])
    
    # Initialize geocoder with optimized settings
    geolocator = Photon(user_agent="epdt_scenario_creator", timeout=15)
    
    # Retry configuration
    max_retries = 2
    base_delay = 1.5  # Slightly longer delay for better Photon performance
    
    for attempt in range(max_retries + 1):
        try:
            # Progressive delay: 1.5s, 3s, 4.5s
            if attempt > 0:
                delay = base_delay * (1 + attempt * 0.5)
                logger.info(f"Retrying geocoding for {address} (attempt {attempt + 1}) after {delay}s delay...")
                time.sleep(delay)
            else:
                time.sleep(base_delay)  # Always respect rate limits
            
            location = geolocator.geocode(address, timeout=15)
            if location:
                coords = {"lat": location.latitude, "lon": location.longitude}
                cache[address] = coords
                logger.info(f"Geocoded: {address} -> ({coords['lat']:.6f}, {coords['lon']:.6f})")
                return (location.latitude, location.longitude)
            else:
                logger.warning(f"Could not geocode address: {address}")
                break  # No point retrying if address not found
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < max_retries:
                logger.warning(f"Geocoding timeout for {address} (attempt {attempt + 1}): {e}")
                continue  # Retry on timeout/service error
            else:
                logger.error(f"Geocoding failed after {max_retries + 1} attempts for {address}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error geocoding {address}: {e}")
            break  # Don't retry on unexpected errors
    
    return None


def construct_full_address(row: pd.Series) -> str:
    """
    Construct a full address string from Excel row data.
    
    Args:
        row: Pandas series containing address components
        
    Returns:
        Full address string suitable for geocoding
    """
    address_parts = []
    
    # Add street address
    if 'ADDRESS' in row and pd.notna(row['ADDRESS']):
        address_parts.append(str(row['ADDRESS']).strip())
    
    # Add postal code (ensure it's formatted as integer, not float)
    if 'POSTAL CODE' in row and pd.notna(row['POSTAL CODE']):
        postal_code = row['POSTAL CODE']
        # Convert float postal codes to integers to avoid .0 suffix
        if isinstance(postal_code, (int, float)):
            postal_code = int(postal_code)
        address_parts.append(str(postal_code).strip())
    
    # Add country
    if 'COUNTRY' in row and pd.notna(row['COUNTRY']):
        address_parts.append(str(row['COUNTRY']).strip())
    
    return ", ".join(address_parts)


def parse_time_window(time_str: str) -> Optional[float]:
    """
    Parse a time string (HH:MM) into minutes from start of day.
    
    Args:
        time_str: Time string in HH:MM format
        
    Returns:
        Minutes from start of day, or None if parsing failed
    """
    if pd.isna(time_str) or not time_str:
        return None
    
    try:
        time_obj = datetime.strptime(str(time_str).strip(), "%H:%M")
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        logger.warning(f"Could not parse time: {time_str}")
        return None


def create_task_from_row(row: pd.Series, geocode_cache: Dict[str, Dict[str, float]]) -> Optional[Task]:
    """
    Create a Task object from an Excel row using column mapping.
    
    Args:
        row: Pandas series containing task data
        geocode_cache: Geocoding cache for address lookups
        
    Returns:
        Task object or None if creation failed
    """
    try:
        # Parse basic information using safe parsing
        company_name = safe_parse_value(row, 'NAME', 'UNKNOWN', str)
        task_id = f"TASK_{company_name.replace(' ', '_')[:20]}_{row.name}"
        order_id = f"ORDER_{company_name.replace(' ', '_')[:20]}_{row.name}"
        
        # Parse task type
        task_type_str = safe_parse_value(row, 'DELIVERY OR PICKUP', 'DELIVERY', str).upper()
        if task_type_str in ['PICKUP', 'PICK']:
            task_type = TaskType.PICKUP
        elif task_type_str in ['DELIVERY', 'DELIVER']:
            task_type = TaskType.DELIVERY
        else:
            task_type = TaskType.DELIVERY
            logger.warning(f"Unknown task type '{task_type_str}' for {company_name}, defaulting to DELIVERY")
        
        # Get coordinates
        full_address = construct_full_address(row)
        coordinates = get_coordinates(full_address, geocode_cache)
        
        if not coordinates:
            logger.error(f"Could not geocode address for {company_name}: {full_address}")
            return None
        
        lat, lon = coordinates
        
        # Parse numeric fields using safe parsing
        load_kg = safe_parse_value(row, 'LOAD KG', 0.0, float)
        load_volume = safe_parse_value(row, 'LOAD VOLUME M^3', 0.0, float)
        service_time = safe_parse_value(row, 'SERVICE TIME', DEFAULT_SERVICE_TIME, float)
        pallets = safe_parse_value(row, 'PALLET', 0, int)
        
        # Parse boolean fields
        requires_low_temp = safe_parse_value(row, 'LOW TEMP', False, bool)
        requires_loader = safe_parse_value(row, 'LOADER', False, bool)
        
        # For pickup tasks, demand is positive; for delivery, negative
        if task_type == TaskType.PICKUP:
            demand = abs(load_kg)
            volume = abs(load_volume)
            pallets_adjusted = abs(pallets)  # Positive for pickup
        else:  # DELIVERY
            demand = -abs(load_kg)
            volume = -abs(load_volume)
            pallets_adjusted = -abs(pallets)  # Negative for delivery
        
        # Parse time windows (simplified for now)
        time_window_hours = safe_parse_value(row, 'TIME WINDOW HOURS', None, str)
        earliest_time = None
        latest_time = None
        
        if time_window_hours and '-' in time_window_hours:
            try:
                start_str, end_str = time_window_hours.split('-', 1)
                # Simple hour parsing (assumes format like "8-12" or "08:00-12:00")
                if ':' in start_str:
                    earliest_time = parse_time_window(start_str.strip())
                    latest_time = parse_time_window(end_str.strip())
                else:
                    # Just hours
                    start_hour = int(start_str.strip())
                    end_hour = int(end_str.strip())
                    earliest_time = start_hour * 60
                    latest_time = end_hour * 60
            except Exception as e:
                logger.warning(f"Could not parse time window '{time_window_hours}' for {company_name}: {e}")
        
        # Set priority based on special requirements
        priority = 1  # Default priority
        if requires_low_temp:
            priority = 2  # Higher priority for temperature-sensitive cargo
        
        # Create task
        task = Task(
            id=task_id,
            location_id=full_address,  # Use full address as location ID
            task_type=task_type,
            order_id=order_id,
            lat=lat,
            lon=lon,
            service_time=service_time,
            earliest_time=earliest_time,
            latest_time=latest_time,
            soft_time_window=True,  # Default to soft time windows
            demand=demand,
            volume=volume,
            pallets=pallets_adjusted,  # Use adjusted pallets with proper sign
            priority=priority,
            requires_low_temp=requires_low_temp,
            requires_loader=requires_loader
        )
        
        logger.debug(f"Created task: {task_id} ({task_type.value}) at ({lat:.6f}, {lon:.6f})")
        return task
        
    except Exception as e:
        logger.error(f"Error creating task from row: {e}")
        return None


def create_vehicle_from_row(row: pd.Series) -> Optional[Vehicle]:
    """
    Create a Vehicle object from an Excel row using column mapping.
    
    Args:
        row: Pandas series containing vehicle data
        
    Returns:
        Vehicle object or None if creation failed
    """
    try:
        # Parse basic information using safe parsing
        vehicle_id = safe_parse_value(row, 'NUMBER PLATE', 'UNKNOWN', str)
        depot_id = 'DEPOT-PADOVA'  # Default depot
        
        # Parse capacity constraints
        weight_capacity = safe_parse_value(row, 'MAX LOAD KG', 3500.0, float)
        volume_capacity = safe_parse_value(row, 'MAX LOAD VOLUME M^3', 25.0, float)
        pallet_capacity = safe_parse_value(row, 'PALLET', None, int)
        
        # Parse vehicle type
        vehicle_type_str = safe_parse_value(row, 'TYPE OF VEHICLE', 'FURGONE', str).upper()
        if vehicle_type_str in ['FURGONE', 'VAN']:
            vehicle_type = 'standard'
            cost_per_km = 1.0
            fixed_cost = 50.0
        elif vehicle_type_str in ['CAMION', 'TRUCK']:
            vehicle_type = 'heavy'
            cost_per_km = 1.5
            fixed_cost = 100.0
        else:
            vehicle_type = 'standard'
            cost_per_km = 1.0
            fixed_cost = 50.0
        
        # Parse boolean requirements
        lifo_required = safe_parse_value(row, 'LAST IN FIRST OUT', False, bool)
        
        # Parse capabilities
        capabilities = set()
        
        if safe_parse_value(row, 'LOADER', False, bool):
            capabilities.add('loader')
        
        if safe_parse_value(row, 'LOW TEMP', False, bool):
            capabilities.add('low_temp')
        
        if safe_parse_value(row, 'HANGERS', False, bool):
            capabilities.add('hangers')
        
        # Set regulations-based constraints
        regulations = safe_parse_value(row, 'REGULATIONS', False, bool)
        if regulations:
            # Vehicle subject to driving time regulations
            max_driving_time = 540.0  # 9 hours
            max_work_time = 780.0     # 13 hours
        else:
            # More flexible for local deliveries
            max_driving_time = 720.0  # 12 hours
            max_work_time = 960.0     # 16 hours
        
        vehicle = Vehicle(
            id=vehicle_id,
            depot_id=depot_id,
            weight_capacity=weight_capacity,
            volume_capacity=volume_capacity,
            pallet_capacity=pallet_capacity,
            cost_per_km=cost_per_km,
            fixed_cost=fixed_cost,
            vehicle_type=vehicle_type,
            capabilities=capabilities,
            lifo_required=lifo_required,
            max_driving_time=max_driving_time,
            max_work_time=max_work_time
        )
        
        logger.debug(f"Created vehicle: {vehicle_id} ({vehicle_type}) with capacity {weight_capacity}kg/{volume_capacity}m³")
        return vehicle
        
    except Exception as e:
        logger.error(f"Error creating vehicle from row: {e}")
        return None


def create_scenario_from_excel(file_path: str) -> Tuple[List[Order], List[Vehicle]]:
    """
    Read an Excel file and create a VRP scenario for the EPDT algorithm.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Tuple of (orders_list, vehicles_list)
        
    Raises:
        ScenarioCreationError: If scenario creation fails
    """
    logger.info(f"Reading scenario from {file_path}...")
    
    try:
        # Read Excel sheets with correct names
        orders_df = pd.read_excel(file_path, sheet_name="CONSEGNE")
        vehicles_df = pd.read_excel(file_path, sheet_name="VEICOLI")
        
        logger.info(f"Loaded {len(orders_df)} delivery locations and {len(vehicles_df)} vehicles")
        
    except Exception as e:
        raise ScenarioCreationError(f"Failed to read Excel file: {e}")
    
    # Load geocoding cache
    geocode_cache = load_geocode_cache()
    
    # --- Vehicle Creation ---
    vehicles = []
    logger.info("Creating vehicles...")
    
    for idx, row in vehicles_df.iterrows():
        vehicle = create_vehicle_from_row(row)
        if vehicle:
            vehicles.append(vehicle)
        else:
            logger.warning(f"Skipped vehicle at row {idx + 2}")  # +2 for 1-indexed + header
    
    logger.info(f"Created {len(vehicles)} vehicles")
    
    # --- Order and Task Creation with Depot Bay Pairs ---
    orders = []
    logger.info("Creating orders and tasks...")
    
    # Depot coordinates (Padova, Italy)
    depot_lat, depot_lon = DEFAULT_DEPOT_COORDS
    
    # Each row in CONSEGNE represents a location that needs service
    # We'll create paired tasks: customer task + corresponding depot bay task
    
    for idx, row in orders_df.iterrows():
        customer_task = create_task_from_row(row, geocode_cache)
        if customer_task:
            # Create corresponding depot bay task
            depot_task = create_depot_bay_task(customer_task, depot_lat, depot_lon)
            
            if depot_task:
                # Create order with paired tasks
                if customer_task.is_pickup():
                    # Customer pickup requires depot delivery (unloading)
                    pickup_tasks = [customer_task]
                    delivery_tasks = [depot_task]
                else:
                    # Customer delivery requires depot pickup (loading)
                    pickup_tasks = [depot_task]
                    delivery_tasks = [customer_task]
                
                # Determine order properties from customer task
                priority = customer_task.priority
                is_urgent = priority >= 3
                is_mandatory = True  # All orders are now mandatory for testing/debugging
                
                order = Order(
                    id=customer_task.order_id,
                    pickup_tasks=pickup_tasks,
                    delivery_tasks=delivery_tasks,
                    priority=priority,
                    is_urgent=is_urgent,
                    is_mandatory=is_mandatory
                )
                
                orders.append(order)
                logger.debug(f"Created order {order.id} with paired tasks: customer {customer_task.task_type.value} + depot {depot_task.task_type.value}")
            else:
                logger.warning(f"Failed to create depot bay task for {customer_task.id}")
        else:
            logger.warning(f"Skipped location at row {idx + 2}")  # +2 for 1-indexed + header
    
    logger.info(f"Created {len(orders)} orders")
    
    # --- Finalization ---
    save_geocode_cache(geocode_cache)
    
    # Validation
    if not orders:
        raise ScenarioCreationError("No valid orders were created")
    
    if not vehicles:
        raise ScenarioCreationError("No valid vehicles were created")
    
    logger.info(f"Scenario creation complete: {len(orders)} orders, {len(vehicles)} vehicles")
    return orders, vehicles


def create_depot_bay_task(customer_task: Task, depot_lat: float, depot_lon: float) -> Optional[Task]:
    """
    Create a corresponding depot bay task for a customer task.
    
    Args:
        customer_task: The original customer task (pickup or delivery)
        depot_lat: Depot latitude coordinate
        depot_lon: Depot longitude coordinate
        
    Returns:
        Depot bay task or None if creation failed
    """
    try:
        # Determine depot task type (opposite of customer task)
        if customer_task.is_pickup():
            # Customer pickup → Depot delivery (unload cargo at depot)
            depot_task_type = TaskType.DELIVERY
            depot_task_id = f"DEPOT_DELIVERY_{customer_task.order_id}"
            depot_demand = -customer_task.demand  # Negative for delivery
            depot_volume = -customer_task.volume  # Negative for delivery
            depot_pallets = -customer_task.pallets  # Negative for delivery (unloading pallets)
        else:
            # Customer delivery → Depot pickup (load cargo from depot)
            depot_task_type = TaskType.PICKUP
            depot_task_id = f"DEPOT_PICKUP_{customer_task.order_id}"
            depot_demand = -customer_task.demand  # Convert negative delivery to positive pickup
            depot_volume = -customer_task.volume  # Convert negative delivery volume to positive pickup
            depot_pallets = -customer_task.pallets  # Convert negative delivery pallets to positive pickup
        
        # Create depot bay task
        depot_task = Task(
            id=depot_task_id,
            location_id="DEPOT_BAY_ASTI",
            task_type=depot_task_type,
            order_id=customer_task.order_id,
            lat=depot_lat,
            lon=depot_lon,
            service_time=5.0,  # 5 minutes for depot operations
            earliest_time=None,  # No time constraints for depot
            latest_time=None,
            soft_time_window=True,
            demand=depot_demand,
            volume=depot_volume,
            pallets=depot_pallets,  # Fixed: pallets now follow same sign logic as weight/volume
            priority=customer_task.priority,
            requires_low_temp=customer_task.requires_low_temp,
            requires_loader=customer_task.requires_loader
        )
        
        logger.debug(f"Created depot bay task: {depot_task_id} ({depot_task_type.value}) for customer task {customer_task.id}")
        return depot_task
        
    except Exception as e:
        logger.error(f"Error creating depot bay task for {customer_task.id}: {e}")
        return None


def validate_scenario(orders: List[Order], vehicles: List[Vehicle]) -> bool:
    """
    Validate a created scenario for basic consistency.
    
    Args:
        orders: List of order objects
        vehicles: List of vehicle objects
        
    Returns:
        True if scenario is valid, False otherwise
    """
    logger.info("Validating scenario...")
    
    # Check orders
    for order in orders:
        if not order.get_all_tasks():
            logger.error(f"Order {order.id} has no tasks")
            return False
        
        # Check task coordinates
        for task in order.get_all_tasks():
            if not (-90 <= task.lat <= 90) or not (-180 <= task.lon <= 180):
                logger.error(f"Task {task.id} has invalid coordinates: ({task.lat}, {task.lon})")
                return False
    
    # Check vehicles
    for vehicle in vehicles:
        if vehicle.weight_capacity <= 0 or vehicle.volume_capacity <= 0:
            logger.error(f"Vehicle {vehicle.id} has invalid capacity")
            return False
    
    logger.info("Scenario validation passed")
    return True


def main():
    """Main function for testing the scenario creator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create EPDT scenario from Excel file')
    parser.add_argument('excel_file', help='Path to Excel file')
    parser.add_argument('--validate', action='store_true', help='Run validation after creation')
    parser.add_argument('--output', help='Output file for scenario (optional)')
    parser.add_argument('--no-geocoding', action='store_true', help='Skip geocoding (use fake coordinates for testing)')
    
    args = parser.parse_args()
    
    try:
        if args.no_geocoding:
            # Use test version without geocoding
            from test_scenario_creator import create_test_scenario
            orders, vehicles = create_test_scenario(args.excel_file)
        else:
            orders, vehicles = create_scenario_from_excel(args.excel_file)
        
        if args.validate:
            if not validate_scenario(orders, vehicles):
                logger.error("Scenario validation failed")
                return 1
        
        if args.output:
            # Save scenario to file (implementation would depend on desired format)
            logger.info(f"Scenario would be saved to {args.output}")
        
        print(f"Successfully created scenario with {len(orders)} orders and {len(vehicles)} vehicles")
        return 0
        
    except ScenarioCreationError as e:
        logger.error(f"Scenario creation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
