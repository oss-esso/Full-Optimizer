"""
Enhanced Scenario Creator implementing TODO #23 specifications

This module implements the exact requirements from TODO #23 for parsing
Excel files with proper ORDER grouping using pandas.groupby('ORDER').

Specifications implemented:
1. CONSEGNE sheet: Tasks grouped by ORDER column into Order objects  
2. AUTISTI sheet: Driver objects with license constraints
3. VEICOLI sheet: Vehicle objects with proper capability matching
"""

import json
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import os

# Add the algo directory to the path
current_dir = Path(__file__).parent
algo_dir = current_dir.parent / "algo"
sys.path.append(str(algo_dir))

from epdt_data_structures import Order, Task, Vehicle, Driver, TaskType
import requests

# Configuration constants
GEOCODE_CACHE_FILE = "geocode_cache.json"
DEFAULT_SERVICE_TIME = 15.0
DEFAULT_DEPOT_COORDS = (44.9009, 8.2057)  # Asti, Italy

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScenarioCreationError(Exception):
    """Custom exception for scenario creation errors."""
    pass


def load_geocode_cache() -> Dict[str, Dict[str, float]]:
    """Load geocoding cache from file."""
    try:
        with open(GEOCODE_CACHE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_geocode_cache(cache: Dict[str, Dict[str, float]]) -> None:
    """Save geocoding cache to file."""
    with open(GEOCODE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_coordinates(address: str, cache: Dict[str, Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Get coordinates for an address using Nominatim API with caching.
    
    TODO #22 implementation: Uses Nominatim exclusively for geocoding.
    """
    if address in cache:
        coords = cache[address]
        return coords.get('lat'), coords.get('lon')
    
    try:
        # Use Nominatim API with proper headers
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': address,
            'format': 'json',
            'limit': 1
        }
        headers = {
            'User-Agent': 'EPDT-Scenario-Creator/1.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            
            # Cache the result
            cache[address] = {'lat': lat, 'lon': lon}
            return lat, lon
        else:
            logger.warning(f"No geocoding results for: {address}")
            return None, None
            
    except Exception as e:
        logger.error(f"Geocoding failed for {address}: {e}")
        return None, None


def parse_time_window(time_str: str) -> int:
    """
    Convert time string (HH:MM:SS) to minutes from start of day.
    
    Args:
        time_str: Time in format "HH:MM:SS" or "HH:MM"
        
    Returns:
        Total minutes from start of day (e.g., "09:00:00" -> 540)
    """
    if pd.isna(time_str) or not time_str:
        return 0
        
    try:
        # Handle both HH:MM:SS and HH:MM formats
        if ':' in str(time_str):
            parts = str(time_str).split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        else:
            # Assume it's already in minutes
            return int(time_str)
    except (ValueError, IndexError):
        logger.warning(f"Could not parse time: {time_str}")
        return 0


def concatenate_address(row: pd.Series) -> str:
    """
    Concatenate address components as specified in TODO #23.
    
    Format: "Via Roma 10, 10121, Torino, TO, Italy"
    """
    components = []
    
    # Street and house number
    street = str(row.get('STREET', '')).strip()
    house_number = str(row.get('HOUSE NUMBER', '')).strip()
    if street and house_number:
        components.append(f"{street} {house_number}")
    elif street:
        components.append(street)
    
    # Postal code
    postal_code = str(row.get('POSTAL CODE', '')).strip()
    if postal_code:
        components.append(postal_code)
    
    # City
    city = str(row.get('CITY', '')).strip()
    if city:
        components.append(city)
    
    # Province
    province = str(row.get('PROVINCE', '')).strip()
    if province:
        components.append(province)
    
    # Country
    country = str(row.get('COUNTRY', '')).strip()
    if country:
        components.append(country)
    
    return ', '.join(components)


def create_task_from_consegne_row(row: pd.Series, geocode_cache: Dict[str, Dict[str, float]]) -> Optional[Task]:
    """
    Create a Task object from a CONSEGNE sheet row according to TODO #23 specification.
    
    Key requirements:
    - DELIVERY tasks have negative load values
    - PICKUP tasks have positive load values
    - Time windows converted to minutes
    - Address concatenation as specified
    """
    try:
        # Extract ORDER ID (primary key for grouping)
        order_id = str(row.get('ORDER', '')).strip()
        if not order_id:
            logger.warning(f"Row missing ORDER column: {row.name}")
            return None
        
        # Company name
        company_name = str(row.get('COMPANY', '')).strip()
        
        # Generate task ID
        task_id = f"TASK_{order_id}_{row.name}"
        
        # Address concatenation
        address = concatenate_address(row)
        if not address:
            logger.warning(f"Could not create address for task {task_id}")
            return None
        
        # Geocoding
        lat, lon = get_coordinates(address, geocode_cache)
        if lat is None or lon is None:
            logger.warning(f"Could not geocode address for task {task_id}: {address}")
            return None
        
        # Parse task type
        task_type_str = str(row.get('TASK', '')).strip().upper()
        if task_type_str == 'DELIVERY':
            task_type = TaskType.DELIVERY
        elif task_type_str == 'PICKUP':
            task_type = TaskType.PICKUP
        else:
            logger.warning(f"Unknown task type: {task_type_str} for task {task_id}")
            return None
        
        # Parse time windows (convert to minutes)
        earliest_day = int(row.get('EARLIEST DAY', 1))
        latest_day = int(row.get('LATEST DAY', 1))
        time_window_start = parse_time_window(row.get('TIME WINDOW START', '09:00:00'))
        time_window_end = parse_time_window(row.get('TIME WINDOW END', '18:00:00'))
        
        # Service time
        service_time = float(row.get('SERVICE TIME', DEFAULT_SERVICE_TIME))
        
        # Load values - CRUCIAL: negative for DELIVERY, positive for PICKUP
        load_kg = float(row.get('LOAD KG', 0))
        load_volume = float(row.get('LOAD VOLUME M^3', 0))
        pallets = float(row.get('PALLETS', 0))
        
        if task_type == TaskType.DELIVERY:
            # Make negative for delivery tasks (capacity is freed up)
            load_kg = -abs(load_kg)
            load_volume = -abs(load_volume)
            pallets = -abs(pallets)
        else:
            # Keep positive for pickup tasks
            load_kg = abs(load_kg)
            load_volume = abs(load_volume)
            pallets = abs(pallets)
        
        # Capabilities
        capabilities = set()
        if str(row.get('LOW_TEMP', '')).strip().upper() == 'YES':
            capabilities.add('LOW_TEMP')
        if str(row.get('LOADER', '')).strip().upper() == 'YES':
            capabilities.add('LOADER')
        if str(row.get('HANGERS', '')).strip().upper() == 'YES':
            capabilities.add('HANGERS')
        
        # Create task
        task = Task(
            id=task_id,
            order_id=order_id,
            location_id=f"LOC_{task_id}",
            task_type=task_type,
            latitude=lat,
            longitude=lon,
            earliest_day=earliest_day,
            latest_day=latest_day,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            service_time=service_time,
            demand=load_kg,
            volume=load_volume,
            pallets=pallets,
            priority=1,
            required_capabilities=capabilities
        )
        
        logger.debug(f"Created task {task_id}: {task_type.value} with demand={load_kg}")
        return task
        
    except Exception as e:
        logger.error(f"Error creating task from CONSEGNE row {row.name}: {e}")
        return None


def create_vehicle_from_veicoli_row(row: pd.Series) -> Optional[Vehicle]:
    """
    Create a Vehicle object from VEICOLI sheet row according to TODO #23 specification.
    """
    try:
        # Vehicle ID (Number plate)
        vehicle_id = str(row.get('NUMBER PLATE', '')).strip()
        if not vehicle_id:
            logger.warning(f"Row missing NUMBER PLATE: {row.name}")
            return None
        
        # Vehicle type
        vehicle_type = str(row.get('TYPE OF VEHICLE', 'Van')).strip()
        
        # Capacity limits
        max_load_kg = float(row.get('MAX LOAD KG', 1000))
        max_load_volume = float(row.get('MAX LOAD VOLUME M^3', 10))
        pallet_capacity = int(row.get('PALLET', 0))
        
        # Costs
        cost_per_km = float(row.get('COST PER KM', 0.5))
        fixed_cost = float(row.get('FIXED COST', 50))
        
        # Capabilities
        capabilities = set()
        if str(row.get('LOW_TEMP', '')).strip().upper() == 'YES':
            capabilities.add('LOW_TEMP')
        if str(row.get('LOADER', '')).strip().upper() == 'YES':
            capabilities.add('LOADER')
        if str(row.get('HANGERS', '')).strip().upper() == 'YES':
            capabilities.add('HANGERS')
        
        # Regulations (Hours of Service rules)
        regulations = str(row.get('REGULATIONS', 'NO')).strip().upper() == 'YES'
        
        # LIFO loading
        lifo_required = str(row.get('LAST IN FIRST OUT', 'NO')).strip().upper() == 'YES'
        
        vehicle = Vehicle(
            id=vehicle_id,
            vehicle_type=vehicle_type,
            max_weight=max_load_kg,
            max_volume=max_load_volume,
            max_pallets=pallet_capacity,
            cost_per_km=cost_per_km,
            fixed_cost=fixed_cost,
            capabilities=capabilities,
            home_depot_id='main_depot',
            regulations_apply=regulations,
            lifo_required=lifo_required
        )
        
        logger.debug(f"Created vehicle {vehicle_id}: {vehicle_type} with capacity {max_load_kg}kg")
        return vehicle
        
    except Exception as e:
        logger.error(f"Error creating vehicle from VEICOLI row {row.name}: {e}")
        return None


def create_driver_from_autisti_row(row: pd.Series) -> Optional[Driver]:
    """
    Create a Driver object from AUTISTI sheet row according to TODO #23 specification.
    
    Key requirement: LICENSE is a hard constraint for vehicle assignment.
    """
    try:
        # Driver information
        driver_name = str(row.get('DRIVER', '')).strip()
        if not driver_name:
            logger.warning(f"Row missing DRIVER name: {row.name}")
            return None
        
        # License (hard constraint)
        license = str(row.get('LICENSE', 'B')).strip()
        
        # Preferred vehicle (from LICENSE PLATE column)
        preferred_vehicle_id = str(row.get('LICENSE PLATE', '')).strip()
        
        # Cost per hour
        cost_per_hour = float(row.get('COST PER HOUR', 25.0))
        
        # Driver ID
        driver_id = f"DRIVER_{driver_name.replace(' ', '_')}"
        
        # Default hours based on license type
        if license in ['CE', 'C']:
            max_shift_hours = 13.0
            max_driving_hours = 9.0
        else:
            max_shift_hours = 12.0
            max_driving_hours = 8.0
        
        driver = Driver(
            id=driver_id,
            name=driver_name,
            license=license,
            default_vehicle_id=preferred_vehicle_id,
            cost_per_hour=cost_per_hour,
            max_shift_hours=max_shift_hours,
            max_driving_hours=max_driving_hours,
            home_depot_id='main_depot',
            qualifications=set()  # Will be enhanced later
        )
        
        logger.debug(f"Created driver {driver_id}: {driver_name} with license {license}")
        return driver
        
    except Exception as e:
        logger.error(f"Error creating driver from AUTISTI row {row.name}: {e}")
        return None


def create_scenario_from_excel_v23(file_path: str) -> Tuple[List[Order], List[Vehicle], List[Driver]]:
    """
    Create scenario from Excel file implementing TODO #23 specifications.
    
    Key feature: Uses pandas.groupby('ORDER') to group tasks into Order objects.
    """
    logger.info(f"Reading scenario from {file_path} using TODO #23 specifications...")
    
    try:
        # Read Excel sheets
        consegne_df = pd.read_excel(file_path, sheet_name="CONSEGNE")
        veicoli_df = pd.read_excel(file_path, sheet_name="VEICOLI")
        autisti_df = pd.read_excel(file_path, sheet_name="AUTISTI")
        
        logger.info(f"Loaded {len(consegne_df)} task rows, {len(veicoli_df)} vehicles, {len(autisti_df)} drivers")
        
    except Exception as e:
        raise ScenarioCreationError(f"Failed to read Excel file: {e}")
    
    # Load geocoding cache
    geocode_cache = load_geocode_cache()
    
    # --- Create Vehicles ---
    vehicles = []
    logger.info("Creating vehicles from VEICOLI sheet...")
    
    for idx, row in veicoli_df.iterrows():
        vehicle = create_vehicle_from_veicoli_row(row)
        if vehicle:
            vehicles.append(vehicle)
        else:
            logger.warning(f"Skipped vehicle at row {idx + 2}")
    
    logger.info(f"Created {len(vehicles)} vehicles")
    
    # --- Create Drivers ---
    drivers = []
    logger.info("Creating drivers from AUTISTI sheet...")
    
    for idx, row in autisti_df.iterrows():
        driver = create_driver_from_autisti_row(row)
        if driver:
            drivers.append(driver)
        else:
            logger.warning(f"Skipped driver at row {idx + 2}")
    
    logger.info(f"Created {len(drivers)} drivers")
    
    # --- Create Orders using groupby('ORDER') as specified in TODO #23 ---
    orders = []
    logger.info("Creating orders from CONSEGNE sheet using groupby('ORDER')...")
    
    # Group tasks by ORDER column (key requirement from TODO #23)
    order_groups = consegne_df.groupby('ORDER')
    
    for order_id, group_df in order_groups:
        logger.debug(f"Processing order {order_id} with {len(group_df)} tasks")
        
        pickup_tasks = []
        delivery_tasks = []
        
        # Create tasks for each row in this order group
        for idx, row in group_df.iterrows():
            task = create_task_from_consegne_row(row, geocode_cache)
            if task:
                if task.task_type == TaskType.PICKUP:
                    pickup_tasks.append(task)
                else:
                    delivery_tasks.append(task)
            else:
                logger.warning(f"Failed to create task from row {idx + 2} in order {order_id}")
        
        # Create Order object if we have valid tasks
        if pickup_tasks or delivery_tasks:
            # Determine priority from first task
            all_tasks = pickup_tasks + delivery_tasks
            priority = all_tasks[0].priority if all_tasks else 1
            
            order = Order(
                id=str(order_id),
                pickup_tasks=pickup_tasks,
                delivery_tasks=delivery_tasks,
                priority=priority,
                is_urgent=priority >= 3,
                is_mandatory=True
            )
            
            orders.append(order)
            logger.debug(f"Created order {order_id}: {len(pickup_tasks)} pickups, {len(delivery_tasks)} deliveries")
        else:
            logger.warning(f"No valid tasks created for order {order_id}")
    
    logger.info(f"Created {len(orders)} orders from {len(order_groups)} order groups")
    
    # Save geocoding cache
    save_geocode_cache(geocode_cache)
    
    # Validation
    if not orders:
        raise ScenarioCreationError("No valid orders were created")
    if not vehicles:
        raise ScenarioCreationError("No valid vehicles were created")
    if not drivers:
        raise ScenarioCreationError("No valid drivers were created")
    
    logger.info(f"Scenario creation complete: {len(orders)} orders, {len(vehicles)} vehicles, {len(drivers)} drivers")
    return orders, vehicles, drivers


if __name__ == "__main__":
    # Test the implementation
    try:
        orders, vehicles, drivers = create_scenario_from_excel_v23("../src/furgoni2.xlsx")
        print(f"Successfully created scenario: {len(orders)} orders, {len(vehicles)} vehicles, {len(drivers)} drivers")
    except Exception as e:
        print(f"Error: {e}")
