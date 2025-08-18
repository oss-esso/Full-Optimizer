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
from datetime import time as datetime_time
import logging
from pathlib import Path
import sys
import os
import argparse

# Add the algo directory to the path to import data structures
current_dir = Path(__file__).parent
algo_dir = current_dir.parent / "algo"
sys.path.append(str(algo_dir))

from epdt_data_structures import Order, Task, Vehicle, Driver, TaskType
import requests

# Configuration constants
GEOCODE_CACHE_FILE = "geocode_cache.json"
DEFAULT_SERVICE_TIME = 15.0  # Default service time in minutes
DEFAULT_DEPOT_COORDS = (44.9009, 8.2057)  # Asti, Italy - Via del Lavoro 38

# Column mapping for CONSEGNE sheet
CONSEGNE_COLUMN_MAPPING = {
    'ORDER': 'order_id',  # Fixed: actual column name in furgoni2.xlsx
    'ORDER_ID': 'order_id',
    'COMPANY': 'company_name',  # Fixed: actual column name in furgoni2.xlsx  
    'COMPANY_NAME': 'company_name',
    'STREET': 'street',
    'HOUSE NUMBER': 'house_number',
    'CITY': 'city',
    'PROVINCE': 'province',
    'POSTAL CODE': 'postal_code',
    'COUNTRY': 'country',
    'EARLIEST DAY': 'earliest_day',
    'LATEST DAY': 'latest_day',
    'TIME WINDOW START': 'time_window_start',
    'TIME WINDOW END': 'time_window_end',
    'SERVICE TIME': 'service_time',
    'DELIVERY OR PICKUP': 'task_type',
    'LOAD KG': 'load_kg',
    'LOAD KG': 'load_kg',  # Support both formats (with and without space)
    'LOAD VOLUME M^3': 'load_volume',
    'LOAD VOLUME M^3': 'load_volume',  # Support both formats
    'PALLETS': 'pallets',
    'PALLET': 'pallets',  # Support both formats
    'REQUIRED CAPABILITIES': 'required_capabilities',
    # Legacy column names for backward compatibility
    'NAME': 'company_name',
    'ADDRESS': 'address', 
    'TIME WINDOW DATE (24=0)': 'time_window_date',
    'TIME WINDOW HOURS': 'time_window_hours',
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
    'COST PER KM': 'cost_per_km',
    'FIXED COST': 'fixed_cost',
    'CAPABILITIES': 'capabilities',
    'REGULATIONS': 'regulations',
    # Legacy column names for backward compatibility
    'LOADER': 'loader',
    'LOW TEMP': 'low_temp',
    'LAST IN FIRST OUT': 'lifo_required',
    'HANGERS': 'hangers'
}

#

# Column mapping for AUTISTI sheet (alternative driver format)
AUTISTI_COLUMN_MAPPING = {
    'NUMBER PLATE': 'vehicle_id',  # Links driver to vehicle
    'DRIVER': 'driver_name',
    'LICENSE': 'license',
    'COST PER HOUR': 'cost_per_hour'
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScenarioCreationError(Exception):
    """Custom exception for scenario creation errors."""
    pass


def find_column_in_row(row: pd.Series, possible_columns: List[str]) -> Optional[str]:
    """
    Find the first available column from a list of possible column names.
    
    Args:
        row: Pandas series containing the data
        possible_columns: List of possible column names to check
        
    Returns:
        First available column name, or None if none found
    """
    for col in possible_columns:
        if col in row and pd.notna(row[col]):
            return col
    return None

def safe_parse_value_with_mapping(row: pd.Series, target_field: str, default_value, value_type=None, boolean_values=None):
    """
    Safely parse a value using column mapping with fallback to direct column names.
    
    Args:
        row: Pandas series containing the data
        target_field: Target field name (e.g., 'load_kg')
        default_value: Default value if parsing fails
        value_type: Type to convert to (int, float, str, bool)
        boolean_values: List of values that should be considered True for boolean conversion
        
    Returns:
        Parsed value or default_value if parsing fails
    """
    # Get possible column names for this target field
    possible_columns = []
    
    # Add columns that map to this target field
    for excel_col, mapped_field in CONSEGNE_COLUMN_MAPPING.items():
        if mapped_field == target_field:
            possible_columns.append(excel_col)
    
    # Find the first available column
    column = find_column_in_row(row, possible_columns)
    
    if column is None:
        return default_value
        
    return safe_parse_value(row, column, default_value, value_type, boolean_values)

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


def normalize_address_string(address: str) -> str:
    """
    Normalize an address string by cleaning up common formatting issues.
    
    Args:
        address: Raw address string
        
    Returns:
        Normalized address string
    """
    if not address:
        return ""
    
    # Convert to string and strip
    address = str(address).strip()
    
    # Remove multiple consecutive spaces
    import re
    address = re.sub(r'\s+', ' ', address)
    
    # Fix common Italian address formatting issues
    address = address.replace('  ', ' ')
    address = address.replace(' ,', ',')
    address = address.replace(', ,', ',')
    
    # Handle common abbreviations and formatting
    address = address.replace(' D\'', ' d\'')  # Fix apostrophe formatting
    address = address.replace('D\'', 'd\'')
    
    # Fix Italian address patterns
    address = address.replace('LOCALITA\'', 'Località')
    address = address.replace('LOCALITÀ', 'Località')
    address = address.replace('VIA', 'Via')
    address = address.replace('CORSO', 'Corso')
    address = address.replace('PIAZZA', 'Piazza')
    address = address.replace('LARGO', 'Largo')
    address = address.replace('STRADA', 'Strada')
    address = address.replace('VIALE', 'Viale')
    address = address.replace('PIAZZALE', 'Piazzale')
    
    return address.strip()


def improve_italian_address_format(address: str, postal_code: str = "", country: str = "") -> List[str]:
    """
    Generate improved Italian address formats by applying specific corrections and patterns.
    
    Args:
        address: Raw address string
        postal_code: Postal code
        country: Country name
        
    Returns:
        List of improved address format variations
    """
    variations = []
    
    if not address or 'ITALY' not in country.upper():
        return variations
    
    # Normalize the address first
    addr = normalize_address_string(address)
    
    # Italian province abbreviations mapping for common problematic cities
    province_mappings = {
        'SERRAVALLE SCRIVIA': ('Serravalle Scrivia', 'AL'),
        'LA MORRA CUNEO': ('La Morra', 'CN'),
        'LA MORRA': ('La Morra', 'CN'),
        'MEZZOLOMBARDO': ('Mezzolombardo', 'TN'),
        'CASTELLO SOPRA TICINO': ('Castello sopra Ticino', 'NO'),
        'VIGLIIANO D\'ASTI': ('Villafranca d\'Asti', 'AT'),  # Common mistake
        'COLOGNO MONZESE': ('Cologno Monzese', 'MI'),
        'OVADA ALTA': ('Ovada', 'AL'),
        'PIEVE DI TECO': ('Pieve di Teco', 'IM'),
        'BIANDRATE': ('Biandrate', 'NO'),
        'RIVALTA': ('Rivalta di Torino', 'TO'),  # Most likely Rivalta
        'LOANTE CEPPINO': ('Lonate Ceppino', 'VA'),  # Likely typo
    }
    
    # Try to extract street and city components
    import re
    
    # Pattern to separate street address from city
    # Look for patterns like "VIA SOMETHING CITY" or "LOCALITÀ SOMETHING CITY"
    street_city_patterns = [
        r'^((?:Via|Corso|Piazza|Largo|Strada|Viale|Piazzale|Località)\s+.+?)\s+([A-Z][A-Z\s\']+)$',
        r'^(.+?)\s+(SERRAVALLE\s+SCRIVIA|LA\s+MORRA\s*CUNEO?|MEZZOLOMBARDO|CASTELLO\s+SOPRA\s+TICINO|VIGLIIANO\s+D\'ASTI|COLOGNO\s+MONZESE|OVADA\s+ALTA|PIEVE\s+DI\s+TECO|BIANDRATE|LOANTE\s+CEPPINO)$'
    ]
    
    street_part = addr
    city_part = ""
    
    for pattern in street_city_patterns:
        match = re.match(pattern, addr, re.IGNORECASE)
        if match:
            street_part = match.group(1).strip()
            city_part = match.group(2).strip().upper()
            break
    
    # Generate variations with proper Italian formatting
    if city_part:
        # Clean city name
        city_clean = city_part.replace('CUNEO', '').strip()
        
        # Check if we have a mapping for this city
        if city_clean in province_mappings:
            proper_city, province = province_mappings[city_clean]
            
            # Variation 1: Street + proper city name + postal + country
            if postal_code:
                variations.append(f"{street_part}, {proper_city}, {postal_code}, {country}")
                # Variation 2: Include province
                variations.append(f"{street_part}, {proper_city} ({province}), {postal_code}, {country}")
                # Variation 3: Province format
                variations.append(f"{street_part}, {postal_code} {proper_city} ({province}), {country}")
            
            # Variation 4: Without postal code
            variations.append(f"{street_part}, {proper_city}, {country}")
    
    # Handle special address patterns
    special_fixes = {
        'VIA DELLA MODA 1 SERRAVALLE SCRIVIA': 'Via della Moda 1, Serravalle Scrivia (AL)',
        'LOCALITÀ BATASIOLO 85/A LA MORRA CUNEO': 'Località Batasiolo 85/A, La Morra (CN)',
        'VIA CAVALLEGGERI D\'UDINE 5 MEZZOLOMBARDO': 'Via Cavalleggeri d\'Udine 5, Mezzolombardo (TN)',
        'VIA 11 SETTEMBRE 2001 VOLGARGNE': 'Via 11 Settembre 2001, Dolcè (VR)',  # Volgargne is a hamlet of Dolcè
        'VIA MANDOLOSSA 43/45 MANDOLOSSA': 'Via Mandolossa 43/45, Gussago (BS)',  # Mandolossa is in Gussago
        'LARGO EULA 1 PIEVE DI TECO': 'Largo Eula 1, Pieve di Teco (IM)',
        'VIA MARZABOTTO 16 CASTELLO SOPRA TICINO': 'Via Marzabotto 16, Castello sopra Ticino (NO)',
        'VIA ASTI MARE 6/B VIGLIIANO D\'ASTI': 'Via Asti Mare 6/B, Villafranca d\'Asti (AT)',
        'VIA MARCO BIAGI 3 RIVALTA': 'Via Marco Biagi 3, Rivalta di Torino (TO)',
        'PIAZZALE ORMIG REGIONE CARLOVI 12A OVADA ALTA': 'Piazzale ORMIG, Ovada (AL)',
        'VIA SPAGNA 156 COLOGNO MONZESE': 'Via Spagna 156, Cologno Monzese (MI)',
        'VIA B FRANKLIN 8 LOANTE CEPPINO': 'Via Benjamin Franklin 8, Lonate Ceppino (VA)'
    }
    
    # Check for exact matches in special fixes
    addr_upper = addr.upper()
    for pattern, fix in special_fixes.items():
        if pattern in addr_upper:
            if postal_code:
                variations.append(f"{fix}, {postal_code}, {country}")
            variations.append(f"{fix}, {country}")
            break
    
    return variations


def generate_alternative_addresses(row: pd.Series) -> List[str]:
    """
    Generate alternative address formats using the new component-based approach.
    Focus on precise street addresses with proper component ordering.
    
    Args:
        row: Pandas series containing address data
        
    Returns:
        List of alternative address strings to try
    """
    alternatives = []
    
    if 'ADDRESS' not in row or pd.isna(row['ADDRESS']):
        return alternatives
    
    # Parse address components
    raw_address = str(row['ADDRESS']).strip()
    street, city = parse_address_components(raw_address)
    
    postal_code = ""
    if 'POSTAL CODE' in row and pd.notna(row['POSTAL CODE']):
        postal_code = str(int(row['POSTAL CODE']) if isinstance(row['POSTAL CODE'], (int, float)) else row['POSTAL CODE']).strip()
    
    country = ""
    if 'COUNTRY' in row and pd.notna(row['COUNTRY']):
        country = normalize_address_string(str(row['COUNTRY']))
    
    province = ""
    if 'ITALY' in country.upper() and city:
        province = get_italian_province_code(city)
    
    # Alternative 1: Standard format with province
    if street and city and postal_code and province:
        alternatives.append(construct_nominatim_address(street, postal_code, city, province, country))
    
    # Alternative 2: Without province code
    if street and city and postal_code:
        alternatives.append(construct_nominatim_address(street, postal_code, city, "", country))
    
    # Alternative 3: Different ordering - postal code first (European style)
    if street and city and postal_code:
        parts = [postal_code, street, city]
        if province:
            parts[-1] = f"{city} ({province})"
        if country:
            parts.append(country)
        alternatives.append(", ".join(parts))
    
    # Alternative 4: Street and city without postal code
    if street and city:
        alternatives.append(construct_nominatim_address(street, "", city, province, country))
    
    # Alternative 5: Try with simplified street address (remove complex house numbers)
    if street:
        import re
        simplified_street = re.sub(r'\s+\d+[A-Z]*(/[A-Z0-9]*)?$', '', street).strip()
        if simplified_street != street and simplified_street and city:
            alternatives.append(construct_nominatim_address(simplified_street, postal_code, city, province, country))
    
    # Alternative 6: Try original format as fallback
    normalized_original = normalize_address_string(raw_address)
    if postal_code and country:
        alternatives.append(f"{normalized_original}, {postal_code}, {country}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_alternatives = []
    for alt in alternatives:
        if alt not in seen:
            seen.add(alt)
            unique_alternatives.append(alt)
    
    return unique_alternatives


def construct_fallback_address(row: pd.Series) -> str:
    """
    Construct a conservative fallback address using the component-based approach.
    Only falls back to city-level when no street address can be identified.
    
    Args:
        row: Pandas series containing address data
        
    Returns:
        Fallback address string or empty string if no fallback is appropriate
    """
    if 'ADDRESS' not in row or pd.isna(row['ADDRESS']):
        return ""
    
    # Parse the address components
    raw_address = str(row['ADDRESS']).strip()
    street, city = parse_address_components(raw_address)
    
    # Only create fallback if we couldn't identify a proper street address
    if not street or len(street.strip()) < 5:  # Very short street suggests it might just be a city
        postal_code = ""
        country = ""
        
        if 'POSTAL CODE' in row and pd.notna(row['POSTAL CODE']):
            postal_code = str(int(row['POSTAL CODE']) if isinstance(row['POSTAL CODE'], (int, float)) else row['POSTAL CODE']).strip()
        
        if 'COUNTRY' in row and pd.notna(row['COUNTRY']):
            country = normalize_address_string(str(row['COUNTRY']))
        
        # Only create city-level fallback if we have identified a city
        if city:
            province = ""
            if 'ITALY' in country.upper():
                province = get_italian_province_code(city)
            
            return construct_nominatim_address("", postal_code, city, province, country)
        
        # If no city identified, try postal code only
        elif postal_code and country:
            return f"{postal_code}, {country}"
    
    return ""  # No appropriate fallback


def construct_postal_only_address(row: pd.Series) -> str:
    """
    Construct a postal code + country only address as absolute last resort.
    Should rarely be used.
    
    Args:
        row: Pandas series containing address data
        
    Returns:
        Postal code address string or empty string
    """
    parts = []
    
    # Add postal code
    if 'POSTAL CODE' in row and pd.notna(row['POSTAL CODE']):
        postal_code = str(int(row['POSTAL CODE']) if isinstance(row['POSTAL CODE'], (int, float)) else row['POSTAL CODE']).strip()
        parts.append(postal_code)
    
    # Add country
    if 'COUNTRY' in row and pd.notna(row['COUNTRY']):
        country = normalize_address_string(str(row['COUNTRY']))
        parts.append(country)
    
    return ", ".join(filter(None, parts))


def try_geocode_with_photon(address: str, cache: Dict[str, Dict[str, float]]) -> Optional[Tuple[float, float]]:
    """Try to geocode an address using the Photon API."""
    if address in cache:
        coords = cache[address]
        return (coords["lat"], coords["lon"])

    photon_url = "https://photon.komoot.io/api/"
    try:
        time.sleep(0.5) # Rate limiting
        response = requests.get(photon_url, params={'q': address, 'limit': 1}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and data.get('features'):
            # IMPORTANT: Photon returns [lon, lat]
            lon, lat = data['features'][0]['geometry']['coordinates']
            coords = {"lat": float(lat), "lon": float(lon)}
            cache[address] = coords
            logger.info(f"Success with Photon: {address} -> ({lat:.6f}, {lon:.6f})")
            return (lat, lon)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Photon API request failed for address '{address}': {e}")
    except (ValueError, KeyError, IndexError) as e:
        logger.warning(f"Could not parse Photon response for address '{address}': {e}")
    return None


def get_coordinates_with_fallback(address: str, cache: Dict[str, Dict[str, float]], row: pd.Series) -> Optional[Tuple[float, float]]:
    """
    Enhanced geocoding with multiple fallback strategies for maximum precision.
    """
    # Strategy 1: Try exact address with Nominatim
    coords = try_geocode_single_address(address, cache)
    if coords:
        #logger.info(f"Geocoded successfully with Nominatim: {address} -> ({coords[0]:.6f}, {coords[1]:.6f})")
        return coords
    
    # Strategy 2: Try alternative address formats with Nominatim
    alternative_addresses = generate_alternative_addresses(row)
    for alt_address in alternative_addresses:
        if alt_address != address:
            coords = try_geocode_single_address(alt_address, cache)
            if coords:
                logger.info(f"Success with improved format (Nominatim): {alt_address} -> ({coords[0]:.6f}, {coords[1]:.6f})")
                return coords

    # Strategy 3: Try original full address with Photon as a fallback
    coords = try_geocode_with_photon(address, cache)
    if coords:
        return coords

    # Strategy 4: Conservative fallback (city-level) with Nominatim
    fallback_address = construct_fallback_address(row)
    if fallback_address and fallback_address != address:
        coords = try_geocode_single_address(fallback_address, cache)
        if coords:
            logger.warning(f"Fallback to city-level geocoding (Nominatim): {fallback_address} -> ({coords[0]:.6f}, {coords[1]:.6f})")
            return coords
    
    # Strategy 5: Last resort - postal code only with Nominatim
    postal_only = construct_postal_only_address(row)
    if postal_only:
        coords = try_geocode_single_address(postal_only, cache)
        if coords:
            logger.error(f"Last resort postal-only geocoding (Nominatim): {postal_only} -> ({coords[0]:.6f}, {coords[1]:.6f})")
            return coords
    
    return None


def try_geocode_single_address(address: str, cache: Dict[str, Dict[str, float]]) -> Optional[Tuple[float, float]]:
    """
    Try to geocode a single address with the standard Nominatim approach.
    
    Args:
        address: Address string to geocode
        cache: Geocoding cache dictionary
        
    Returns:
        Tuple of (latitude, longitude) or None if geocoding failed
    """
    # Check cache first
    if address in cache:
        coords = cache[address]
        return (coords["lat"], coords["lon"])
    
    # Nominatim API endpoint
    nominatim_url = "https://nominatim.openstreetmap.org/search"
    
    # Retry configuration
    max_retries = 2
    base_delay = 1.0  # Nominatim rate limiting compliance
    
    for attempt in range(max_retries + 1):
        try:
            # Progressive delay: 1s, 2s, 3s
            if attempt > 0:
                delay = base_delay * (1 + attempt)
                time.sleep(delay)
            else:
                time.sleep(base_delay)  # Always respect rate limits
            
            # Prepare request parameters
            params = {
                'q': address,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1,
                'accept-language': 'en'
            }
            
            # Set proper User-Agent header for Nominatim usage policy compliance
            headers = {
                'User-Agent': 'EPDT-Scenario-Creator/1.0'
            }
            
            # Make request to Nominatim
            response = requests.get(nominatim_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                # Extract coordinates from first result
                result = data[0]
                lat = float(result['lat'])
                lon = float(result['lon'])
                
                # Cache the result
                coords = {"lat": lat, "lon": lon}
                cache[address] = coords
                return (lat, lon)
            else:
                break  # No point retrying if address not found
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                continue  # Retry on network/timeout errors
            else:
                break
        except (ValueError, KeyError) as e:
            break  # Don't retry on parsing errors
        except Exception as e:
            break  # Don't retry on unexpected errors
    
    return None


def get_coordinates(address: str, cache: Dict[str, Dict[str, float]]) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude coordinates for an address using enhanced Nominatim API.
    
    This is a wrapper function that maintains the original interface but delegates 
    to the enhanced geocoding function when row data is available.
    
    Args:
        address: Full address string to geocode
        cache: Geocoding cache dictionary
        
    Returns:
        Tuple of (latitude, longitude) or None if geocoding failed
    """
    # Use the basic single address geocoding for backwards compatibility
    return try_geocode_single_address(address, cache)


def parse_address_components(address: str) -> Tuple[str, str]:
    """
    Parse an address string to separate street address from city name.
    
    Args:
        address: Raw address string that may contain both street and city
        
    Returns:
        Tuple of (street_address, city_name)
    """
    if not address:
        return "", ""
    
    address = normalize_address_string(address)
    
    # Import re here to avoid issues
    import re
    
    # Special handling for non-Italian addresses first
    non_italian_patterns = [
        # French addresses
        (r'^(.+?)\s+(BOUC-BEL\s+AIR)$', 'Bouc-Bel-Air'),
        (r'^(.*)\s+(BOUC-BEL)\s+(AIR)(.*)$', 'Bouc-Bel-Air'),
    ]
    
    for pattern, city_name in non_italian_patterns:
        match = re.match(pattern, address, re.IGNORECASE)
        if match:
            # For the complex BOUC-BEL pattern, reconstruct the street part
            if "BOUC-BEL" in pattern:
                street = match.group(1).strip() if match.group(1) else ""
                if not street:  # If no street part, use empty
                    street = ""
            else:
                street = match.group(1).strip()
            return street, city_name
    
    # Specific patterns for known problematic Italian addresses
    specific_patterns = [
        # Pattern: Street + specific known cities  
        (r'^(.+?)\s+(SERRAVALLE\s+SCRIVIA)$', 'Serravalle Scrivia'),
        (r'^(.+?)\s+(LA\s+MORRA\s*CUNEO?)$', 'La Morra'),
        (r'^(.+?)\s+(MEZZOLOMBARDO)$', 'Mezzolombardo'),
        (r'^(.+?)\s+(CASTELLO\s+SOPRA\s+TICINO)$', 'Castello sopra Ticino'),
        (r'^(.+?)\s+(VIGLIIANO\s+D\'ASTI)$', 'Villafranca d\'Asti'),
        (r'^(.+?)\s+(COLOGNO\s+MONZESE)$', 'Cologno Monzese'),
        (r'^(.+?)\s+(OVADA\s+ALTA)$', 'Ovada'),
        (r'^(.+?)\s+(PIEVE\s+DI\s+TECO)$', 'Pieve di Teco'),
        (r'^(.+?)\s+(BIANDRATE)$', 'Biandrate'),
        (r'^(.+?)\s+(LOANTE\s+CEPPINO)$', 'Lonate Ceppino'),
        (r'^(.+?)\s+(TREBASELEGHE)$', 'Trebaseleghe'),
        (r'^(.+?)\s+(BERNAREGGIO)$', 'Bernareggio'),
        (r'^(.+?)\s+(VIGONZA)$', 'Vigonza'),
        (r'^(.+?)\s+(NICHELINO)$', 'Nichelino'),
        (r'^(.+?)\s+(BRESCIA)$', 'Brescia'),
        (r'^(.+?)\s+(SULBIATE)$', 'Sulbiate'),
        (r'^(.+?)\s+(CAMERANO)$', 'Camerano'),
        (r'^(.+?)\s+(CHIAVARI)$', 'Chiavari'),
        (r'^(.+?)\s+(TRAVERSETOLO)$', 'Traversetolo'),
        (r'^(.+?)\s+(MERETO\s+DI\s+TOMBA)$', 'Mereto di Tomba'),
        (r'^(.+?)\s+(CIVIDALE\s+DEL\s+FRIULI)$', 'Cividale del Friuli'),
        (r'^(.+?)\s+(TRUCCAZZANO)$', 'Truccazzano'),
        (r'^(.+?)\s+(SOLIERA)$', 'Soliera'),
        (r'^(.+?)\s+(CAMBIANO)$', 'Cambiano'),
        # Additional patterns for missing cities
        (r'^(.+?)\s+(CEREA)$', 'Cerea'),
        (r'^(.+?)\s+(VERONA)$', 'Verona'),
        (r'^(.+?)\s+(LAIVES)$', 'Laives'),
        (r'^(.+?)\s+(VALLE\s+LOMELLINA)$', 'Valle Lomellina'),
        (r'^(.+?)\s+(MILANO)$', 'Milano'),
        (r'^(.+?)\s+(MOMO)$', 'Momo'),
        (r'^(.+?)\s+(ARESE)$', 'Arese'),
        (r'^(.+?)\s+(CARPI)$', 'Carpi'),
        (r'^(.+?)\s+(TRIESTE)$', 'Trieste'),
        (r'^(.+?)\s+(CUNEO)$', 'Cuneo'),
        (r'^(.+?)\s+(GHEVIO\s+DI\s+MEINA)$', 'Ghevio di Meina'),
        (r'^(.+?)\s+(VOLGARGNE)$', 'Dolcè'),  # Volgargne is a hamlet of Dolcè
        (r'^(.+?)\s+(MANDOLOSSA)$', 'Gussago'),  # Mandolossa is in Gussago
    ]
    
    # Try specific patterns first
    for pattern, city_name in specific_patterns:
        match = re.match(pattern, address, re.IGNORECASE)
        if match:
            street = match.group(1).strip()
            city = city_name
            return street, city
    
    # Generic patterns for Italian addresses
    generic_patterns = [
        # Pattern: Street with house number + Italian city (2+ words)
        r'^(.+?\s+\d+[A-Z]*(?:/[A-Z0-9]*)?)\s+([A-Z][A-Za-z\s\']+(?:\s+[A-Z][A-Za-z\s\']+)+)$',
        # Pattern: Street without house number + Italian city (2+ words)  
        r'^((?:Via|Corso|Piazza|Largo|Strada|Viale|Piazzale|Località)\s+[^,]+?)\s+([A-Z][A-Za-z\s\']+(?:\s+[A-Z][A-Za-z\s\']+)+)$',
        # Pattern: Any street + single word city (less reliable)
        r'^(.+?)\s+([A-Z][A-Z]+)$',
    ]
    
    for pattern in generic_patterns:
        match = re.match(pattern, address, re.IGNORECASE)
        if match:
            street = match.group(1).strip()
            city = match.group(2).strip()
            
            # Validate that the city part looks like a city (not part of street address)
            if len(city.split()) >= 1 and not re.search(r'\d', city):
                return street, city
    
    # If no pattern matches, return the full address as street and empty city
    return address, ""


def get_italian_province_code(city: str) -> str:
    """
    Get the Italian province code for a given city.
    
    Args:
        city: City name
        
    Returns:
        Province code (e.g., 'TO', 'MI') or empty string if not found
    """
    # Mapping of cities to their province codes
    city_to_province = {
        'SERRAVALLE SCRIVIA': 'AL',
        'LA MORRA': 'CN',
        'MEZZOLOMBARDO': 'TN',
        'CASTELLO SOPRA TICINO': 'NO',
        'VILLAFRANCA D\'ASTI': 'AT',
        'VIGLIIANO D\'ASTI': 'AT',
        'COLOGNO MONZESE': 'MI',
        'OVADA': 'AL',
        'PIEVE DI TECO': 'IM',
        'BIANDRATE': 'NO',
        'LONATE CEPPINO': 'VA',
        'TREBASELEGHE': 'PD',
        'BERNAREGGIO': 'MB',
        'VIGONZA': 'PD',
        'CEREA': 'VR',
        'VERONA': 'VR',
        'LAIVES': 'BZ',
        'VALLE LOMELLINA': 'PV',
        'NICHELINO': 'TO',
        'BRESCIA': 'BS',
        'SULBIATE': 'MB',
        'MOMO': 'NO',
        'GHEVIO DI MEINA': 'NO',
        'MILANO': 'MI',
        'CHIAVARI': 'GE',
        'CAMERANO': 'AN',
        'CARPI': 'MO',
        'TRAVERSETOLO': 'PR',
        'TRIESTE': 'TS',
        'MERETO DI TOMBA': 'UD',
        'CUNEO': 'CN',
        'CIVIDALE DEL FRIULI': 'UD',
        'ARESE': 'MI',
        'TRUCCAZZANO': 'MI',
        'SOLIERA': 'MO',
        'CAMBIANO': 'TO',
    }
    
    city_upper = city.upper().strip()
    return city_to_province.get(city_upper, '')


def construct_nominatim_address(street: str, postal_code: str, city: str, province: str, country: str) -> str:
    """
    Construct an address in the optimal format for Nominatim geocoding.
    
    Args:
        street: Street address with house number
        postal_code: Postal code
        city: City name
        province: Province/state code
        country: Country name
        
    Returns:
        Formatted address string for Nominatim
    """
    parts = []
    
    # Add street address
    if street:
        parts.append(street)
    
    # Add postal code
    if postal_code:
        parts.append(postal_code)
    
    # Add city with province if available
    if city:
        if province:
            parts.append(f"{city} ({province})")
        else:
            parts.append(city)
    
    # Add country
    if country:
        parts.append(country)
    
    return ", ".join(filter(None, parts))


def construct_full_address(row: pd.Series) -> str:
    """
    Construct a full address string from Excel row data with proper component separation.
    Orders components optimally for Nominatim: street, postal code, city, province, country.
    Supports both new format (separate components) and legacy format (ADDRESS column).
    
    Args:
        row: Pandas series containing address components
        
    Returns:
        Full address string suitable for geocoding
    """
    # Check if new format is available (separate address components)
    street = safe_parse_value(row, 'STREET', '', str)
    house_number = safe_parse_value(row, 'HOUSE_NUMBER', '', str)
    city = safe_parse_value(row, 'CITY', '', str)
    province = safe_parse_value(row, 'PROVINCE', '', str)
    postal_code = safe_parse_value(row, 'POSTAL_CODE', '', str)
    country = safe_parse_value(row, 'COUNTRY', '', str)
    
    # If new format has data, use it
    if street or city:
        # Combine street and house number
        if street and house_number:
            full_street = f"{street} {house_number}".strip()
        elif street:
            full_street = street
        else:
            full_street = ""
        
        # Get province code for Italian addresses if not explicitly provided
        if not province and 'ITALY' in country.upper() and city:
            province = get_italian_province_code(city)
        
        return construct_nominatim_address(full_street, postal_code, city, province, country)
    
    # Fall back to legacy format (ADDRESS column)
    if 'ADDRESS' not in row or pd.isna(row['ADDRESS']):
        return ""
    
    # Parse the raw address to separate street and city
    raw_address = str(row['ADDRESS']).strip()
    street, city = parse_address_components(raw_address)
    
    # Get postal code from legacy column names
    postal_code = ""
    if 'POSTAL CODE' in row and pd.notna(row['POSTAL CODE']):
        postal_code = str(int(row['POSTAL CODE']) if isinstance(row['POSTAL CODE'], (int, float)) else row['POSTAL CODE']).strip()
    
    # Get country from legacy column names
    country = ""
    if 'COUNTRY' in row and pd.notna(row['COUNTRY']):
        country = normalize_address_string(str(row['COUNTRY']))
    
    # Get province code for Italian addresses
    province = ""
    if 'ITALY' in country.upper() and city:
        province = get_italian_province_code(city)
    
    # Construct the optimized address
    return construct_nominatim_address(street, postal_code, city, province, country)


def parse_time_window(time_input) -> Optional[float]:
    """
    Parse a time input (string, datetime.time) into minutes from start of day.
    """
    if pd.isna(time_input) or not time_input:
        return None

    if isinstance(time_input, datetime_time):
        return time_input.hour * 60 + time_input.minute

    time_str_clean = str(time_input).strip()
    
    # Try HH:MM:SS format first
    try:
        time_obj = datetime.strptime(time_str_clean, "%H:%M:%S").time()
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        pass
    
    # Fall back to HH:MM format
    try:
        time_obj = datetime.strptime(time_str_clean, "%H:%M").time()
        return time_obj.hour * 60 + time_obj.minute
    except ValueError:
        logger.warning(f"Could not parse time: {time_input}")
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
        # Parse basic information using column mapping
        company_name = safe_parse_value_with_mapping(row, 'company_name', 'UNKNOWN', str)
        
        # Parse order ID using column mapping
        order_id = safe_parse_value_with_mapping(row, 'order_id', None, str)
        if not order_id:
            order_id = f"ORDER_{company_name.replace(' ', '_')[:20]}_{row.name}"
        
        task_id = f"TASK_{order_id}_{row.name}"
        
        # Parse task type
        task_type_str = safe_parse_value(row, 'DELIVERY OR PICKUP', 'DELIVERY', str).upper()
        if task_type_str in ['PICKUP', 'PICK']:
            task_type = TaskType.PICKUP
        elif task_type_str in ['DELIVERY', 'DELIVER']:
            task_type = TaskType.DELIVERY
        else:
            task_type = TaskType.DELIVERY
            logger.warning(f"Unknown task type '{task_type_str}' for {company_name}, defaulting to DELIVERY")
        
        # Get coordinates using enhanced geocoding with fallback strategies
        full_address = construct_full_address(row)
        coordinates = get_coordinates_with_fallback(full_address, geocode_cache, row)
        
        if not coordinates:
            logger.error(f"Could not geocode address for {company_name}: {full_address}")
            return None
        
        lat, lon = coordinates
        
        # Parse order info using column mapping
        order_id = safe_parse_value_with_mapping(row, 'order_id', f"ORDER_{row.name}", str)
        company_name = safe_parse_value_with_mapping(row, 'company_name', f"COMPANY_{row.name}", str)
        
        # Parse numeric fields using column mapping
        load_kg = safe_parse_value_with_mapping(row, 'load_kg', 0.0, float)
        load_volume = safe_parse_value_with_mapping(row, 'load_volume', 0.0, float)
        service_time = safe_parse_value(row, 'SERVICE TIME', DEFAULT_SERVICE_TIME, float)
        pallets = safe_parse_value_with_mapping(row, 'pallets', 0, int)
        
        # Parse capabilities - new format uses REQUIRED_CAPABILITIES, legacy uses individual boolean fields
        required_capabilities = set()
        capabilities_str = safe_parse_value(row, 'REQUIRED_CAPABILITIES', '', str)
        if capabilities_str:
            caps = [cap.strip().upper() for cap in capabilities_str.split(',') if cap.strip()]
            required_capabilities.update(caps)
        
        # Legacy format support (individual boolean columns)
        requires_low_temp = safe_parse_value(row, 'LOW TEMP', False, bool)
        requires_loader = safe_parse_value(row, 'LOADER', False, bool)
        
        # Convert legacy boolean fields to capabilities
        if requires_low_temp:
            required_capabilities.add('LOW TEMP')
        if requires_loader:
            required_capabilities.add('LOADER')
        
        # For backward compatibility, also set individual flags
        requires_low_temp = requires_low_temp or 'LOW TEMP' in required_capabilities
        requires_loader = requires_loader or 'LOADER' in required_capabilities
        
        # For pickup tasks, demand is positive; for delivery, negative
        if task_type == TaskType.PICKUP:
            demand = abs(load_kg)
            volume = abs(load_volume)
            pallets_adjusted = abs(pallets)  # Positive for pickup
        else:  # DELIVERY
            demand = -abs(load_kg)
            volume = -abs(load_volume)
            pallets_adjusted = -abs(pallets)  # Negative for delivery
        
        # --- Time Window Parsing ---
        earliest_time = None
        latest_time = None

        # New format: Multi-day with hours
        earliest_day = safe_parse_value(row, 'EARLIEST DAY', 1, int)
        latest_day = safe_parse_value(row, 'LATEST DAY', earliest_day, int)
        time_start_input = safe_parse_value(row, 'TIME WINDOW START', '00:00', None) # Keep as object
        time_end_input = safe_parse_value(row, 'TIME WINDOW END', '23:59', None)   # Keep as object

        time_start_minutes = parse_time_window(time_start_input)
        time_end_minutes = parse_time_window(time_end_input)

        if time_start_minutes is not None:
            # Convert to absolute minutes from the beginning of the planning horizon (Day 1, 00:00)
            # Day 1 corresponds to an offset of 0 days.
            earliest_time = (earliest_day - 1) * 1440 + time_start_minutes

        if time_end_minutes is not None:
            # Day 1 corresponds to an offset of 0 days.
            latest_time = (latest_day - 1) * 1440 + time_end_minutes

        # Legacy format: time range string (e.g., "8-12" or "08:00-12:00")
        if earliest_time is None and latest_time is None:
            time_window_hours = safe_parse_value(row, 'TIME WINDOW HOURS', None, str)
            if time_window_hours and '-' in time_window_hours:
                try:
                    start_str, end_str = time_window_hours.split('-', 1)
                    if ':' in start_str:
                        earliest_time = parse_time_window(start_str.strip())
                        latest_time = parse_time_window(end_str.strip())
                    else:
                        start_hour = int(start_str.strip())
                        end_hour = int(end_str.strip())
                        earliest_time = start_hour * 60
                        latest_time = end_hour * 60
                except Exception as e:
                    logger.warning(f"Could not parse legacy time window '{time_window_hours}' for {company_name}: {e}")
        
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
        depot_id = 'DEPOT-ASTI'  # Default depot
        
        # Parse capacity constraints
        weight_capacity = safe_parse_value(row, 'MAX LOAD KG', 3500.0, float)
        volume_capacity = safe_parse_value(row, 'MAX LOAD VOLUME M^3', 25.0, float)
        pallet_capacity = safe_parse_value(row, 'PALLET', None, int)
        
        # Parse vehicle type and costs
        vehicle_type_str = safe_parse_value(row, 'TYPE OF VEHICLE', 'FURGONE', str).upper()
        
        # Parse costs from Excel (new format) or use defaults (legacy format)
        cost_per_km = safe_parse_value(row, 'COST PER KM', None, float)
        fixed_cost = safe_parse_value(row, 'FIXED COST', None, float)
        
        # Set defaults based on vehicle type if not specified in Excel
        if vehicle_type_str in ['FURGONE', 'VAN']:
            vehicle_type = 'standard'
            if cost_per_km is None:
                cost_per_km = 0.55  # Default from proposed format
            if fixed_cost is None:
                fixed_cost = 50.0
        elif vehicle_type_str in ['CAMION', 'TRUCK']:
            vehicle_type = 'heavy'
            if cost_per_km is None:
                cost_per_km = 0.75  # Higher cost for trucks
            if fixed_cost is None:
                fixed_cost = 100.0
        else:
            vehicle_type = 'standard'
            if cost_per_km is None:
                cost_per_km = 0.55
            if fixed_cost is None:
                fixed_cost = 50.0
        
        # Parse boolean requirements
        lifo_required = safe_parse_value(row, 'LAST IN FIRST OUT', False, bool)
        
        # Parse capabilities
        capabilities = set()
        
        # New format: comma-separated CAPABILITIES column
        capabilities_str = safe_parse_value(row, 'CAPABILITIES', '', str)
        if capabilities_str:
            caps = [cap.strip().upper() for cap in capabilities_str.split(',') if cap.strip()]
            capabilities.update(caps)
        
        # Legacy format: individual boolean columns (for backward compatibility)
        if safe_parse_value(row, 'LOADER', False, bool):
            capabilities.add('LOADER')
        
        if safe_parse_value(row, 'LOW TEMP', False, bool):
            capabilities.add('LOW TEMP')
        
        if safe_parse_value(row, 'HANGERS', False, bool):
            capabilities.add('HANGERS')
        
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


def create_driver_from_row(row: pd.Series) -> Optional[Driver]:
    """
    Create a Driver object from an Excel row using column mapping.
    
    Args:
        row: Pandas series containing driver data
        
    Returns:
        Driver object or None if creation failed
    """
    try:
        # Parse basic information using safe parsing
        driver_id = safe_parse_value(row, 'DRIVER_ID', 'UNKNOWN', str)
        driver_name = safe_parse_value(row, 'DRIVER_NAME', 'Unknown Driver', str)
        cost_per_hour = safe_parse_value(row, 'COST_PER_HOUR', 25.0, float)
        max_shift_hours = safe_parse_value(row, 'MAX_SHIFT_HOURS', 13.0, float)
        max_driving_hours = safe_parse_value(row, 'MAX_DRIVING_HOURS', 9.0, float)
        
        # Parse capabilities (comma-separated string)
        capabilities_str = safe_parse_value(row, 'CAPABILITIES', '', str)
        capabilities = set()
        if capabilities_str:
            # Split by comma and clean up each capability
            caps = [cap.strip().upper() for cap in capabilities_str.split(',') if cap.strip()]
            capabilities.update(caps)
        
        # Set default license based on capabilities (simplified logic)
        license = 'B'  # Default license
        if 'TRUCK_LICENSE' in capabilities or 'CE_LICENSE' in capabilities:
            license = 'CE'
        
        # Create driver with default vehicle assignment (will be updated during assignment)
        driver = Driver(
            id=driver_id,
            name=driver_name,
            license=license,
            default_vehicle_id='',  # Will be assigned during optimization
            cost_per_hour=cost_per_hour,
            max_shift_hours=max_shift_hours,
            max_driving_hours=max_driving_hours,
            home_depot_id='main_depot',  # Default depot
            qualifications=capabilities
        )
        
        logger.debug(f"Created driver: {driver_id} ({driver_name}) with capabilities: {capabilities}")
        return driver
        
    except Exception as e:
        logger.error(f"Error creating driver from row: {e}")
        return None


def create_driver_from_autisti_row(row: pd.Series) -> Optional[Driver]:
    """
    Create a Driver object from an AUTISTI sheet row (legacy format).
    
    Args:
        row: Pandas series containing AUTISTI data
        
    Returns:
        Driver object or None if creation failed
    """
    try:
        # Parse basic information from AUTISTI format
        vehicle_id = safe_parse_value(row, 'NUMBER PLATE', 'UNKNOWN', str)
        driver_name = safe_parse_value(row, 'DRIVER', 'Unknown Driver', str)
        license = safe_parse_value(row, 'LICENSE', 'B', str)
        cost_per_hour = safe_parse_value(row, 'COST PER HOUR', 25.0, float)
        
        # Generate driver ID from name and vehicle
        driver_id = f"DRV_{driver_name.replace(' ', '_').upper()}_{vehicle_id}"
        
        # Set capabilities based on license
        capabilities = set()
        if license == 'CE':
            capabilities.add('TRUCK_LICENSE')
        
        # Set default hours based on license type
        if license == 'CE':
            max_shift_hours = 13.0
            max_driving_hours = 9.0
        else:
            max_shift_hours = 12.0
            max_driving_hours = 8.0
        
        # Create driver linked to specific vehicle
        driver = Driver(
            id=driver_id,
            name=driver_name,
            license=license,
            default_vehicle_id=vehicle_id,  # Linked to specific vehicle
            cost_per_hour=cost_per_hour,
            max_shift_hours=max_shift_hours,
            max_driving_hours=max_driving_hours,
            home_depot_id='main_depot',
            qualifications=capabilities
        )
        
        logger.debug(f"Created driver from AUTISTI: {driver_id} ({driver_name}) -> {vehicle_id}")
        return driver
        
    except Exception as e:
        logger.error(f"Error creating driver from AUTISTI row: {e}")
        return None


def create_scenario_from_excel(file_path: str) -> Tuple[List[Order], List[Vehicle], List[Driver]]:
    """
    Read an Excel file and create a VRP scenario for the EPDT algorithm.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Tuple of (orders_list, vehicles_list, drivers_list)
        
    Raises:
        ScenarioCreationError: If scenario creation fails
    """
    logger.info(f"Reading scenario from {file_path}...")
    
    try:
        orders_df = pd.read_excel(file_path, sheet_name="CONSEGNE")
        vehicles_df = pd.read_excel(file_path, sheet_name="VEICOLI")
        drivers_df = pd.read_excel(file_path, sheet_name="DRIVERS") if "DRIVERS" in pd.ExcelFile(file_path).sheet_names else None
    except Exception as e:
        raise ScenarioCreationError(f"Failed to read Excel file: {e}")

    geocode_cache = load_geocode_cache()
    
    vehicles = [v for v in [create_vehicle_from_row(row) for _, row in vehicles_df.iterrows()] if v]
    logger.info(f"Created {len(vehicles)} vehicles")

    if drivers_df is not None:
        drivers = [d for d in [create_driver_from_row(row) for _, row in drivers_df.iterrows()] if d]
        logger.info(f"Created {len(drivers)} drivers from DRIVERS sheet")
    else:
        drivers = [] # No default drivers

    orders = []
    logger.info("Creating orders and tasks...")

    order_id_col = 'ORDER' if 'ORDER' in orders_df.columns else 'ORDER_ID'
    if order_id_col not in orders_df.columns:
        raise ScenarioCreationError("Could not find order ID column ('ORDER' or 'ORDER_ID') in CONSEGNE sheet.")

    grouped_by_order = orders_df.groupby(order_id_col)
    
    for order_id, group in grouped_by_order:
        pickup_tasks = []
        delivery_tasks = []
        order_priority = 1
        is_urgent = False

        for _, row in group.iterrows():
            customer_task = create_task_from_row(row, geocode_cache)
            if customer_task:
                if customer_task.is_pickup():
                    pickup_tasks.append(customer_task)
                else:
                    delivery_tasks.append(customer_task)
                
                order_priority = max(order_priority, customer_task.priority)
                if customer_task.priority >= 3:
                    is_urgent = True

        if pickup_tasks or delivery_tasks:
            order = Order(
                id=str(order_id),
                pickup_tasks=pickup_tasks,
                delivery_tasks=delivery_tasks,
                priority=order_priority,
                is_urgent=is_urgent,
                is_mandatory=True
            )
            orders.append(order)

    logger.info(f"Created {len(orders)} orders from grouped tasks.")
    save_geocode_cache(geocode_cache)

    if not orders or not vehicles:
        raise ScenarioCreationError("No valid orders or vehicles were created")

    logger.info(f"Scenario creation complete: {len(orders)} orders, {len(vehicles)} vehicles, {len(drivers)} drivers")
    return orders, vehicles, drivers


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
            orders, vehicles, drivers = create_test_scenario(args.excel_file)
        else:
            orders, vehicles, drivers = create_scenario_from_excel(args.excel_file)
        
        if args.validate:
            if not validate_scenario(orders, vehicles):
                logger.error("Scenario validation failed")
                return 1
        
        if args.output:
            # Save scenario to file (implementation would depend on desired format)
            logger.info(f"Scenario would be saved to {args.output}")
        
        print(f"Successfully created scenario with {len(orders)} orders, {len(vehicles)} vehicles, and {len(drivers)} drivers")
        return 0
        
    except ScenarioCreationError as e:
        logger.error(f"Scenario creation failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
