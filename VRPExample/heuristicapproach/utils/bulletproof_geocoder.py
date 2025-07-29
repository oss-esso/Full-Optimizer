"""
Bulletproof Geocoding System

A robust geocoding system with multiple providers, exponential backoff,
comprehensive error handling, and intelligent fallback strategies.
Designed to achieve near 100% geocoding success rate for Italian addresses.

Features:
- Multiple geocoding providers with intelligent fallback
- Exponential backoff retry logic
- Advanced Italian address parsing
- Comprehensive caching system
- Detailed logging and success tracking
- Address validation and normalization
- Progressive address simplification strategies

Author: GitHub Copilot
Date: 2025-01-29
"""

import time
import json
import logging
import re
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from geopy.geocoders import Nominatim, GoogleV3, Here, OpenCage, Photon
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable, GeocoderRateLimited


@dataclass
class GeocodeResult:
    """Structure for geocoding results"""
    latitude: float
    longitude: float
    formatted_address: str
    provider: str
    confidence: float
    raw_response: Dict[str, Any]


@dataclass
class ProviderConfig:
    """Configuration for a geocoding provider"""
    name: str
    geocoder_class: type
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # seconds between requests
    timeout: int = 10
    enabled: bool = True
    max_retries: int = 3


class BulletproofGeocoder:
    """
    Advanced geocoding system with multiple providers and robust error handling
    """
    
    def __init__(self, cache_file: str = "bulletproof_geocode_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.success_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful_geocodes': 0,
            'failed_geocodes': 0,
            'provider_stats': {}
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Italian province mapping for enhanced accuracy
        self.italian_provinces = {
            'Torino': 'TO', 'Milano': 'MI', 'Napoli': 'NA', 'Roma': 'RM',
            'Palermo': 'PA', 'Genova': 'GE', 'Bologna': 'BO', 'Firenze': 'FI',
            'Catania': 'CT', 'Venezia': 'VE', 'Verona': 'VR', 'Messina': 'ME',
            'Padova': 'PD', 'Trieste': 'TS', 'Brescia': 'BS', 'Taranto': 'TA',
            'Reggio Calabria': 'RC', 'Modena': 'MO', 'Prato': 'PO', 'Parma': 'PR',
            'Cagliari': 'CA', 'Livorno': 'LI', 'Perugia': 'PG', 'Foggia': 'FG',
            'Ravenna': 'RA', 'Rimini': 'RN', 'Salerno': 'SA', 'Ferrara': 'FE',
            'Sassari': 'SS', 'Latina': 'LT', 'Giugliano in Campania': 'NA',
            'Monza': 'MB', 'Bergamo': 'BG', 'Forlì': 'FC', 'Trento': 'TN',
            'Vicenza': 'VI', 'Terni': 'TR', 'Bolzano': 'BZ', 'Novara': 'NO',
            'Piacenza': 'PC', 'Ancona': 'AN', 'Andria': 'BT', 'Arezzo': 'AR',
            'Udine': 'UD', 'Cesena': 'FC', 'Lecce': 'LE', 'Pesaro': 'PU',
            'Bari': 'BA', 'Alessandria': 'AL', 'La Spezia': 'SP', 'Catanzaro': 'CZ',
            'Pistoia': 'PT', 'Brindisi': 'BR', 'Caserta': 'CE', 'Pesaro': 'PU',
            'Cuneo': 'CN', 'Bernareggio': 'MB', 'La Morra': 'CN', 'Dolcè': 'VR',
            'Fiano Romano': 'RM', 'Cavour': 'TO'
        }
        
        # Initialize providers with configuration
        self.providers = self._initialize_providers()
        
    def _initialize_providers(self) -> List[ProviderConfig]:
        """Initialize geocoding providers in priority order"""
        providers = [
            ProviderConfig(
                name="Nominatim",
                geocoder_class=Nominatim,
                rate_limit=1.0,  # 1 second between requests (Nominatim policy)
                timeout=15,
                max_retries=3
            ),
            ProviderConfig(
                name="Photon",
                geocoder_class=Photon,
                rate_limit=0.5,  # Faster alternative OSM-based service
                timeout=10,
                max_retries=2
            ),
            # Add more providers as needed (require API keys)
            # ProviderConfig(
            #     name="GoogleV3",
            #     geocoder_class=GoogleV3,
            #     api_key="YOUR_GOOGLE_API_KEY",
            #     rate_limit=0.1,
            #     timeout=10,
            #     enabled=False  # Enable when API key is available
            # ),
        ]
        
        # Initialize geocoder instances
        for provider in providers:
            if provider.enabled:
                try:
                    if provider.name == "Nominatim":
                        provider.geocoder = Nominatim(
                            user_agent="EPDT_VRP_Optimizer_v2.0",
                            timeout=provider.timeout
                        )
                    elif provider.name == "Photon":
                        provider.geocoder = Photon(
                            timeout=provider.timeout
                        )
                    elif provider.name == "GoogleV3" and provider.api_key:
                        provider.geocoder = GoogleV3(
                            api_key=provider.api_key,
                            timeout=provider.timeout
                        )
                    
                    self.logger.info(f"Initialized provider: {provider.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {provider.name}: {e}")
                    provider.enabled = False
        
        return providers
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load geocoding cache from file"""
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save geocoding cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save cache: {e}")
    
    def _normalize_address(self, address: str) -> str:
        """Normalize address string for better geocoding success"""
        if not address:
            return ""
        
        # Convert to string and strip whitespace
        address = str(address).strip()
        
        # Remove extra whitespace
        address = re.sub(r'\s+', ' ', address)
        
        # Fix common Italian address patterns
        address = re.sub(r'\bVia\s+(\w)', r'Via \1', address, flags=re.IGNORECASE)
        address = re.sub(r'\bViale\s+(\w)', r'Viale \1', address, flags=re.IGNORECASE)
        address = re.sub(r'\bPiazza\s+(\w)', r'Piazza \1', address, flags=re.IGNORECASE)
        address = re.sub(r'\bCorso\s+(\w)', r'Corso \1', address, flags=re.IGNORECASE)
        
        # Clean up special characters that might cause issues
        address = address.replace('"', '').replace("'", "'")
        
        return address
    
    def _parse_italian_address(self, address: str) -> Dict[str, str]:
        """
        Enhanced Italian address parsing with multiple strategies
        """
        components = {
            'street': '',
            'number': '',
            'postal_code': '',
            'city': '',
            'province': '',
            'country': 'Italy'
        }
        
        # Normalize the address
        normalized = self._normalize_address(address)
        
        # Pattern 1: Standard format with postal code
        pattern1 = r'^(.+?),?\s*(\d{5}),?\s*([^,()]+?)(?:\s*\(([A-Z]{2})\))?,?\s*(?:ITALY|Italia)?$'
        match1 = re.match(pattern1, normalized, re.IGNORECASE)
        
        if match1:
            street_part = match1.group(1).strip()
            postal_code = match1.group(2)
            city = match1.group(3).strip()
            province = match1.group(4) if match1.group(4) else ''
            
            # Extract number from street
            number_match = re.search(r'(\d+(?:/[A-Z])?)\s*$', street_part)
            if number_match:
                components['number'] = number_match.group(1)
                components['street'] = street_part[:number_match.start()].strip()
            else:
                components['street'] = street_part
            
            components['postal_code'] = postal_code
            components['city'] = city
            components['province'] = province
            
            return components
        
        # Pattern 2: Simple format without postal code
        pattern2 = r'^(.+?),?\s*([^,()]+?)(?:\s*\(([A-Z]{2})\))?,?\s*(?:ITALY|Italia)?$'
        match2 = re.match(pattern2, normalized, re.IGNORECASE)
        
        if match2:
            street_part = match2.group(1).strip()
            city = match2.group(2).strip()
            province = match2.group(3) if match2.group(3) else ''
            
            # Extract number from street
            number_match = re.search(r'(\d+(?:/[A-Z])?)\s*$', street_part)
            if number_match:
                components['number'] = number_match.group(1)
                components['street'] = street_part[:number_match.start()].strip()
            else:
                components['street'] = street_part
            
            components['city'] = city
            components['province'] = province
            
            return components
        
        # Fallback: treat entire address as street
        components['street'] = normalized
        return components
    
    def _detect_country(self, address: str) -> str:
        """
        Detect the country from the address string
        """
        address_upper = address.upper()
        
        # Explicit country indicators
        if any(country in address_upper for country in ['FRANCE', 'FRANCIA', 'FRENCH']):
            return 'France'
        elif any(country in address_upper for country in ['ITALY', 'ITALIA', 'ITALIAN']):
            return 'Italy'
        elif any(country in address_upper for country in ['SWITZERLAND', 'SVIZZERA', 'SWISS']):
            return 'Switzerland'
        elif any(country in address_upper for country in ['AUSTRIA', 'AUSTRIAN']):
            return 'Austria'
        elif any(country in address_upper for country in ['SLOVENIA', 'SLOVENIAN']):
            return 'Slovenia'
        elif any(country in address_upper for country in ['SAN MARINO', 'SANMARINO']):
            return 'San Marino'
        
        # Known French cities/patterns - EXPANDED LIST
        french_indicators = [
            'BOUC-BEL-AIR', 'BOUC BEL AIR', 'BOUC-BEL', 'MARSEILLE', 'PARIS', 'LYON', 'TOULOUSE',
            'NICE', 'BORDEAUX', 'LILLE', 'STRASBOURG', 'MONTPELLIER', 'NANTES', 'CANNES',
            'ANTIBES', 'AIX-EN-PROVENCE', 'AVIGNON', 'PERPIGNAN', 'TOULON', 'GRENOBLE',
            'DIJON', 'ANGERS', 'SAINT-ETIENNE', 'LE HAVRE', 'RENNES', 'REIMS', 'TOURS',
            # French postal codes patterns
            '13320'  # Bouc-Bel-Air postal code
        ]
        if any(indicator in address_upper for indicator in french_indicators):
            return 'France'
        
        # Known Swiss cities/patterns
        swiss_indicators = [
            'ZURICH', 'GENEVA', 'BASEL', 'BERN', 'LAUSANNE', 'LUGANO', 'ST. GALLEN',
            'WINTERTHUR', 'LUCERNE', 'ST. MORITZ', 'ZERMATT', 'INTERLAKEN'
        ]
        if any(indicator in address_upper for indicator in swiss_indicators):
            return 'Switzerland'
        
        # Known Austrian cities
        austrian_indicators = [
            'VIENNA', 'SALZBURG', 'INNSBRUCK', 'GRAZ', 'LINZ'
        ]
        if any(indicator in address_upper for indicator in austrian_indicators):
            return 'Austria'
        
        # San Marino indicators
        san_marino_indicators = [
            'DOMAGNO', 'BORGO MAGGIORE', 'SERRAVALLE', 'ACQUAVIVA', 'CHIESANUOVA',
            'DOMAGNANO', 'FAETANO', 'FIORENTINO', 'MONTEGIARDINO'
        ]
        if any(indicator in address_upper for indicator in san_marino_indicators):
            return 'San Marino'
        
        # Default to Italy for most addresses in this context
        return 'Italy'
    
    def _generate_address_variants(self, address: str) -> List[str]:
        """
        Generate multiple address variants for improved geocoding success
        """
        variants = [address]
        
        # Detect country first
        detected_country = self._detect_country(address)
        
        # Parse address components (works for Italian-style addresses)
        components = self._parse_italian_address(address)
        
        # Handle French addresses differently
        if detected_country == 'France':
            # For French addresses, try simpler variants
            normalized = self._normalize_address(address)
            
            # Add France explicitly
            variants.append(f"{normalized}, France")
            
            # Try without extra formatting
            if 'BOUC-BEL-AIR' in normalized.upper():
                variants.extend([
                    "Bouc-Bel-Air, France",
                    "Bouc Bel Air, France",
                    "13320 Bouc-Bel-Air, France"  # Common postal code
                ])
            
            return self._deduplicate_variants(variants)
        
        # Handle other countries
        if detected_country != 'Italy':
            normalized = self._normalize_address(address)
            variants.append(f"{normalized}, {detected_country}")
            return self._deduplicate_variants(variants)
        
        # Italian address handling (existing logic)
        if components['city'] and components['street']:
            # Variant 1: Street, City, Country
            variant1 = f"{components['street']}"
            if components['number']:
                variant1 += f" {components['number']}"
            variant1 += f", {components['city']}, Italy"
            variants.append(variant1)
            
            # Variant 2: With province if available
            if components['province']:
                variant2 = f"{components['street']}"
                if components['number']:
                    variant2 += f" {components['number']}"
                variant2 += f", {components['city']} ({components['province']}), Italy"
                variants.append(variant2)
            
            # Variant 3: With postal code if available
            if components['postal_code']:
                variant3 = f"{components['street']}"
                if components['number']:
                    variant3 += f" {components['number']}"
                variant3 += f", {components['postal_code']} {components['city']}, Italy"
                variants.append(variant3)
            
            # Variant 4: City only (last resort)
            variants.append(f"{components['city']}, Italy")
        
        return self._deduplicate_variants(variants)
    
    def _deduplicate_variants(self, variants: List[str]) -> List[str]:
        """Remove duplicates while preserving order"""
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        return unique_variants
    
    def _exponential_backoff(self, attempt: int, base_delay: float = 0.1, max_delay: float = 5.0) -> float:
        """Calculate exponential backoff delay"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        # Add some jitter to avoid thundering herd
        jitter = random.uniform(0, 0.1 * delay)
        return delay + jitter
    
    def _geocode_with_provider(self, provider: ProviderConfig, address: str) -> Optional[GeocodeResult]:
        """
        Geocode an address using a specific provider with retry logic
        """
        if not provider.enabled or not hasattr(provider, 'geocoder'):
            return None
        
        for attempt in range(provider.max_retries):
            try:
                # Rate limiting
                time.sleep(provider.rate_limit)
                
                # Perform geocoding
                location = provider.geocoder.geocode(address, timeout=provider.timeout)
                
                if location:
                    # Update provider statistics
                    if provider.name not in self.success_stats['provider_stats']:
                        self.success_stats['provider_stats'][provider.name] = {
                            'requests': 0, 'successes': 0, 'failures': 0
                        }
                    
                    self.success_stats['provider_stats'][provider.name]['requests'] += 1
                    self.success_stats['provider_stats'][provider.name]['successes'] += 1
                    
                    return GeocodeResult(
                        latitude=location.latitude,
                        longitude=location.longitude,
                        formatted_address=location.address,
                        provider=provider.name,
                        confidence=1.0,  # Could be enhanced with provider-specific confidence
                        raw_response=location.raw
                    )
                else:
                    self.logger.warning(f"{provider.name}: No result for '{address}'")
                    
            except GeocoderRateLimited:
                self.logger.warning(f"{provider.name}: Rate limited, waiting...")
                delay = self._exponential_backoff(attempt, base_delay=2.0)
                time.sleep(delay)
                continue
                
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                self.logger.warning(f"{provider.name}: {type(e).__name__} for '{address}' (attempt {attempt + 1})")
                if attempt < provider.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    time.sleep(delay)
                    continue
                
            except Exception as e:
                self.logger.error(f"{provider.name}: Unexpected error for '{address}': {e}")
                break
        
        # Update failure statistics
        if provider.name not in self.success_stats['provider_stats']:
            self.success_stats['provider_stats'][provider.name] = {
                'requests': 0, 'successes': 0, 'failures': 0
            }
        
        self.success_stats['provider_stats'][provider.name]['requests'] += 1
        self.success_stats['provider_stats'][provider.name]['failures'] += 1
        
        return None
    
    def geocode(self, address: str, use_cache: bool = True) -> Optional[GeocodeResult]:
        """
        Main geocoding method with comprehensive fallback strategies
        """
        if not address or not address.strip():
            return None
        
        self.success_stats['total_requests'] += 1
        
        # Normalize address
        normalized_address = self._normalize_address(address)
        
        # Check cache first
        if use_cache and normalized_address in self.cache:
            self.success_stats['cache_hits'] += 1
            cached_result = self.cache[normalized_address]
            if cached_result.get('latitude') and cached_result.get('longitude'):
                return GeocodeResult(
                    latitude=cached_result['latitude'],
                    longitude=cached_result['longitude'],
                    formatted_address=cached_result.get('formatted_address', ''),
                    provider=cached_result.get('provider', 'cache'),
                    confidence=cached_result.get('confidence', 1.0),
                    raw_response=cached_result.get('raw_response', {})
                )
        
        # Generate address variants
        address_variants = self._generate_address_variants(normalized_address)
        detected_country = self._detect_country(normalized_address)
        
        self.logger.info(f"Geocoding '{address}' with {len(address_variants)} variants (detected: {detected_country})")
        
        # Try each provider with each address variant
        for variant in address_variants:
            for provider in self.providers:
                if not provider.enabled:
                    continue
                
                result = self._geocode_with_provider(provider, variant)
                
                if result:
                    # Validate coordinates with country-specific bounds
                    if self._validate_coordinates(result.latitude, result.longitude, detected_country):
                        # Cache the successful result
                        self.cache[normalized_address] = {
                            'latitude': result.latitude,
                            'longitude': result.longitude,
                            'formatted_address': result.formatted_address,
                            'provider': result.provider,
                            'confidence': result.confidence,
                            'raw_response': result.raw_response,
                            'detected_country': detected_country,
                            'timestamp': time.time()
                        }
                        self._save_cache()
                        
                        self.success_stats['successful_geocodes'] += 1
                        self.logger.info(f"✅ Geocoded '{address}' -> ({result.latitude:.6f}, {result.longitude:.6f}) via {result.provider} ({detected_country})")
                        
                        return result
        
        # All providers and variants failed
        self.success_stats['failed_geocodes'] += 1
        self.logger.error(f"❌ Failed to geocode '{address}' with all providers and variants")
        
        # Cache the failure to avoid repeated attempts
        self.cache[normalized_address] = {
            'latitude': None,
            'longitude': None,
            'error': 'All providers failed',
            'timestamp': time.time()
        }
        self._save_cache()
        
        return None
    
    def _validate_coordinates(self, lat: float, lon: float, detected_country: str = 'Italy') -> bool:
        """Validate that coordinates are reasonable for the detected country"""
        if lat is None or lon is None:
            return False
        
        # Define bounds for different countries
        country_bounds = {
            'Italy': {
                'lat_min': 35.0,  # South (Sicily)
                'lat_max': 47.5,  # North (Alps)
                'lon_min': 6.0,   # West
                'lon_max': 19.0   # East
            },
            'France': {
                'lat_min': 41.0,  # South (Corsica)
                'lat_max': 51.5,  # North
                'lon_min': -5.0,  # West
                'lon_max': 10.0   # East
            },
            'Switzerland': {
                'lat_min': 45.5,
                'lat_max': 48.0,
                'lon_min': 5.5,
                'lon_max': 11.0
            },
            'Austria': {
                'lat_min': 46.0,
                'lat_max': 49.5,
                'lon_min': 9.0,
                'lon_max': 17.5
            }
        }
        
        bounds = country_bounds.get(detected_country, country_bounds['Italy'])
        
        return (bounds['lat_min'] <= lat <= bounds['lat_max'] and
                bounds['lon_min'] <= lon <= bounds['lon_max'])
    
    def batch_geocode(self, addresses: List[str], progress_callback=None) -> List[Optional[GeocodeResult]]:
        """
        Geocode a batch of addresses with progress tracking
        """
        results = []
        total = len(addresses)
        
        self.logger.info(f"Starting batch geocoding of {total} addresses")
        
        for i, address in enumerate(addresses):
            result = self.geocode(address)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
            
            # Progress logging every 10 addresses
            if (i + 1) % 10 == 0:
                success_rate = (self.success_stats['successful_geocodes'] / 
                              self.success_stats['total_requests']) * 100
                self.logger.info(f"Progress: {i + 1}/{total} ({success_rate:.1f}% success rate)")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive geocoding statistics"""
        total_requests = self.success_stats['total_requests']
        if total_requests == 0:
            return {"message": "No requests processed yet"}
        
        success_rate = (self.success_stats['successful_geocodes'] / total_requests) * 100
        cache_hit_rate = (self.success_stats['cache_hits'] / total_requests) * 100
        
        stats = {
            'total_requests': total_requests,
            'successful_geocodes': self.success_stats['successful_geocodes'],
            'failed_geocodes': self.success_stats['failed_geocodes'],
            'cache_hits': self.success_stats['cache_hits'],
            'success_rate': f"{success_rate:.2f}%",
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'provider_statistics': {}
        }
        
        # Provider-specific statistics
        for provider_name, provider_stats in self.success_stats['provider_stats'].items():
            if provider_stats['requests'] > 0:
                provider_success_rate = (provider_stats['successes'] / provider_stats['requests']) * 100
                stats['provider_statistics'][provider_name] = {
                    'requests': provider_stats['requests'],
                    'successes': provider_stats['successes'],
                    'failures': provider_stats['failures'],
                    'success_rate': f"{provider_success_rate:.2f}%"
                }
        
        return stats
    
    def clear_cache(self):
        """Clear the geocoding cache"""
        self.cache = {}
        self._save_cache()
        self.logger.info("Geocoding cache cleared")


# Example usage and testing functions
def test_bulletproof_geocoder():
    """Test the bulletproof geocoder with sample Italian addresses"""
    
    # Test addresses (including previously failing ones)
    test_addresses = [
        "Via DEL COMMERCIO 20, 20881, Bernareggio (MB), ITALY",
        "Via 11 SETTEMBRE 2001, 37020, Dolcè, ITALY",
        "Località BATASIOLO 85/A, 12064, La Morra (CN), ITALY",
        "Via Roma 1, 10023 Chieri, Italy",
        "Piazza del Duomo, Milano, Italy",
        "Via delle Botteghe Oscure 1, Roma, Italy",
        "Corso Buenos Aires 123, Milano, Italy"
    ]
    
    geocoder = BulletproofGeocoder()
    
    print("🚀 Testing Bulletproof Geocoder")
    print("=" * 50)
    
    results = []
    for address in test_addresses:
        print(f"\n📍 Testing: {address}")
        result = geocoder.geocode(address)
        
        if result:
            print(f"✅ Success: ({result.latitude:.6f}, {result.longitude:.6f})")
            print(f"   Provider: {result.provider}")
            print(f"   Address: {result.formatted_address}")
        else:
            print("❌ Failed")
        
        results.append(result)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("📊 Final Statistics:")
    stats = geocoder.get_statistics()
    for key, value in stats.items():
        if key != 'provider_statistics':
            print(f"   {key}: {value}")
    
    if 'provider_statistics' in stats:
        print("   Provider Statistics:")
        for provider, provider_stats in stats['provider_statistics'].items():
            print(f"     {provider}: {provider_stats}")
    
    return results


if __name__ == "__main__":
    test_bulletproof_geocoder()
