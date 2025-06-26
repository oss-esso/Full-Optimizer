# Truck Speed Profile Integration Summary

## Overview
Successfully integrated truck speed profile logic from the demo into the VRP scenarios file (`vrp_scenarios.py`) so that truck specifications defined in the Vehicle class are automatically used for OSM/OSRM routing speed adjustments.

## What Was Implemented

### 1. Default Truck Speed Profiles
Added `DEFAULT_TRUCK_SPEED_RATIOS` configuration to `vrp_scenarios.py`:

```python
DEFAULT_TRUCK_SPEED_RATIOS = {
    'standard': {  # Light trucks (4-ton capacity)
        'motorway': 80 / 130,      # 80 km/h vs 130 km/h cars
        'trunk': 70 / 100,         # 70 km/h vs 100 km/h cars  
        'primary': 60 / 90,        # 60 km/h vs 90 km/h cars
        # ... other road types
        'default': 0.80            # 20% slower than cars
    },
    'heavy': {     # Heavy trucks (24-ton capacity)
        'motorway': 70 / 130,      # 70 km/h vs 130 km/h cars
        'trunk': 60 / 100,         # 60 km/h vs 100 km/h cars
        'primary': 50 / 90,        # 50 km/h vs 90 km/h cars  
        # ... other road types
        'default': 0.65            # 35% slower than cars
    }
}
```

### 2. Enhanced Vehicle Class
Extended the `Vehicle` class in `vrp_data_models.py` with truck speed profile fields:

```python
@dataclass
class Vehicle:
    # ... existing fields ...
    
    # Truck speed profile settings for OSM routing
    truck_speed_ratios: Optional[Dict[str, float]] = None  # Custom speed ratios by road type
    use_truck_speeds: bool = True  # Whether to apply truck speed adjustments in OSM routing
```

### 3. VRPInstance Speed Profile Methods
Added methods to `VRPInstance` class for automatic speed profile extraction:

- `get_truck_speed_profile()` - Calculates weighted fleet composition speed profile
- `should_use_truck_speeds()` - Determines if truck speeds should be used
- `get_vehicle_speed_specs()` - Gets detailed speed specs for all vehicles

### 4. Updated Scenarios with Truck Specs
Modified `MODA_small` and `MODA_first` scenarios to assign truck speed profiles:

```python
# Heavy truck with truck speed profile
vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=24000, depot_id=depot_id, max_time=540)
vehicle.vehicle_type = "heavy"
vehicle.use_truck_speeds = True
vehicle.truck_speed_ratios = DEFAULT_TRUCK_SPEED_RATIOS['heavy'].copy()

# Standard truck with truck speed profile  
vehicle = Vehicle(f"vehicle_{vehicle_id}", capacity=4000, depot_id=depot_id, max_time=600)
vehicle.vehicle_type = "standard"
vehicle.use_truck_speeds = True
vehicle.truck_speed_ratios = DEFAULT_TRUCK_SPEED_RATIOS['standard'].copy()
```

### 5. Enhanced OSM Distance Calculator
Updated `OSMDistanceCalculator` in the optimizer to:

- Accept truck speed ratio parameters in constructor
- Apply weighted speed adjustments in bulk matrix calculations
- Analyze road types from OSRM steps for detailed speed adjustments
- Support both individual route calculations and bulk matrix operations

### 6. Automatic Integration in Optimizer
The `CleanVRPOptimizer` now automatically:

```python
# Extract truck speed profile from VRPInstance
truck_speed_ratios = vrp_instance.get_truck_speed_profile()
use_truck_speeds = vrp_instance.should_use_truck_speeds()

# Initialize OSM calculator with scenario-based speed profile
self.distance_calculator = OSMDistanceCalculator(
    self.locations, 
    osrm_url=osrm_url,
    truck_speed_ratios=truck_speed_ratios,
    use_truck_speeds=use_truck_speeds
)
```

## How It Works

### Fleet Composition Weighting
The system automatically calculates weighted average speed ratios based on vehicle capacity distribution:

**Example:** MODA_small scenario (2 x 24-ton heavy + 3 x 4-ton standard trucks)
- Total capacity: 60,000 kg (48,000 kg heavy + 12,000 kg standard)
- Weight distribution: 80% heavy trucks, 20% standard trucks
- Final speed ratios: Weighted average of heavy and standard profiles

### Speed Adjustment Application
1. **Bulk calculations:** Uses average fleet speed ratio for fast matrix operations
2. **Individual routes:** Analyzes OSRM step data for road type composition and applies weighted speed adjustments
3. **Real-time adjustment:** Route time = Car time / Speed ratio (slower speed = longer time)

## Test Results

**MODA_small Scenario Test:**
- Fleet: 2 heavy trucks (24t) + 3 standard trucks (4t)
- Speed profile: Weighted average (motorway: 0.554x, primary: 0.578x, default: 0.680x)
- Real route: depot_1 → pickup_1
  - Distance: 2.66 km
  - Car time: 4.1 min → Truck time: 7.1 min
  - Adjustment factor: 1.709x (trucks 71% slower on this route)

## Benefits

1. **Automatic Integration:** No manual speed profile configuration needed
2. **Fleet-Aware:** Different truck types get appropriate speed profiles
3. **Realistic Routing:** Travel times reflect actual truck performance on different road types
4. **Scenario-Driven:** Speed profiles are part of scenario definitions, ensuring consistency
5. **Mixed Fleet Support:** Weighted averaging handles heterogeneous fleets correctly

## Files Modified

1. `vrp_scenarios.py` - Added truck speed profiles and updated vehicle creation
2. `vrp_data_models.py` - Enhanced Vehicle class and VRPInstance methods
3. `vrp_optimizer_clean copy.py` - Updated OSM calculator and optimizer integration
4. `test_truck_speed_integration.py` - Integration test script

## Usage

The integration is transparent to users. Simply create scenarios with vehicle specifications and the system automatically:

1. Extracts appropriate truck speed profiles based on vehicle types
2. Configures OSM routing with fleet-appropriate speed adjustments
3. Applies realistic travel time calculations for truck routing

No additional configuration or manual speed profile setup is required.
