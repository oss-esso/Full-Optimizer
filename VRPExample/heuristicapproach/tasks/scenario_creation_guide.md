# Development Guide: Creating a Scenario from an Excel File

This document outlines the development plan for a Python script that will read a company-specific Excel file, process its data, and generate a valid scenario for the EPDT algorithm.

**Objective:** To bridge the gap between real-world operational data (in Excel) and the input requirements of the EPDT solver, including the automated conversion of addresses to geographic coordinates.

## 1. File and Module Plan

*   **New Script Name:** `scenario_creator.py`
*   **Location:** A new `utils/` directory at the root of the `heuristicapproach` folder seems appropriate to keep helper scripts organized.
*   **Dependencies:** The script will require the following Python libraries:
    *   `pandas`: For reading and processing the Excel file.
    *   `openpyxl`: A dependency for pandas to work with `.xlsx` files.
    *   `geopy`: For converting street addresses into latitude and longitude coordinates.

    These can be installed via pip:
    ```bash
    pip install pandas openpyxl geopy
    ```

## 2. Expected Excel File Structure

The script will expect an Excel file with a specific structure. We can start with two main sheets:

**Sheet 1: `Orders`**

This sheet will contain the details of each order. Each row represents a task (a pickup or a delivery).

| Order ID | Task ID | Task Type | Address              | City       | Postal Code | Country | Weight (kg) | Volume (m³) | TW Start (HH:MM) | TW End (HH:MM) | Service Time (min) | Priority  |
|----------|---------|-----------|----------------------|------------|-------------|---------|-------------|-------------|------------------|----------------|--------------------|-----------|
| ORD-001  | T1-P1   | PICKUP    | Via Roma 1           | Padova     | 35121       | IT      | 150         | 1.2         | 09:00            | 11:00          | 20                 | urgent    |
| ORD-001  | T1-D1   | DELIVERY  | Via Milano 2         | Verona     | 37121       | IT      | -150        | -1.2        | 14:00            | 17:00          | 15                 | urgent    |
| ORD-002  | T2-P1   | PICKUP    | Corso del Popolo 10  | Padova     | 35131       | IT      | 200         | 2.0         | 10:00            | 12:00          | 25                 | mandatory |
| ...      | ...     | ...       | ...                  | ...        | ...         | ...     | ...         | ...         | ...              | ...            | ...                | ...       |

**Sheet 2: `Vehicles`**

This sheet will define the available vehicle fleet.

| Vehicle ID | Depot ID | Weight Capacity (kg) | Volume Capacity (m³) | Cost per km | Fixed Cost | Vehicle Type | LIFO Required |
|------------|----------|----------------------|----------------------|-------------|------------|--------------|---------------|
| V001       | DEPOT-PD | 3500                 | 25                   | 1.2         | 50         | standard     | false         |
| V002       | DEPOT-PD | 7500                 | 40                   | 1.8         | 100        | heavy        | true          |
| ...        | ...      | ...                  | ...                  | ...         | ...        | ...          | ...           |

## 3. Development Steps (Partial Outline)

Here is a partial outline for the `scenario_creator.py` script.

### Step 3.1: Geocoding with Caching

Since geocoding API calls can be slow and costly, it's essential to implement a caching mechanism. We will store the results of geocoding calls in a JSON file to avoid re-querying the same address.

```python
import json
import time
from geopy.geocoders import Nominatim

GEOCODE_CACHE_FILE = "geocode_cache.json"

def load_geocode_cache():
    try:
        with open(GEOCODE_CACHE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_geocode_cache(cache):
    with open(GEOCODE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_coordinates(address, cache):
    if address in cache:
        return cache[address]

    geolocator = Nominatim(user_agent="epdt_scenario_creator")
    try:
        time.sleep(1) # Respect Nominatim's rate limit
        location = geolocator.geocode(address)
        if location:
            coords = {"lat": location.latitude, "lon": location.longitude}
            cache[address] = coords
            return coords
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
    return None
```

### Step 3.2: Main Scenario Creation Function

This function will orchestrate the entire process.

```python
import pandas as pd
from epdt_data_structures import Order, Task, Vehicle, TaskType

def create_scenario_from_excel(file_path: str):
    """
    Reads an Excel file and creates a VRP scenario for the EPDT algorithm.
    """
    print(f"Reading scenario from {file_path}...")
    orders_df = pd.read_excel(file_path, sheet_name="Orders")
    vehicles_df = pd.read_excel(file_path, sheet_name="Vehicles")

    geocode_cache = load_geocode_cache()

    # --- Vehicle Creation (To be implemented) ---
    # Iterate through vehicles_df and create Vehicle objects.

    # --- Order and Task Creation (To be implemented) ---
    # Group orders_df by "Order ID".
    # For each group, create an Order object.
    # For each row in the group, create a Task object.
    #   - Construct the full address from columns.
    #   - Call get_coordinates() to get lat/lon.
    #   - Populate all other Task fields.

    # --- Finalization (To be implemented) ---
    # Save the geocode cache.
    # Return the lists of orders and vehicles.

    print("Scenario creation complete.")
    # return orders, vehicles

if __name__ == "__main__":
    # Example of how to run the script
    # create_scenario_from_excel("path/to/your/company_file.xlsx")
    pass
```

## 4. Next Steps

*   **Flesh out the implementation:** Complete the logic for creating `Vehicle` and `Order`/`Task` objects from the pandas DataFrames.
*   **Error Handling:** Add robust error handling for missing data in the Excel file and for geocoding failures.
*   **Configuration:** Move hardcoded values (like the cache filename) into a configuration file.
*   **Integration:** Integrate this script with the main test runner (`run_scenario_test.py`) to allow loading scenarios directly from Excel.
