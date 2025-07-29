# Proposed Data Format for VRP with Driver Assignment

This document outlines a comprehensive data format for the Excel input file. This structure separates vehicle, driver, and order data, enabling a more powerful and realistic Vehicle Routing Problem (VRP) with Driver Assignment.

It incorporates:
- Multi-stop orders.
- Multi-day time windows.
- Separated vehicle and driver costs.
- Capability matching for tasks, vehicles, and drivers.

## 1. New Excel Structure: Three Sheets

The Excel file will now be composed of three distinct sheets:
1.  **`CONSEGNE`**: Defines all the tasks (pickups and deliveries) to be performed.
2.  **`VEICOLI`**: Defines the fleet of available vehicles and their specific costs and capabilities.
3.  **`DRIVERS`**: Defines the available drivers, their costs, and their capabilities.

---

## 2. `CONSEGNE` Sheet Format

This sheet details every task. Tasks are grouped into orders using `ORDER_ID`.

| Column Name | Required | Description | Example |
| :--- | :--- | :--- | :--- |
| `ORDER_ID` | Yes | Groups tasks into a single order. | `ORD-101` |
| `COMPANY_NAME` | Yes | Name of the customer or location. | `Customer A` |
| `STREET` | Yes | Street name for the task location. | `Via Roma` |
| `HOUSE_NUMBER` | Yes | Street number for the task location. | `10` |
| `CITY` | Yes | City for the task location. | `Torino` |
| `PROVINCE` | Yes | Province abbreviation (e.g., TO). | `TO` |
| `POSTAL_CODE` | Yes | Postal code for the task location. | `10121` |
| `COUNTRY` | Yes | Country for the task location. | `Italy` |
| `EARLIEST_DAY` | No | First day task can be serviced (e.g., 1, 2...). Defaults to 1. | `1` |
| `LATEST_DAY` | No | Last day task can be serviced. Defaults to `EARLIEST_DAY`. | `4` |
| `TIME_WINDOW_START`| No | Earliest time of day for the task (HH:MM). | `09:00` |
| `TIME_WINDOW_END` | No | Latest time of day for the task (HH:MM). | `18:00` |
| `SERVICE_TIME` | No | Service time in minutes. Defaults to 15. | `20` |
| `DELIVERY_OR_PICKUP`| Yes | Must be `DELIVERY` or `PICKUP`. | `DELIVERY` |
| `LOAD_KG` | No | Weight of the goods for this task. | `150.5` |
| `LOAD_VOLUME_M3` | No | Volume of the goods for this task. | `1.5` |
| `PALLETS` | No | Number of pallets for this task. | `2` |
| `REQUIRED_CAPABILITIES` | No | Comma-separated list of requirements (e.g., `LOW_TEMP`, `LOADER`, `ADR_CERTIFIED`). | `LOW_TEMP, LOADER` |

---

## 3. `DRIVERS` Sheet Format

This new sheet defines the driver pool with their specific costs, licenses, and preferences.

| Column Name | Required | Description | Example |
| :--- | :--- | :--- | :--- |
| `LICENSE PLATE` | Yes | The number plate of the driver's **preferred** or usually assigned vehicle. This is used to create initial driver-vehicle pairings, but the algorithm can swap them for optimization. | `AB123CD` |
| `DRIVER NAME` | Yes | The unique name or identifier for the driver. | `Mario Rossi` |
| `LICENSE` | Yes | The driver's license category (e.g., B, C, CE). This acts as a **hard constraint**, determining which vehicles the driver is legally allowed to operate. | `CE` |
| `COST PER HOUR` | Yes | The hourly cost of the driver (wage, benefits, etc.). This is a primary component of the route cost. | `28.50` |

---

## 4. `VEICOLI` Sheet Format

This sheet defines the vehicle fleet. Note that `COST_PER_HOUR` has been moved to the `DRIVERS` sheet.

| Column Name | Required | Description | Example |
| :--- | :--- | :--- | :--- |
| `NUMBER PLATE` | Yes | Unique identifier for the vehicle. | `AB123CD` |
| `TYPE OF VEHICLE` | Yes | Type of vehicle (e.g., Van, Truck). This can be mapped to a license requirement (e.g., a 'Truck' might require a 'CE' license). | `Truck` |
| `MAX LOAD KG` | Yes | Maximum weight capacity in kilograms. | `3500` |
| `PALLET` | No | Maximum pallet capacity. | `8` |
| `MAX LOAD VOLUME M^3`| No | Maximum volume capacity in cubic meters. | `15` |
| `COST_PER_KM` | Yes | The cost incurred for each kilometer driven. | `0.55` |
| `FIXED_COST` | Yes | A fixed cost for using the vehicle for a route. | `50` |
| `CAPABILITIES` | No | Comma-separated list of vehicle features (e.g., `LOW_TEMP`, `LOADER`, `HANGERS`). | `LOW_TEMP, LOADER` |
| `REGULATIONS` | No | Whether the vehicle is subject to strict driving regulations (`YES`/`NO`). | `YES` |

---

## 5. Integration with the Algorithm

This structure enables a more sophisticated optimization process.

1.  **Data Loading**: `scenario_creator.py` will be updated to read all three sheets (`CONSEGNE`, `VEICOLI`, `DRIVERS`) and create corresponding lists of `Order`, `Vehicle`, and `Driver` objects.

2.  **Driver-Vehicle Assignment**: The core algorithm must be enhanced to perform **driver-vehicle assignment**. When building a route, it will select a `(driver, vehicle)` pair. This is no longer just a VRP but a VRP with Driver Assignment.
    *   **Hard Constraint**: The driver's `LICENSE` must be compatible with the `TYPE OF VEHICLE`.
    *   **Soft Constraint/Preference**: The initial solution can be biased towards matching the driver's preferred `LICENSE PLATE`.

3.  **Capability Matching**: The assignment will be constrained by capabilities. A task requiring `LOW_TEMP` can only be assigned to a `(driver, vehicle)` pair where the vehicle has the `LOW_TEMP` capability.

4.  **Unified Cost Calculation**: The route cost function will be more accurate, combining costs from the assigned driver and vehicle:

    `Total Route Cost = (route_km * vehicle.cost_per_km) + (route_hours * driver.cost_per_hour) + vehicle.fixed_cost + penalties`

This model provides a much more realistic and flexible foundation for finding high-quality, cost-effective solutions.

---

## 6. Source File

**Note:** The reference Excel file for this data format is located at: `@src/furgoni2.xlsx`.