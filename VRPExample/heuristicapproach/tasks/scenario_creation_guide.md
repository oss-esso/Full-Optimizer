# Refactoring Guide: `scenario_creator.py`

This document outlines the plan to refactor the `scenario_creator.py` script. The goal is to align it with the updated data structures in `epdt_data_structures.py` and to dynamically handle Excel column headers as defined in `utils/headers.txt`.

**Objective:** To make the scenario creation process more robust and adaptable to changes in the input Excel file.

## 1. Analysis of `utils/headers.txt`

The `utils/headers.txt` file defines the expected column headers for the `CONSEGNE` and `VEICOLI` sheets.

**CONSEGNE (Orders/Tasks):**
- `NAME`: Customer name.
- `ADDRESS`: Street address.
- `COUNTRY`: Country.
- `TIME WINDOW DATE (24=0)`: Date of the time window.
- `TIME WINDOW HOURS`: Time window in HH-HH format.
- `SERVICE TIME`: Service time in minutes.
- `DELIVERY OR PICKUP`: Task type (delivery or pickup).
- `LOAD KG`: Weight of the load.
- `LOAD VOLUME M^3`: Volume of the load.
- `PALLET`: Number of pallets.
- `LOW TEMP`: Requires low temperature.
- `LOADER`: Requires a loader.

**VEICOLI (Vehicles):**
- `NUMBER PLATE`: Vehicle's number plate.
- `TYPE OF VEHICLE`: Type of vehicle.
- `MAX LOAD KG`: Maximum load weight.
- `PALLET`: Pallet capacity.
- `MAX LOAD VOLUME M^3`: Maximum load volume.
- `LOADER`: Has a loader.
- `LOW TEMP`: Has low temperature capability.
- `LAST IN FIRST OUT`: LIFO constraint.
- `HANGERS`: Has hangers.
- `REGULATIONS`: Subject to regulations.

## 2. Refactoring Plan for `scenario_creator.py`

### Step 2.1: Update `create_task_from_row` function

This function needs to be updated to:

1.  **Use the correct column names** from `utils/headers.txt` instead of hardcoded values.
2.  **Map the new attributes** to the `Task` data structure:
    *   `PALLET` -> `task.pallets`
    *   `LOW TEMP` -> `task.requires_low_temp`
    *   `LOADER` -> `task.requires_loader`

### Step 2.2: Update `create_vehicle_from_row` function

This function needs to be updated to:

1.  **Use the correct column names** from `utils/headers.txt`.
2.  **Map the new attributes** to the `Vehicle` data structure:
    *   `PALLET` -> `vehicle.pallet_capacity`
    *   `LOADER` -> `vehicle.capabilities` (add 'loader')
    *   `LOW TEMP` -> `vehicle.capabilities` (add 'low_temp')
    *   `HANGERS` -> `vehicle.capabilities` (add 'hangers')
    *   `LAST IN FIRST OUT` -> `vehicle.lifo_required`

### Step 2.3: Update `create_scenario_from_excel` function

This function needs to be updated to:

1.  **Read the correct sheet names**: `CONSEGNE` and `VEICOLI`.

## 3. Implementation Details

*   **Column Mapping:** Create a dictionary to map the column names from `utils/headers.txt` to the corresponding attribute in the `Task` and `Vehicle` data classes. This will make the code more maintainable.
*   **Error Handling:** Add robust error handling to manage missing columns or unexpected values in the Excel file.
*   **Type Conversion:** Ensure that the data from the Excel file is converted to the correct data type before being assigned to the data class attributes.

By following this guide, the LLM agent should be able to refactor the `scenario_creator.py` script to be more robust and aligned with the project's data structures.