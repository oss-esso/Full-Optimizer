# Advanced Hours of Service (HoS) Implementation

## Overview

This document describes the implementation of the advanced, realistic multi-day and weekly Hours of Service (HoS) simulation as required in section 9.5 of the TODO document. The implementation replaces the simplistic HoS check with a detailed, iterative simulation that correctly implements the full scope of European driving and working time regulations per Regulation (EC) 561/2006 and Directive 2002/15/EC.

## Enhanced DriverState Class

The `DriverState` class has been enhanced with comprehensive state variables for tracking all aspects of European HoS regulations:

### Core State Variables
- `drive_since_break`: Accumulated driving since the last 45min break (or split break part)
- `work_since_break`: Accumulated working time since the last 30/45min break  
- `drive_today`: Accumulated driving in the current daily period
- `work_today`: Accumulated working time in the current daily period
- `drive_this_week`: Accumulated driving from Monday 00:00 to current time
- `drive_last_week`: Total driving time from the previous week

### Advanced State Variables (New)
- `time_in_daily_period`: Time elapsed since the end of the last daily rest (must not exceed 24h)
- `work_this_week`: Accumulated working time from Monday 00:00
- `time_since_weekly_rest`: Time elapsed since the last weekly rest ended (must not exceed 144 hours)

### Extension and Rest Tracking (Enhanced)
- `daily_driving_extensions_used`: Count of 10h driving days used this week (max 2)
- `daily_work_extensions_used`: Count of 14h work days used this week (max 2) 
- `daily_rest_reductions_used`: Count of 9h daily rests used between weekly rests (max 3)
- `is_weekly_rest_reduction_taken`: Flag indicating if a reduced weekly rest was taken in the last two-week period

## Core Implementation: _simulate_hos_advanced()

The new `_simulate_hos_advanced()` function implements the iterative simulation logic as specified:

### Key Features

1. **Stateful Iterative Simulation**: For any activity (travel or service), the function enters a loop that continues as long as the activity has time remaining.

2. **Dynamic Limit Calculation**: Inside the loop, it calculates `drivable_time` as the minimum of all applicable limits:
   - `4.5h - drive_since_break`
   - `6h - work_since_break` (if the activity is work/driving)
   - `9h - drive_today` (or `10h` if an extension is available)
   - `56h - drive_this_week`
   - `90h - (drive_last_week + drive_this_week)`
   - `60h - work_this_week`
   - Time until daily rest becomes mandatory (`24h - time_in_daily_period`)
   - Time until weekly rest becomes mandatory (`144h - time_since_weekly_rest`)
   - The actual time remaining for the current activity

3. **Progressive State Updates**: The function advances simulation time by `drivable_time` and updates all relevant driver state counters.

4. **Mandatory Rest Simulation**: When regulatory limits are hit, the function simulates required rests/breaks:
   - **Driving Break**: 45-minute break (or 30-minute if 15-minute already taken)
   - **Daily Rest**: 11-hour rest (or reduced 9-hour if permissible)
   - **Weekly Rest**: 45-hour rest (or reduced 24-hour)

## Supporting Functions

### Activity Simulation
- `_simulate_activity()`: Core iterative simulation for individual activities
- `_calculate_max_continuous_time()`: Calculates maximum time before mandatory stop
- `_advance_driver_state()`: Updates state counters after activity completion

### Rest/Break Simulation  
- `_simulate_mandatory_rest()`: Determines and simulates required rest type
- `_simulate_weekly_rest()`: Handles weekly rest periods with reduction options
- `_simulate_daily_rest()`: Handles daily rest periods with reduction tracking
- `_simulate_break()`: Handles driving/work breaks (45 minutes)

### Utility Functions
- `_simulate_daily_transition()`: Handles day changes with mandatory daily rest
- `_reset_daily_counters()`: Resets daily state after rest periods
- `_get_max_drive_today()` / `_get_max_work_today()`: Dynamic limit calculation with extensions

## Integration

The advanced HoS simulation is integrated into the route feasibility checking by replacing the call to `_check_hos_multiday()` with:

```python
# H5: Multi-day Hours of Service check - Advanced simulation with iterative breaks/rests
feasible, _ = _simulate_hos_advanced(route, driver_state, sorted_tasks)
if not feasible:
    return False
```

## Compliance with Requirements

### ✅ Stateful Iterative Simulation
- Implements loop-based processing for activities with time remaining
- Correctly handles long activities that require intermediate breaks/rests

### ✅ Comprehensive State Tracking  
- All required state variables from the specification are implemented
- Proper initialization at planning horizon start

### ✅ Dynamic Limit Calculation
- Calculates `drivable_time` as minimum of all applicable regulatory limits
- Handles extensions, reductions, and strategic limit usage

### ✅ Progressive Simulation
- Advances time incrementally with proper state updates
- Simulates mandatory rests during long travel legs

### ✅ European HoS Regulation Compliance
- Implements Regulation (EC) 561/2006 and Directive 2002/15/EC
- Handles daily, weekly, and bi-weekly limits
- Supports extension mechanisms and reduced rest options

## Benefits

1. **Accuracy**: Routes are now feasible across multiple days and weeks with realistic HoS constraints
2. **Compliance**: Full European regulatory compliance ensures real-world applicability  
3. **Flexibility**: Handles complex multi-day scenarios with strategic rest placement
4. **Optimization**: Enables more accurate route optimization by considering actual driver costs

## Future Enhancements

The implementation provides a foundation for:
- Integration with cost calculation functions for more accurate driver costs
- Real-time route adjustment based on HoS state
- Strategic rest placement optimization
- Multi-driver scenarios with coordinated rest scheduling
