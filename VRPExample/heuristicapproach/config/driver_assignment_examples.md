# Driver Assignment Configuration Examples

## Default Configuration
```json
{
    "default_cost_per_hour": 25.0,
    "default_depot_id": "main_depot",
    "penalty_wrong_depot": 50.0,
    "bonus_default_vehicle": 20.0,
    "dummy_assignment_cost": 1000.0,
    "penalty_route_complexity": 2.0,
    "complexity_threshold": 10,
    "time_preference_penalty": 10.0
}
```

## High Performance Configuration (Prioritizes Experience)
```json
{
    "default_cost_per_hour": 25.0,
    "default_depot_id": "main_depot",
    "penalty_wrong_depot": 30.0,
    "bonus_default_vehicle": 15.0,
    "dummy_assignment_cost": 1000.0,
    "penalty_route_complexity": 5.0,
    "complexity_threshold": 8,
    "time_preference_penalty": 15.0
}
```

## Cost-Optimized Configuration (Minimizes Costs)
```json
{
    "default_cost_per_hour": 25.0,
    "default_depot_id": "main_depot",
    "penalty_wrong_depot": 100.0,
    "bonus_default_vehicle": 30.0,
    "dummy_assignment_cost": 1000.0,
    "penalty_route_complexity": 1.0,
    "complexity_threshold": 15,
    "time_preference_penalty": 5.0
}
```

## Flexible Configuration (Balanced Approach)
```json
{
    "default_cost_per_hour": 25.0,
    "default_depot_id": "main_depot",
    "penalty_wrong_depot": 40.0,
    "bonus_default_vehicle": 25.0,
    "dummy_assignment_cost": 1000.0,
    "penalty_route_complexity": 3.0,
    "complexity_threshold": 12,
    "time_preference_penalty": 8.0
}
```
