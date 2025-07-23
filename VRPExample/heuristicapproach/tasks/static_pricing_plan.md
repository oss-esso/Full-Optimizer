# Static Pricing Plan for EPDT-Based Deliveries

This document outlines a strategy for creating a pricing plan for a static set of deliveries, where all orders are known in advance. This approach complements the dynamic pricing plan and is suitable for large batch processing.

## 1. Core Principle

The fundamental principle of the static pricing plan is that the price for each order should reflect its contribution to the total cost of the most efficient set of routes that serves all orders. This ensures that the pricing is fair and that the overall profitability of the batch is maximized.

## 2. Pricing Strategy

The strategy consists of three main steps:

1.  **Global VRP Solution:** First, solve the Vehicle Routing Problem for the entire set of orders using the EPDT algorithm. This will produce a globally optimized solution with the most efficient routes.
2.  **Route Cost Calculation:** For each route in the optimized solution, calculate its total cost. This is done using the `calculate_z2_score` function, which includes all relevant costs and penalties.
3.  **Cost Allocation to Orders:** The total cost of each route is then distributed among the orders it serves. This is the most critical step and requires a fair allocation metric.

### Step 1: Global VRP Solution

Run the `l1_heuristic` on the entire set of orders to obtain the optimal `Solution` object. This solution will contain the most efficient set of routes for the given batch of orders.

### Step 2: Route Cost Calculation

For each `Route` object in the `solution.routes` dictionary, calculate its total cost using the `calculate_z2_score` function. This function provides a comprehensive cost for the route, including travel, service, and penalties for soft constraint violations.

### Step 3: Cost Allocation to Orders

This step involves distributing the total cost of each route among the orders on that route. A simple proportional allocation is the most straightforward approach. The allocation can be based on a weighted "fairness" metric that considers multiple factors. Here is a proposed metric:

**Allocation Share for an Order `o` on Route `r`:**

`AllocationShare(o, r) = w_time * (service_time(o) / total_service_time(r)) + w_demand * (demand(o) / total_demand(r)) + w_tasks * (num_tasks(o) / total_tasks(r))`

Where:

*   `w_time`, `w_demand`, and `w_tasks` are weights that sum to 1. These can be tuned based on what the business considers the primary cost drivers.
*   `service_time(o)` is the sum of the service times of all tasks in order `o`.
*   `total_service_time(r)` is the sum of the service times of all tasks on route `r`.
*   `demand(o)` is the total demand (weight or volume) of order `o`.
*   `total_demand(r)` is the total demand of all orders on route `r`.
*   `num_tasks(o)` is the number of tasks in order `o`.
*   `total_tasks(r)` is the total number of tasks on route `r`.

**Example Weights:**

*   If time is the most important factor: `w_time = 0.6`, `w_demand = 0.3`, `w_tasks = 0.1`
*   If demand is the most important factor: `w_time = 0.2`, `w_demand = 0.7`, `w_tasks = 0.1`

## 3. Final Price Calculation

Once the allocation share for each order is calculated, the final price can be determined:

**Cost for Order `o`:**

`Cost(o) = calculate_z2_score(r) * AllocationShare(o, r)`

**Final Price for Order `o`:**

`FinalPrice(o) = Cost(o) * (1 + ProfitMargin)`

Where `ProfitMargin` is the desired profit margin (e.g., 0.20 for 20%).

## 4. Example Implementation (Pseudo-code)

```python
# 1. Solve the global VRP
global_solution = l1_heuristic(all_orders, all_vehicles, params)

# 2. Calculate prices for each order
order_prices = {}
for route in global_solution.routes.values():
    if not route.tasks:
        continue

    route_cost = calculate_z2_score(route)
    total_service_time = sum(t.service_time for t in route.tasks)
    total_demand = sum(t.demand for t in route.tasks if t.is_pickup())
    total_tasks = len(route.tasks)

    orders_on_route = {}
    for task in route.tasks:
        if task.order_id not in orders_on_route:
            orders_on_route[task.order_id] = {
                "service_time": 0,
                "demand": 0,
                "num_tasks": 0
            }
        orders_on_route[task.order_id]["service_time"] += task.service_time
        if task.is_pickup():
            orders_on_route[task.order_id]["demand"] += task.demand
        orders_on_route[task.order_id]["num_tasks"] += 1

    for order_id, order_data in orders_on_route.items():
        # Define weights for the fairness metric
        w_time = 0.5
        w_demand = 0.3
        w_tasks = 0.2

        # Calculate allocation share
        time_share = order_data["service_time"] / total_service_time if total_service_time > 0 else 0
        demand_share = order_data["demand"] / total_demand if total_demand > 0 else 0
        tasks_share = order_data["num_tasks"] / total_tasks if total_tasks > 0 else 0

        allocation_share = (w_time * time_share) + (w_demand * demand_share) + (w_tasks * tasks_share)

        # Calculate order cost and final price
        order_cost = route_cost * allocation_share
        profit_margin = 0.20
        final_price = order_cost * (1 + profit_margin)

        order_prices[order_id] = final_price

# Now you have a dictionary of prices for each order
print(order_prices)
```
