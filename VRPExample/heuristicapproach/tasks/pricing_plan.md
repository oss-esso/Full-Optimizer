# Dynamic Pricing Plan for EPDT-Based Deliveries

This document outlines a strategy for creating a dynamic pricing plan for deliveries, leveraging the capabilities of the EPDT algorithm and the parameters defined in `config/epdt_params.json`. The approach is based on the concepts of marginal cost calculation and machine learning-based price prediction, as discussed in the thesis `tesi_definitiva_Nicola_Gastaldon.pdf` (Chapters 8 and 9).

## 1. Core Concepts

The pricing strategy is built on two core concepts:

1.  **Marginal Cost:** The price of a new delivery should be based on the *actual cost* of adding it to the existing routes. This is the most accurate way to ensure profitability.
2.  **Dynamic Pricing:** The price is not static. It changes based on the real-time state of the system, including vehicle availability, current route efficiency, and the characteristics of the new order.

## 2. Pricing Strategy

The pricing strategy is a two-step process:

1.  **Calculate the Marginal Cost:** This is the minimum price you should charge to avoid losing money on a delivery.
2.  **Determine the Final Price:** This is the price you present to the client, which includes your profit margin and any adjustments based on the order's characteristics.

### Step 1: Calculate the Marginal Cost

The EPDT algorithm can calculate the marginal cost of a new order by simulating its insertion into the current set of routes. The process is as follows:

1.  **Get the current solution score:** Before adding the new order, calculate the Z1 score of the current solution using `calculate_z1_score`.
2.  **Simulate the insertion:** Use the `l1_heuristic` to find the best way to insert the new order into the existing routes. This will produce a new solution.
3.  **Calculate the new solution score:** Calculate the Z1 score of the new solution.
4.  **The difference is the marginal cost:** The difference between the new score and the old score is the marginal cost of the new order.

### Step 2: Determine the Final Price

Once you have the marginal cost, you can determine the final price to quote the client. This is where the parameters from `config/epdt_params.json` come into play.

**Base Price = Marginal Cost + Profit Margin**

A reasonable starting point for the profit margin is a percentage of the marginal cost (e.g., 20%).

**Adjustments Based on Order Characteristics:**

The following factors, represented by the weights and penalties in `epdt_params.json`, can be used to adjust the final price:

*   **Urgency (`Lo`):** For urgent orders, a premium should be added to the price. The value of `Lo` can be used as a guideline for this premium.
*   **Time Windows (`time_window_violation_penalty`):** Orders with very tight time windows are more difficult to schedule and should have a higher price. The `time_window_violation_penalty` can be used to quantify this.
*   **Capacity (`capacity_violation_penalty`):** If an order uses a large portion of a vehicle's capacity, it limits the ability to serve other orders, so the price should be higher.
*   **Preferred Subsets of Orders (`wk_ID`):** If a new order breaks up a preferred subset of orders, the price should be increased to compensate for the loss of efficiency.
*   **Preferred Vehicle (`wk_IF`):** If the client requests a specific vehicle, and that vehicle is not the most efficient choice, the price should be adjusted to cover the extra cost.
*   **Preferred End-of-Day Position (`wk_IH`):** If a new order forces a vehicle to end its day in a non-optimal location, the price should be increased to cover the cost of repositioning the vehicle for the next day.
*   **Maximum Route Duration (`wk_IJ`):** If a new order causes a route to exceed its preferred maximum duration, the price should be increased to compensate for the potential for driver overtime and other costs.

### 3. Advanced Pricing with Machine Learning

The thesis also describes a machine learning approach to predict a suitable price range for a new order. This can be used to further refine the pricing strategy:

1.  **Train a Model:** Train a machine learning model (e.g., a Decision Tree Classifier as mentioned in the thesis) on historical data to predict a price range based on order features like distance, weight, volume, and urgency.
2.  **Get a Price Range:** For a new order, use the trained model to get a predicted price range.
3.  **Combine with Marginal Cost:** Use the predicted price range to guide the final price negotiation. For example, if the marginal cost is at the low end of the predicted range, you have more flexibility to offer a competitive price. If the marginal cost is at the high end, you know you need to charge a premium.

## 4. Example Pricing Calculation

Here is a pseudo-code example of how to calculate the price for a new order:

```python
# 1. Calculate Marginal Cost
current_score = calculate_z1_score(current_solution)
new_solution = l1_heuristic(current_solution, new_order)
new_score = calculate_z1_score(new_solution)
marginal_cost = new_score - current_score

# 2. Determine Base Price
profit_margin = 0.20 # 20% profit margin
base_price = marginal_cost * (1 + profit_margin)

# 3. Adjust Price Based on Factors
final_price = base_price
if new_order.is_urgent:
    final_price += params["Lo"] * 0.1 # Add 10% of the urgent penalty

if new_order_violates_soft_constraint("max_duration"):
    final_price += params["wk_IJ"] * 0.5 # Add 50% of the duration penalty

# ... other adjustments based on other factors ...

# 4. (Optional) Refine with Machine Learning
predicted_price_range = ml_model.predict(new_order)
if final_price < predicted_price_range.lower_bound:
    final_price = predicted_price_range.lower_bound
elif final_price > predicted_price_range.upper_bound:
    # Consider if the high price is justified or if the order should be rejected
    pass

return final_price
```
