"""
Static Pricing Calculator for EPDT-Based VRP Solutions

This module implements a comprehensive pricing system that calculates break-even prices
for routes and orders based on their proportional share of costs including distance,
time, and other operational factors.

Key Features:
- Route-level cost calculation using Z2 scoring
- Fair cost allocation to orders based on multiple factors (time, distance, demand)
- Break-even pricing with configurable profit margins
- Detailed pricing breakdown and reports
"""

import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CostBreakdown:
    """Detailed breakdown of costs for a route"""
    driver_salary: float = 0.0
    vehicle_operating_cost: float = 0.0
    fuel_cost: float = 0.0
    time_window_penalties: float = 0.0
    weight_penalties: float = 0.0
    hos_penalties: float = 0.0
    distance_cost: float = 0.0
    base_vehicle_cost: float = 0.0
    other_costs: float = 0.0
    
    @property
    def total_cost(self) -> float:
        return (self.driver_salary + self.vehicle_operating_cost + self.fuel_cost +
                self.time_window_penalties + self.weight_penalties + self.hos_penalties +
                self.distance_cost + self.base_vehicle_cost + self.other_costs)


@dataclass
class OrderPricingInfo:
    """Information for pricing an individual order"""
    order_id: str
    route_vehicle_id: str
    service_time: float
    travel_time: float  # Time spent traveling for this order
    demand_weight: float
    demand_volume: float
    num_tasks: int
    allocation_share: float
    route_cost_share: float
    break_even_price: float
    final_price_with_margin: float


@dataclass
class RoutePricingInfo:
    """Information for pricing a complete route"""
    vehicle_id: str
    total_cost: float
    cost_breakdown: CostBreakdown
    total_distance: float
    total_time: float
    total_service_time: float
    total_travel_time: float
    total_demand_weight: float
    total_demand_volume: float
    num_orders: int
    num_tasks: int
    orders: List[OrderPricingInfo]
    break_even_revenue: float
    target_revenue_with_margin: float


class StaticPricingCalculator:
    """
    Calculator for determining break-even and profitable pricing for VRP solutions
    """
    
    def __init__(self, 
                 profit_margin: float = 0.0,
                 time_weight: float = 0.4,
                 distance_weight: float = 0.3,
                 demand_weight: float = 0.2,
                 tasks_weight: float = 0.1):
        """
        Initialize pricing calculator
        
        Args:
            profit_margin: Target profit margin (0.0 = break-even, 0.2 = 20% profit)
            time_weight: Weight for time-based allocation (should sum to 1.0 with other weights)
            distance_weight: Weight for distance-based allocation
            demand_weight: Weight for demand-based allocation
            tasks_weight: Weight for task count-based allocation
        """
        self.profit_margin = profit_margin
        self.time_weight = time_weight
        self.distance_weight = distance_weight
        self.demand_weight = demand_weight  
        self.tasks_weight = tasks_weight
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = time_weight + distance_weight + demand_weight + tasks_weight
        if total_weight != 1.0:
            self.time_weight /= total_weight
            self.distance_weight /= total_weight
            self.demand_weight /= total_weight
            self.tasks_weight /= total_weight
    
    def calculate_route_pricing(self, solution, orders: List) -> Dict[str, RoutePricingInfo]:
        """
        Calculate pricing information for all routes using SIMPLE formula:
        Cost = (Hours driven × Driver wage) + (km × Vehicle cost per km) + Fixed vehicle cost
        
        Args:
            solution: VRP solution object with routes
            orders: List of all orders
            
        Returns:
            Dictionary mapping vehicle_id to RoutePricingInfo
        """
        route_pricing = {}
        
        # Debug: Show Excel data being used
        print(f"\nDEBUG: Excel data being used for pricing:")
        print(f"="*50)
        
        for vehicle_id, route in solution.routes.items():
            if not route.tasks or len([t for t in route.tasks if not self._is_depot_task(t)]) == 0:
                continue  # Skip empty routes or routes with only depot tasks
                
            # Get route metrics (time and distance)
            route_info = self._analyze_route_metrics(route, orders)
            
            # Simple break-even calculation using Excel data
            hours_driven = route_info['total_time'] / 60.0  # Convert minutes to hours
            distance_km = route_info['total_distance']
            
            # Get costs from Excel data (VEICOLI and AUTISTI sheets)
            # Get driver wage from AUTISTI sheet (COST PER HOUR column)
            if hasattr(route, 'driver') and route.driver and hasattr(route.driver, 'cost_per_hour'):
                driver_wage_per_hour = route.driver.cost_per_hour  # From AUTISTI sheet
            else:
                driver_wage_per_hour = getattr(route.vehicle, 'cost_per_hour', 25.0)  # Fallback
            
            # Get vehicle costs from VEICOLI sheet  
            vehicle_cost_per_km = getattr(route.vehicle, 'cost_per_km', 0.50)     # COST PER KM column
            fixed_vehicle_cost = getattr(route.vehicle, 'daily_cost', 100.0)      # FIXED COST column
            
            # If the vehicle doesn't have these attributes, try alternative names from Excel
            if not hasattr(route.vehicle, 'cost_per_km'):
                vehicle_cost_per_km = getattr(route.vehicle, 'COST_PER_KM', 0.50)
            if not hasattr(route.vehicle, 'daily_cost'):
                fixed_vehicle_cost = getattr(route.vehicle, 'FIXED_COST', 100.0)
            
            # Debug: Show Excel values for first few routes
            if len(route_pricing) < 3:  # Show first 3 routes
                driver_name = route.driver.name if hasattr(route, 'driver') and route.driver else "No driver"
                print(f"Route {vehicle_id} - Driver: {driver_name}")
                print(f"  Driver cost/hour: €{driver_wage_per_hour:.2f} (from AUTISTI sheet)")
                print(f"  Vehicle cost/km: €{vehicle_cost_per_km:.2f} (from VEICOLI sheet)")  
                print(f"  Fixed cost: €{fixed_vehicle_cost:.2f} (from VEICOLI sheet)")
            
            # SIMPLE FORMULA: Hours×Wage + km×Cost_per_km + Fixed_cost
            driver_cost = hours_driven * driver_wage_per_hour
            distance_cost = distance_km * vehicle_cost_per_km  
            fixed_cost = fixed_vehicle_cost
            total_route_cost = driver_cost + distance_cost + fixed_cost
            
            # Debug info for wage verification
            if vehicle_id == 'XA321KW':  # Debug for the specific route you mentioned
                print(f"DEBUG Route {vehicle_id}:")
                print(f"  Hours: {hours_driven:.2f}")
                print(f"  Driver wage/hour from Excel: €{driver_wage_per_hour:.2f}")
                print(f"  Distance: {distance_km:.2f} km")
                print(f"  Cost per km from Excel: €{vehicle_cost_per_km:.2f}")
                print(f"  Fixed cost from Excel: €{fixed_vehicle_cost:.2f}")
                print(f"  Calculated: Driver=€{driver_cost:.2f}, Distance=€{distance_cost:.2f}, Fixed=€{fixed_cost:.2f}")
                print(f"  Total: €{total_route_cost:.2f}")
            
            # Create simplified cost breakdown
            cost_breakdown = CostBreakdown()
            cost_breakdown.driver_salary = driver_cost
            cost_breakdown.distance_cost = distance_cost
            cost_breakdown.base_vehicle_cost = fixed_cost
            cost_breakdown.vehicle_operating_cost = 0.0  # Not used in simple formula
            cost_breakdown.fuel_cost = 0.0  # Included in cost_per_km
            cost_breakdown.time_window_penalties = 0.0  # Disabled
            cost_breakdown.weight_penalties = 0.0  # Disabled
            cost_breakdown.hos_penalties = 0.0  # Disabled
            cost_breakdown.other_costs = 0.0  # None
            
            # Calculate order allocations using the simple total cost
            order_pricing_list = self._calculate_order_allocations(
                route, route_info, total_route_cost, orders
            )
            
            # Create route pricing info
            route_pricing_info = RoutePricingInfo(
                vehicle_id=vehicle_id,
                total_cost=total_route_cost,
                cost_breakdown=cost_breakdown,
                total_distance=route_info['total_distance'],
                total_time=route_info['total_time'],
                total_service_time=route_info['total_service_time'],
                total_travel_time=route_info['total_travel_time'],
                total_demand_weight=route_info['total_demand_weight'],
                total_demand_volume=route_info['total_demand_volume'],
                num_orders=route_info['num_orders'],
                num_tasks=route_info['num_tasks'],
                orders=order_pricing_list,
                break_even_revenue=total_route_cost,
                target_revenue_with_margin=total_route_cost * (1 + self.profit_margin)
            )
            
            route_pricing[vehicle_id] = route_pricing_info
            
        return route_pricing
    
    def _calculate_detailed_cost_breakdown(self, route) -> CostBreakdown:
        """Calculate detailed cost breakdown for a route"""
        
        breakdown = CostBreakdown()
        
        # Basic vehicle parameters
        vehicle = route.vehicle
        
        # Calculate route timing
        total_time_minutes = 0.0
        total_distance_km = 0.0
        
        # Import route calculation functions if available
        try:
            from route_provider import calculate_travel_time_between_tasks
        except ImportError:
            # Fallback function
            def calculate_travel_time_between_tasks(task1, task2, vehicle):
                return 30.0  # Default 30 minutes
        
        # Calculate travel times and distances
        prev_task = None
        for task in route.tasks:
            if prev_task is not None:
                travel_time = calculate_travel_time_between_tasks(prev_task, task, vehicle)
                total_time_minutes += travel_time
                total_distance_km += travel_time * 1.0  # Approximate: 1 km per minute average speed
            
            # Add service time for non-depot tasks
            if not self._is_depot_task(task):
                service_time = getattr(task, 'service_time', 5.0)
                total_time_minutes += service_time
            
            prev_task = task
        
        # Calculate cost components
        
        # 1. Driver salary (based on time)
        driver_hourly_rate = getattr(vehicle, 'cost_per_hour', 25.0)  # Default €25/hour
        breakdown.driver_salary = (total_time_minutes / 60.0) * driver_hourly_rate
        
        # 2. Vehicle operating cost (depreciation, maintenance, insurance)
        vehicle_operating_rate = getattr(vehicle, 'operating_cost_per_hour', 8.0)  # Default €8/hour
        breakdown.vehicle_operating_cost = (total_time_minutes / 60.0) * vehicle_operating_rate
        
        # 3. Fuel cost (based on distance and vehicle type)
        fuel_consumption_per_km = self._get_fuel_consumption(vehicle)
        fuel_price_per_liter = 1.4  # Default €1.40/liter
        breakdown.fuel_cost = total_distance_km * fuel_consumption_per_km * fuel_price_per_liter
        
        # 4. Base vehicle cost (fixed cost per day)
        base_daily_cost = getattr(vehicle, 'daily_cost', 50.0)  # Default €50/day
        breakdown.base_vehicle_cost = base_daily_cost
        
        # 5. Distance-based cost (toll roads, wear and tear)
        distance_cost_per_km = getattr(vehicle, 'cost_per_km', 0.15)  # Default €0.15/km
        breakdown.distance_cost = total_distance_km * distance_cost_per_km
        
        # 6. Penalties (NOW DISABLED for realistic pricing)
        # Since we disabled penalties in the Z2 scoring for realistic cost calculation,
        # set all penalties to 0
        breakdown.time_window_penalties = 0.0  # Disabled
        breakdown.weight_penalties = 0.0       # Disabled  
        breakdown.hos_penalties = 0.0          # Disabled
        breakdown.other_costs = 0.0           # No additional penalties
        
        return breakdown
    
    def _get_fuel_consumption(self, vehicle) -> float:
        """Get fuel consumption per km based on vehicle type"""
        vehicle_type = getattr(vehicle, 'vehicle_type', 'standard').lower()
        
        if vehicle_type == 'heavy':
            return 0.35  # 35 liters per 100 km
        elif vehicle_type == 'standard':  
            return 0.12  # 12 liters per 100 km
        else:
            return 0.15  # Default 15 liters per 100 km
    
    def _get_z2_score_safely(self, route) -> float:
        """Safely get Z2 score for a route"""
        try:
            from second_level import calculate_z2_score
            return calculate_z2_score(route)
        except:
            # Fallback calculation if Z2 not available
            return 100.0  # Default fallback cost
    
    def _analyze_route_metrics(self, route, orders: List) -> Dict[str, Any]:
        """Analyze route to extract key metrics for pricing"""
        
        # Import route calculation functions
        try:
            from route_provider import calculate_travel_time_between_tasks
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algo'))
            try:
                from route_provider import calculate_travel_time_between_tasks
            except ImportError:
                # Fallback function if route provider not available
                def calculate_travel_time_between_tasks(task1, task2, vehicle):
                    return 30.0  # Default 30 minutes
        
        metrics = {
            'total_distance': 0.0,
            'total_time': 0.0,
            'total_service_time': 0.0,
            'total_travel_time': 0.0,
            'total_demand_weight': 0.0,
            'total_demand_volume': 0.0,
            'num_orders': 0,
            'num_tasks': 0,
            'order_details': {}
        }
        
        # Filter non-depot tasks
        non_depot_tasks = [t for t in route.tasks if not self._is_depot_task(t)]
        
        if not non_depot_tasks:
            return metrics
        
        # Calculate travel times between consecutive tasks
        prev_task = None
        for task in route.tasks:
            if prev_task is not None:
                travel_time = calculate_travel_time_between_tasks(prev_task, task, route.vehicle)
                metrics['total_travel_time'] += travel_time
                metrics['total_time'] += travel_time
            
            # Add service time for non-depot tasks
            if not self._is_depot_task(task):
                service_time = getattr(task, 'service_time', 5.0)
                metrics['total_service_time'] += service_time
                metrics['total_time'] += service_time
                metrics['num_tasks'] += 1
                
                # Track order-specific metrics
                order_id = getattr(task, 'order_id', 'unknown')
                if order_id not in metrics['order_details']:
                    metrics['order_details'][order_id] = {
                        'service_time': 0.0,
                        'travel_time': 0.0,
                        'demand_weight': 0.0,
                        'demand_volume': 0.0,
                        'num_tasks': 0
                    }
                
                metrics['order_details'][order_id]['service_time'] += service_time
                metrics['order_details'][order_id]['num_tasks'] += 1
                
                # Add demand (only for pickup tasks to avoid double counting)
                if self._is_pickup_task(task):
                    demand_weight = abs(getattr(task, 'demand', 0.0))
                    demand_volume = abs(getattr(task, 'volume', 0.0))
                    
                    metrics['total_demand_weight'] += demand_weight
                    metrics['total_demand_volume'] += demand_volume
                    
                    metrics['order_details'][order_id]['demand_weight'] += demand_weight
                    metrics['order_details'][order_id]['demand_volume'] += demand_volume
            
            prev_task = task
        
        # Count unique orders
        metrics['num_orders'] = len(metrics['order_details'])
        
        # Calculate realistic distance based on travel time
        # More realistic: assume average speed of 50 km/h for intercity routes
        average_speed_kmh = 50.0
        metrics['total_distance'] = (metrics['total_travel_time'] / 60.0) * average_speed_kmh
        
        return metrics
    
    def _calculate_order_allocations(self, route, route_info: Dict, total_route_cost: float, orders: List) -> List[OrderPricingInfo]:
        """Calculate cost allocation for each order in the route"""
        
        order_pricing_list = []
        
        for order_id, order_metrics in route_info['order_details'].items():
            
            # Calculate allocation shares
            time_share = (order_metrics['service_time'] / route_info['total_service_time'] 
                         if route_info['total_service_time'] > 0 else 0)
            
            # For distance/travel allocation, distribute travel time proportionally by service time
            distance_share = time_share  # Simplified - could be enhanced with actual route analysis
            
            demand_share = (order_metrics['demand_weight'] / route_info['total_demand_weight'] 
                           if route_info['total_demand_weight'] > 0 else 0)
            
            tasks_share = (order_metrics['num_tasks'] / route_info['num_tasks'] 
                          if route_info['num_tasks'] > 0 else 0)
            
            # Calculate weighted allocation share
            allocation_share = (
                self.time_weight * time_share +
                self.distance_weight * distance_share +
                self.demand_weight * demand_share +
                self.tasks_weight * tasks_share
            )
            
            # Calculate pricing
            route_cost_share = total_route_cost * allocation_share
            break_even_price = route_cost_share  # Break-even price equals allocated cost
            final_price = break_even_price * (1 + self.profit_margin)
            
            # Create order pricing info
            order_pricing = OrderPricingInfo(
                order_id=order_id,
                route_vehicle_id=route.vehicle.id,
                service_time=order_metrics['service_time'],
                travel_time=order_metrics.get('travel_time', 0.0),
                demand_weight=order_metrics['demand_weight'],
                demand_volume=order_metrics['demand_volume'],
                num_tasks=order_metrics['num_tasks'],
                allocation_share=allocation_share,
                route_cost_share=route_cost_share,
                break_even_price=break_even_price,
                final_price_with_margin=final_price
            )
            
            order_pricing_list.append(order_pricing)
        
        return order_pricing_list
    
    def _is_depot_task(self, task) -> bool:
        """Check if a task is a depot task"""
        if hasattr(task, 'is_depot_start') and task.is_depot_start():
            return True
        if hasattr(task, 'is_depot_return') and task.is_depot_return():
            return True
        location_name = getattr(task, 'location_id', '').upper()
        return 'DEPOT' in location_name
    
    def _is_pickup_task(self, task) -> bool:
        """Check if a task is a pickup task"""
        task_type = getattr(task, 'task_type', None)
        if hasattr(task_type, 'name'):
            return task_type.name.upper() == 'PICKUP'
        if hasattr(task, 'demand'):
            return getattr(task, 'demand', 0) > 0
        return False
    
    def generate_pricing_report(self, route_pricing: Dict[str, RoutePricingInfo]) -> str:
        """Generate a comprehensive pricing report"""
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE PRICING REPORT")
        report.append("="*80)
        
        # Summary statistics
        total_routes = len(route_pricing)
        total_orders = sum(len(rp.orders) for rp in route_pricing.values())
        total_cost = sum(rp.total_cost for rp in route_pricing.values())
        total_break_even_revenue = sum(rp.break_even_revenue for rp in route_pricing.values())
        total_target_revenue = sum(rp.target_revenue_with_margin for rp in route_pricing.values())
        
        report.append(f"\nSUMMARY:")
        report.append(f"   - Routes analyzed: {total_routes}")
        report.append(f"   - Orders priced: {total_orders}")
        report.append(f"   - Total operational cost: €{total_cost:.2f}")
        report.append(f"   - Break-even revenue needed: €{total_break_even_revenue:.2f}")
        if self.profit_margin > 0:
            report.append(f"   - Target revenue ({self.profit_margin:.1%} margin): €{total_target_revenue:.2f}")
            report.append(f"   - Expected profit: €{total_target_revenue - total_cost:.2f}")
        
        report.append(f"\nPRICING ALLOCATION WEIGHTS:")
        report.append(f"   - Time/Service: {self.time_weight:.1%}")
        report.append(f"   - Distance/Travel: {self.distance_weight:.1%}")
        report.append(f"   - Demand/Weight: {self.demand_weight:.1%}")
        report.append(f"   - Task Count: {self.tasks_weight:.1%}")
        
        # Detailed route breakdown
        report.append(f"\n" + "="*60)
        report.append("DETAILED ROUTE PRICING BREAKDOWN")
        report.append("="*60)
        
        for vehicle_id, route_info in route_pricing.items():
            report.append(f"\n--- Route {vehicle_id} ---")
            report.append(f"   Total Cost: €{route_info.total_cost:.2f}")
            report.append(f"   Distance: {route_info.total_distance:.1f} km")
            report.append(f"   Time: {route_info.total_time:.1f} min")
            report.append(f"   Orders: {route_info.num_orders}, Tasks: {route_info.num_tasks}")
            
            # Cost breakdown
            breakdown = route_info.cost_breakdown
            report.append(f"\n   COST BREAKDOWN:")
            report.append(f"     • Driver Salary: €{breakdown.driver_salary:.2f} ({breakdown.driver_salary/route_info.total_cost*100:.1f}%)")
            report.append(f"     • Vehicle Operating: €{breakdown.vehicle_operating_cost:.2f} ({breakdown.vehicle_operating_cost/route_info.total_cost*100:.1f}%)")
            report.append(f"     • Fuel Cost: €{breakdown.fuel_cost:.2f} ({breakdown.fuel_cost/route_info.total_cost*100:.1f}%)")
            report.append(f"     • Base Vehicle Cost: €{breakdown.base_vehicle_cost:.2f} ({breakdown.base_vehicle_cost/route_info.total_cost*100:.1f}%)")
            report.append(f"     • Distance Cost: €{breakdown.distance_cost:.2f} ({breakdown.distance_cost/route_info.total_cost*100:.1f}%)")
            if breakdown.time_window_penalties > 0:
                report.append(f"     • Time Window Penalties: €{breakdown.time_window_penalties:.2f} ({breakdown.time_window_penalties/route_info.total_cost*100:.1f}%)")
            if breakdown.weight_penalties > 0:
                report.append(f"     • Weight Penalties: €{breakdown.weight_penalties:.2f} ({breakdown.weight_penalties/route_info.total_cost*100:.1f}%)")
            if breakdown.hos_penalties > 0:
                report.append(f"     • HoS Penalties: €{breakdown.hos_penalties:.2f} ({breakdown.hos_penalties/route_info.total_cost*100:.1f}%)")
            if breakdown.other_costs > 0:
                report.append(f"     • Other Costs: €{breakdown.other_costs:.2f} ({breakdown.other_costs/route_info.total_cost*100:.1f}%)")
            
            report.append(f"   Break-even Revenue: €{route_info.break_even_revenue:.2f}")
            if self.profit_margin > 0:
                report.append(f"   Target Revenue: €{route_info.target_revenue_with_margin:.2f}")
            
            report.append(f"\n   ORDER PRICING:")
            for order in route_info.orders:
                margin_info = f" → €{order.final_price_with_margin:.2f}" if self.profit_margin > 0 else ""
                report.append(f"     • Order {order.order_id}: {order.allocation_share:.1%} share, "
                             f"€{order.break_even_price:.2f}{margin_info}")
                report.append(f"       ({order.service_time:.0f}min service, {order.demand_weight:.0f}kg, "
                             f"{order.num_tasks} tasks)")
        
        # Cost category summary
        report.append(f"\n" + "="*60)
        report.append("COST CATEGORY SUMMARY")
        report.append("="*60)
        
        total_driver_salary = sum(rp.cost_breakdown.driver_salary for rp in route_pricing.values())
        total_vehicle_operating = sum(rp.cost_breakdown.vehicle_operating_cost for rp in route_pricing.values())
        total_fuel = sum(rp.cost_breakdown.fuel_cost for rp in route_pricing.values())
        total_base_vehicle = sum(rp.cost_breakdown.base_vehicle_cost for rp in route_pricing.values())
        total_distance = sum(rp.cost_breakdown.distance_cost for rp in route_pricing.values())
        total_penalties = sum(rp.cost_breakdown.time_window_penalties + rp.cost_breakdown.weight_penalties + 
                             rp.cost_breakdown.hos_penalties + rp.cost_breakdown.other_costs for rp in route_pricing.values())
        
        report.append(f"\nTOTAL COST BREAKDOWN:")
        report.append(f"   - Driver Salaries: €{total_driver_salary:.2f} ({total_driver_salary/total_cost*100:.1f}%)")
        report.append(f"   - Vehicle Operating: €{total_vehicle_operating:.2f} ({total_vehicle_operating/total_cost*100:.1f}%)")
        report.append(f"   - Fuel Costs: €{total_fuel:.2f} ({total_fuel/total_cost*100:.1f}%)")
        report.append(f"   - Base Vehicle Costs: €{total_base_vehicle:.2f} ({total_base_vehicle/total_cost*100:.1f}%)")
        report.append(f"   - Distance-based Costs: €{total_distance:.2f} ({total_distance/total_cost*100:.1f}%)")
        if total_penalties > 0:
            report.append(f"   - Penalties & Violations: €{total_penalties:.2f} ({total_penalties/total_cost*100:.1f}%)")
        
        return "\n".join(report)
    
    def get_order_pricing_summary(self, route_pricing: Dict[str, RoutePricingInfo]) -> Dict[str, Dict]:
        """Get a summary of pricing for each order (useful for integration with other systems)"""
        
        order_summary = {}
        
        for route_info in route_pricing.values():
            for order in route_info.orders:
                order_summary[order.order_id] = {
                    'route_vehicle': order.route_vehicle_id,
                    'break_even_price': order.break_even_price,
                    'final_price': order.final_price_with_margin,
                    'allocation_share': order.allocation_share,
                    'service_time': order.service_time,
                    'demand_weight': order.demand_weight,
                    'num_tasks': order.num_tasks
                }
        
        return order_summary
