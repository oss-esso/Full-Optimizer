#!/usr/bin/env python3

# Quick test to see why all routes are infeasible
from algo.second_level import is_feasible
from algo.epdt_data_structures import Route, Task, Vehicle, TaskType
import sys

# Simple test - create a route that should definitely be feasible
def test_simple_route():
    vehicle = Vehicle(
        id="test",
        depot_id="depot", 
        weight_capacity=5000.0,
        volume_capacity=50.0,
        lifo_required=False
    )
    
    pickup = Task(
        id="p1",
        location_id="loc1",
        task_type=TaskType.PICKUP,
        order_id="order1",
        lat=45.0,
        lon=9.0,
        service_time=10.0,
        demand=100.0,
        volume=1.0
    )
    
    delivery = Task(
        id="d1", 
        location_id="loc2",
        task_type=TaskType.DELIVERY,
        order_id="order1",
        lat=45.1,
        lon=9.1,
        service_time=10.0,
        demand=-100.0,
        volume=-1.0
    )
    
    route = Route(vehicle=vehicle, tasks=[pickup, delivery])
    
    print("Testing simple route...")
    result = is_feasible(route)
    print(f"Simple route result: {result}")

if __name__ == "__main__":
    test_simple_route()
