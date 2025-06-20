"""
VRP Optimization Example Package

This package demonstrates the application of quantum-enhanced optimization methods
to Vehicle Routing Problems (VRP), including the Ride Pooling Problem (RPP).

Based on the paper:
"Modeling routing problems in QUBO with application to ride‑hailing"
by Michele Cattelan & Sheir Yarkoni

Modules:
- vrp_data_models: Data structures for VRP problems
- vrp_scenarios: Pre-defined VRP scenarios for testing
- vrp_optimizer: Quantum-enhanced VRP optimization implementation
- vrp_main: Main script to run VRP optimization examples
"""

from .vrp_data_models import VRPInstance, VRPResult, VRPObjective, Location, Vehicle, RideRequest
from .vrp_scenarios import VRPScenarioGenerator, get_all_scenarios
from .vrp_optimizer import VRPQuantumOptimizer

__version__ = "1.0.0"
__author__ = "OQI Project"
__description__ = "VRP optimization using quantum-enhanced methods"

__all__ = [
    'VRPInstance',
    'VRPResult', 
    'VRPObjective',
    'Location',
    'Vehicle',
    'RideRequest',
    'VRPScenarioGenerator',
    'get_all_scenarios',
    'VRPQuantumOptimizer'
]
