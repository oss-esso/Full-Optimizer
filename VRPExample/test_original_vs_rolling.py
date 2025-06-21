#!/usr/bin/env python3

import os
import sys
import logging

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vrp_scenarios import create_moda_small_scenario, create_moda_first_scenario
from vrp_optimizer_fixed import VRPQuantumOptimizer
from vrp_optimizer_rolling_window import VRPOptimizerRollingWindow
from vrp_data_models import VRPObjective

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_original_vs_rolling_window():
    """Compare the original fixed optimizer vs rolling window optimizer."""
    
    logger.info("=" * 70)
    logger.info("COMPARING ORIGINAL FIXED vs ROLLING WINDOW OPTIMIZERS")
    logger.info("=" * 70)
    
    scenarios = [
        ("MODA_small", create_moda_small_scenario),
        ("MODA_first", create_moda_first_scenario)
    ]
    
    for scenario_name, scenario_func in scenarios:
        logger.info(f"\n{'='*20} {scenario_name} {'='*20}")
        
        try:
            # Generate scenario once
            instance = scenario_func()
            
            # Test original fixed optimizer (with OR-Tools)
            logger.info(f"\n--- Testing {scenario_name} with ORIGINAL FIXED OPTIMIZER ---")
            
            original_optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
            
            # Try to call the OR-Tools method if it exists
            try:
                if hasattr(original_optimizer, '_solve_with_ortools'):
                    original_result = original_optimizer._solve_with_ortools()
                else:
                    # Fall back to quantum method
                    original_result = original_optimizer.optimize_with_quantum_benders()
                
                logger.info(f"Original optimizer result:")
                logger.info(f"  - Status: {original_result.status}")
                logger.info(f"  - Runtime: {original_result.runtime:.2f} ms")
                
                if original_result.status == "optimal":
                    metrics = original_result.metrics
                    logger.info(f"  - Total distance: {metrics.get('total_distance', 0):.2f}")
                    logger.info(f"  - Vehicles used: {metrics.get('vehicles_used', 0)}")
                    
                    # Count active routes
                    active_routes = {k: v for k, v in original_result.routes.items() if len(v) > 2}
                    logger.info(f"  - Active routes: {len(active_routes)}")
                    
                    for vehicle_id, route in active_routes.items():
                        route_length = len(route) - 2  # Exclude start/end depot
                        logger.info(f"    - Vehicle {vehicle_id}: {route_length} stops")
                else:
                    logger.warning(f"  - Failed: {original_result.metrics}")
                    
            except Exception as e:
                logger.error(f"Original optimizer failed: {str(e)}")
                original_result = None
            
            # Test rolling window optimizer
            logger.info(f"\n--- Testing {scenario_name} with ROLLING WINDOW OPTIMIZER ---")
            
            rolling_optimizer = VRPOptimizerRollingWindow(instance, VRPObjective.MINIMIZE_DISTANCE)
            rolling_result = rolling_optimizer.optimize_with_rolling_window()
            
            logger.info(f"Rolling window optimizer result:")
            logger.info(f"  - Status: {rolling_result.status}")
            logger.info(f"  - Runtime: {rolling_result.runtime:.2f} ms")
            
            if rolling_result.status == "optimal":
                metrics = rolling_result.metrics
                logger.info(f"  - Total distance: {metrics.get('total_distance', 0):.2f}")
                logger.info(f"  - Vehicles used: {metrics.get('vehicles_used', 0)}")
                
                # Count active routes
                active_routes = {k: v for k, v in rolling_result.routes.items() if len(v) > 2}
                logger.info(f"  - Active routes: {len(active_routes)}")
                
                for vehicle_id, route in active_routes.items():
                    route_length = len(route) - 2  # Exclude start/end depot
                    logger.info(f"    - Vehicle {vehicle_id}: {route_length} stops")
            else:
                logger.warning(f"  - Failed: {rolling_result.metrics}")
            
            # Compare results
            logger.info(f"\n--- COMPARISON for {scenario_name} ---")
            if original_result and rolling_result:
                logger.info(f"Original: {original_result.status} | Rolling: {rolling_result.status}")
                
                if original_result.status == "optimal" and rolling_result.status == "optimal":
                    orig_vehicles = original_result.metrics.get('vehicles_used', 0)
                    roll_vehicles = rolling_result.metrics.get('vehicles_used', 0)
                    orig_distance = original_result.metrics.get('total_distance', 0)
                    roll_distance = rolling_result.metrics.get('total_distance', 0)
                    
                    logger.info(f"Vehicles used: Original={orig_vehicles}, Rolling={roll_vehicles}")
                    logger.info(f"Total distance: Original={orig_distance:.2f}, Rolling={roll_distance:.2f}")
                    
                    if orig_vehicles == roll_vehicles and abs(orig_distance - roll_distance) < 0.1:
                        logger.info("✅ Both optimizers found similar solutions")
                    else:
                        logger.info("⚠️  Different solutions found")
                        
                elif original_result.status == "optimal" and rolling_result.status != "optimal":
                    logger.warning("⚠️  Original worked but Rolling failed - regression detected!")
                    
                elif original_result.status != "optimal" and rolling_result.status == "optimal":
                    logger.info("✅ Rolling window improved over original")
                    
                else:
                    logger.warning("❌ Both optimizers failed")
            
        except Exception as e:
            logger.error(f"Error testing {scenario_name}: {str(e)}")
            import traceback
            traceback.print_exc()

def test_original_ortools_method():
    """Test specifically the OR-Tools method from the original optimizer."""
    
    logger.info(f"\n" + "=" * 70)
    logger.info("TESTING ORIGINAL OR-TOOLS METHOD DIRECTLY")
    logger.info("=" * 70)
    
    scenarios = [
        ("MODA_small", create_moda_small_scenario),
        ("MODA_first", create_moda_first_scenario)
    ]
    
    for scenario_name, scenario_func in scenarios:
        logger.info(f"\n--- Testing {scenario_name} with Original OR-Tools Method ---")
        
        try:
            instance = scenario_func()
            optimizer = VRPQuantumOptimizer(instance, VRPObjective.MINIMIZE_DISTANCE)
            
            # Check if the original has OR-Tools method
            if hasattr(optimizer, '_solve_with_ortools'):
                logger.info("Found _solve_with_ortools method, calling it...")
                result = optimizer._solve_with_ortools()
            else:
                logger.info("No _solve_with_ortools method found, trying other methods...")
                # Try to find any OR-Tools related method
                methods = [method for method in dir(optimizer) if 'ortools' in method.lower()]
                logger.info(f"Available methods with 'ortools': {methods}")
                
                # Fallback to quantum method
                result = optimizer.optimize_with_quantum_benders()
            
            logger.info(f"Result: {result.status}")
            if result.status == "optimal":
                metrics = result.metrics
                vehicles_used = metrics.get('vehicles_used', 0)
                total_distance = metrics.get('total_distance', 0)
                logger.info(f"  - Vehicles used: {vehicles_used}")
                logger.info(f"  - Total distance: {total_distance:.2f}")
                
                # Check for OR-Tools specific metrics
                ortools_metrics = {k: v for k, v in metrics.items() if 'ortools' in k.lower()}
                if ortools_metrics:
                    logger.info(f"  - OR-Tools metrics: {ortools_metrics}")
            else:
                logger.warning(f"  - Failed: {result.metrics}")
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Comparing Original Fixed vs Rolling Window Optimizers")
    print("=====================================================")
    
    # Test both optimizers
    test_original_vs_rolling_window()
    
    # Test original OR-Tools method specifically
    test_original_ortools_method()
    
    print("\nComparison completed!")
