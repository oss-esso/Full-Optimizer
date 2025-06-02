#!/usr/bin/env python3
"""
Test script to verify the enhanced simulated annealing features.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_vs_enhanced_sa():
    """Compare basic SA vs enhanced SA on the same problem."""
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        
        logger.info("Testing basic vs enhanced simulated annealing...")
        
        # Create optimizer instance
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        logger.info("✓ Optimizer created and data loaded")
        
        # Test basic SA (default)
        logger.info("Running BASIC simulated annealing...")
        basic_result = optimizer.optimize_with_simulated_annealing_benders(
            max_iterations=200,
            initial_temperature=50.0,
            cooling_rate=0.95,
            enhanced_sa=False  # Use basic SA
        )
        
        logger.info(f"Basic SA - Objective: {basic_result.objective_value:.6f}")
        logger.info(f"Basic SA - Runtime: {basic_result.runtime:.3f}s")
        logger.info(f"Basic SA - Foods selected: {basic_result.metrics.get('total_foods_selected', 'N/A')}")
        
        # Test enhanced SA
        logger.info("Running ENHANCED simulated annealing...")
        enhanced_result = optimizer.optimize_with_simulated_annealing_benders(
            max_iterations=200,
            initial_temperature=50.0,
            cooling_rate=0.95,
            enhanced_sa=True,  # Use enhanced SA
            adaptive_cooling=True,
            use_restart=True,
            neighborhood_type="multi_flip"
        )
        
        logger.info(f"Enhanced SA - Objective: {enhanced_result.objective_value:.6f}")
        logger.info(f"Enhanced SA - Runtime: {enhanced_result.runtime:.3f}s")
        logger.info(f"Enhanced SA - Foods selected: {enhanced_result.metrics.get('total_foods_selected', 'N/A')}")
        
        # Compare results
        improvement = ((basic_result.objective_value - enhanced_result.objective_value) / 
                      abs(basic_result.objective_value)) * 100 if basic_result.objective_value != 0 else 0
        
        logger.info("=" * 60)
        logger.info("COMPARISON RESULTS:")
        logger.info(f"Basic SA Objective:     {basic_result.objective_value:.6f}")
        logger.info(f"Enhanced SA Objective:  {enhanced_result.objective_value:.6f}")
        logger.info(f"Improvement:            {improvement:.2f}%")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced SA test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_enhanced_sa_features():
    """Test specific enhanced SA features."""
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        
        logger.info("Testing enhanced SA specific features...")
        
        # Create optimizer instance
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        # Test different cooling schedules
        cooling_schedules = ["exponential", "adaptive"]
        neighborhood_types = ["single_flip", "multi_flip"]
        
        results = []
        
        for cooling in cooling_schedules:
            for neighborhood in neighborhood_types:
                logger.info(f"Testing: {cooling} cooling + {neighborhood} neighborhood")
                
                result = optimizer.optimize_with_simulated_annealing_benders(
                    max_iterations=100,
                    initial_temperature=30.0,
                    cooling_rate=0.9,
                    enhanced_sa=True,
                    cooling_schedule=cooling,
                    neighborhood_type=neighborhood,
                    adaptive_cooling=(cooling == "adaptive"),
                    use_restart=True
                )
                
                results.append({
                    'config': f"{cooling}+{neighborhood}",
                    'objective': result.objective_value,
                    'runtime': result.runtime,
                    'foods_selected': result.metrics.get('total_foods_selected', 0)
                })
                
                logger.info(f"  Result: {result.objective_value:.6f} ({result.runtime:.3f}s)")
        
        # Display results summary
        logger.info("=" * 60)
        logger.info("ENHANCED SA FEATURES TEST RESULTS:")
        for r in results:
            logger.info(f"{r['config']:20} | Obj: {r['objective']:8.4f} | "
                       f"Time: {r['runtime']:6.3f}s | Foods: {r['foods_selected']:2d}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced SA features test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_enhanced_sa_availability():
    """Test that enhanced SA components are available."""
    try:
        # Test imports
        from my_functions.enhanced_simulated_annealing import (
            EnhancedSimulatedAnnealing, SAConfig, CoolingSchedule, NeighborhoodType
        )
        logger.info("✓ Enhanced SA imports successful")
        
        # Test enum values
        logger.info(f"Available cooling schedules: {[cs.value for cs in CoolingSchedule]}")
        logger.info(f"Available neighborhood types: {[nt.value for nt in NeighborhoodType]}")
        
        # Test basic functionality
        import numpy as np
        
        def simple_objective(x):
            return np.sum((x - 0.5)**2)
        
        config = SAConfig(
            initial_temperature=10.0,
            cooling_schedule=CoolingSchedule.ADAPTIVE,
            neighborhood_type=NeighborhoodType.SINGLE_FLIP,
            max_iterations=50,
            track_history=True
        )
        
        enhanced_sa = EnhancedSimulatedAnnealing(
            objective_function=simple_objective,
            initial_params=np.random.random(5),
            config=config
        )
        
        result = enhanced_sa.optimize()
        metrics = enhanced_sa.get_performance_metrics()
        
        logger.info(f"✓ Enhanced SA standalone test: {result.objective_value:.6f}")
        logger.info(f"✓ Performance metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced SA availability test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all enhanced SA tests."""
    logger.info("=" * 60)
    logger.info("ENHANCED SIMULATED ANNEALING TEST SUITE")
    logger.info("=" * 60)
    
    test1 = test_enhanced_sa_availability()
    test2 = test_basic_vs_enhanced_sa()
    test3 = test_enhanced_sa_features()
    
    logger.info("=" * 60)
    if test1 and test2 and test3:
        logger.info("✓ ALL ENHANCED SA TESTS PASSED!")
        logger.info("✓ Enhanced simulated annealing is fully functional!")
    else:
        logger.error("✗ SOME ENHANCED SA TESTS FAILED")
    logger.info("=" * 60)
    
    return test1 and test2 and test3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
