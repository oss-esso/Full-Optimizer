#!/usr/bin/env python3
"""
Quick test of enhanced SA functionality in the simulated annealing method.
"""

import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_sa():
    """Test the enhanced SA functionality."""
    try:
        # Test direct import of enhanced SA
        from my_functions.enhanced_simulated_annealing import (
            EnhancedSimulatedAnnealing, SAConfig, CoolingSchedule, NeighborhoodType
        )
        
        logger.info("✓ Enhanced SA imports successful")
        
        # Create a simple test function
        def test_objective(x):
            return np.sum((x - 0.5)**2)
        
        # Test enhanced SA with different configurations
        configurations = [
            ("Basic Exponential", SAConfig(
                cooling_schedule=CoolingSchedule.EXPONENTIAL,
                neighborhood_type=NeighborhoodType.SINGLE_FLIP,
                max_iterations=200
            )),
            ("Adaptive Cooling", SAConfig(
                cooling_schedule=CoolingSchedule.ADAPTIVE,
                neighborhood_type=NeighborhoodType.MULTI_FLIP,
                adaptive_cooling=True,
                max_iterations=200
            )),
            ("Logarithmic + Restart", SAConfig(
                cooling_schedule=CoolingSchedule.LOGARITHMIC,
                neighborhood_type=NeighborhoodType.SINGLE_FLIP,
                use_restart=True,
                restart_threshold=50,
                max_iterations=200
            ))
        ]
        
        results = []
        
        for name, config in configurations:
            logger.info(f"Testing configuration: {name}")
            
            # Random initial solution
            initial = np.random.random(10)
            
            # Create enhanced SA
            enhanced_sa = EnhancedSimulatedAnnealing(test_objective, initial, config)
            
            # Run optimization
            result = enhanced_sa.optimize()
            
            # Get metrics
            metrics = enhanced_sa.get_performance_metrics()
            
            results.append((name, result.objective_value, metrics))
            
            logger.info(f"  Result: {result.objective_value:.6f}")
            logger.info(f"  Iterations: {metrics['total_iterations']}")
            logger.info(f"  Acceptance rate: {metrics['acceptance_rate']:.2%}")
            logger.info(f"  Restarts: {metrics['restarts']}")
            logger.info(f"  Reheats: {metrics['reheats']}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("ENHANCED SA TEST RESULTS SUMMARY")
        logger.info("="*60)
        
        for name, obj_value, metrics in results:
            logger.info(f"{name:20s}: {obj_value:.6f} (iter={metrics['total_iterations']:3d}, "
                       f"acc={metrics['acceptance_rate']:.1%}, restart={metrics['restarts']})")
        
        logger.info("✓ All enhanced SA configurations tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced SA test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_method_integration():
    """Test that the enhanced SA can be used in the method."""
    try:
        # Import method directly
        from src.methods.simulated_annealing_method import optimize_with_simulated_annealing_benders
        
        logger.info("✓ Method import successful")
        
        # Create a mock optimizer object with minimal required attributes
        class MockOptimizer:
            def __init__(self):
                self.farms = ['Farm1', 'Farm2']
                self.foods = {'Food1': {'nutritional_value': 0.8, 'environmental_impact': 0.3}, 
                             'Food2': {'nutritional_value': 0.6, 'environmental_impact': 0.4}}
                self.parameters = {'objective_weights': {}}
                
                # Setup logging
                import logging
                self.logger = logging.getLogger('MockOptimizer')
        
        mock_optimizer = MockOptimizer()
        
        # Test basic SA (should work)
        logger.info("Testing basic SA...")
        result_basic = optimize_with_simulated_annealing_benders(
            mock_optimizer,
            max_iterations=50,
            enhanced_sa=False
        )
        
        if result_basic:
            logger.info(f"✓ Basic SA result: {result_basic.objective_value:.6f}")
        else:
            logger.error("✗ Basic SA failed")
            return False
        
        # Test enhanced SA
        logger.info("Testing enhanced SA...")
        result_enhanced = optimize_with_simulated_annealing_benders(
            mock_optimizer,
            max_iterations=50,
            enhanced_sa=True,
            adaptive_cooling=True,
            use_restart=True
        )
        
        if result_enhanced:
            logger.info(f"✓ Enhanced SA result: {result_enhanced.objective_value:.6f}")
        else:
            logger.error("✗ Enhanced SA failed")
            return False
        
        logger.info("✓ Both basic and enhanced SA methods work!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Method integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    logger.info("ENHANCED SIMULATED ANNEALING TEST")
    logger.info("="*50)
    
    success1 = test_enhanced_sa()
    success2 = test_method_integration()
    
    logger.info("="*50)
    if success1 and success2:
        logger.info("✓ ALL TESTS PASSED!")
    else:
        logger.error("✗ SOME TESTS FAILED")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
