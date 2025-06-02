#!/usr/bin/env python3
"""
Test script to verify the simulated annealing method is properly integrated.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simulated_annealing_integration():
    """Test that the simulated annealing method is properly integrated."""
    try:
        # Import the SimpleFoodOptimizer
        from src.Qoptimizer import SimpleFoodOptimizer
        
        logger.info("✓ SimpleFoodOptimizer imported successfully")
        
        # Create optimizer instance
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        logger.info("✓ Optimizer created and data loaded")
        
        # Check if the simulated annealing method is bound
        if hasattr(optimizer, 'optimize_with_simulated_annealing_benders'):
            logger.info("✓ optimize_with_simulated_annealing_benders method is bound")
            
            # Try calling the method (quick test without full optimization)
            try:
                # Test method signature by checking if it's callable
                method = getattr(optimizer, 'optimize_with_simulated_annealing_benders')
                if callable(method):
                    logger.info("✓ Method is callable")
                    
                    # Test with minimal parameters for a quick check
                    logger.info("Testing simulated annealing method with minimal parameters...")
                    result = optimizer.optimize_with_simulated_annealing_benders(
                        max_iterations=10,  # Very short test
                        initial_temperature=10.0,
                        cooling_rate=0.9
                    )
                    
                    if result:
                        logger.info("✓ Simulated annealing method executed successfully")
                        logger.info(f"  Result type: {type(result)}")
                        if hasattr(result, 'objective_value'):
                            logger.info(f"  Objective value: {result.objective_value}")
                        if hasattr(result, 'solver_info'):
                            logger.info(f"  Solver: {result.solver_info.get('method', 'Unknown')}")
                        return True
                    else:
                        logger.error("✗ Method returned None")
                        return False
                        
                else:
                    logger.error("✗ Method is not callable")
                    return False
                    
            except Exception as e:
                logger.error(f"✗ Error calling simulated annealing method: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False
                
        else:
            logger.error("✗ optimize_with_simulated_annealing_benders method is NOT bound")
            
            # List available methods
            available_methods = [method for method in dir(optimizer) if method.startswith('optimize_with')]
            logger.info(f"Available optimization methods: {available_methods}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_solve_method():
    """Test that the solve method supports simulated annealing."""
    try:
        from src.Qoptimizer import SimpleFoodOptimizer
        
        optimizer = SimpleFoodOptimizer(complexity_level='simple')
        optimizer.load_food_data()
        
        # Test solve method with simulated-annealing
        logger.info("Testing solve method with 'simulated-annealing' parameter...")
        
        # Create dummy parameters for the solve method (these won't be used by SA method)
        import numpy as np
        F = len(optimizer.farms)
        C = len(optimizer.foods)
        f = np.zeros((F*C, 1))
        A = np.zeros((1, F*C))
        b = np.zeros((1, 1))
        C_matrix = np.zeros((1, F*C))
        c = np.zeros((F*C, 1))
        
        result = optimizer.solve(
            method='simulated-annealing',
            f=f, A=A, b=b, C=C_matrix, c=c,
            debug=True
        )
        
        if result:
            logger.info("✓ Solve method with simulated-annealing works")
            return True
        else:
            logger.error("✗ Solve method returned None")
            return False
            
    except Exception as e:
        logger.error(f"✗ Solve method test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("SIMULATED ANNEALING INTEGRATION TEST")
    logger.info("=" * 60)
    
    success1 = test_simulated_annealing_integration()
    success2 = test_solve_method()
    
    logger.info("=" * 60)
    if success1 and success2:
        logger.info("✓ ALL TESTS PASSED - Simulated Annealing is properly integrated!")
    else:
        logger.error("✗ SOME TESTS FAILED - Integration needs work")
    logger.info("=" * 60)
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
