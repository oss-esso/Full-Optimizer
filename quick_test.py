#!/usr/bin/env python3
"""
Quick test to verify both adapters are working correctly.
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing adapter imports...")
    
    # Test simulated annealing import
    try:
        from my_functions.simulated_annealing_adapter import (
            SimulatedAnnealingAdapter, SimulatedAnnealingConfig
        )
        logger.info("✓ Simulated annealing adapter imported successfully")
        sa_available = True
    except ImportError as e:
        logger.error(f"✗ Simulated annealing import failed: {e}")
        sa_available = False
    
    # Test D-Wave import
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        logger.info("✓ D-Wave adapter imported successfully")
        dwave_available = True
    except ImportError as e:
        logger.warning(f"⚠ D-Wave adapter import failed: {e}")
        dwave_available = False
    
    if sa_available:
        logger.info("Testing simulated annealing...")
        
        # Simple 2x2 QUBO test
        Q = np.array([[1, -2], [-2, 1]])
        offset = 0.0
        
        config = SimulatedAnnealingConfig(
            initial_temperature=10.0,
            max_iterations=100,
            log_interval=1000  # Reduce logging
        )
        
        adapter = SimulatedAnnealingAdapter(config=config)
        result = adapter._solve_qubo_sa(Q, offset, 2)
        
        if result.get('error'):
            logger.error(f"SA test failed: {result['error']}")
        else:
            logger.info(f"SA test passed: energy={result['energy']:.4f}, solution={result['solution']}")
    
    logger.info(f"Summary: SA={sa_available}, D-Wave={dwave_available}")
    return sa_available

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
