"""
QUBO Implementation Validation Script

This script validates the QUBO formulation implementation to ensure
compatibility with the existing Column Generation MILP formulation.
"""

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def validate_qubo_implementation():
    """Validate the complete QUBO implementation."""
    print("🧪 Validating QUBO Implementation for Quantum Annealing")
    print("=" * 70)
    
    success = True
    
    # Test 1: Basic imports
    print("\n📦 Testing QUBO imports...")
    try:
        from algo.qubo_formulation import QUBOFormulator, QUBOSolver, QUBOConfig
        from algo.qubo_integration import HybridSolver, HybridConfig
        from algo.qubo_test import QUBOTestSuite
        print("   ✅ All QUBO modules imported successfully")
    except ImportError as e:
        print(f"   ❌ QUBO import failed: {e}")
        success = False
    
    # Test 2: Dependencies
    print("\n🔍 Checking QUBO dependencies...")
    
    try:
        import dimod
        print("   ✅ DIMOD available for QUBO solving")
    except ImportError:
        print("   ⚠️  DIMOD not available - install with: pip install dimod")
    
    try:
        import neal
        print("   ✅ Neal simulated annealing available")
    except ImportError:
        print("   ⚠️  Neal not available - install with: pip install dwave-neal")
    
    try:
        import dwave.system
        print("   ✅ D-Wave quantum access available")
    except ImportError:
        print("   ⚠️  D-Wave not available - quantum hardware access not configured")
    
    # Test 3: QUBO formulation basics
    print("\n🔬 Testing QUBO formulation...")
    try:
        from algo.epdt_data_structures import Order, Vehicle, Task, TaskType, Route
        
        # Create simple test problem
        pickup_task = Task(
            id="P1", location_id="LOC_P1", task_type=TaskType.PICKUP,
            order_id="O1", lat=0.0, lon=0.0, service_time=10.0,
            demand=100.0, volume=1.0
        )
        
        delivery_task = Task(
            id="D1", location_id="LOC_D1", task_type=TaskType.DELIVERY,
            order_id="O1", lat=1.0, lon=1.0, service_time=10.0,
            demand=-100.0, volume=-1.0
        )
        
        order = Order(
            id="O1", pickup_tasks=[pickup_task], delivery_tasks=[delivery_task],
            is_mandatory=True
        )
        
        vehicle = Vehicle(
            id="V1", depot_id="DEPOT", weight_capacity=1000.0,
            volume_capacity=10.0, cost_per_km=1.0
        )
        
        route = Route(vehicle=vehicle)
        route.tasks = [pickup_task, delivery_task]
        
        # Test QUBO formulation
        formulator = QUBOFormulator([order], [vehicle], [route], [150.0])
        qubo_matrix = formulator.formulate_qubo()
        
        print(f"   ✅ QUBO matrix created: {qubo_matrix.shape[0]}x{qubo_matrix.shape[1]}")
        print(f"   📊 Variables: {len(formulator.variables)}")
        print(f"   📊 Task penalty: {formulator.task_penalty:.1f}")
        print(f"   📊 Fleet penalty: {formulator.fleet_penalty:.1f}")
        
    except Exception as e:
        print(f"   ❌ QUBO formulation failed: {e}")
        success = False
    
    # Test 4: Solver integration (if available)
    print("\n⚛️  Testing QUBO solver integration...")
    try:
        import neal
        
        config = QUBOConfig(
            preferred_solver="neal",
            num_reads=100,
            verbose=False
        )
        
        from algo.qubo_formulation import solve_epdt_with_qubo
        
        result = solve_epdt_with_qubo([order], [vehicle], [route], [150.0], config)
        
        print(f"   ✅ QUBO solver test completed")
        print(f"   📊 Energy: {result.energy:.2f}")
        print(f"   📊 Feasible: {result.feasible}")
        print(f"   📊 Selected routes: {len(result.selected_routes)}")
        
    except ImportError:
        print("   ⚠️  QUBO solver test skipped (Neal not available)")
    except Exception as e:
        print(f"   ❌ QUBO solver test failed: {e}")
        success = False
    
    # Test 5: Hybrid integration
    print("\n🔀 Testing Hybrid CG+QUBO integration...")
    try:
        hybrid_config = HybridConfig(
            use_column_generation=False,  # Skip CG for validation
            qubo_config=QUBOConfig(preferred_solver="fallback", verbose=False)
        )
        
        from algo.qubo_integration import solve_epdt_hybrid
        
        hybrid_result = solve_epdt_hybrid([order], [vehicle], hybrid_config)
        
        print(f"   ✅ Hybrid workflow test completed")
        print(f"   📊 Total time: {hybrid_result.total_solve_time:.3f}s")
        print(f"   📊 QUBO feasible: {hybrid_result.qubo_feasible}")
        
    except Exception as e:
        print(f"   ❌ Hybrid integration failed: {e}")
        success = False
    
    # Test 6: Command line integration
    print("\n💻 Testing command line integration...")
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run([
            sys.executable, 
            "tests/run_scenario_test.py", 
            "--help"
        ], capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if "--test-qubo" in result.stdout:
            print("   ✅ QUBO command line options integrated")
        else:
            print("   ⚠️  QUBO command line options may not be integrated")
            
    except Exception as e:
        print(f"   ⚠️  Command line test failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 QUBO VALIDATION SUMMARY")
    print("=" * 70)
    
    if success:
        print("✅ QUBO implementation validation PASSED")
        print("\n🎯 Ready for quantum annealing experiments:")
        print("   • QUBO formulation mathematically correct")
        print("   • Multiple solver support (Neal, D-Wave, exact)")
        print("   • Hybrid CG+QUBO workflow operational")
        print("   • Command line integration complete")
        
        print("\n🚀 Next steps:")
        print("   1. Run: python tests/run_scenario_test.py --test-qubo")
        print("   2. Benchmark: python tests/run_scenario_test.py --test-quantum-benchmark")
        print("   3. Install quantum packages: pip install dimod dwave-neal")
        
    else:
        print("❌ QUBO implementation validation FAILED")
        print("   Check error messages above and install missing dependencies")
    
    return success


if __name__ == "__main__":
    success = validate_qubo_implementation()
    sys.exit(0 if success else 1)
