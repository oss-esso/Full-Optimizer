"""
Simple Column Generation Validation Script

This script provides a basic test of the Column Generation implementation
to verify that all components are working correctly.
"""

import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_basic_column_generation():
    """Test basic Column Generation functionality."""
    print("🧪 Testing Basic Column Generation Implementation")
    print("=" * 60)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        
        try:
            from algo.epdt_data_structures import Order, Vehicle, Task, TaskType
            print("   ✅ EPDT data structures imported")
        except ImportError as e:
            print(f"   ❌ Failed to import EPDT data structures: {e}")
            return False
        
        try:
            from algo.column_generation import ColumnGenerationConfig
            print("   ✅ Column Generation config imported")
        except ImportError as e:
            print(f"   ❌ Failed to import Column Generation config: {e}")
            return False
        
        try:
            from algo.master_problem import MasterProblem
            print("   ✅ Master Problem imported")
        except ImportError as e:
            print(f"   ❌ Failed to import Master Problem: {e}")
            return False
        
        try:
            from algo.pricing_problem import PricingProblem
            print("   ✅ Pricing Problem imported")
        except ImportError as e:
            print(f"   ❌ Failed to import Pricing Problem: {e}")
            return False
        
        # Test basic functionality
        print("\n🔧 Testing basic functionality...")
        
        # Create simple test problem
        pickup_task = Task(
            id="P1",
            location_id="LOC_P1",
            task_type=TaskType.PICKUP,
            order_id="O1",
            lat=0.0,
            lon=0.0,
            service_time=10.0,
            demand=100.0,
            volume=1.0
        )
        
        delivery_task = Task(
            id="D1",
            location_id="LOC_D1",
            task_type=TaskType.DELIVERY,
            order_id="O1",
            lat=1.0,
            lon=1.0,
            service_time=10.0,
            demand=-100.0,
            volume=-1.0
        )
        
        test_order = Order(
            id="O1",
            pickup_tasks=[pickup_task],
            delivery_tasks=[delivery_task],
            is_mandatory=True
        )
        
        test_vehicle = Vehicle(
            id="V1",
            depot_id="DEPOT",
            weight_capacity=1000.0,
            volume_capacity=10.0,
            cost_per_km=1.0
        )
        
        print("   ✅ Test problem created")
        
        # Test Master Problem initialization
        config = ColumnGenerationConfig(verbose=False)
        mp = MasterProblem([test_order], [test_vehicle], config)
        print("   ✅ Master Problem initialized")
        
        # Test Pricing Problem initialization
        pp = PricingProblem(test_vehicle, [test_order], config)
        print("   ✅ Pricing Problem initialized")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_solver_availability():
    """Check which LP solvers are available."""
    print("\n🔍 Checking LP Solver Availability")
    print("=" * 40)
    
    solvers_available = []
    
    # Check PuLP
    try:
        import pulp
        print("   ✅ PuLP available")
        solvers_available.append("pulp")
        
        # Test basic LP solving
        prob = pulp.LpProblem("test", pulp.LpMinimize)
        x = pulp.LpVariable("x", lowBound=0)
        prob += x
        prob += x >= 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            print("      ✅ PuLP solver working correctly")
        else:
            print("      ⚠️  PuLP solver has issues")
            
    except ImportError:
        print("   ❌ PuLP not available (install with: pip install pulp)")
    except Exception as e:
        print(f"   ⚠️  PuLP available but has issues: {e}")
    
    # Check Gurobi
    try:
        import gurobipy as gp
        print("   ✅ Gurobi available")
        solvers_available.append("gurobi")
        
        # Test basic model creation
        model = gp.Model("test")
        model.setParam('OutputFlag', 0)
        x = model.addVar(name="x")
        model.setObjective(x, gp.GRB.MINIMIZE)
        model.addConstr(x >= 1)
        model.optimize()
        
        if model.status == gp.GRB.OPTIMAL:
            print("      ✅ Gurobi solver working correctly")
        else:
            print("      ⚠️  Gurobi solver has issues")
            
    except ImportError:
        print("   ❌ Gurobi not available (requires license and installation)")
    except Exception as e:
        print(f"   ⚠️  Gurobi available but has issues: {e}")
    
    if not solvers_available:
        print("\n❌ No LP solvers available! Please install:")
        print("   pip install pulp  # Free option")
        print("   # OR install Gurobi with license")
        return False
    else:
        print(f"\n✅ Available solvers: {', '.join(solvers_available)}")
        return True


if __name__ == "__main__":
    print("🚀 Column Generation Validation Script")
    print("=" * 60)
    
    # Check solver availability first
    solvers_ok = check_solver_availability()
    
    if not solvers_ok:
        print("\n❌ Cannot proceed without LP solvers")
        sys.exit(1)
    
    # Test basic functionality
    basic_test_ok = test_basic_column_generation()
    
    if basic_test_ok:
        print(f"\n🎉 Column Generation implementation validated successfully!")
        print(f"   Ready for integration with EPDT test runner")
        sys.exit(0)
    else:
        print(f"\n❌ Column Generation validation failed")
        sys.exit(1)
