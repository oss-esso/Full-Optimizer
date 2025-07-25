"""
Production-Level Test Script for TODO2.md Section 14 Requirements

This script implements comprehensive performance analysis and optimization
targeting sub-30 second runtime with detailed bottleneck identification.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
sys.path.insert(0, heuristic_root)
sys.path.insert(0, os.path.join(heuristic_root, 'src'))
sys.path.insert(0, os.path.join(heuristic_root, 'algo'))

def run_production_test():
    """Execute production-level performance test"""
    
    print(f"🏭 PRODUCTION-LEVEL EPDT PERFORMANCE TEST")
    print(f"{'='*60}")
    print(f"Target: Sub-30 second runtime")
    print(f"Requirements: TODO2.md Section 14")
    print(f"{'='*60}")
    
    try:
        # Import required modules
        from moda_scenarios import create_furgoni_scenario
        from data_adapter import convert_instance_to_epdt_input_fixed as convert_instance
        from first_level import l1_heuristic
        
        # Import production optimizations
        from algo.production_optimizer import ProductionOptimizer
        from algo.performance_profiler import ProductionProfiler, SolutionAnalyzer
        
        print(f"✅ All modules imported successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"Falling back to basic test...")
        return run_basic_performance_test()
    
    # Step 1: Load scenario
    print(f"\n📋 Step 1: Loading scenario...")
    start_time = time.time()
    
    scenario = create_furgoni_scenario()
    print(f"✅ Scenario loaded in {time.time() - start_time:.2f}s")
    
    # Step 2: Convert to EPDT format  
    print(f"\n🔄 Step 2: Converting to EPDT format...")
    convert_start = time.time()
    
    orders, vehicles = convert_instance(scenario)
    print(f"✅ Conversion completed in {time.time() - convert_start:.2f}s")
    print(f"   📦 Orders: {len(orders)}")
    print(f"   🚛 Vehicles: {len(vehicles)}")
    
    # Step 3: Configure for production
    print(f"\n⚙️  Step 3: Production configuration...")
    
    # Ultra-aggressive parameters for sub-30s target
    params = {
        'tabu_tenure': 5,
        'M1': 3,  # Only 3 main iterations
        'M2': 10,  # Only 10 neighborhood iterations
        'max_neighbors_per_type': 20,  # Drastically reduced
        'early_termination_threshold': 2,  # Stop after 2 non-improving
        'max_runtime_seconds': 25,  # Hard limit at 25s
        'fast_scoring': True,
        'intelligent_pruning': True
    }
    
    print(f"🎯 Production parameters: {params}")
    
    # Step 4: Run with production profiling
    print(f"\n🚀 Step 4: Running production-optimized algorithm...")
    
    profiler = ProductionProfiler()
    profiler.start_profiling()
    
    # Initialize production optimizer
    optimizer = ProductionOptimizer()
    
    # Run optimized algorithm
    algorithm_start = time.time()
    
    # Option 1: Use production optimizer
    try:
        solution = optimizer.optimize_l1_performance(orders, vehicles, params)
        optimization_method = "Production Optimizer"
    except Exception as e:
        print(f"⚠️  Production optimizer failed: {e}")
        print(f"Falling back to standard l1_heuristic...")
        solution = l1_heuristic(orders, vehicles, params)
        optimization_method = "Standard L1 Heuristic"
    
    algorithm_time = time.time() - algorithm_start
    
    # Stop profiling
    performance_metrics = profiler.stop_profiling()
    
    # Step 5: Comprehensive analysis
    print(f"\n📊 Step 5: Production Analysis Results")
    print(f"{'='*50}")
    
    print(f"🕐 Algorithm Runtime: {algorithm_time:.2f}s")
    print(f"🔧 Method Used: {optimization_method}")
    
    # Check production target
    if algorithm_time <= 30:
        print(f"✅ SUCCESS: Meets production target (<30s)")
        status = "PRODUCTION READY"
    else:
        print(f"❌ FAILED: Exceeds production target by {algorithm_time - 30:.1f}s")
        status = "OPTIMIZATION REQUIRED"
    
    print(f"💾 Peak Memory: {performance_metrics.memory_peak_mb:.1f}MB")
    
    # Solution quality analysis
    quality_metrics = SolutionAnalyzer.analyze_solution_quality(solution, orders, vehicles)
    
    print(f"\n📈 Solution Quality:")
    print(f"   📦 Assignment Rate: {quality_metrics.assignment_rate:.1%}")
    print(f"   🚛 Vehicle Utilization: {quality_metrics.vehicle_utilization_weight:.1%}")
    print(f"   🚚 Vehicles Used: {quality_metrics.vehicles_used}/{quality_metrics.total_vehicles}")
    
    # Bottleneck analysis
    print(f"\n🎯 Performance Bottlenecks:")
    for i, (func_name, time_spent) in enumerate(performance_metrics.bottleneck_functions[:3], 1):
        percentage = (time_spent / algorithm_time) * 100
        print(f"   {i}. {func_name}: {time_spent:.2f}s ({percentage:.1f}%)")
    
    # Production recommendations
    print(f"\n🚨 Production Recommendations:")
    if algorithm_time > 30:
        print(f"   1. CRITICAL: Reduce runtime by {algorithm_time - 30:.1f}s")
        
        # Identify top bottleneck
        if performance_metrics.bottleneck_functions:
            top_bottleneck = performance_metrics.bottleneck_functions[0]
            print(f"   2. OPTIMIZE: Focus on {top_bottleneck[0]}")
        
        print(f"   3. Consider further reducing M1/M2 parameters")
        print(f"   4. Implement more aggressive neighborhood pruning")
        print(f"   5. Use pre-computed distance matrices")
    else:
        print(f"   1. ✅ Runtime target achieved")
        print(f"   2. Monitor solution quality in production")
        print(f"   3. Consider enabling additional optimizations")
    
    # Save production report
    report = {
        'timestamp': time.time(),
        'status': status,
        'runtime_seconds': algorithm_time,
        'target_seconds': 30,
        'meets_target': algorithm_time <= 30,
        'optimization_method': optimization_method,
        'performance_metrics': {
            'memory_peak_mb': performance_metrics.memory_peak_mb,
            'bottleneck_functions': performance_metrics.bottleneck_functions[:5]
        },
        'solution_quality': {
            'assignment_rate': quality_metrics.assignment_rate,
            'vehicle_utilization': quality_metrics.vehicle_utilization_weight,
            'vehicles_used': quality_metrics.vehicles_used,
            'total_vehicles': quality_metrics.total_vehicles
        },
        'parameters': params
    }
    
    # Save to results directory
    os.makedirs('results', exist_ok=True)
    timestamp = int(time.time())
    report_file = f"results/production_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Production report saved to: {report_file}")
    
    return status == "PRODUCTION READY"

def run_basic_performance_test():
    """Fallback test when production modules are not available"""
    
    print(f"\n⚠️  Running basic performance test (production modules unavailable)")
    
    try:
        # Basic imports only
        import sys
        import os
        
        # Add path for basic modules
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
        
        from run_scenario_test import run_furgoni_test
        
        print(f"🔍 Running basic scenario test...")
        start_time = time.time()
        
        # Run basic test
        run_furgoni_test()
        
        runtime = time.time() - start_time
        
        print(f"\n📊 Basic Test Results:")
        print(f"🕐 Total Runtime: {runtime:.2f}s")
        
        if runtime <= 30:
            print(f"✅ Meets basic performance target")
            return True
        else:
            print(f"❌ Exceeds performance target by {runtime - 30:.1f}s")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"🏭 Starting TODO2.md Section 14 Production Test")
    print(f"Target: Achieve sub-30 second runtime with comprehensive analysis")
    
    success = run_production_test()
    
    if success:
        print(f"\n🎉 PRODUCTION TEST PASSED!")
        print(f"✅ Algorithm meets production performance requirements")
    else:
        print(f"\n🚨 PRODUCTION TEST FAILED!")
        print(f"❌ Further optimization required")
        
    print(f"\n🎯 Next Steps:")
    print(f"1. Review production report for optimization opportunities")
    print(f"2. Implement recommended performance improvements")
    print(f"3. Re-test until production target is achieved")
