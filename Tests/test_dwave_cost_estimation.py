#!/usr/bin/env python3
"""
Test script for D-Wave cost estimation functionality.

This script tests the D-Wave QPU adapter's cost estimation capabilities
using real scenario data from scenarios.py. It demonstrates:
- Free cost estimation for all complexity levels
- Problem size scaling analysis
- Budget planning and recommendations
- Quantum vs classical method comparison
"""

import sys
import os
import time
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run comprehensive D-Wave cost estimation tests."""
    print("=" * 80)
    print("D-WAVE COST ESTIMATION TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Import required modules
        from src.scenarios import load_food_data
        from my_functions.dwave_qpu_adapter import (
            DWaveQPUAdapter, DWaveConfig, get_free_dwave_analysis,
            estimate_dwave_cost_for_problem
        )
        
        print("✓ All imports successful")
        
        # Test 1: Basic D-Wave adapter initialization
        test_basic_initialization()
        
        # Test 2: Free complexity analysis
        test_free_complexity_analysis()
        
        # Test 3: Scenario-based cost estimation
        test_scenario_cost_estimation()
        
        # Test 4: Budget planning scenarios
        test_budget_planning()
        
        # Test 5: Problem scaling analysis
        test_problem_scaling()
        
        # Test 6: Solver validation (NEW)
        test_solver_validation()
        
        # Test 7: Generate comprehensive report
        generate_cost_estimation_report()
        
        print("\n" + "=" * 80)
        print("✓ ALL D-WAVE COST ESTIMATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please ensure D-Wave libraries are available or test will use fallback mode")
        return False
    except Exception as e:
        print(f"✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_basic_initialization():
    """Test basic D-Wave adapter initialization and connection."""
    print("\n" + "-" * 60)
    print("TEST 1: BASIC D-WAVE ADAPTER INITIALIZATION")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Test with different configurations
        configs = [
            ("Default Config", DWaveConfig()),
            ("Budget-Conscious", DWaveConfig(max_budget_usd=25.0, num_reads=500)),
            ("High-Performance", DWaveConfig(max_budget_usd=200.0, num_reads=2000)),
            ("Estimation-Only", DWaveConfig(estimate_cost_only=True))
        ]
        
        for config_name, config in configs:
            print(f"\n--- Testing {config_name} ---")
            
            adapter = DWaveQPUAdapter(config=config, logger=logger)
            
            # Test connection
            connection_test = adapter.test_connection()
            print(f"Connection Status: {connection_test.get('status', 'unknown')}")
            print(f"Simulator Available: {connection_test.get('simulator_available', False)}")
            print(f"QPU Available: {connection_test.get('qpu_available', False)}")
            
            # Test basic functionality
            metrics = adapter._get_performance_metrics()
            print(f"Initial Metrics: {metrics}")
            
        print("✓ Basic initialization tests passed")
        
    except Exception as e:
        print(f"✗ Basic initialization test failed: {e}")
        raise

def test_free_complexity_analysis():
    """Test free complexity analysis for all predefined levels."""
    print("\n" + "-" * 60)
    print("TEST 2: FREE COMPLEXITY ANALYSIS")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import get_free_dwave_analysis
        
        # Test different budget scenarios
        budget_scenarios = [10.0, 50.0, 100.0, 500.0]
        
        for budget in budget_scenarios:
            print(f"\n--- Budget Analysis: ${budget} USD ---")
            
            analysis = get_free_dwave_analysis(
                max_budget_usd=budget,
                use_real_qpu=False,  # Use simulator for testing
                logger=logger
            )
            
            print(f"Analysis Duration: {analysis['analysis_duration']:.2f} seconds")
            print(f"Levels Analyzed: {analysis['summary']['total_levels_analyzed']}")
            print(f"Feasible Levels: {len(analysis['summary']['feasible_levels'])}")
            print(f"Affordable Levels: {len(analysis['summary']['affordable_levels'])}")
            
            print(f"\nCost Range: ${analysis['summary']['cost_range']['min']:.4f} - "
                  f"${analysis['summary']['cost_range']['max']:.4f}")
            
            print("Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")
            
            # Show detailed breakdown for this budget
            print(f"\nAffordable Complexity Levels (Budget: ${budget}):")
            for level in analysis['summary']['affordable_levels']:
                estimation = analysis['estimations'][level]
                print(f"  {level.upper()}: {estimation.num_variables} vars, "
                      f"${estimation.estimated_cost_usd:.4f}")
        
        print("\n✓ Free complexity analysis tests passed")
        
    except Exception as e:
        print(f"✗ Free complexity analysis test failed: {e}")
        raise

def test_scenario_cost_estimation():
    """Test cost estimation using real scenario data."""
    print("\n" + "-" * 60)
    print("TEST 3: SCENARIO-BASED COST ESTIMATION")
    print("-" * 60)
    
    try:
        from src.scenarios import load_food_data
        from my_functions.dwave_qpu_adapter import estimate_dwave_cost_for_problem
        
        # Test all scenario complexity levels
        complexity_levels = ['simple', 'intermediate', 'full']
        
        scenario_results = {}
        
        for complexity in complexity_levels:
            print(f"\n--- Testing {complexity.upper()} Scenario ---")
            
            # Load scenario data
            farms, foods, food_groups, config = load_food_data(complexity)
            
            num_farms = len(farms)
            num_foods = len(foods)
            problem_size = num_farms * num_foods
            
            print(f"Scenario Details:")
            print(f"  Farms: {num_farms} ({farms})")
            print(f"  Foods: {num_farms} ({list(foods.keys())})")
            print(f"  Problem Size: {problem_size} variables")
            
            # Test different num_reads scenarios
            read_scenarios = [100, 500, 1000, 2000]
            
            scenario_costs = {}
            
            for num_reads in read_scenarios:
                recommendation = estimate_dwave_cost_for_problem(
                    num_farms=num_farms,
                    num_foods=num_foods,
                    num_reads=num_reads,
                    budget_usd=100.0
                )
                
                scenario_costs[num_reads] = recommendation
                
                print(f"\n  Reads: {num_reads}")
                print(f"    Estimated Cost: ${recommendation['estimated_cost']:.4f}")
                print(f"    Affordable: {recommendation['is_affordable']}")
                print(f"    Closest Level: {recommendation['closest_complexity_level']}")
            
            scenario_results[complexity] = {
                'farms': farms,
                'foods': list(foods.keys()),
                'problem_size': problem_size,
                'costs': scenario_costs,
                'config': config
            }
        
        # Cross-scenario comparison
        print(f"\n--- CROSS-SCENARIO COMPARISON ---")
        print(f"{'Scenario':<12} {'Size':<6} {'Cost@1000':<10} {'Affordable':<10}")
        print("-" * 50)
        
        for complexity, data in scenario_results.items():
            cost_1000 = data['costs'][1000]['estimated_cost']
            affordable = data['costs'][1000]['is_affordable']
            print(f"{complexity:<12} {data['problem_size']:<6} "
                  f"${cost_1000:<9.4f} {affordable}")
        
        print("\n✓ Scenario-based cost estimation tests passed")
        
    except Exception as e:
        print(f"✗ Scenario cost estimation test failed: {e}")
        raise

def test_budget_planning():
    """Test budget planning and recommendation functionality."""
    print("\n" + "-" * 60)
    print("TEST 4: BUDGET PLANNING SCENARIOS")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Test different budget constraints
        budget_scenarios = [
            ("Student Budget", 5.0),
            ("Research Budget", 25.0),
            ("Small Business", 100.0),
            ("Enterprise", 500.0)
        ]
        
        for scenario_name, budget in budget_scenarios:
            print(f"\n--- {scenario_name}: ${budget} USD ---")
            
            config = DWaveConfig(max_budget_usd=budget)
            adapter = DWaveQPUAdapter(config=config, logger=logger)
            
            # Test recommendations for different problem sizes
            problem_sizes = [
                (3, 6, "Small Problem"),
                (5, 10, "Medium Problem"), 
                (10, 15, "Large Problem"),
                (20, 25, "Very Large Problem")
            ]
            
            recommendations = []
            
            for farms, foods, description in problem_sizes:
                rec = adapter.get_complexity_recommendation(farms, foods, budget)
                recommendations.append((description, rec))
                
                print(f"\n  {description} ({farms}×{foods} = {rec['problem_size']} vars):")
                print(f"    Estimated Cost: ${rec['estimated_cost']:.4f}")
                print(f"    Affordable: {rec['is_affordable']}")
                
                if not rec['is_affordable'] and rec['alternatives']:
                    print(f"    Alternatives: {', '.join(rec['alternatives'][:2])}")
                
                if rec['affordable_levels']:
                    best_affordable = rec['affordable_levels'][0]
                    print(f"    Best Affordable: {best_affordable['level']} "
                          f"({best_affordable['size']} vars, ${best_affordable['cost']:.4f})")
            
            # Budget utilization summary
            affordable_count = sum(1 for _, rec in recommendations if rec['is_affordable'])
            print(f"\n  Budget Summary:")
            print(f"    Affordable Problems: {affordable_count}/{len(recommendations)}")
            print(f"    Budget Utilization: Varies by problem size")
        
        print("\n✓ Budget planning tests passed")
        
    except Exception as e:
        print(f"✗ Budget planning test failed: {e}")
        raise

def test_problem_scaling():
    """Test how costs scale with problem size."""
    print("\n" + "-" * 60)
    print("TEST 5: PROBLEM SCALING ANALYSIS")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        config = DWaveConfig(num_reads=1000)
        adapter = DWaveQPUAdapter(config=config, logger=logger)
        
        # Test different problem sizes
        problem_sizes = [
            (2, 3, 6),      # Tiny
            (3, 5, 15),     # Small
            (5, 8, 40),     # Medium
            (8, 12, 96),    # Large
            (10, 15, 150),  # Very Large
            (15, 20, 300),  # Huge
        ]
        
        scaling_results = []
        
        print(f"{'Size':<8} {'Variables':<10} {'Cost ($)':<10} {'Qubits':<8} {'Feasible':<9} {'Warnings'}")
        print("-" * 80)
        
        for farms, foods, expected_vars in problem_sizes:
            actual_vars = farms * foods
            
            estimation = adapter.estimate_qpu_cost(
                problem_size=actual_vars,
                complexity_level=f"{farms}x{foods}"
            )
            
            scaling_results.append({
                'farms': farms,
                'foods': foods,
                'variables': actual_vars,
                'cost': estimation.estimated_cost_usd,
                'qubits': estimation.estimated_qubits,
                'feasible': estimation.is_feasible,
                'warnings': estimation.warnings
            })
            
            warnings_str = f"{len(estimation.warnings)} warnings" if estimation.warnings else "None"
            print(f"{farms}×{foods:<4} {actual_vars:<10} ${estimation.estimated_cost_usd:<9.4f} "
                  f"{estimation.estimated_qubits:<8} {estimation.is_feasible:<9} {warnings_str}")
            
            # Print first warning if any
            if estimation.warnings:
                print(f"    → {estimation.warnings[0]}")
        
        # Analyze scaling trends
        print(f"\n--- SCALING ANALYSIS ---")
        
        # Calculate cost per variable
        cost_per_var = [r['cost'] / r['variables'] for r in scaling_results if r['variables'] > 0]
        if cost_per_var:
            print(f"Cost per variable: ${min(cost_per_var):.6f} - ${max(cost_per_var):.6f}")
            print(f"Average cost per variable: ${sum(cost_per_var) / len(cost_per_var):.6f}")
        
        # Find feasibility limits
        feasible_results = [r for r in scaling_results if r['feasible']]
        infeasible_results = [r for r in scaling_results if not r['feasible']]
        
        if feasible_results:
            max_feasible = max(feasible_results, key=lambda x: x['variables'])
            print(f"Maximum feasible size: {max_feasible['variables']} variables "
                  f"({max_feasible['farms']}×{max_feasible['foods']})")
        
        if infeasible_results:
            print(f"Infeasible problems:")
            for r in infeasible_results:
                print(f"  {r['farms']}×{r['foods']} ({r['variables']} vars, {r['qubits']} qubits)")
                if r['warnings']:
                    print(f"    Reason: {r['warnings'][0]}")
        
        print("\n✓ Problem scaling analysis tests passed")
        
    except Exception as e:
        print(f"✗ Problem scaling analysis test failed: {e}")
        raise

def test_solver_validation():
    """Test and compare ExactSolver vs SimulatedAnnealingSampler."""
    print("\n" + "-" * 60)
    print("TEST 6: SOLVER VALIDATION - EXACT vs SIMULATED ANNEALING")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import DWaveQPUAdapter, DWaveConfig
        
        # Import dimod solvers
        try:
            from dimod.reference.samplers import ExactSolver
            from dimod import SimulatedAnnealingSampler, BinaryQuadraticModel
            exact_solver_available = True
        except ImportError:
            print("⚠️  ExactSolver not available - skipping solver validation")
            return
        
        config = DWaveConfig(num_reads=1000)
        adapter = DWaveQPUAdapter(config=config, logger=logger)
        
        # Test different problem sizes (small enough for ExactSolver)
        test_problems = [
            (2, 3, "Tiny Problem"),     # 6 variables
            (3, 5, "Small Problem"),    # 15 variables  
            (3, 6, "Simple Scenario"),  # 18 variables (our actual scenario)
        ]
        
        validation_results = []
        
        print(f"{'Problem':<15} {'Size':<6} {'Exact Energy':<12} {'SA Energy':<12} {'Gap':<8} {'Success'}")
        print("-" * 75)
        
        for farms, foods, description in test_problems:
            problem_size = farms * foods
            
            # Skip if too large for ExactSolver (typically > 20 variables)
            if problem_size > 20:
                print(f"{description:<15} {problem_size:<6} {'TOO LARGE':<12} {'N/A':<12} {'N/A':<8} {'N/A'}")
                continue
            
            try:
                # Create a test QUBO problem based on food optimization structure
                Q_matrix = create_test_food_qubo(farms, foods)
                
                # Create BQM
                bqm = BinaryQuadraticModel('BINARY')
                for i in range(problem_size):
                    if Q_matrix[i, i] != 0:
                        bqm.add_variable(i, Q_matrix[i, i])
                
                for i in range(problem_size):
                    for j in range(i + 1, problem_size):
                        if Q_matrix[i, j] != 0:
                            bqm.add_interaction(i, j, Q_matrix[i, j])
                
                # Solve with ExactSolver
                exact_solver = ExactSolver()
                exact_result = exact_solver.sample(bqm)
                optimal_energy = exact_result.first.energy
                optimal_sample = dict(exact_result.first.sample)
                
                # Solve with SimulatedAnnealingSampler multiple times
                sa_solver = SimulatedAnnealingSampler()
                sa_energies = []
                success_count = 0
                num_trials = 10
                
                for trial in range(num_trials):
                    sa_result = sa_solver.sample(bqm, num_reads=100, seed=trial)
                    sa_energy = sa_result.first.energy
                    sa_energies.append(sa_energy)
                    
                    # Consider success if within 1% of optimal
                    if abs(sa_energy - optimal_energy) <= abs(optimal_energy * 0.01):
                        success_count += 1
                
                # Calculate statistics
                best_sa_energy = min(sa_energies)
                avg_sa_energy = np.mean(sa_energies)
                energy_gap = abs(best_sa_energy - optimal_energy)
                success_rate = (success_count / num_trials) * 100
                
                validation_results.append({
                    'problem': description,
                    'size': problem_size,
                    'optimal_energy': optimal_energy,
                    'best_sa_energy': best_sa_energy,
                    'avg_sa_energy': avg_sa_energy,
                    'energy_gap': energy_gap,
                    'success_rate': success_rate,
                    'optimal_sample': optimal_sample
                })
                
                print(f"{description:<15} {problem_size:<6} {optimal_energy:<12.4f} {best_sa_energy:<12.4f} "
                      f"{energy_gap:<8.4f} {success_rate:<6.0f}%")
                
            except Exception as e:
                print(f"{description:<15} {problem_size:<6} {'ERROR':<12} {'ERROR':<12} {'ERROR':<8} {'ERROR'}")
                logger.warning(f"Error testing {description}: {e}")
        
        # Analysis and recommendations
        print(f"\n--- SOLVER VALIDATION ANALYSIS ---")
        
        if validation_results:
            avg_gap = np.mean([r['energy_gap'] for r in validation_results])
            avg_success = np.mean([r['success_rate'] for r in validation_results])
            
            print(f"Average energy gap: {avg_gap:.4f}")
            print(f"Average success rate: {avg_success:.1f}%")
            
            # Quality assessment
            if avg_success >= 90:
                print("✓ Excellent: SimulatedAnnealing finds optimal solutions consistently")
            elif avg_success >= 70:
                print("✓ Good: SimulatedAnnealing finds good solutions most of the time")
            elif avg_success >= 50:
                print("⚠️  Fair: SimulatedAnnealing sometimes struggles to find optimal solutions")
            else:
                print("❌ Poor: SimulatedAnnealing frequently fails to find good solutions")
            
            print(f"\nRecommendations:")
            if avg_gap < 0.01:
                print("- Current SA settings are excellent for cost estimation")
            elif avg_gap < 0.1:
                print("- Consider increasing num_reads for better accuracy")
            else:
                print("- May need to tune SA parameters or use different solver approach")
            
            # Store results for report generation
            global solver_validation_data
            solver_validation_data = validation_results
        
        print("\n✓ Solver validation tests completed")
        
    except Exception as e:
        print(f"✗ Solver validation test failed: {e}")
        import traceback
        traceback.print_exc()

def create_test_food_qubo(num_farms: int, num_foods: int) -> np.ndarray:
    """
    Create a test QUBO matrix based on food optimization problem structure.
    
    Args:
        num_farms: Number of farms
        num_foods: Number of food types
        
    Returns:
        QUBO matrix representing the optimization problem
    """
    problem_size = num_farms * num_foods
    Q = np.zeros((problem_size, problem_size))
    
    # Add objective terms (favor certain food-farm combinations)
    for i in range(problem_size):
        farm_idx = i // num_foods
        food_idx = i % num_foods
        
        # Add linear terms (food preferences vary by farm)
        preference = 1.0 + 0.5 * np.sin(farm_idx + food_idx)  # Varies between 0.5 and 1.5
        Q[i, i] = -preference  # Negative because we want to maximize preference
    
    # Add constraint penalties (each farm should select at least one food)
    penalty_strength = 5.0
    
    for farm_idx in range(num_farms):
        # Constraint: sum of foods for this farm >= 1
        # Penalty form: -penalty * (sum - 1)^2 = -penalty * (sum^2 - 2*sum + 1)
        # Which expands to: penalty * (2*sum - sum^2 - 1)
        
        farm_vars = [farm_idx * num_foods + food_idx for food_idx in range(num_foods)]
        
        # Add quadratic penalty terms
        for i in farm_vars:
            for j in farm_vars:
                if i == j:
                    Q[i, j] += penalty_strength  # Quadratic terms
                else:
                    Q[i, j] += penalty_strength / 2  # Cross terms (divided by 2 due to symmetry)
        
        # Add linear penalty terms
        for i in farm_vars:
            Q[i, i] -= 2 * penalty_strength  # Linear terms
    
    return Q

# Global variable to store validation results for report
solver_validation_data = []

def generate_cost_estimation_report():
    """Generate a comprehensive cost estimation report."""
    print("\n" + "-" * 60)
    print("TEST 7: GENERATING COMPREHENSIVE REPORT")
    print("-" * 60)
    
    try:
        from my_functions.dwave_qpu_adapter import get_free_dwave_analysis
        from src.scenarios import load_food_data
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Run comprehensive analysis
        print("Running comprehensive D-Wave analysis...")
        
        analysis = get_free_dwave_analysis(
            max_budget_usd=100.0,
            use_real_qpu=False,
            logger=logger
        )
        
        # Load all scenarios for comparison
        scenario_data = {}
        for complexity in ['simple', 'intermediate', 'full']:
            try:
                farms, foods, food_groups, config = load_food_data(complexity)
                scenario_data[complexity] = {
                    'farms': farms,
                    'foods': list(foods.keys()),
                    'problem_size': len(farms) * len(foods),
                    'config': config
                }
            except Exception as e:
                print(f"Warning: Could not load {complexity} scenario: {e}")
        
        # Generate JSON report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_complexity_levels': len(analysis['estimations']),
                'analysis_duration': analysis['analysis_duration'],
                'budget_tested': analysis['budget_usd']
            },
            'dwave_analysis': analysis,
            'scenario_data': scenario_data,
            'solver_validation': solver_validation_data if 'solver_validation_data' in globals() else [],
            'recommendations': {
                'for_beginners': [
                    "Start with 'tiny' or 'small' complexity levels",
                    "Use simulated annealing for initial testing",
                    "Budget $10-25 for initial quantum experiments"
                ],
                'for_researchers': [
                    "Medium complexity problems are good for research",
                    "Budget $25-100 for comprehensive studies", 
                    "Consider QAOA² for larger problems"
                ],
                'for_enterprises': [
                    "Large problems require significant QPU budget",
                    "Consider hybrid classical-quantum approaches",
                    "Budget $100+ for production workloads"
                ]
            }
        }
        
        # Save JSON report
        json_path = os.path.join(results_dir, 'dwave_cost_estimation_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_report = generate_html_report(report_data)
        html_path = os.path.join(results_dir, 'dwave_cost_estimation_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✓ Reports generated:")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        
        # Print summary
        print(f"\n--- REPORT SUMMARY ---")
        print(f"Analysis Duration: {analysis['analysis_duration']:.2f} seconds")
        print(f"Complexity Levels Analyzed: {len(analysis['estimations'])}")
        print(f"Feasible Levels: {len(analysis['summary']['feasible_levels'])}")
        print(f"Affordable Levels: {len(analysis['summary']['affordable_levels'])}")
        print(f"Cost Range: ${analysis['summary']['cost_range']['min']:.4f} - "
              f"${analysis['summary']['cost_range']['max']:.4f}")
        
        # Add solver validation summary
        if solver_validation_data:
            avg_success = np.mean([r['success_rate'] for r in solver_validation_data])
            print(f"Solver Validation: {len(solver_validation_data)} problems tested, {avg_success:.1f}% average success rate")
        
        print("\n✓ Comprehensive report generation completed")
        
    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        raise

def generate_html_report(report_data: Dict[str, Any]) -> str:
    """Generate HTML report for cost estimation results."""
    
    timestamp = report_data['timestamp']
    analysis = report_data['dwave_analysis']
    scenarios = report_data['scenario_data']
    solver_validation = report_data.get('solver_validation', [])
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>D-Wave Cost Estimation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h3 {{ margin: 0 0 10px 0; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .feasible {{ color: #27ae60; font-weight: bold; }}
        .not-feasible {{ color: #e74c3c; font-weight: bold; }}
        .affordable {{ color: #27ae60; }}
        .expensive {{ color: #e74c3c; }}
        .excellent {{ color: #27ae60; font-weight: bold; }}
        .good {{ color: #f39c12; font-weight: bold; }}
        .poor {{ color: #e74c3c; font-weight: bold; }}
        .recommendations {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .timestamp {{ text-align: center; color: #7f8c8d; font-style: italic; }}
        .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>D-Wave Cost Estimation Report</h1>
        <div class="timestamp">Generated on {timestamp}</div>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Analysis Duration</h3>
                <div class="value">{analysis['analysis_duration']:.2f}s</div>
            </div>
            <div class="summary-card">
                <h3>Budget Tested</h3>
                <div class="value">${analysis['budget_usd']}</div>
            </div>
            <div class="summary-card">
                <h3>Complexity Levels</h3>
                <div class="value">{len(analysis['estimations'])}</div>
            </div>
            <div class="summary-card">
                <h3>Affordable Levels</h3>
                <div class="value">{len(analysis['summary']['affordable_levels'])}</div>
            </div>
        </div>
"""
    
    # Add solver validation section if available
    if solver_validation:
        avg_success = np.mean([r['success_rate'] for r in solver_validation])
        html += f"""
        <h2>Solver Validation Results</h2>
        <p>Comparison between ExactSolver (optimal) and SimulatedAnnealingSampler (heuristic) on small problems:</p>
        <table>
            <tr>
                <th>Problem</th>
                <th>Size</th>
                <th>Optimal Energy</th>
                <th>SA Best Energy</th>
                <th>Energy Gap</th>
                <th>Success Rate</th>
                <th>Quality</th>
            </tr>
"""
        
        for result in solver_validation:
            success_rate = result['success_rate']
            if success_rate >= 90:
                quality_class = "excellent"
                quality_text = "Excellent"
            elif success_rate >= 70:
                quality_class = "good"
                quality_text = "Good"
            else:
                quality_class = "poor"
                quality_text = "Needs Improvement"
            
            html += f"""
            <tr>
                <td>{result['problem']}</td>
                <td>{result['size']}</td>
                <td>{result['optimal_energy']:.4f}</td>
                <td>{result['best_sa_energy']:.4f}</td>
                <td>{result['energy_gap']:.4f}</td>
                <td>{success_rate:.1f}%</td>
                <td class="{quality_class}">{quality_text}</td>
            </tr>
"""
        
        html += f"""
        </table>
        <div class="recommendations">
            <p><strong>Validation Summary:</strong> Average success rate is {avg_success:.1f}%. 
            This indicates how often SimulatedAnnealing finds optimal or near-optimal solutions compared to the exact solver.</p>
        </div>
"""
    
    # Continue with existing HTML structure...
    html += """
        
        <h2>Complexity Level Analysis</h2>
        <table>
            <tr>
                <th>Complexity Level</th>
                <th>Variables</th>
                <th>Estimated Qubits</th>
                <th>Estimated Cost (USD)</th>
                <th>Feasible</th>
                <th>Affordable</th>
                <th>Warnings</th>
            </tr>
"""
    
    for level_name, estimation in analysis['estimations'].items():
        feasible_class = "feasible" if estimation.is_feasible else "not-feasible"
        affordable_class = "affordable" if level_name in analysis['summary']['affordable_levels'] else "expensive"
        warnings_text = f"{len(estimation.warnings)} warnings" if estimation.warnings else "None"
        
        html += f"""
            <tr>
                <td><strong>{level_name.upper()}</strong></td>
                <td>{estimation.num_variables}</td>
                <td>{estimation.estimated_qubits}</td>
                <td class="{affordable_class}">${estimation.estimated_cost_usd:.4f}</td>
                <td class="{feasible_class}">{'Yes' if estimation.is_feasible else 'No'}</td>
                <td class="{affordable_class}">{'Yes' if level_name in analysis['summary']['affordable_levels'] else 'No'}</td>
                <td>{warnings_text}</td>
            </tr>
"""
        
        # Add warnings if any
        if estimation.warnings:
            for warning in estimation.warnings:
                html += f"""
            <tr>
                <td colspan="7" class="warning">Warning for {level_name}: {warning}</td>
            </tr>
"""
    
    html += """
        </table>
        
        <h2>Key Insights</h2>
        <div class="recommendations">
            <ul>
                <li><strong>Cost Range:</strong> ${analysis['summary']['cost_range']['min']:.4f} - ${analysis['summary']['cost_range']['max']:.4f}</li>
                <li><strong>Most Affordable:</strong> {analysis['summary']['affordable_levels'][0] if analysis['summary']['affordable_levels'] else 'None within budget'}</li>
                <li><strong>Feasibility:</strong> {len(analysis['summary']['feasible_levels'])}/{len(analysis['estimations'])} levels are feasible</li>
                <li><strong>Simulator Available:</strong> {'Yes' if analysis['sampler_info']['simulator_available'] else 'No'}</li>
                <li><strong>Real QPU Configured:</strong> {'Yes' if analysis['sampler_info']['qpu_configured'] else 'No'}</li>"""
    
    if solver_validation:
        html += f"""
                <li><strong>Solver Validation:</strong> {len(solver_validation)} test problems with {avg_success:.1f}% average success rate</li>"""
    
    html += """
            </ul>
            
            <h3>Feasibility Analysis</h3>
            <p>Problems are marked as <strong>feasible</strong> if they can fit on D-Wave QPU hardware (under 5,000 qubits).</p>
            <p>Problems are marked as <strong>affordable</strong> if their estimated cost is within your specified budget.</p>
            
            {f'<div class="warning">Note: All problems show the same cost because the current pricing model uses a fixed overhead. In practice, costs would vary more significantly with problem size and complexity.</div>' if len(set(est.estimated_cost_usd for est in analysis['estimations'].values())) == 1 else ''}
        </div>
        
        <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            D-Wave Cost Estimation Report | Food Production Optimization System
        </footer>
    </div>
</body>
</html>
"""
    
    return html

if __name__ == "__main__":
    success = main()
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\n🎉 All tests passed! Check the Results folder for detailed reports.")
        print("\nNext steps:")
        print("1. Review the HTML report for cost breakdowns")
        print("2. Use the JSON data for further analysis")
        print("3. Plan your quantum experiments within budget")
        print("4. Consider starting with smaller complexity levels")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
    
    input("\nPress Enter to exit...")
