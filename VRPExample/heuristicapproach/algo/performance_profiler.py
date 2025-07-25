"""
Production-Level Performance Profiler for EPDT Algorithm
Implements comprehensive profiling and analysis as specified in TODO2.md Section 14
"""

import cProfile
import pstats
import io
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import tracemalloc

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for production monitoring"""
    total_runtime: float
    initialization_time: float
    optimization_time: float
    neighborhood_times: Dict[str, float]
    scoring_time: float
    feasibility_check_time: float
    memory_peak_mb: float
    function_call_counts: Dict[str, int]
    bottleneck_functions: List[Tuple[str, float]]  # (function_name, cumulative_time)
    
@dataclass 
class SolutionQualityMetrics:
    """Solution quality analysis for heuristic vs optimal comparison"""
    total_distance: float
    vehicle_utilization_weight: float
    vehicle_utilization_volume: float
    time_window_violations: int
    total_delay_hours: float
    assignment_rate: float
    vehicles_used: int
    total_vehicles: int
    cost_score: float

@dataclass
class FeasibilityAnalysis:
    """Detailed analysis of constraint violations and rejections"""
    capacity_violations: List[Dict[str, Any]]
    time_window_violations: List[Dict[str, Any]]
    hos_violations: List[Dict[str, Any]]
    infeasible_insertions: List[Dict[str, Any]]
    constraint_strictness: Dict[str, float]  # constraint_name -> rejection_rate

class ProductionProfiler:
    """Production-level profiler implementing TODO2.md Section 14 requirements"""
    
    def __init__(self):
        self.profiler = None
        self.start_time = None
        self.phase_times = {}
        self.memory_tracker = None
        self.performance_log = []
        
    def start_profiling(self):
        """Start comprehensive profiling session"""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.start_time = time.time()
        
        # Start memory tracking
        tracemalloc.start()
        
    def mark_phase(self, phase_name: str):
        """Mark a phase for timing analysis"""
        current_time = time.time()
        if self.start_time:
            self.phase_times[phase_name] = current_time - self.start_time
            
    def stop_profiling(self) -> PerformanceMetrics:
        """Stop profiling and generate comprehensive metrics"""
        if self.profiler:
            self.profiler.disable()
            
        # Get memory info
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Analyze profiling results
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        # Extract bottleneck functions
        stats = ps.get_stats_profile()
        bottlenecks = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[2]}:{func[0]}:{func[1]}"
            bottlenecks.append((func_name, ct))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return PerformanceMetrics(
            total_runtime=total_time,
            initialization_time=self.phase_times.get('initialization', 0),
            optimization_time=self.phase_times.get('optimization', 0),
            neighborhood_times=self._extract_neighborhood_times(),
            scoring_time=self.phase_times.get('scoring', 0),
            feasibility_check_time=self.phase_times.get('feasibility', 0),
            memory_peak_mb=peak / 1024 / 1024,
            function_call_counts=self._extract_call_counts(stats),
            bottleneck_functions=bottlenecks[:10]  # Top 10 bottlenecks
        )
    
    def _extract_neighborhood_times(self) -> Dict[str, float]:
        """Extract timing for each neighborhood function"""
        neighborhood_times = {}
        for phase, time_val in self.phase_times.items():
            if 'neighborhood' in phase.lower():
                neighborhood_times[phase] = time_val
        return neighborhood_times
    
    def _extract_call_counts(self, stats) -> Dict[str, int]:
        """Extract function call counts"""
        call_counts = {}
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[2]}"
            call_counts[func_name] = cc
        return call_counts

class SolutionAnalyzer:
    """Comprehensive solution quality analysis"""
    
    @staticmethod
    def analyze_solution_quality(solution, orders: List, vehicles: List) -> SolutionQualityMetrics:
        """Analyze solution quality metrics for production monitoring"""
        
        # Calculate total distance
        total_distance = 0
        vehicles_used = 0
        
        for vehicle_id, route in solution.routes.items():
            if route.tasks:
                vehicles_used += 1
                # Calculate route distance (simplified)
                total_distance += SolutionAnalyzer._calculate_route_distance(route)
        
        # Calculate utilization
        total_weight_capacity = sum(v.weight_capacity for v in vehicles)
        total_volume_capacity = sum(v.volume_capacity for v in vehicles)
        
        used_weight = 0
        used_volume = 0
        
        for vehicle_id, route in solution.routes.items():
            for task in route.tasks:
                if hasattr(task, 'order') and task.order:
                    used_weight += task.order.weight
                    used_volume += task.order.volume
        
        # Time window analysis
        tw_violations = 0
        total_delay = 0
        
        for vehicle_id, route in solution.routes.items():
            for task in route.tasks:
                if hasattr(task, 'time_window_violation') and task.time_window_violation:
                    tw_violations += 1
                    total_delay += task.time_window_violation
        
        # Assignment rate
        assigned_orders = len([o for o in orders if o.assigned])
        assignment_rate = assigned_orders / len(orders) if orders else 0
        
        return SolutionQualityMetrics(
            total_distance=total_distance,
            vehicle_utilization_weight=used_weight / total_weight_capacity if total_weight_capacity > 0 else 0,
            vehicle_utilization_volume=used_volume / total_volume_capacity if total_volume_capacity > 0 else 0,
            time_window_violations=tw_violations,
            total_delay_hours=total_delay / 3600,  # Convert to hours
            assignment_rate=assignment_rate,
            vehicles_used=vehicles_used,
            total_vehicles=len(vehicles),
            cost_score=getattr(solution, 'z1_score', 0)
        )
    
    @staticmethod
    def _calculate_route_distance(route) -> float:
        """Calculate total distance for a route"""
        # Simplified distance calculation
        # In production, use actual distance matrix
        return len(route.tasks) * 10  # Placeholder
    
    @staticmethod
    def compare_solutions(heuristic_metrics: SolutionQualityMetrics, 
                         mock_metrics: SolutionQualityMetrics) -> Dict[str, Any]:
        """Compare heuristic vs mock solution quality"""
        
        comparison = {
            'distance_efficiency': (mock_metrics.total_distance - heuristic_metrics.total_distance) / mock_metrics.total_distance if mock_metrics.total_distance > 0 else 0,
            'utilization_difference_weight': heuristic_metrics.vehicle_utilization_weight - mock_metrics.vehicle_utilization_weight,
            'utilization_difference_volume': heuristic_metrics.vehicle_utilization_volume - mock_metrics.vehicle_utilization_volume,
            'assignment_rate_difference': heuristic_metrics.assignment_rate - mock_metrics.assignment_rate,
            'vehicles_efficiency': (mock_metrics.vehicles_used - heuristic_metrics.vehicles_used) / mock_metrics.vehicles_used if mock_metrics.vehicles_used > 0 else 0,
            'time_window_performance': mock_metrics.time_window_violations - heuristic_metrics.time_window_violations
        }
        
        return comparison

class ConstraintAnalyzer:
    """Analyze constraint strictness and feasibility issues"""
    
    def __init__(self):
        self.constraint_violations = defaultdict(list)
        self.rejection_counts = defaultdict(int)
        self.total_checks = defaultdict(int)
    
    def record_violation(self, constraint_type: str, violation_details: Dict[str, Any]):
        """Record a constraint violation for analysis"""
        self.constraint_violations[constraint_type].append(violation_details)
        self.rejection_counts[constraint_type] += 1
    
    def record_check(self, constraint_type: str):
        """Record a constraint check (for calculating rejection rates)"""
        self.total_checks[constraint_type] += 1
    
    def generate_analysis(self) -> FeasibilityAnalysis:
        """Generate comprehensive feasibility analysis"""
        
        constraint_strictness = {}
        for constraint_type in self.total_checks:
            if self.total_checks[constraint_type] > 0:
                constraint_strictness[constraint_type] = self.rejection_counts[constraint_type] / self.total_checks[constraint_type]
            else:
                constraint_strictness[constraint_type] = 0
        
        return FeasibilityAnalysis(
            capacity_violations=self.constraint_violations.get('capacity', []),
            time_window_violations=self.constraint_violations.get('time_window', []),
            hos_violations=self.constraint_violations.get('hos', []),
            infeasible_insertions=self.constraint_violations.get('insertion', []),
            constraint_strictness=constraint_strictness
        )

def save_production_report(metrics: PerformanceMetrics, 
                          quality: SolutionQualityMetrics,
                          feasibility: FeasibilityAnalysis,
                          comparison: Dict[str, Any],
                          filepath: str):
    """Save comprehensive production-level performance report"""
    
    report = {
        'timestamp': time.time(),
        'performance_metrics': asdict(metrics),
        'solution_quality': asdict(quality),
        'feasibility_analysis': asdict(feasibility),
        'solution_comparison': comparison,
        'production_recommendations': _generate_recommendations(metrics, quality, feasibility)
    }
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)

def _generate_recommendations(metrics: PerformanceMetrics, 
                            quality: SolutionQualityMetrics,
                            feasibility: FeasibilityAnalysis) -> List[str]:
    """Generate production optimization recommendations"""
    
    recommendations = []
    
    # Performance recommendations
    if metrics.total_runtime > 30:
        recommendations.append(f"CRITICAL: Runtime {metrics.total_runtime:.1f}s exceeds production target of 30s")
        
        # Identify top bottlenecks
        if metrics.bottleneck_functions:
            top_bottleneck = metrics.bottleneck_functions[0]
            recommendations.append(f"OPTIMIZE: Top bottleneck is {top_bottleneck[0]} ({top_bottleneck[1]:.2f}s)")
    
    # Memory recommendations
    if metrics.memory_peak_mb > 500:
        recommendations.append(f"MEMORY: Peak usage {metrics.memory_peak_mb:.1f}MB may cause issues in production")
    
    # Quality recommendations
    if quality.assignment_rate < 0.95:
        recommendations.append(f"QUALITY: Assignment rate {quality.assignment_rate:.1%} below target 95%")
    
    if quality.vehicle_utilization_weight < 0.7:
        recommendations.append(f"EFFICIENCY: Vehicle weight utilization {quality.vehicle_utilization_weight:.1%} below target 70%")
    
    # Constraint recommendations
    for constraint, strictness in feasibility.constraint_strictness.items():
        if strictness > 0.8:
            recommendations.append(f"CONSTRAINT: {constraint} rejection rate {strictness:.1%} may be too strict")
    
    return recommendations
