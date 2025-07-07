#!/usr/bin/env python3
"""
Simple VRP Diagnostic Tool

This tool analyzes VRP scenarios to identify potential bottlenecks
without running the full optimization.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from vrp_data_models import VRPInstance
from vrp_scenarios import create_moda_first_scenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VRPDiagnosticTool:
    """Tool for diagnosing VRP scenario feasibility issues."""
    
    def __init__(self):
        """Initialize the diagnostic tool."""
        pass
    
    def analyze_scenario(self, instance: VRPInstance) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a VRP scenario.
        
        Returns detailed diagnostic information about potential bottlenecks.
        """
        logger.info("=" * 80)
        logger.info("VRP SCENARIO DIAGNOSTIC ANALYSIS")
        logger.info("=" * 80)
        
        analysis = {}
        
        # Basic scenario statistics
        basic_stats = self._analyze_basic_statistics(instance)
        analysis['basic_statistics'] = basic_stats
        logger.info("✅ Basic statistics analysis completed")
        
        # Vehicle capacity analysis
        capacity_analysis = self._analyze_capacity_constraints(instance)
        analysis['capacity_analysis'] = capacity_analysis
        logger.info("✅ Capacity analysis completed")
        
        # Time window analysis
        time_analysis = self._analyze_time_constraints(instance)
        analysis['time_analysis'] = time_analysis
        logger.info("✅ Time constraint analysis completed")
        
        # Service time analysis
        service_analysis = self._analyze_service_times(instance)
        analysis['service_analysis'] = service_analysis
        logger.info("✅ Service time analysis completed")
        
        # Distance/travel time analysis
        distance_analysis = self._analyze_distances(instance)
        analysis['distance_analysis'] = distance_analysis
        logger.info("✅ Distance analysis completed")
        
        # Pickup-dropoff feasibility
        if hasattr(instance, 'ride_requests'):
            pdp_analysis = self._analyze_pickup_dropoff_feasibility(instance)
            analysis['pickup_dropoff_analysis'] = pdp_analysis
            logger.info("✅ Pickup-dropoff feasibility analysis completed")
        
        # Generate overall feasibility assessment
        feasibility = self._assess_overall_feasibility(analysis)
        analysis['feasibility_assessment'] = feasibility
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        analysis['recommendations'] = recommendations
        
        logger.info("=" * 80)
        
        return analysis
    
    def _analyze_basic_statistics(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze basic scenario statistics."""
        num_locations = len(instance.locations)
        num_vehicles = len(instance.vehicles)
        num_requests = len(instance.ride_requests) if hasattr(instance, 'ride_requests') else 0
        
        # Count different location types
        depot_count = 0
        service_area_count = 0
        pickup_count = 0
        dropoff_count = 0
        
        for loc in instance.locations.values():
            if 'depot' in loc.id.lower():
                depot_count += 1
            elif 'sa_' in loc.id.lower() or 'service' in loc.id.lower():
                service_area_count += 1
            elif 'pickup' in loc.id.lower():
                pickup_count += 1
            elif 'dropoff' in loc.id.lower():
                dropoff_count += 1
        
        return {
            'total_locations': num_locations,
            'num_vehicles': num_vehicles,
            'num_requests': num_requests,
            'location_breakdown': {
                'depots': depot_count,
                'service_areas': service_area_count,
                'pickups': pickup_count,
                'dropoffs': dropoff_count
            },
            'stops_per_vehicle_avg': (pickup_count + dropoff_count) / num_vehicles if num_vehicles > 0 else 0
        }
    
    def _analyze_capacity_constraints(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze vehicle capacity vs demand."""
        vehicles = list(instance.vehicles.values())
        
        # Vehicle capacity statistics
        total_capacity = sum(v.capacity for v in vehicles)
        min_capacity = min(v.capacity for v in vehicles)
        max_capacity = max(v.capacity for v in vehicles)
        avg_capacity = total_capacity / len(vehicles)
        
        # Demand analysis
        total_demand = 0
        if hasattr(instance, 'ride_requests'):
            total_demand = sum(getattr(req, 'passengers', 1) for req in instance.ride_requests)
        
        return {
            'vehicle_statistics': {
                'total_capacity': total_capacity,
                'min_capacity': min_capacity,
                'max_capacity': max_capacity,
                'avg_capacity': avg_capacity,
                'num_vehicles': len(vehicles)
            },
            'demand_statistics': {
                'total_demand': total_demand,
                'capacity_utilization': total_demand / total_capacity if total_capacity > 0 else float('inf'),
                'feasible': total_demand <= total_capacity
            }
        }
    
    def _analyze_time_constraints(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze time window constraints."""
        locations = list(instance.locations.values())
        
        # Find locations with time windows
        time_constrained = [loc for loc in locations 
                           if hasattr(loc, 'time_window_start') and loc.time_window_start is not None]
        
        if not time_constrained:
            return {'has_time_windows': False}
        
        # Analyze time window spans
        starts = [loc.time_window_start for loc in time_constrained]
        ends = [loc.time_window_end for loc in time_constrained]
        spans = [end - start for start, end in zip(starts, ends)]
        
        # Analyze vehicle time constraints
        vehicles = list(instance.vehicles.values())
        max_work_times = [getattr(v, 'max_total_work_time', 600) for v in vehicles]
        
        return {
            'has_time_windows': True,
            'window_statistics': {
                'num_constrained_locations': len(time_constrained),
                'earliest_start': min(starts),
                'latest_end': max(ends),
                'total_time_span': max(ends) - min(starts),
                'avg_window_span': sum(spans) / len(spans),
                'min_window_span': min(spans),
                'max_window_span': max(spans)
            },
            'vehicle_time_limits': {
                'min_work_time': min(max_work_times),
                'max_work_time': max(max_work_times),
                'avg_work_time': sum(max_work_times) / len(max_work_times)
            }
        }
    
    def _analyze_service_times(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze service time requirements."""
        locations = list(instance.locations.values())
        
        service_times = [getattr(loc, 'service_time', 0) for loc in locations]
        non_zero_service_times = [st for st in service_times if st > 0]
        
        if not non_zero_service_times:
            return {'has_service_times': False}
        
        total_service_time = sum(service_times)
        num_vehicles = len(instance.vehicles)
        avg_service_per_vehicle = total_service_time / num_vehicles
        
        # Analyze by location type
        pickup_service_times = []
        dropoff_service_times = []
        service_area_times = []
        
        for loc in locations:
            service_time = getattr(loc, 'service_time', 0)
            if service_time > 0:
                if 'pickup' in loc.id.lower():
                    pickup_service_times.append(service_time)
                elif 'dropoff' in loc.id.lower():
                    dropoff_service_times.append(service_time)
                elif 'sa_' in loc.id.lower() or 'service' in loc.id.lower():
                    service_area_times.append(service_time)
        
        return {
            'has_service_times': True,
            'total_service_time': total_service_time,
            'avg_service_per_vehicle': avg_service_per_vehicle,
            'service_time_breakdown': {
                'pickup_avg': sum(pickup_service_times) / len(pickup_service_times) if pickup_service_times else 0,
                'dropoff_avg': sum(dropoff_service_times) / len(dropoff_service_times) if dropoff_service_times else 0,
                'service_area_avg': sum(service_area_times) / len(service_area_times) if service_area_times else 0,
                'pickup_count': len(pickup_service_times),                'dropoff_count': len(dropoff_service_times),
                'service_area_count': len(service_area_times)
            }
        }
    
    def _analyze_distances(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze distance matrix and travel requirements."""
        if not hasattr(instance, 'distance_matrix') or instance.distance_matrix is None:
            return {'has_distance_matrix': False}
        
        matrix = instance.distance_matrix
        n = len(matrix)
        
        # Calculate distance statistics
        all_distances = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    all_distances.append(matrix[i][j])
        
        if not all_distances:
            return {'has_distance_matrix': False}
        
        return {
            'has_distance_matrix': True,
            'matrix_size': n,
            'distance_statistics': {
                'min_distance': min(all_distances),
                'max_distance': max(all_distances),
                'avg_distance': sum(all_distances) / len(all_distances),
                'total_distance_if_all_connected': sum(all_distances)
            }
        }
    
    def _analyze_pickup_dropoff_feasibility(self, instance: VRPInstance) -> Dict[str, Any]:
        """Analyze pickup-dropoff timing feasibility."""
        if not hasattr(instance, 'ride_requests'):
            return {'has_pickup_dropoff': False}
        
        feasible_pairs = 0
        tight_pairs = 0
        impossible_pairs = 0
        timing_violations = []
        
        for req in instance.ride_requests:
            pickup_loc = instance.locations[req.pickup_location]
            dropoff_loc = instance.locations[req.dropoff_location]
            
            if (hasattr(pickup_loc, 'time_window_end') and hasattr(dropoff_loc, 'time_window_start') and
                pickup_loc.time_window_end is not None and dropoff_loc.time_window_start is not None):
                
                # Calculate minimum time needed
                pickup_service = getattr(pickup_loc, 'service_time', 0)
                earliest_pickup_finish = pickup_loc.time_window_end + pickup_service
                latest_dropoff_start = dropoff_loc.time_window_start
                
                available_travel_time = latest_dropoff_start - earliest_pickup_finish
                  # Estimate minimum travel time needed (simplified)
                if hasattr(instance, 'distance_matrix') and instance.distance_matrix is not None:
                    pickup_idx = list(instance.locations.keys()).index(pickup_loc.id)
                    dropoff_idx = list(instance.locations.keys()).index(dropoff_loc.id)
                    estimated_travel_time = instance.distance_matrix[pickup_idx][dropoff_idx]
                else:
                    # Fallback: use Euclidean distance
                    dx = pickup_loc.x - dropoff_loc.x
                    dy = pickup_loc.y - dropoff_loc.y
                    estimated_travel_time = (dx*dx + dy*dy)**0.5
                
                if available_travel_time < 0:
                    impossible_pairs += 1
                    timing_violations.append({
                        'request_id': req.id,
                        'available_time': available_travel_time,
                        'estimated_travel_time': estimated_travel_time,
                        'violation_type': 'impossible'
                    })
                elif available_travel_time < estimated_travel_time:
                    tight_pairs += 1
                    timing_violations.append({
                        'request_id': req.id,
                        'available_time': available_travel_time,
                        'estimated_travel_time': estimated_travel_time,
                        'violation_type': 'tight'
                    })
                else:
                    feasible_pairs += 1
        
        total_pairs = len(instance.ride_requests)
        
        return {
            'has_pickup_dropoff': True,
            'total_pairs': total_pairs,
            'feasible_pairs': feasible_pairs,
            'tight_pairs': tight_pairs,
            'impossible_pairs': impossible_pairs,
            'timing_violations': timing_violations,
            'feasibility_rate': feasible_pairs / total_pairs if total_pairs > 0 else 0
        }
    
    def _assess_overall_feasibility(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall scenario feasibility."""
        issues = []
        warnings = []
        
        # Check capacity
        if 'capacity_analysis' in analysis:
            cap = analysis['capacity_analysis']['demand_statistics']
            if not cap['feasible']:
                issues.append('insufficient_capacity')
            elif cap['capacity_utilization'] > 0.9:
                warnings.append('high_capacity_utilization')
        
        # Check time constraints
        if 'pickup_dropoff_analysis' in analysis:
            pdp = analysis['pickup_dropoff_analysis']
            if pdp['impossible_pairs'] > 0:
                issues.append('impossible_time_windows')
            elif pdp['tight_pairs'] > pdp['feasible_pairs']:
                warnings.append('many_tight_time_windows')
        
        # Check service time vs available time
        if 'service_analysis' in analysis and 'time_analysis' in analysis:
            service = analysis['service_analysis']
            time_limits = analysis['time_analysis']['vehicle_time_limits']
            
            if service['avg_service_per_vehicle'] > time_limits['avg_work_time'] * 0.8:
                warnings.append('high_service_time_ratio')
        
        likelihood = 'high'
        if issues:
            likelihood = 'low'
        elif warnings:
            likelihood = 'medium'
        
        return {
            'feasibility_likelihood': likelihood,
            'identified_issues': issues,
            'warnings': warnings
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        feasibility = analysis['feasibility_assessment']
        
        if 'insufficient_capacity' in feasibility['identified_issues']:
            recommendations.append("Increase vehicle capacity or add more vehicles")
        
        if 'impossible_time_windows' in feasibility['identified_issues']:
            recommendations.append("Adjust time windows to ensure pickup-dropoff sequences are feasible")
        
        if 'high_capacity_utilization' in feasibility['warnings']:
            recommendations.append("Consider adding buffer capacity to handle demand fluctuations")
        
        if 'many_tight_time_windows' in feasibility['warnings']:
            recommendations.append("Expand time windows or reduce service times to improve scheduling flexibility")
        
        if 'high_service_time_ratio' in feasibility['warnings']:
            recommendations.append("Reduce service times or increase vehicle work time limits")
        
        # General recommendations based on scenario complexity
        basic = analysis['basic_statistics']
        if basic['stops_per_vehicle_avg'] > 10:
            recommendations.append("Consider splitting scenario into smaller sub-problems")
        
        if feasibility['feasibility_likelihood'] == 'high':
            recommendations.extend([
                "Scenario appears feasible - try increasing OR-Tools solution time",
                "Experiment with different OR-Tools search strategies"
            ])
        
        return recommendations

def main():
    """Run diagnostic analysis on MODA_first scenario."""
    print("VRP Diagnostic Tool")
    print("=" * 50)
    
    # Create scenario
    scenario = create_moda_first_scenario()
    
    # Run diagnostic analysis
    diagnostic_tool = VRPDiagnosticTool()
    analysis = diagnostic_tool.analyze_scenario(scenario)
    
    # Print summary report
    print("\nDIAGNOSTIC REPORT")
    print("=" * 50)
    
    basic = analysis['basic_statistics']
    print(f"📊 Scenario Statistics:")
    print(f"   - Total locations: {basic['total_locations']}")
    print(f"   - Vehicles: {basic['num_vehicles']}")
    print(f"   - Ride requests: {basic['num_requests']}")
    print(f"   - Avg stops per vehicle: {basic['stops_per_vehicle_avg']:.1f}")
    
    if 'capacity_analysis' in analysis:
        cap = analysis['capacity_analysis']
        print(f"\\n🚛 Capacity Analysis:")
        print(f"   - Total capacity: {cap['vehicle_statistics']['total_capacity']}")
        print(f"   - Total demand: {cap['demand_statistics']['total_demand']}")
        print(f"   - Utilization: {cap['demand_statistics']['capacity_utilization']:.1%}")
        print(f"   - Feasible: {'✅' if cap['demand_statistics']['feasible'] else '❌'}")
    
    if 'time_analysis' in analysis:
        time_info = analysis['time_analysis']
        if time_info['has_time_windows']:
            print(f"\\n⏰ Time Analysis:")
            print(f"   - Time constrained locations: {time_info['window_statistics']['num_constrained_locations']}")
            print(f"   - Total time span: {time_info['window_statistics']['total_time_span']:.0f} minutes")
            print(f"   - Avg window span: {time_info['window_statistics']['avg_window_span']:.0f} minutes")
            print(f"   - Vehicle work time limit: {time_info['vehicle_time_limits']['avg_work_time']:.0f} minutes")
    
    if 'service_analysis' in analysis:
        service = analysis['service_analysis']
        if service['has_service_times']:
            print(f"\\n🔧 Service Time Analysis:")
            print(f"   - Total service time: {service['total_service_time']:.0f} minutes")
            print(f"   - Avg per vehicle: {service['avg_service_per_vehicle']:.0f} minutes")
            breakdown = service['service_time_breakdown']
            print(f"   - Pickup service: {breakdown['pickup_avg']:.0f} min x {breakdown['pickup_count']} locations")
            print(f"   - Dropoff service: {breakdown['dropoff_avg']:.0f} min x {breakdown['dropoff_count']} locations")
            print(f"   - Service areas: {breakdown['service_area_avg']:.0f} min x {breakdown['service_area_count']} locations")
    
    if 'pickup_dropoff_analysis' in analysis:
        pdp = analysis['pickup_dropoff_analysis']
        if pdp['has_pickup_dropoff']:
            print(f"\\n🔄 Pickup-Dropoff Analysis:")
            print(f"   - Total pairs: {pdp['total_pairs']}")
            print(f"   - Feasible: {pdp['feasible_pairs']} ✅")
            print(f"   - Tight timing: {pdp['tight_pairs']} ⚠️")
            print(f"   - Impossible: {pdp['impossible_pairs']} ❌")
            print(f"   - Feasibility rate: {pdp['feasibility_rate']:.1%}")
    
    feasibility = analysis['feasibility_assessment']
    print(f"\\n🎯 Overall Assessment:")
    print(f"   - Feasibility likelihood: {feasibility['feasibility_likelihood'].upper()}")
    print(f"   - Issues: {feasibility['identified_issues']}")
    print(f"   - Warnings: {feasibility['warnings']}")
    
    print(f"\\n💡 Recommendations:")
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"   {i}. {rec}")

if __name__ == "__main__":
    main()
