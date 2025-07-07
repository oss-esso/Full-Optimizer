#!/usr/bin/env python3
"""
TSP Multi-Day Solver adapted for Furgoni scenario.
Based on tsp_multiple_days.py but using furgoni delivery data.
"""

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from functools import partial
from datetime import datetime, time, timedelta
import argparse
import furgoni_tsp_multi_day as F  # Use our furgoni data module

def timedelta_format(td):
    """Format timedelta as HH:MM string."""
    ts = int(td.total_seconds())
    tm = time(hour=(ts//3600), minute=(ts%3600//60))
    return tm.strftime("%H:%M")

def main():
    parser = argparse.ArgumentParser(description='Solve Furgoni delivery routing problem with multi-day scheduling')
    parser.add_argument('--days', type=int, dest='days', default=3,
                        help='Number of days to schedule. Default is 3 days')
    parser.add_argument('--start', type=int, dest='start', default=6,
                        help='The earliest any trip can start on any day, in hours. Default 6')
    parser.add_argument('--end', type=int, dest='end', default=18,
                        help='The latest any trip can end on any day, in hours. Default 18 (6pm)')
    parser.add_argument('--waittime', type=int, dest='service', default=30,
                        help='Default service time at locations, in minutes. Default is 30')
    parser.add_argument('-t', '--timelimit', type=int, dest='timelimit', default=30,
                        help='Maximum run time for solver, in seconds. Default is 30 seconds.')
    parser.add_argument('--debug', action='store_true', dest='debug', default=False,
                        help="Turn on solver logging.")
    parser.add_argument('--no_guided_local', action='store_true', dest='guided_local_off',
                        default=False,
                        help='Whether or not to use the guided local search metaheuristic')
    parser.add_argument('--skip_mornings', action='store_true', dest='skip_mornings',
                        default=False,
                        help='Whether or not to use dummy morning nodes. Default is false')
    parser.add_argument('--summary', action='store_true', dest='show_summary',
                        default=False,
                        help='Show detailed scenario summary before solving')

    args = parser.parse_args()
    
    # Show scenario summary if requested
    if args.show_summary:
        F.print_scenario_summary()
    
    # Convert hours to seconds
    day_start = args.start * 3600
    day_end = args.end * 3600

    if args.days <= 0:
        print("--days parameter must be 1 or more")
        assert args.days > 0

    num_days = args.days - 1

    node_service_time = args.service * 60  # Convert to seconds
    overnight_time = (day_start - day_end)  # Time from end of day to start of next day

    disjunction_penalty = 10000000

    Slack_Max = (day_end - day_start) - day_start  # Maximum slack time
    Capacity = day_end  # Maximum time that can be used in one day

    num_nodes = F.num_nodes()
    
    print(f"\n🚚 FURGONI MULTI-DAY TSP SOLVER")
    print(f"{'='*50}")
    print(f"📊 Problem Setup:")
    print(f"   - Locations to visit: {num_nodes}")
    print(f"   - Days available: {args.days}")
    print(f"   - Daily working hours: {args.start}:00 - {args.end}:00")
    print(f"   - Default service time: {args.service} minutes")
    print(f"   - Solver time limit: {args.timelimit} seconds")
    
    # Create dummy nodes for returning to the depot every night
    night_nodes = list(range(num_nodes, num_nodes + num_days))

    # Create dummy nodes linked to night nodes that fix the AM depart time
    morning_nodes = list(range(num_nodes + num_days, num_nodes + num_days + num_days))
    if args.skip_mornings:
        morning_nodes = []

    total_nodes = num_nodes + len(night_nodes) + len(morning_nodes)
    
    print(f"   - Night nodes: {len(night_nodes)} ({night_nodes})")
    print(f"   - Morning nodes: {len(morning_nodes)} ({morning_nodes})")
    print(f"   - Total nodes: {total_nodes}")
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(total_nodes, 1, [0], [1])

    print(f'📋 Created manager with {total_nodes} total nodes = {num_nodes} real + {len(night_nodes)} night + {len(morning_nodes)} morning')
    
    # Create Routing Model with caching for better performance
    model_parameters = pywrapcp.DefaultRoutingModelParameters()
    model_parameters.max_callback_cache_size = 2 * total_nodes * total_nodes
    routing = pywrapcp.RoutingModel(manager, model_parameters)

    # Set up transit callback (cost function)
    transit_callback_fn = partial(F.transit_callback,
                                  manager,
                                  day_end,
                                  night_nodes,
                                  morning_nodes)

    transit_callback_index = routing.RegisterTransitCallback(transit_callback_fn)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    print('✅ Set arc cost evaluator for all vehicles')

    # Set up time callback 
    time_callback_fn = partial(F.time_callback,
                               manager,
                               node_service_time,
                               overnight_time,
                               night_nodes,
                               morning_nodes)

    time_callback_index = routing.RegisterTransitCallback(time_callback_fn)

    # Create time dimension
    routing.AddDimension(
        time_callback_index,
        Slack_Max,  # Upper bound for slack (wait times at locations)
        Capacity,   # Upper bound for total time over vehicle's route
        False,      # Don't set cumulative variable to zero at start
        'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    print('✅ Created time dimension')

    # Remove slack for regular nodes and morning nodes, but keep for depot and night nodes
    for node in range(2, num_nodes):
        index = manager.NodeToIndex(node)
        time_dimension.SlackVar(index).SetValue(0)
    
    for node in morning_nodes:
        index = manager.NodeToIndex(node)
        time_dimension.SlackVar(index).SetValue(0)

    # Allow all locations except the first two (depot start/end) to be droppable
    print("🎯 Setting up droppable locations...")
    droppable_count = 0
    for node in range(2, num_nodes):
        routing.AddDisjunction([manager.NodeToIndex(node)], disjunction_penalty)
        droppable_count += 1

    # Allow all overnight and morning nodes to be dropped for free
    for node in night_nodes:
        routing.AddDisjunction([manager.NodeToIndex(node)], 0)
    
    for node in morning_nodes:
        routing.AddDisjunction([manager.NodeToIndex(node)], 0)
    
    print(f"   - {droppable_count} locations can be dropped (with penalty)")
    print(f"   - {len(night_nodes)} night nodes can be dropped for free")
    print(f"   - {len(morning_nodes)} morning nodes can be dropped for free")

    # Add time window constraints for each regular node
    print("⏰ Setting up time window constraints...")
    for node in range(2, num_nodes):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(day_start, day_end)

    # Time constraints for overnight and morning nodes
    for node in range(num_nodes, total_nodes):
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(day_start, day_end)

    # Add time window constraints for vehicle start/end nodes
    for veh in range(1):  # Single vehicle
        start_index = routing.Start(veh)
        time_dimension.CumulVar(start_index).SetMin(day_start)
        end_index = routing.End(veh)
        time_dimension.CumulVar(end_index).SetMax(day_end)

    print('✅ Time constraints configured')

    # Ensure days happen in order using counting dimension
    print("📅 Setting up day ordering constraints...")
    routing.AddConstantDimension(1,           # increment by 1
                                 total_nodes + 1,  # max count
                                 True,        # start count at zero
                                 "Counting")
    count_dimension = routing.GetDimensionOrDie('Counting')

    # Use count dimension to enforce ordering of overnight and morning nodes
    solver = routing.solver()
    
    # Ensure night nodes happen in order
    for i in range(len(night_nodes)):
        inode = night_nodes[i]
        iidx = manager.NodeToIndex(inode)
        iactive = routing.ActiveVar(iidx)

        for j in range(i + 1, len(night_nodes)):
            jnode = night_nodes[j]
            jidx = manager.NodeToIndex(jnode)
            jactive = routing.ActiveVar(jidx)
            solver.Add(iactive >= jactive)
            solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                       count_dimension.CumulVar(jidx) * iactive * jactive)

        # If night node is active AND it's not the last night,
        # must transition to corresponding morning node
        if i < len(morning_nodes):
            i_morning_idx = manager.NodeToIndex(morning_nodes[i])
            i_morning_active = routing.ActiveVar(i_morning_idx)
            solver.Add(iactive == i_morning_active)
            solver.Add(count_dimension.CumulVar(iidx) + 1 ==
                       count_dimension.CumulVar(i_morning_idx))

    # Ensure morning nodes happen in order
    for i in range(len(morning_nodes)):
        inode = morning_nodes[i]
        iidx = manager.NodeToIndex(inode)
        iactive = routing.ActiveVar(iidx)

        for j in range(i + 1, len(morning_nodes)):
            jnode = morning_nodes[j]
            jidx = manager.NodeToIndex(jnode)
            jactive = routing.ActiveVar(jidx)

            solver.Add(iactive >= jactive)
            solver.Add(count_dimension.CumulVar(iidx) * iactive * jactive <=
                       count_dimension.CumulVar(jidx) * iactive * jactive)

    print('✅ Day ordering constraints configured')

    # Configure search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    
    if not args.guided_local_off:
        search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    search_parameters.time_limit.seconds = args.timelimit
    search_parameters.log_search = args.debug

    print(f"🔧 Search Configuration:")
    print(f"   - Strategy: PARALLEL_CHEAPEST_INSERTION")
    print(f"   - Metaheuristic: {'GUIDED_LOCAL_SEARCH' if not args.guided_local_off else 'None'}")
    print(f"   - Time limit: {args.timelimit} seconds")
    print(f"   - Debug logging: {args.debug}")

    # Solve the problem
    print(f"\n🔍 Solving Furgoni multi-day delivery problem...")
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        print("❌ No solution found!")
        print("💡 Try:")
        print("   - Increasing time limit (--timelimit)")
        print("   - Adding more days (--days)")
        print("   - Extending working hours (--start/--end)")
        return
    
    print(f"✅ Solution found!")
    print(f"🎯 Objective value: {solution.ObjectiveValue():,}")

    # Analyze and print the results
    result = {
        'Dropped': [],
        'Scheduled': []
    }

    # Find dropped locations
    dropped_count = 0
    for index in range(routing.Size()):
        if routing.IsStart(index) or routing.IsEnd(index):
            continue
        node = manager.IndexToNode(index)
        if node in night_nodes or node in morning_nodes:
            continue
        if solution.Value(routing.NextVar(index)) == index:
            result['Dropped'].append(node)
            dropped_count += 1

    # Extract the scheduled route
    scheduled_count = 0
    cumultime = 0
    index = routing.Start(0)
    current_day = 1
    
    while not routing.IsEnd(index):
        cumultime = time_dimension.CumulVar(index)
        count = count_dimension.CumulVar(index)
        node = manager.IndexToNode(index)
        
        # Handle special nodes
        node_display = node
        location_info = F.get_location_info(node) if node < F.num_nodes() else {}
        
        if node in night_nodes:
            night_day = night_nodes.index(node) + 1
            node_display = f'END DAY {night_day} (staying overnight)'
            location_info = {'id': f'night_day_{night_day}', 'address': 'Overnight at current location'}
        elif node in morning_nodes:
            morning_day = morning_nodes.index(node) + 2
            node_display = f'START DAY {morning_day} (morning departure)'  
            location_info = {'id': f'morning_day_{morning_day}', 'address': 'Resume from overnight location'}

        mintime = timedelta(seconds=solution.Min(cumultime))
        maxtime = timedelta(seconds=solution.Max(cumultime))
        
        result['Scheduled'].append([node_display, solution.Value(count),
                                    timedelta_format(mintime),
                                    timedelta_format(maxtime),
                                    location_info.get('address', 'N/A')])
        
        if node < F.num_nodes() and node not in [0, 1]:
            scheduled_count += 1
        
        index = solution.Value(routing.NextVar(index))

    # Add final location (end depot)
    cumultime = time_dimension.CumulVar(index)
    count = count_dimension.CumulVar(index)
    mintime = timedelta(seconds=solution.Min(cumultime))
    maxtime = timedelta(seconds=solution.Max(cumultime))
    end_info = {'address': 'Return to depot'}
    
    result['Scheduled'].append([manager.IndexToNode(index),
                                solution.Value(count),
                                timedelta_format(mintime),
                                timedelta_format(maxtime),
                                end_info['address']])

    # Print comprehensive results
    print(f"\n📊 SOLUTION SUMMARY:")
    print(f"{'='*60}")
    print(f"🎯 Locations scheduled: {scheduled_count}/{F.num_nodes() - 2}")  # Exclude depot start/end
    print(f"📍 Locations dropped: {dropped_count}")
    print(f"⏱️  Total solution cost: {solution.ObjectiveValue():,}")

    if result['Dropped']:
        print(f"\n❌ Dropped Locations ({len(result['Dropped'])}):")
        for node in result['Dropped']:
            info = F.get_location_info(node)
            print(f"   - Node {node}: {info.get('id', 'N/A')} - {info.get('address', 'N/A')[:60]}...")

    print(f"\n✅ Scheduled Route:")
    print(f"{'='*80}")
    print(f"{'Node':<15} {'Order':<6} {'Arrival':<8} {'Latest':<8} {'Location':<50}")
    print(f"{'-'*80}")
    
    for line in result['Scheduled']:
        node_str = str(line[0])[:14]
        order_str = str(line[1])
        arrival_str = line[2]
        latest_str = line[3] 
        location_str = line[4][:49] if len(line) > 4 else 'N/A'
        
        print(f"{node_str:<15} {order_str:<6} {arrival_str:<8} {latest_str:<8} {location_str:<50}")

    # Calculate and display statistics
    total_scheduled = len([x for x in result['Scheduled'] if not isinstance(x[0], str) or not any(keyword in str(x[0]) for keyword in ['END DAY', 'START DAY'])])
    
    print(f"\n📈 DELIVERY STATISTICS:")
    print(f"   - Route stops: {total_scheduled}")
    print(f"   - Success rate: {scheduled_count/(F.num_nodes()-2)*100:.1f}%")
    print(f"   - Multi-day transitions: {len([x for x in result['Scheduled'] if isinstance(x[0], str) and ('END DAY' in str(x[0]) or 'START DAY' in str(x[0]))])}")
    
    working_hours = (args.end - args.start)
    total_available_time = working_hours * args.days
    print(f"   - Available time: {total_available_time} hours over {args.days} days")
    
    print(f"\n🎉 Furgoni multi-day delivery optimization complete!")

if __name__ == '__main__':
    main()
