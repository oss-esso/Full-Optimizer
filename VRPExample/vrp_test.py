import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

FirstSolutionStrategy = routing_enums_pb2.FirstSolutionStrategy
LocalSearchMetaheuristic = routing_enums_pb2.LocalSearchMetaheuristic
RoutingSearchStatus = routing_enums_pb2.RoutingSearchStatus


def create_data_model():
    data = {}
    num_depots = 1
    patient_len = 16
    vehicle_capacity = 3
    data["num_Locations"] = 2 * patient_len + num_depots

    # fmt:off
    # data['demands'] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3]
    # for the Demands Tried adding Vehicle_Capacity(-3) instead of -1, Vehicle Didnt Visit Any Nodes. But adding -1 yielded result
    data['demands']=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

    data['time_windows']=[(0, 86400), (23400, 25200), (23400, 25200), (23400, 25200), (23400, 25200), (23400, 25200), (28800, 28800), (28800, 28800), (32400, 32400), (32400, 32400), (36000, 36000), (36000, 36000), (36000, 36000), (39600, 39600), (39600, 39600), (43200, 43200), (43200, 43200), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400), (0, 86400)]
    data['num_vehicles']= 40
    data['vehicle_capacity'] = [vehicle_capacity for _ in range(data['num_vehicles'])]
    data['depot']=0
    data['pickup_dropoff']=[(1, 17), (2, 18), (3, 19), (4, 20), (5, 21), (6, 22), (7, 23), (8, 24), (9, 25), (10, 26), (11, 27), (12, 28), (13, 29), (14, 30), (15, 31), (16, 32)]
    data['duplicate_nodes']=patient_len+1 # INdex of Start of Duplicate Node
    data['distance_matrix'] = [
    [    0 ,18432 , 8219 , 7741 ,21036 ,11099 ,28697 ,21912 ,31279 ,39777 ,35803 , 8667 ,71688 ,12592 ,11929 ,21985 ,18432 ],
    [17866 ,    0 ,10701 ,16151 ,21228 ,29756 ,25445 ,22105 ,31472 ,36525 ,48423 ,10499 ,68436 ,14823 ,14225 ,12439 ,    0 ],
    [ 8084 ,10986 ,    0 , 6369 ,13032 ,19974 ,21474 ,13909 ,23623 ,32555 ,38641 , 1221 ,64465 , 4799 , 4140 ,14762 ,10986 ],
    [ 7268 ,16304 , 6090 ,    0 ,18907 ,19158 ,26568 ,19783 ,31148 ,37649 ,32296 , 6538 ,69559 ,10463 , 9800 ,19856 ,16304 ],
    [19622 ,21970 ,12496 ,17908 ,    0 ,31512 ,13404 , 4291 ,13905 ,26053 ,37574 ,11758 ,56084 , 8945 , 9289 ,18362 ,21970 ],
    [11208 ,30040 ,19827 ,19349 ,32644 ,    0 ,40305 ,33520 ,42887 ,51385 ,47721 ,20275 ,83296 ,24200 ,23536 ,33593 ,30040 ],
    [28383 ,26104 ,21364 ,26669 ,15291 ,40273 ,    0 ,13247 ,15930 ,14162 ,46324 ,20465 ,46072 ,21785 ,22129 ,15111 ,26104 ],
    [20790 ,23138 ,13664 ,19075 , 4208 ,32680 ,11417 ,    0 ,10501 ,24831 ,37917 ,12926 ,54862 ,10113 ,10457 ,20970 ,23138 ],
    [31108 ,33154 ,23981 ,29142 ,14263 ,42997 ,17630 ,11016 ,    0 ,24328 ,31712 ,23243 ,50310 ,19040 ,19699 ,28100 ,33154 ],
    [39053 ,36774 ,32034 ,37339 ,26278 ,50943 ,12623 ,24234 ,25334 ,    0 ,69610 ,31135 ,32389 ,36012 ,35353 ,25781 ,36774 ],
    [37147 ,49357 ,39144 ,31154 ,39309 ,45936 ,47054 ,38162 ,32604 ,57812 ,    0 ,39592 ,87181 ,43516 ,42853 ,52910 ,49357 ],
    [ 8344 ,10812 , 1179 , 6629 ,12035 ,20234 ,20257 ,12911 ,22229 ,31337 ,38901 ,    0 ,63248 , 5003 , 4344 ,13545 ,10812 ],
    [70605 ,68325 ,63585 ,68890 ,54839 ,82495 ,43646 ,52795 ,49929 ,26387 ,80816 ,62687 ,    0 ,61333 ,61677 ,57333 ,68325 ],
    [12370 ,14944 , 4734 ,10655 , 8988 ,24259 ,24531 , 9864 ,18076 ,35611 ,42926 , 4653 ,62123 ,    0 ,  658 ,17819 ,14944 ],
    [11711 ,14285 , 4075 , 9996 , 9332 ,23600 ,23872 ,10208 ,18734 ,34952 ,42267 , 3994 ,62467 ,  658 ,    0 ,17160 ,14285 ],
    [21792 ,12856 ,14773 ,20077 ,19004 ,33682 ,15385 ,19880 ,26223 ,26465 ,52349 ,13874 ,58376 ,18750 ,18091 ,    0 ,12856 ],
    [17866 ,    0 ,10701 ,16151 ,21228 ,29756 ,25445 ,22105 ,31472 ,36525 ,48423 ,10499 ,68436 ,14823 ,14225 ,12439 ,    0 ],
    ]

    data['duration_matrix']=[
    [   0 , 1468 ,  564 ,  528 , 1535 ,  864 , 1797 , 1630 , 2442 , 2552 , 1737 ,  614 , 3806 ,  916 ,  870 , 1501 , 1468 ],
    [1391 ,    0 ,  939 , 1229 , 1685 , 2157 , 1949 , 1781 , 2592 , 2704 , 2610 ,  921 , 3958 , 1194 , 1167 , 1167 ,    0 ],
    [ 612 , 1026 ,    0 ,  450 , 1165 , 1378 , 1475 , 1260 , 2017 , 2230 , 1832 ,  173 , 3485 ,  445 ,  388 , 1180 , 1026 ],
    [ 515 , 1306 ,  402 ,    0 , 1373 , 1282 , 1635 , 1468 , 2238 , 2390 , 1506 ,  452 , 3644 ,  755 ,  708 , 1339 , 1306 ],
    [1488 , 1699 , 1077 , 1327 ,    0 , 2255 , 1216 ,  450 , 1297 , 1998 , 2616 ,  968 , 2941 ,  895 ,  929 , 1470 , 1699 ],
    [ 801 , 2204 , 1301 , 1265 , 2272 ,    0 , 2533 , 2367 , 3178 , 3288 , 2511 , 1351 , 4542 , 1653 , 1607 , 2238 , 2204 ],
    [1805 , 1968 , 1438 , 1644 , 1069 , 2572 ,    0 ,  958 , 1318 , 1113 , 2858 , 1318 , 2368 , 1572 , 1607 , 1051 , 1968 ],
    [1620 , 1831 , 1208 , 1458 ,  445 , 2386 , 1062 ,    0 , 1060 , 1879 , 2512 , 1100 , 2821 , 1026 , 1061 , 1557 , 1831 ],
    [2427 , 2680 , 2015 , 2189 , 1345 , 3194 , 1433 , 1097 ,    0 , 2052 , 2065 , 1907 , 2937 , 1802 , 1860 , 2218 , 2680 ],
    [2549 , 2712 , 2182 , 2387 , 1980 , 3315 , 1130 , 1869 , 1986 ,    0 , 3769 , 2062 , 1751 , 2421 , 2363 , 1794 , 2712 ],
    [1906 , 2784 , 1880 , 1494 , 2655 , 2516 , 3025 , 2534 , 2047 , 3862 ,    0 , 1930 , 4808 , 2233 , 2186 , 2817 , 2784 ],
    [ 617 ,  989 ,  165 ,  456 , 1059 , 1384 , 1308 , 1154 , 1937 , 2063 , 1837 ,    0 , 3317 ,  451 ,  393 , 1012 ,  989 ],
    [3862 , 4024 , 3495 , 3700 , 2895 , 4628 , 2432 , 2784 , 2858 , 1643 , 4725 , 3375 ,    0 , 3397 , 3432 , 3107 , 4024 ],
    [ 913 , 1297 ,  428 ,  751 ,  922 , 1679 , 1684 , 1017 , 1752 , 2439 , 2132 ,  431 , 3435 ,    0 ,   57 , 1388 , 1297 ],
    [ 855 , 1240 ,  371 ,  693 ,  957 , 1622 , 1626 , 1052 , 1810 , 2381 , 2075 ,  374 , 3470 ,   57 ,    0 , 1331 , 1240 ],
    [1488 , 1267 , 1120 , 1326 , 1529 , 2254 , 1121 , 1624 , 2266 , 1876 , 2707 , 1000 , 3130 , 1360 , 1302 ,    0 , 1267 ],
    [1391 ,    0 ,  939 , 1229 , 1685 , 2157 , 1949 , 1781 , 2592 , 2704 , 2610 ,  921 , 3958 , 1194 , 1167 , 1167 ,    0 ],
    ]
    # fmt:off
    assert data['num_Locations']==len(data['time_windows'])
    assert len(data['distance_matrix'])*2-1 == data['num_Locations']
    assert len(data['duration_matrix'])*2-1 == data['num_Locations']
    assert data['num_Locations']==len(data['demands'])
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    status = routing.status()
    print(f"Status: {RoutingSearchStatus.Value.Name(status)}")
    if (
        status != RoutingSearchStatus.ROUTING_OPTIMAL
        and status != RoutingSearchStatus.ROUTING_SUCCESS
    ):
        print("No solution found!")
        return
    print(f"Objective: {solution.ObjectiveValue()}")
    # Display dropped nodes.
    dropped_nodes = "Dropped nodes:"
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if solution.Value(routing.NextVar(node)) == node:
            dropped_nodes += f" {manager.IndexToNode(node)}"
    print(dropped_nodes)
    # Display routes.
    time_dimension = routing.GetDimensionOrDie("Time")
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    total_distance = 0
    total_time = 0
    total_load = 0
    for vehicle_id in range(manager.GetNumberOfVehicles()):
        nodes_visited = []  # To Keep Track of Node Visited by Vehicle
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        nodes_visited.append(manager.IndexToNode(index))
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            capacity_var = capacity_dimension.CumulVar(index)
            plan_output += (
                f"Node_{manager.IndexToNode(index)}"
                f" {route_distance}m"
                f" TW:[{time_var.Min()},{time_var.Max()}]"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                f" Load({solution.Value(capacity_var)}/{capacity_var.Max()})"
                " -> "
            )
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            nodes_visited.append(manager.IndexToNode(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        if len(nodes_visited) > 2:  # Condition does not prints vehicels that are not assigned
            time_var = time_dimension.CumulVar(index)
            capacity_var = capacity_dimension.CumulVar(index)
            plan_output += (
                f"Node_{manager.IndexToNode(index)}"
                f" {route_distance}m"
                f" Time({solution.Min(time_var)},{solution.Max(time_var)})"
                f" Load({solution.Value(capacity_var)}/{capacity_var.Max()})"
                "\n"
            )
            plan_output += f"Distance of the route: {route_distance}m\n"
            plan_output += f"Time of the route: {solution.Min(time_var)}min\n"
            plan_output += f"Load of the route: {solution.Value(capacity_var)}\n"
            print(plan_output)
            total_distance += route_distance
            total_time += solution.Min(time_var)
            total_load += solution.Value(capacity_var)
            print(nodes_visited)
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total time of all routes: {total_time}min")
    print(f"Total load of all routes: {total_load}")
    # print(f"Total Distance of all routes: {total_distance}m")


def main():
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(
        data["num_Locations"], data["num_vehicles"], data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node in range(data["duplicate_nodes"], data["num_Locations"]):
            from_node = 0
        if to_node in range(data["duplicate_nodes"], data["num_Locations"]):
            to_node = 0
        return data["distance_matrix"][from_node][to_node]

    data_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(data_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    capacity = "Capacity"
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, data["vehicle_capacity"], True, capacity
    )
    capacity_dimension = routing.GetDimensionOrDie(capacity)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node in range(data["duplicate_nodes"], data["num_Locations"]):
            from_node = 0
        if to_node in range(data["duplicate_nodes"], data["num_Locations"]):
            to_node = 0
        return data["duration_matrix"][from_node][to_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)

    time_dim = "Time"
    routing.AddDimension(
        time_callback_index,
        300_000,
        100_000_000_000,  # Some Big Number so that Model should not limit itself.
        False,
        time_dim,
    )
    time_dimension = routing.GetDimensionOrDie(time_dim)

    for location_idx, time_window in enumerate(data["time_windows"]):
        if location_idx == 0:
            continue
        index = manager.NodeToIndex(location_idx)

        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])
        routing.AddToAssignment(time_dimension.SlackVar(index))

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        time_dimension.CumulVar(index).SetRange(
            data["time_windows"][0][0], data["time_windows"][0][1]
        )
        routing.AddToAssignment(time_dimension.SlackVar(index))
    for i in range(data["num_vehicles"]):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.Start(i))
        )
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

    for request in data["pickup_dropoff"]:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])

        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )
        routing.solver().Add(
            time_dimension.CumulVar(pickup_index)
            <= time_dimension.CumulVar(delivery_index)
        )

        # Not Sure about whether this is right or wrong. Added Here Baesd Upon the issue #685
        min_dur = time_callback(pickup_index, delivery_index)
        max_dur = int(1.3 * min_dur)
        dur_expr = time_dimension.CumulVar(delivery_index) - time_dimension.CumulVar(
            index
        )
        routing.solver().Add(dur_expr <= max_dur)
        # Consraint End-----------

        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.AddDisjunction([pickup_index, delivery_index], 20_000, 2)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(3)
    # search_parameters.log_search = True

    solution = routing.SolveWithParameters(search_parameters)
    print_solution(manager, routing, solution)


if __name__ == "__main__":
    main()