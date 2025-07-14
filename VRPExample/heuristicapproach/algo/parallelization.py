
from typing import List, Optional, Iterator, Callable, TYPE_CHECKING
import multiprocessing
import concurrent.futures
import copy

if TYPE_CHECKING:
    from route import Route
    from order import Order
    from solution import Solution
    from vehicle import Vehicle

from first_level import l1_heuristic, best_insertion_initializer, get_move_attributes, calculate_z1_score
from granular_tabu_search import granular_multiple_order_relocation_neighborhood

def l1_heuristic_parallel(orders: List['Order'], vehicles: List['Vehicle'], params: dict) -> 'Solution':
    """
    Parallel version of the L1 heuristic using ProcessPoolExecutor (Section 5.10).
    
    This implements both Parallel Evaluation (PE) and Parallel Neighborhood (PN)
    exploration strategies for improved performance on multi-core systems.
    
    Args:
        orders: List of orders to be assigned
        vehicles: List of available vehicles  
        params: Algorithm parameters including parallelization settings
        
    Returns:
        Optimized solution after parallel Tabu Search
    """
    # Check if parallelization is enabled
    if not params.get('enable_parallelization', False):
        return l1_heuristic(orders, vehicles, params)
    
    # Get parallelization parameters
    max_workers = params.get('max_workers', multiprocessing.cpu_count())
    parallel_strategy = params.get('parallel_strategy', 'PE')  # 'PE' or 'PN'
    
    if parallel_strategy == 'PE':
        return _l1_heuristic_parallel_evaluation(orders, vehicles, params, max_workers)
    elif parallel_strategy == 'PN':
        return _l1_heuristic_parallel_neighborhood(orders, vehicles, params, max_workers)
    else:
        # Fallback to sequential version
        return l1_heuristic(orders, vehicles, params)


def _l1_heuristic_parallel_evaluation(orders: List['Order'], vehicles: List['Vehicle'], 
                                     params: dict, max_workers: int) -> 'Solution':
    """
    L1 heuristic with Parallel Evaluation (PE) strategy.
    
    In PE, neighborhood exploration is sequential, but solution evaluation
    is parallelized across multiple processes.
    """
    # 1. Create initial solution
    initial_solution = best_insertion_initializer(orders, vehicles, params)
    
    # 2. Initialize state
    best_solution = copy.deepcopy(initial_solution)
    center_solution = copy.deepcopy(initial_solution)
    tabu_list = collections.deque(maxlen=params['tabu_tenure'])
    non_improving_iters = 0
    total_iters = 0
    
    # Initialize ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        # 3. Main loop
        while non_improving_iters < params['M1'] and total_iters < params['M2']:
            total_iters += 1
            improvement_found = False
            best_neighbors_pool = []
            
            # Include advanced neighborhoods based on parameters
            neighborhoods = [single_order_relocation_neighborhood, two_orders_swap_neighborhood]
            if params.get('enable_advanced_neighborhoods', False):
                neighborhoods.extend([multiple_order_relocation_neighborhood, two_opt_routes_neighborhood])
            if params.get('enable_granular_search', False) and NETWORKX_AVAILABLE:
                neighborhoods.append(granular_multiple_order_relocation_neighborhood)
            
            # 4. VND Loop - Variable Neighborhood Descent
            for neighborhood_func in neighborhoods:
                
                # Collect all neighbors for parallel evaluation
                neighbors = list(neighborhood_func(center_solution))
                
                if not neighbors:
                    continue
                
                # Parallel evaluation of all neighbors
                neighbor_scores = list(executor.map(
                    _evaluate_solution_wrapper, 
                    [(neighbor, params) for neighbor in neighbors]
                ))
                
                # Process results
                best_neighbor_in_N = None
                best_neighbor_score = float('-inf')
                
                for neighbor, neighbor_score in zip(neighbors, neighbor_scores):
                    # Check if move is tabu
                    move_attrs = get_move_attributes(center_solution, neighbor)
                    is_tabu = move_attrs in tabu_list
                    
                    # Apply aspiration criteria
                    aspiration = neighbor_score > _evaluate_solution_wrapper((best_solution, params))
                    
                    if (not is_tabu or aspiration) and (best_neighbor_in_N is None or neighbor_score > best_neighbor_score):
                        best_neighbor_in_N = neighbor
                        best_neighbor_score = neighbor_score
                    elif not is_tabu:
                        best_neighbors_pool.append((neighbor, neighbor_score))
                
                # Check for improvement
                center_score = _evaluate_solution_wrapper((center_solution, params))
                if best_neighbor_in_N and best_neighbor_score > center_score:
                    # Update tabu list with move attributes
                    move_attrs = get_move_attributes(center_solution, best_neighbor_in_N)
                    tabu_list.append(move_attrs)
                    
                    center_solution = best_neighbor_in_N
                    non_improving_iters = 0
                    improvement_found = True
                    
                    # Update global best if needed
                    best_score = _evaluate_solution_wrapper((best_solution, params))
                    if best_neighbor_score > best_score:
                        best_solution = copy.deepcopy(center_solution)
                    
                    break  # VND restart
            
            # 5. Diversification / Non-improving move
            if not improvement_found:
                non_improving_iters += 1
                
                if best_neighbors_pool:
                    previous_center = center_solution
                    
                    if params.get('exploration_strategy', 'deterministic') == 'deterministic':
                        best_neighbors_pool.sort(key=lambda x: x[1], reverse=True)
                        selected_neighbor = best_neighbors_pool[0][0]
                    else:
                        import random
                        weights = [score - min(s for _, s in best_neighbors_pool) + 1 
                                  for _, score in best_neighbors_pool]
                        total_weight = sum(weights)
                        weights = [w/total_weight for w in weights]
                        selected_idx = random.choices(range(len(best_neighbors_pool)), weights=weights, k=1)[0]
                        selected_neighbor = best_neighbors_pool[selected_idx][0]
                    
                    center_solution = selected_neighbor
                    move_attrs = get_move_attributes(previous_center, center_solution)
                    tabu_list.append(move_attrs)
    
    return best_solution


def _l1_heuristic_parallel_neighborhood(orders: List['Order'], vehicles: List['Vehicle'], 
                                       params: dict, max_workers: int) -> 'Solution':
    """
    L1 heuristic with Parallel Neighborhood (PN) strategy.
    
    In PN, different neighborhoods are explored in parallel, with early
    termination when an improving solution is found.
    """
    # 1. Create initial solution
    initial_solution = best_insertion_initializer(orders, vehicles, params)
    
    # 2. Initialize state
    best_solution = copy.deepcopy(initial_solution)
    center_solution = copy.deepcopy(initial_solution)
    tabu_list = collections.deque(maxlen=params['tabu_tenure'])
    non_improving_iters = 0
    total_iters = 0
    
    # Create manager for inter-process communication
    manager = multiprocessing.Manager()
    improvement_found_event = manager.Event()
    best_result_queue = manager.Queue()
    
    # 3. Main loop
    while non_improving_iters < params['M1'] and total_iters < params['M2']:
        total_iters += 1
        improvement_found = False
        
        # Include advanced neighborhoods based on parameters
        neighborhoods = [single_order_relocation_neighborhood, two_orders_swap_neighborhood]
        if params.get('enable_advanced_neighborhoods', False):
            neighborhoods.extend([multiple_order_relocation_neighborhood, two_opt_routes_neighborhood])
        if params.get('enable_granular_search', False) and NETWORKX_AVAILABLE:
            neighborhoods.append(granular_multiple_order_relocation_neighborhood)
        
        # Clear previous results
        improvement_found_event.clear()
        while not best_result_queue.empty():
            best_result_queue.get()
        
        # 4. Parallel neighborhood exploration
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            
            # Submit neighborhood exploration tasks
            futures = []
            for neighborhood_func in neighborhoods:
                future = executor.submit(
                    _explore_neighborhood_parallel,
                    neighborhood_func,
                    center_solution,
                    tabu_list,
                    best_solution,
                    params,
                    improvement_found_event,
                    best_result_queue
                )
                futures.append(future)
            
            # Wait for results or improvement
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=1.0)  # Short timeout for responsiveness
                    if result and result.get('improvement_found', False):
                        improvement_found = True
                        center_solution = result['best_neighbor']
                        best_solution = result.get('global_best', best_solution)
                        move_attrs = result['move_attrs']
                        tabu_list.append(move_attrs)
                        non_improving_iters = 0
                        
                        # Cancel other tasks
                        improvement_found_event.set()
                        for other_future in futures:
                            other_future.cancel()
                        break
                        
                except concurrent.futures.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error in parallel neighborhood exploration: {e}")
                    continue
        
        # 5. Diversification if no improvement found
        if not improvement_found:
            non_improving_iters += 1
            # Implement diversification strategy here
            # For simplicity, we'll use the sequential fallback
            pass
    
    return best_solution


def _evaluate_solution_wrapper(args) -> float:
    """
    Wrapper function for parallel solution evaluation.
    
    Args:
        args: Tuple of (solution, params)
        
    Returns:
        Z1 score for the solution
    """
    solution, params = args
    return calculate_z1_score(solution, params)


def _explore_neighborhood_parallel(neighborhood_func: Callable, center_solution: 'Solution',
                                 tabu_list: collections.deque, best_solution: 'Solution',
                                 params: dict, improvement_event, result_queue) -> dict:
    """
    Explore a neighborhood in parallel with early termination.
    
    Args:
        neighborhood_func: Neighborhood exploration function
        center_solution: Current center solution
        tabu_list: Current tabu list
        best_solution: Current best solution
        params: Algorithm parameters
        improvement_event: Event to signal when improvement is found
        result_queue: Queue for sharing results
        
    Returns:
        Dictionary with exploration results
    """
    best_neighbor = None
    best_score = float('-inf')
    move_attrs = None
    global_best_updated = False
    
    try:
        for neighbor in neighborhood_func(center_solution):
            # Check if another process found improvement
            if improvement_event.is_set():
                break
            
            neighbor_score = calculate_z1_score(neighbor, params)
            
            # Check tabu status
            neighbor_move_attrs = get_move_attributes(center_solution, neighbor)
            is_tabu = neighbor_move_attrs in tabu_list
            
            # Apply aspiration criteria
            best_solution_score = calculate_z1_score(best_solution, params)
            aspiration = neighbor_score > best_solution_score
            
            if not is_tabu or aspiration:
                if neighbor_score > best_score:
                    best_neighbor = neighbor
                    best_score = neighbor_score
                    move_attrs = neighbor_move_attrs
                    
                    # Check if this improves the global best
                    if neighbor_score > best_solution_score:
                        global_best_updated = True
                    
                    # Check if this improves over center solution
                    center_score = calculate_z1_score(center_solution, params)
                    if neighbor_score > center_score:
                        # Found improvement - signal other processes
                        improvement_event.set()
                        result = {
                            'improvement_found': True,
                            'best_neighbor': best_neighbor,
                            'move_attrs': move_attrs,
                            'best_score': best_score
                        }
                        if global_best_updated:
                            result['global_best'] = best_neighbor
                        
                        result_queue.put(result)
                        return result
        
    except Exception as e:
        print(f"Error in neighborhood exploration: {e}")
        return {'improvement_found': False}
    
    return {
        'improvement_found': False,
        'best_neighbor': best_neighbor,
        'move_attrs': move_attrs,
        'best_score': best_score
    }
