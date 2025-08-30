"""
Vehicle utility functions for the EPDT heuristic solver.

This module contains utility functions for analyzing vehicle capabilities and constraints.
"""

def get_pallets_change(task):
    """Get the pallets change from a task."""
    return getattr(task, 'pallets', 0.0)


def get_vehicle_capabilities(vehicle):
    """Extract vehicle capabilities from vehicle object."""
    capabilities = {
        'loader': False,
        'low_temp': False, 
        'hangers': False,
        'lifo_required': False,
        'regulations': []
    }
    
    if not vehicle:
        return capabilities
    
    # DEBUG: Print vehicle capabilities to understand what's actually loaded (COMMENTED OUT)
    vehicle_capabilities = getattr(vehicle, 'capabilities', set())
    if False and hasattr(vehicle, 'id') and vehicle.id in ['GA625VG', 'GA621VG', 'FX194HX', 'GE026FZ', 'FX192HX']:
        print(f"DEBUG CAPABILITIES: Vehicle {vehicle.id} has capabilities set: {vehicle_capabilities}")
        print(f"DEBUG CAPABILITIES: Vehicle {vehicle.id} attributes: {[attr for attr in dir(vehicle) if 'temp' in attr.lower() or 'low' in attr.lower() or 'cap' in attr.lower()]}")
        # Let's check what the final capabilities dict looks like after our detection
        temp_caps = {
            'loader': False,
            'low_temp': False, 
            'hangers': False,
            'lifo_required': False,
            'regulations': []
        }
        temp_caps['low_temp'] = getattr(vehicle, 'low_temp', False) or \
                               getattr(vehicle, 'has_low_temp', False) or \
                               ('LOW TEMP' in str(getattr(vehicle, 'capabilities', '')).upper()) or \
                               ('LOW TEMP' in vehicle_capabilities) or ('LOW_TEMP' in vehicle_capabilities) or \
                               ('low_temp' in vehicle_capabilities) or ('low temp' in vehicle_capabilities)
        # DEBUG COMMENTED OUT: print(f"DEBUG CAPABILITIES: Vehicle {vehicle.id} final capabilities: {temp_caps}")
    
    # Check for loader capability
    capabilities['loader'] = getattr(vehicle, 'loader', False) or \
                            getattr(vehicle, 'has_loader', False) or \
                            ('LOADER' in str(getattr(vehicle, 'capabilities', '')).upper()) or \
                            ('LOADER' in vehicle_capabilities) or ('loader' in vehicle_capabilities)
    
    # Check for low temperature capability - FIXED: Check for "LOW TEMP" with space  
    capabilities['low_temp'] = getattr(vehicle, 'low_temp', False) or \
                              getattr(vehicle, 'has_low_temp', False) or \
                              ('LOW TEMP' in str(getattr(vehicle, 'capabilities', '')).upper()) or \
                              ('LOW TEMP' in vehicle_capabilities) or ('LOW_TEMP' in vehicle_capabilities) or \
                              ('low_temp' in vehicle_capabilities) or ('low temp' in vehicle_capabilities)
    
    # Check for hangers capability
    capabilities['hangers'] = getattr(vehicle, 'hangers', False) or \
                             getattr(vehicle, 'has_hangers', False) or \
                             ('HANGERS' in str(getattr(vehicle, 'capabilities', '')).upper()) or \
                             ('HANGERS' in vehicle_capabilities) or ('hangers' in vehicle_capabilities)
    
    # Check for LIFO requirement
    capabilities['lifo_required'] = getattr(vehicle, 'lifo_required', False) or \
                                   getattr(vehicle, 'requires_lifo', False) or \
                                   ('LIFO' in str(getattr(vehicle, 'regulations', '')).upper())
    
    # Get regulations as list
    regulations = getattr(vehicle, 'regulations', '')
    if regulations:
        capabilities['regulations'] = [reg.strip() for reg in str(regulations).split(',') if reg.strip()]
    
    # DEBUG: Print final capabilities for debugging (COMMENTED OUT)
    if False and hasattr(vehicle, 'id') and vehicle.id in ['GA625VG', 'GA621VG', 'FX194HX']:
        print(f"DEBUG CAPABILITIES: Vehicle {vehicle.id} final capabilities: {capabilities}")
    
    return capabilities


def get_route_order_requirements(orders, route):
    """Extract requirements from all orders in a route."""
    requirements = {
        'loader': False,
        'low_temp': False,
        'hangers': False,
        'special_requirements': []
    }
    
    if not orders or not route or not route.tasks:
        return requirements
    
    # Get order IDs from route tasks
    order_ids = set()
    for task in route.tasks:
        if hasattr(task, 'order_id') and task.order_id:
            order_ids.add(str(task.order_id))
    
    # Check requirements for each order
    for order in orders:
        if str(order.id) in order_ids:
            # Check all tasks in the order
            all_tasks = []
            if hasattr(order, 'pickup_tasks'):
                all_tasks.extend(order.pickup_tasks)
            if hasattr(order, 'delivery_tasks'):
                all_tasks.extend(order.delivery_tasks)
            
            for task in all_tasks:
                # Check for loader requirement
                if getattr(task, 'requires_loader', False) or \
                   ('LOADER' in str(getattr(task, 'required_capabilities', '')).upper()):
                    requirements['loader'] = True
                
                # Check for low temp requirement
                if getattr(task, 'requires_low_temp', False) or \
                   ('LOW_TEMP' in str(getattr(task, 'required_capabilities', '')).upper()):
                    requirements['low_temp'] = True
                
                # Check for hangers requirement
                if getattr(task, 'requires_hangers', False) or \
                   ('HANGERS' in str(getattr(task, 'required_capabilities', '')).upper()):
                    requirements['hangers'] = True
                
                # Collect special requirements
                special_reqs = getattr(task, 'special_requirements', '')
                if special_reqs:
                    requirements['special_requirements'].extend([req.strip() for req in str(special_reqs).split(',') if req.strip()])
    
    return requirements


def validate_constraints(vehicle_capabilities, order_requirements):
    """
    Validate that vehicle capabilities meet order requirements.
    
    Args:
        vehicle_capabilities: Dict with vehicle capabilities
        order_requirements: Dict with order requirements
        
    Returns:
        List of violation strings (empty list if no violations)
    """
    violations = []
    
    # Check loader requirement
    if order_requirements.get('loader', False) and not vehicle_capabilities.get('loader', False):
        violations.append("Missing LOADER capability")
    
    # Check low temp requirement
    if order_requirements.get('low_temp', False) and not vehicle_capabilities.get('low_temp', False):
        violations.append("Missing LOW_TEMP capability")
    
    # Check hangers requirement
    if order_requirements.get('hangers', False) and not vehicle_capabilities.get('hangers', False):
        violations.append("Missing HANGERS capability")
    
    return violations
