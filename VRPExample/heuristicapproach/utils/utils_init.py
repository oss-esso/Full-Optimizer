"""
Utilities Package for EPDT Algorithm

This package contains utility modules for:
- OSRM route pre-computation and caching
- Performance optimization tools
- Testing and validation utilities
"""

__version__ = "1.0.0"

# Import main functionality for easy access
try:
    from .osrm_utils import (
        init_route_cache_db,
        cache_route_data,
        is_route_cached,
        query_osrm_and_cache,
        get_cache_stats,
        clear_cache
    )
    from .precompute_routes import RoutePrecomputer, LocationInfo
    
    __all__ = [
        'init_route_cache_db',
        'cache_route_data', 
        'is_route_cached',
        'query_osrm_and_cache',
        'get_cache_stats',
        'clear_cache',
        'RoutePrecomputer',
        'LocationInfo'
    ]
except ImportError:
    # Allow partial imports if some dependencies are missing
    __all__ = []
