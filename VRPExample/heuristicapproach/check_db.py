#!/usr/bin/env python3
import sqlite3

try:
    conn = sqlite3.connect('moda_routes.db')
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables in database: {tables}")
    
    if 'route_cache' in tables:
        cursor.execute("SELECT COUNT(*) FROM route_cache")
        count = cursor.fetchone()[0]
        print(f"Routes in cache: {count}")
        
        if count > 0:
            cursor.execute("SELECT start_node_id, end_node_id, distance_km, duration_minutes FROM route_cache LIMIT 3")
            routes = cursor.fetchall()
            print("Sample routes:")
            for route in routes:
                print(f"  {route[0]} -> {route[1]}: {route[2]:.1f} km, {route[3]:.1f} min")
    else:
        print("No route_cache table found")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
