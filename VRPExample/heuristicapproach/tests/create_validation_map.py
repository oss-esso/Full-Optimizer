"""
Interactive Coordinate Map Generator

Creates a clean interactive map focusing on suspicious coordinates
for easy visual verification.
"""

import os
import sys
import json
import folium

# Add necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_coordinates_from_json():
    """Load coordinates from the extracted JSON file."""
    json_path = os.path.join(current_dir, 'extracted_coordinates.json')
    
    if not os.path.exists(json_path):
        print("❌ No extracted coordinates found. Run test_coordinate_validation.py first.")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        coordinates = json.load(f)
    
    print(f"✅ Loaded {len(coordinates)} coordinates from JSON")
    return coordinates

def identify_suspicious_coordinates(coordinates):
    """Identify suspicious coordinates that need manual verification."""
    suspicious = []
    
    # Define Italy bounds (approximate)
    ITALY_BOUNDS = {
        'lat_min': 35.0,  # Southern Sicily
        'lat_max': 47.0,  # Northern Alps
        'lon_min': 6.0,   # Western borders (excluding Sardinia)
        'lon_max': 19.0   # Eastern borders
    }
    
    # Define specific regions for postal code validation
    PIEMONTE_BOUNDS = {
        'lat_min': 44.2,
        'lat_max': 46.4,
        'lon_min': 6.6,
        'lon_max': 9.7
    }
    
    for coord in coordinates:
        lat, lon = coord['lat'], coord['lon']
        address = coord['address']
        issues = []
        
        # Check for completely wrong regions
        if 'CAMBIANO, 10020' in address:
            # This should be in Piemonte
            if not (PIEMONTE_BOUNDS['lat_min'] <= lat <= PIEMONTE_BOUNDS['lat_max'] and
                    PIEMONTE_BOUNDS['lon_min'] <= lon <= PIEMONTE_BOUNDS['lon_max']):
                issues.append("CRITICAL: Address in wrong region of Italy")
        
        # Check for duplicate coordinates
        duplicates = [c for c in coordinates if c['lat'] == lat and c['lon'] == lon and c != coord]
        if duplicates:
            issues.append(f"Duplicate coordinates shared with {len(duplicates)} other location(s)")
        
        # Check for coordinates outside Italy (excluding valid international locations)
        if lat < ITALY_BOUNDS['lat_min'] or lat > ITALY_BOUNDS['lat_max']:
            if 'FRANCE' not in address and 'SAN MARINO' not in address:
                issues.append(f"Latitude {lat} outside Italy bounds")
        
        if lon < ITALY_BOUNDS['lon_min'] or lon > ITALY_BOUNDS['lon_max']:
            if 'FRANCE' not in address and 'SAN MARINO' not in address:
                issues.append(f"Longitude {lon} outside Italy bounds")
        
        if issues:
            suspicious.append({
                **coord,
                'issues': issues,
                'severity': 'CRITICAL' if any('CRITICAL' in issue for issue in issues) else 'WARNING'
            })
    
    return suspicious

def create_focused_map(coordinates, suspicious_coords, output_path):
    """Create a focused interactive map highlighting issues."""
    
    # Calculate center point
    if coordinates:
        center_lat = sum(c['lat'] for c in coordinates) / len(coordinates)
        center_lon = sum(c['lon'] for c in coordinates) / len(coordinates)
    else:
        center_lat, center_lon = 44.5, 11.0  # Central Italy
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add all coordinates with different colors
    for coord in coordinates:
        # Determine color based on whether it's suspicious
        is_suspicious = any(s['address'] == coord['address'] for s in suspicious_coords)
        
        if is_suspicious:
            susp_coord = next(s for s in suspicious_coords if s['address'] == coord['address'])
            if susp_coord['severity'] == 'CRITICAL':
                color = 'red'
                icon = 'exclamation-sign'
            else:
                color = 'orange'
                icon = 'warning-sign'
        else:
            color = 'green'
            icon = 'ok-sign'
        
        # Create popup with coordinate info
        popup_text = f"<b>{coord['address']}</b><br>"
        popup_text += f"Coordinates: ({coord['lat']:.6f}, {coord['lon']:.6f})<br>"
        popup_text += f"Type: {coord['type']}<br>"
        
        if is_suspicious:
            popup_text += "<br><b>⚠️ ISSUES:</b><br>"
            for issue in susp_coord['issues']:
                popup_text += f"• {issue}<br>"
        
        folium.Marker(
            [coord['lat'], coord['lon']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{coord['address'][:50]}{'...' if len(coord['address']) > 50 else ''}",
            icon=folium.Icon(color=color, icon=icon)
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;">
    <h4>Coordinate Status</h4>
    <i class="fa fa-circle" style="color:green"></i> Verified OK<br>
    <i class="fa fa-circle" style="color:orange"></i> Warning<br>
    <i class="fa fa-circle" style="color:red"></i> Critical Issue<br>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_path)
    print(f"🗺️  Focused validation map saved to: {output_path}")

def create_summary_report(suspicious_coords):
    """Create a summary report of suspicious coordinates."""
    print("\n" + "="*80)
    print("🔍 COORDINATE VALIDATION REPORT")
    print("="*80)
    
    if not suspicious_coords:
        print("✅ No suspicious coordinates found!")
        return
    
    critical_count = sum(1 for c in suspicious_coords if c['severity'] == 'CRITICAL')
    warning_count = len(suspicious_coords) - critical_count
    
    print(f"⚠️  Found {len(suspicious_coords)} locations with issues:")
    print(f"   🔴 {critical_count} CRITICAL issues")
    print(f"   🟠 {warning_count} WARNING issues")
    print()
    
    for coord in suspicious_coords:
        severity_icon = "🔴" if coord['severity'] == 'CRITICAL' else "🟠"
        print(f"{severity_icon} {coord['severity']}: {coord['address']}")
        print(f"   📍 Coordinates: ({coord['lat']:.6f}, {coord['lon']:.6f})")
        for issue in coord['issues']:
            print(f"   ⚠️  {issue}")
        print()

def main():
    """Main function."""
    print("🗺️  INTERACTIVE COORDINATE MAP GENERATOR")
    print("="*50)
    
    # Load coordinates
    coordinates = load_coordinates_from_json()
    if not coordinates:
        return
    
    # Identify suspicious coordinates
    suspicious_coords = identify_suspicious_coordinates(coordinates)
    
    # Create focused map
    output_path = os.path.join(current_dir, 'coordinate_validation_focused_map.html')
    create_focused_map(coordinates, suspicious_coords, output_path)
    
    # Create summary report
    create_summary_report(suspicious_coords)
    
    print("\n🎯 VERIFICATION ACTIONS RECOMMENDED:")
    print("1. Open the generated HTML map in your browser")
    print("2. Check red markers (critical issues) first")
    print("3. Verify orange markers (warnings) second")
    print("4. Green markers are likely correct")
    print(f"\n📁 Map file: {output_path}")

if __name__ == "__main__":
    main()
