"""
Integration script to test the bulletproof geocoder with the existing VRP scenario data

This script loads the Excel data and tests the new bulletproof geocoder
against all addresses to validate performance improvements.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
heuristic_root = os.path.join(current_dir, '..')
utils_dir = os.path.join(heuristic_root, 'utils')
src_dir = os.path.join(heuristic_root, 'src')
sys.path.insert(0, utils_dir)
sys.path.insert(0, src_dir)

from bulletproof_geocoder import BulletproofGeocoder

def extract_addresses_from_excel(excel_path: str) -> list:
    """Extract all unique addresses from the Excel file"""
    try:
        df = pd.read_excel(excel_path)
        
        # Use the correct column name for this dataset
        address_column = 'ADDRESS'
        
        if address_column not in df.columns:
            print("Available columns:", df.columns.tolist())
            return []
        
        addresses = df[address_column].dropna().unique().tolist()
        return addresses
        
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def test_bulletproof_on_excel_data():
    """Test the bulletproof geocoder on real Excel data"""
    
    # Locate the Excel file
    excel_path = os.path.join(src_dir, 'furgoni.xlsx')
    if not os.path.exists(excel_path):
        print(f"❌ Excel file not found: {excel_path}")
        return
    
    print("🚀 Testing Bulletproof Geocoder on Excel Data")
    print("=" * 60)
    
    # Extract addresses
    print("📁 Loading addresses from Excel...")
    addresses = extract_addresses_from_excel(excel_path)
    
    if not addresses:
        print("❌ No addresses found in Excel file")
        return
    
    print(f"✅ Found {len(addresses)} unique addresses")
    
    # Initialize bulletproof geocoder
    geocoder = BulletproofGeocoder(cache_file="excel_test_geocode_cache.json")
    
    # Progress callback function
    def progress_callback(current, total, result):
        if current % 5 == 0 or current == total:  # Update every 5 addresses
            success_rate = (geocoder.success_stats['successful_geocodes'] / 
                          geocoder.success_stats['total_requests']) * 100
            print(f"Progress: {current}/{total} ({success_rate:.1f}% success)")
    
    # Perform batch geocoding
    print("\n🔍 Starting batch geocoding...")
    results = geocoder.batch_geocode(addresses, progress_callback=progress_callback)
    
    # Analyze results
    successful_results = [r for r in results if r is not None]
    failed_addresses = [addr for addr, result in zip(addresses, results) if result is None]
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"   Total addresses: {len(addresses)}")
    print(f"   Successful geocodes: {len(successful_results)}")
    print(f"   Failed geocodes: {len(failed_addresses)}")
    print(f"   Success rate: {(len(successful_results) / len(addresses)) * 100:.2f}%")
    
    # Print detailed statistics
    stats = geocoder.get_statistics()
    print(f"\n📈 Detailed Statistics:")
    for key, value in stats.items():
        if key != 'provider_statistics':
            print(f"   {key}: {value}")
    
    if 'provider_statistics' in stats and stats['provider_statistics']:
        print(f"\n🔧 Provider Performance:")
        for provider, provider_stats in stats['provider_statistics'].items():
            print(f"   {provider}: {provider_stats}")
    
    # Show failed addresses for analysis
    if failed_addresses:
        print(f"\n❌ Failed Addresses ({len(failed_addresses)}):")
        for i, addr in enumerate(failed_addresses[:10], 1):  # Show first 10
            print(f"   {i}. {addr}")
        if len(failed_addresses) > 10:
            print(f"   ... and {len(failed_addresses) - 10} more")
    
    # Show sample successful results
    if successful_results:
        print(f"\n✅ Sample Successful Results:")
        sample_results = list(zip(addresses, results))[:5]
        for i, (addr, result) in enumerate(sample_results, 1):
            if result:
                print(f"   {i}. {addr}")
                print(f"      -> ({result.latitude:.6f}, {result.longitude:.6f}) via {result.provider}")
    
    return results

if __name__ == "__main__":
    test_bulletproof_on_excel_data()
