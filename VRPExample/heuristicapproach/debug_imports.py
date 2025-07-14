import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'VRPInstance'))

print("Current working directory:", os.getcwd())
print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying imports...")

try:
    import vrp_data_models
    print("✅ vrp_data_models imported successfully")
except ImportError as e:
    print(f"❌ vrp_data_models import failed: {e}")

try:
    from src.moda_scenarios import create_furgoni_scenario
    print("✅ moda_scenarios imported successfully via src.moda_scenarios")
except ImportError as e:
    print(f"❌ src.moda_scenarios import failed: {e}")

try:
    from moda_scenarios import create_furgoni_scenario
    print("✅ moda_scenarios imported successfully via moda_scenarios")
except ImportError as e:
    print(f"❌ moda_scenarios import failed: {e}")

# Check if moda_scenarios.py exists in src
src_path = os.path.join(os.path.dirname(__file__), 'src', 'moda_scenarios.py')
print(f"\nChecking moda_scenarios.py at: {src_path}")
print(f"File exists: {os.path.exists(src_path)}")
