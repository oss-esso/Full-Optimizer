import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import logging

logger = logging.getLogger(__name__)

def load_food_data(complexity_level: str = 'simple') -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """
    Load food data based on specified complexity level.
    
    Args:
        complexity_level (str): One of 'simple', 'intermediate', or 'full'
        
    Returns:
        Tuple containing farms, foods, food_groups, and config
    """
    if complexity_level == 'simple':
        return _load_simple_food_data()
    elif complexity_level == 'intermediate':
        return _load_intermediate_food_data()
    elif complexity_level == 'full':
        return _load_full_food_data()
    else:
        raise ValueError(f"Invalid complexity level: {complexity_level}. Must be one of: simple, intermediate, full")

def _load_simple_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load simplified food data for testing."""
    # Define farms
    farms = ['Farm1', 'Farm2', 'Farm3']
    
    # Define foods with nutritional values, etc.
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.3,
            'affordability': 0.8,
            'sustainability': 0.7
        },
        'Corn': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'environmental_impact': 0.4,
            'affordability': 0.9,
            'sustainability': 0.6
        },
        'Rice': {
            'nutritional_value': 0.8,
            'nutrient_density': 0.7,
            'environmental_impact': 0.6,
            'affordability': 0.7,
            'sustainability': 0.5
        },
        'Soybeans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.6,
            'sustainability': 0.8
        },
        'Potatoes': {
            'nutritional_value': 0.5,
            'nutrient_density': 0.4,
            'environmental_impact': 0.3,
            'affordability': 0.9,
            'sustainability': 0.7
        },
        'Apples': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.2,
            'affordability': 0.5,
            'sustainability': 0.8
        }
    }
    
    # Define food groups
    food_groups = {
        'Grains': ['Wheat', 'Corn', 'Rice'],
        'Legumes': ['Soybeans'],
        'Vegetables': ['Potatoes'],
        'Fruits': ['Apples']
    }
    
    # Set parameters
    parameters = {
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.25,
            'affordability': 0,
            'sustainability': 0,
            'environmental_impact': 0.5
        },
        'land_availability': {
            'Farm1': 75,
            'Farm2': 100,
            'Farm3': 50
        },
        'food_groups': food_groups
    }
    
    # Update config
    config = {
        'parameters': parameters
    }
    
    logger.info(f"Loaded simple data for {len(farms)} farms and {len(foods)} foods")
    return farms, foods, food_groups, config

def _load_intermediate_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load intermediate food data for testing."""
    # Define farms
    farms = ['Farm1', 'Farm2', 'Farm3']
    
    # Define foods with nutritional values, etc.
    foods = {
        'Wheat': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.3,
            'affordability': 0.8,
            'sustainability': 0.7
        },
        'Corn': {
            'nutritional_value': 0.6,
            'nutrient_density': 0.5,
            'environmental_impact': 0.4,
            'affordability': 0.9,
            'sustainability': 0.6
        },
        'Rice': {
            'nutritional_value': 0.8,
            'nutrient_density': 0.7,
            'environmental_impact': 0.6,
            'affordability': 0.7,
            'sustainability': 0.5
        },
        'Soybeans': {
            'nutritional_value': 0.9,
            'nutrient_density': 0.8,
            'environmental_impact': 0.2,
            'affordability': 0.6,
            'sustainability': 0.8
        },
        'Potatoes': {
            'nutritional_value': 0.5,
            'nutrient_density': 0.4,
            'environmental_impact': 0.3,
            'affordability': 0.9,
            'sustainability': 0.7
        },
        'Apples': {
            'nutritional_value': 0.7,
            'nutrient_density': 0.6,
            'environmental_impact': 0.2,
            'affordability': 0.5,
            'sustainability': 0.8
        }
    }
    
    # Define food groups
    food_groups = {
        'Grains': ['Wheat', 'Corn', 'Rice'],
        'Legumes': ['Soybeans'],
        'Vegetables': ['Potatoes'],
        'Fruits': ['Apples']
    }
    
    # Set parameters with updated configuration
    parameters = {
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'affordability': 0.15,
            'sustainability': 0.15,
            'environmental_impact': 0.25
        },
        'land_availability': {
            'Farm1': 75,
            'Farm2': 100,
            'Farm3': 50
        },
        'minimum_planting_area': {
            'Wheat': 10,
            'Corn': 8,
            'Rice': 12,
            'Soybeans': 7,
            'Potatoes': 5,
            'Apples': 15
        },
        'max_percentage_per_crop': {
            food: 0.4 for food in foods  # Updated to 40% max per crop
        },
        'social_benefit': {
            farm: 0.2 for farm in farms  # 20% minimum land utilization
        },
        'food_group_constraints': {
            group: {
                'min_foods': 1,  # At least 1 food from each group
                'max_foods': len(foods_in_group)  # Up to all foods in group
            }
            for group, foods_in_group in food_groups.items()
        }
    }
    
    # Update config with additional solver settings
    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,  # Tighter convergence tolerance
        'benders_max_iterations': 100,  # More iterations allowed
        'pulp_time_limit': 120,  # 2 minutes time limit for PuLP
        'use_multi_cut': True,  # Enable multi-cut Benders
        'use_trust_region': True,  # Enable trust region stabilization
        'use_anticycling': True,  # Enable anti-cycling measures
        'use_norm_cuts': True,  # Use normalized optimality cuts
        'quantum_settings': {  # Added quantum-specific settings
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }
    
    logger.info(f"Loaded intermediate data for {len(farms)} farms and {len(foods)} foods")
    return farms, foods, food_groups, config

def _load_full_food_data() -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, List[str]], Dict]:
    """Load food data and configuration from Excel for optimization."""
    # Locate Excel file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Determine drive letter from the script directory (e.g., "G:" or "H:")
    drive = script_dir.split(os.sep)[0]
    excel_path = os.path.join(drive, "\\Il mio Drive", "OQI_Project", "Inputs", "Combined_Food_Data.xlsx")
    print(f"Loading food data from: {excel_path}")
    
    # Read Excel
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file not found at {excel_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}")
    
    # Map columns
    col_map = {
        'Food_Name': 'Food_Name',
        'food_group': 'Food_Group',
        'nutritional_value': 'nutritional_value',
        'nutrient_density': 'nutrient_density',
        'environmental_impact': 'environmental_impact',
        'affordability': 'affordability',
        'sustainability': 'sustainability'
    }
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in Excel: {missing}")

    # Select 2 samples per group
    grp_col = 'food_group'
    name_col = 'Food_Name'
    sampled = df.groupby(grp_col).apply(
        lambda x: x.sample(n=min(len(x), 2))
    ).reset_index(drop=True)
    foods_list = sampled[col_map['Food_Name']].tolist()
    
    # Filter and rename
    filt = df[df[name_col].isin(foods_list)][list(col_map.keys())].copy()
    filt.rename(columns=col_map, inplace=True)

    # Convert scores
    objectives = ['nutritional_value', 'nutrient_density', 'environmental_impact', 'affordability', 'sustainability']
    for obj in objectives:
        filt[obj] = pd.to_numeric(filt[obj], errors='coerce').fillna(0.0)

    # Build structures without profitability
    farms = ['Farm1', 'Farm2', 'Farm3', 'Farm4', 'Farm5']
    foods = {}
    for _, row in filt.iterrows():
        food_dict = {obj: float(row[obj]) for obj in objectives}
        # No longer adding profitability
        foods[row['Food_Name']] = food_dict
    
    food_groups: Dict[str, List[str]] = {}
    for _, row in filt.iterrows():
        fg = row['Food_Group'] or 'Unknown'
        food_groups.setdefault(fg, []).append(row['Food_Name'])
    
    # Default config parameters
    parameters = {
        'land_availability': {
            'Farm1': 50, 'Farm2': 75, 'Farm3': 100, 'Farm4': 80, 'Farm5': 50
        },
        'social_benefit': {
            'Farm1': 0.20,   
            'Farm2': 0.25,   
            'Farm3': 0.15,
            'Farm4': 0.20,
            'Farm5': 0.10,
        },
        'minimum_planting_area': {
            "Mango": 0.000929, "Papaya": 0.000400, "Orange": 0.005810, "Banana": 0.005950,
            "Guava": 0.000929, "Watermelon": 0.000334, "Apple": 0.003720, "Avocado": 0.008360,
            "Durian": 0.010000, "Corn": 0.000183, "Potato": 0.000090, "Tofu": 0.000010,
            "Tempeh": 0.000010, "Peanuts": 0.000030, "Chickpeas": 0.000020, "Pumpkin": 0.000100,
            "Spinach": 0.000090, "Tomatoes": 0.000105, "Long bean": 0.000090, "Cabbage": 0.000250,
            "Eggplant": 0.000360, "Cucumber": 0.000500, "Egg": 0.000019, "Beef": 0.728400,
            "Lamb": 0.025000, "Pork": 0.016200, "Chicken": 0.001000
        },
        'max_percentage_per_crop': {
            "Mango": 1.0, "Papaya": 1.0, "Orange": 1.0, "Banana": 1.0,
            "Guava": 1.0, "Watermelon": 1.0, "Apple": 1.0, "Avocado": 1.0,
            "Durian": 1.0, "Corn": 1.0, "Potato": 1.0, "Tofu": 1.0,
            "Tempeh": 1.0, "Peanuts": 1.0, "Chickpeas": 0.5, "Pumpkin": 1.0,
            "Spinach": 1.0, "Tomatoes": 1.0, "Long bean": 0.10,
            "Cabbage": 1.0, "Eggplant": 1.0, "Cucumber": 1.0, "Egg": 1.0,
            "Beef": 1.0, "Lamb": 1.0, "Pork": 1.0, "Chicken": 1.0
        },
        'food_group_constraints': {
            g: {'min_foods': 1, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    
    # Add solver settings to config
    config = {
        'parameters': parameters,
        'benders_tolerance': 1e-3,  # Tighter convergence tolerance
        'benders_max_iterations': 100,  # More iterations allowed
        'pulp_time_limit': 120,  # 2 minutes time limit for PuLP
        'use_multi_cut': True,  # Enable multi-cut Benders
        'use_trust_region': True,  # Enable trust region stabilization
        'use_anticycling': True,  # Enable anti-cycling measures
        'use_norm_cuts': True,  # Use normalized optimality cuts
        'quantum_settings': {  # Added quantum-specific settings
            'max_qubits': 20,
            'use_qaoa_squared': True,
            'force_qaoa_squared': True
        }
    }

    logger.info(f"Loaded full data for {len(farms)} farms and {len(foods)} foods. Parameters generated.")
    
    return farms, foods, food_groups, config

# Test scenario output
if __name__ == "__main__":
    complexity = 'intermediate'
    farms, foods, food_groups, config = load_food_data(complexity)
    
    # Display scenario details
    num_farms = len(farms)
    num_foods = len(foods)
    problem_size = "N/A"  # Placeholder for problem size
    print(f"Scenario Details:")
    print(f"  Farms: {num_farms} ({farms})")
    print(f"  Foods: {num_foods} ({list(foods.keys())})")
    print(f"  Problem Size: {problem_size} variables")