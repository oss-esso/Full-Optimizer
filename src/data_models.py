import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any

class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    NUTRITIONAL_VALUE = "nutritional_value"
    NUTRIENT_DENSITY = "nutrient_density"
    ENVIRONMENTAL_IMPACT = "environmental_impact"
    AFFORDABILITY = "affordability"
    SUSTAINABILITY = "sustainability"

@dataclass
class OptimizationResult:
    """Class for storing optimization results."""
    status: str
    objective_value: float
    solution: Dict[Tuple[str, str], float]
    metrics: Dict[str, float]
    runtime: float
    benders_data: Dict = field(default_factory=dict)
    quantum_metrics: Dict = field(default_factory=dict) 