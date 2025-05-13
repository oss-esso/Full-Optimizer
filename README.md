# OQI Food Production Optimization

This repository contains a Python-based implementation of a food production optimization model using Benders decomposition, PuLP, and quantum-inspired Benders methods.

## Project Structure

- `src/`: Main source code for optimization models.
- `tests/`: Unit and integration tests.
- `examples/`: Example scripts and notebooks.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Examples

```powershell
python src/main.py
```

## Project Organization

We have refactored the monolithic script into separate modules:

- `src/optimizer.py`: Core `FoodProductionOptimizer` class.
- `src/methods/benders_method.py`: Benders decomposition implementation.
- `src/methods/pulp_method.py`: PuLP implementation.
- `src/methods/quantum_methods.py`: Quantum-inspired and QAOA-based methods.
- `src/utils/plotting.py`: Plotting and visualization utilities.
- `src/utils/logging.py`: Logging setup.

## Contributing

Contributions are welcome via pull requests.
