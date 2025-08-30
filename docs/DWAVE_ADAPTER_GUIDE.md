# Guide: Running the D-Wave Test Scenario & Extracting the BQM

## 1. Introduction

This guide explains how to run the example D-Wave test script and details the process of converting a problem into a `BinaryQuadraticModel` (BQM), which is the standard input format for D-Wave solvers.

The primary script for this showcase is `Tests/test_dwave_cost_estimation.py`. It demonstrates a complete workflow from loading a problem scenario to solving it with a D-Wave sampler (in this case, a local simulator).

1. Configuration: Forcing the Simulator

  In each test function (e.g., test_simple_complexity), a DWaveConfig object is created with a key parameter:

   1 # Inside Tests/test_dwave_cost_estimation.py
   2 
   3 dwave_config = DWaveConfig(
   4     solver_type='simulator',  # <--- THIS IS THE KEY
   5     num_reads=500,
   6     estimate_cost_only=False
   7 )
   8 adapter = DWaveQPUAdapter(config=dwave_config, logger=logger)   

   - solver_type='simulator': This explicitly tells the DWaveQPUAdapter to use a classical, CPU-based simulator instead of
     trying to connect to a real D-Wave QPU or a cloud-based hybrid solver.

  2. Adapter Initialization

  The DWaveQPUAdapter (in my_functions/dwave_qpu_adapter.py) reads this configuration. Inside its _initialize_samplers
  method, it sees solver_type='simulator' and selects the appropriate sampler.

    1 # Inside my_functions/dwave_qpu_adapter.py
    2 
    3 def _initialize_samplers(self):
    4     # ...
    5     # Always initialize simulated annealing sampler (CPU-based)
    6     self.sim_sampler = SimulatedAnnealingSampler() # <--- THIS IS THE SIMULATOR
    7     self.logger.info("✓ Initialized simulated annealing sampler (CPU)")
    8     # ...
    9     # Select active sampler based on configuration
   10     self._select_active_sampler()
   11 
   12 def _select_active_sampler(self):
   13     # ...
   14     elif self.config.solver_type == 'simulator':
   15         self.active_sampler = self.sim_sampler # <--- IT GETS SELECTED HERE
   16         self.sampler_type = 'simulator'
   17     # ...

  The dimod.SimulatedAnnealingSampler is a classical algorithm that finds low-energy states for QUBO/BQM problems. It
  mimics the physical process of annealing but runs entirely on your local CPU.

  3. Problem Formulation (QUBO to BQM)

  The test script first creates a QUBO matrix, which is a NumPy array representing the optimization problem.

   1 # Inside Tests/test_dwave_cost_estimation.py
   2 qubo_matrix = create_food_optimization_qubo(farms, foods, config)

  This matrix is then passed to the adapter to be converted into a BinaryQuadraticModel (BQM) object. This is the standard 
  format used by D-Wave's dimod library.

   1 # Inside Tests/test_dwave_cost_estimation.py
   2 bqm = adapter.create_bqm_from_qubo(qubo_matrix)

  4. Solving with the Simulator

  The core of the simulation happens when adapter._solve_bqm(bqm) is called.

    1 # Inside my_functions/dwave_qpu_adapter.py
    2 
    3 def _solve_bqm(self, bqm: BinaryQuadraticModel) -> Dict[str, Any]:
    4     # ...
    5     # It prepares sampler_params for the 'simulator'
    6     sampler_params = {
    7         'num_reads': min(self.config.num_reads, 1000),
    8         'beta_range': [0.1, 10.0],
    9         'num_sweeps': 1000,
   10         'seed': None
   11     }
   12     # ...
   13     # It calls the sample() method on the active sampler, which is the simulator
   14     sampleset = self.active_sampler.sample(bqm, **sampler_params)
   15     # ...
   16     return { ... } # returns the results

   - `self.active_sampler.sample(bqm, ...)`: This is where the magic happens. The SimulatedAnnealingSampler takes the BQM and
     runs a classical simulated annealing algorithm.
   - Simulated Annealing (SA): This is a probabilistic heuristic. It starts with a random solution and iteratively tries to  
     improve it by making small changes (flipping bits). It has a "temperature" parameter that starts high, allowing it to   
     accept "bad" moves occasionally to escape local minima. As the temperature slowly decreases, it becomes less likely to  
     accept bad moves, eventually settling into a low-energy (good) solution.
   - `num_reads`: This parameter tells the simulator how many independent simulated annealing runs to perform. It returns the
     results from all these runs, from which the script selects the one with the lowest energy.

  Summary

  In short, the local simulator in this script does not simulate quantum mechanics. Instead, it uses a well-known classical 
  heuristic algorithm (Simulated Annealing) to find good solutions for the same type of problem (BQM/QUBO) that a quantum   
  annealer would solve. This allows for rapid, free, and offline testing of the entire problem-solving workflow before      
  sending it to actual quantum hardware.


## 2. How to Run the Test

To run the test, execute the following command from the root directory of the project (`Full Optimizer/`):

```bash
python Tests/test_test_dwave_cost_estimation.py
```

The script will print its progress to the console, showing the results for 'simple', 'intermediate', and 'full' complexity scenarios.

## 3. What the Script Does: Step-by-Step

The `test_dwave_cost_estimation.py` script performs the following key steps:

1.  **Load Scenario Data**: It imports the `load_food_data` function from `src/scenarios.py` to get the problem definition (farms, foods, weights, constraints) for a specific complexity level.

2.  **Create a QUBO Matrix**: It defines a helper function `create_food_optimization_qubo` that takes the scenario data and formulates it as a **QUBO (Quadratic Unconstrained Binary Optimization)** matrix. This matrix (`Q`) mathematically represents the optimization problem where the goal is to find a binary vector `x` that minimizes `x^T * Q * x`.

3.  **Initialize the D-Wave Adapter**: It creates an instance of the `DWaveQPUAdapter` class from `my_functions/dwave_qpu_adapter.py`. This adapter is the main interface to the D-Wave ecosystem.

4.  **Create the BQM**: The adapter's `create_bqm_from_qubo` method is called. This is the crucial step where the NumPy QUBO matrix is converted into a `dimod.BinaryQuadraticModel` (BQM) object. The BQM is the standardized format that all D-Wave samplers understand.

5.  **Solve the BQM**: The adapter's internal `_solve_bqm` method is called, which passes the BQM object to the configured D-Wave sampler (in this case, `dimod.SimulatedAnnealingSampler`).

6.  **Process Results**: The script receives the results (lowest energy found, corresponding solution) and prints them to the console.

## 4. How to Extract the Binary Quadratic Model (BQM)

The primary goal is often to get the BQM object itself for use with D-Wave's hardware or other tools. The conversion from a QUBO matrix to a BQM happens inside the `DWaveQPUAdapter`.

Here’s how you can access it:

1.  **Locate the Key Function**: Open the file `my_functions/dwave_qpu_adapter.py`. The key function is `create_bqm_from_qubo`.

    ```python
    # Inside my_functions/dwave_qpu_adapter.py

    def create_bqm_from_qubo(self, Q_matrix: np.ndarray, 
                            offset: float = 0.0,
                            variable_labels: Optional[List[str]] = None) -> BinaryQuadraticModel:
        """
        Create a D-Wave BinaryQuadraticModel from a QUBO matrix.
        ...
        """
        # ... function logic ...
        
        bqm = BinaryQuadraticModel(vartype='BINARY')
        
        # ... logic to add variables and interactions ...
        
        self.logger.debug(f"Created BQM with {len(bqm.variables)} variables...")
        
        return bqm # <-- The BQM object is returned here
    ```

2.  **Intercept the BQM Object**: To get the BQM, you can modify the test script (`Tests/test_dwave_cost_estimation.py`) to capture and inspect the BQM after it's created.

    In `test_dwave_cost_estimation.py`, find the line where the BQM is created. It's inside the main test functions, like `test_simple_complexity`:

    ```python
    # Inside Tests/test_dwave_cost_estimation.py

    # ... inside a test function ...
    qubo_matrix = create_food_optimization_qubo(farms, foods, config)
    bqm = adapter.create_bqm_from_qubo(qubo_matrix) # <-- BQM is created here
    
    # Add the following lines to inspect or save the BQM
    print("\n--- BQM Details ---")
    print(bqm)
    print("-------------------\n")

    # To save it to a file for later use:
    # with open("simple_scenario.bqm", "wb") as f:
    #     f.write(bqm.to_file().read())

    print("\n--- Solving simple problem ---")
    result = adapter._solve_bqm(bqm)
    # ...
    ```

By adding the `print(bqm)` statement, your colleague can see the BQM structure directly in the console output. Saving it to a file allows it to be loaded and used by other D-Wave tools.
