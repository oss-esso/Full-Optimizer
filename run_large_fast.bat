@echo off
echo Running large scenario tests with fast methods only (NO quantum-inspired)
python Tests\test_configuration.py --methods pulp,benders,quantum-enhanced,quantum-enhanced-merge --runs 25 --scenarios large --decomposition_detail --quantum_params max_qubits=25,force_qaoa_squared=True
echo Test completed!
pause 