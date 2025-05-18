@echo off
cd /d "G:\Il mio Drive\OQI_Project\Full Optimizer"
python Tests/test_configuration.py --methods all --runs 10 --scenarios large --quantum_params max_qubits=25,force_qaoa_squared=True 