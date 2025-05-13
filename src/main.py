from src.optimizer import FoodProductionOptimizer, SimpleFoodOptimizer


def main():
    optimizer = SimpleFoodOptimizer()
    optimizer.load_food_data()

    # Solve with Benders
    benders_res = optimizer.optimize_with_benders()
    print("Benders Result:", benders_res)

    # Solve with Quantum-Inspired Benders
    qi_res = optimizer.optimize_with_quantum_inspired_benders()
    print("Quantum-Inspired Benders Result:", qi_res)

    # Solve with Quantum-Enhanced Benders
    q_res = optimizer.optimize_with_quantum_benders()
    print("Quantum-Enhanced Benders Result:", q_res)

    # Solve with PuLP
    pulp_res = optimizer.optimize_with_pulp()
    print("PuLP Result:", pulp_res)


if __name__ == '__main__':
    main()
