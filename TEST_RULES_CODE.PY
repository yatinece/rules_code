import time



import numpy as np
import pygad

# -------------------------------
# Data Simulation (Vectorized)
# -------------------------------
np.random.seed(42)
n_transactions = 10**6      # 1 million transactions
n_features = [1,3,5,10,20,30,40,50,60,70,80,90,100]            # 100 features
fraud_rate = 0.05           # 5% fraud

for n_features in n_features:
    # Save timestamp
    start = time.time()
    # Binary fraud labels
    fraud = (np.random.rand(n_transactions) < fraud_rate).astype(int)

    # Simulate real-valued feature data
    features = np.random.randn(n_transactions, n_features)

    # Create binary flag matrix: flag = 1 if feature > 0
    flags = (features > 0).astype(int)

    # Boost fraud rows with additional flagging chance
    fraud_indices = np.where(fraud == 1)[0]
    extra_flags = (np.random.rand(len(fraud_indices), n_features) < 0.3).astype(int)
    flags[fraud_indices] = np.maximum(flags[fraud_indices], extra_flags)

    # -------------------------------
    # Fitness Function for PyGAD 2.20+
    # -------------------------------
    def fitness_func(ga_instance, solution, solution_idx):
        if np.sum(solution) == 0:
            return -1e6

        selected = np.where(solution == 1)[0]
        flagged = np.any(flags[:, selected] == 1, axis=1)

        TP = np.sum(flagged & (fraud == 1))
        FP = np.sum(flagged & (fraud == 0))

        if TP == 0 or (FP / TP < 5):
            return -1e6  # Penalize if no fraud caught or FP/TP > 5

        return TP  # Maximize frauds captured

    # -------------------------------
    # Run the Genetic Algorithm
    # -------------------------------
    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=30,
        num_genes=n_features,
        gene_type=int,
        gene_space=[0, 1],
        mutation_percent_genes=10,
        mutation_type="random",
        keep_elitism=2,
        random_seed=42,
        suppress_warnings=True
    )

    ga_instance.run()

    # -------------------------------
    # Evaluate Best Solution
    # -------------------------------
    solution, fitness, _ = ga_instance.best_solution()
    selected_features = np.where(solution == 1)[0]
    flagged = np.any(flags[:, selected_features] == 1, axis=1)

    TP = np.sum(flagged & (fraud == 1))
    FP = np.sum(flagged & (fraud == 0))
    FP_TP_ratio = FP / TP if TP > 0 else np.inf

    print("\n===== Best Solution =====")
    print("Selected features:", selected_features.tolist())
    print("Number of selected features:", len(selected_features))
    print("True Positives (Fraud flagged):", TP)
    print("False Positives:", FP)
    print("FP/TP ratio:", round(FP_TP_ratio, 2))
    print("Fitness (Fraud captured):", fitness)


    # Save timestamp
    end = time.time()

    print(f"for fraud rate {fraud_rate} and {n_features} features, time taken: {end - start}")