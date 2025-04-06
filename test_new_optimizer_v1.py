import pandas as pd
import numpy as np
import time
import pygad

# Load dataset
base_dt = pd.read_csv("./creditcardfraud/creditcard.csv")
print("Columns:", base_dt.columns)

# Prepare features (X) and target (y)
# Drop 'Time' and 'Class' from features and convert to float
X = base_dt.drop(columns=['Time', 'Class']).astype(float)
y = base_dt['Class'].astype(int)

print(f"Current global fraud rate is: {y.mean():.4%}")

# Normalize X so that each feature is scaled to [0,1]
X_norm = (X - X.min()) / (X.max() - X.min())
X = X_norm

# Genetic algorithm parameters
num_features = X.shape[1]  # candidate features count
max_features = 7           # maximum number of features allowed in composite rule
# Set desired fraud rate to 0.5% (0.005) to be attainable given global rate ~0.17%
desired_fraud_rate = 0.005

# Chromosome encoding: first num_features for selection (binary), then num_features for thresholds.
chromosome_length = num_features * 2
print(f"Number of candidate features: {num_features}, Max features allowed: {max_features}, Desired fraud rate: {desired_fraud_rate}")

def fitness_func(ga_instance, solution, solution_idx):
    # Decode chromosome:
    # Selection part: use gene value > 0.5 as a candidate for being selected.
    raw_selection = solution[:num_features]
    selection = raw_selection > 0.5  
    thresholds = solution[num_features:]
    
    # Repair: if more than max_features are selected, keep only the top max_features genes.
    if np.sum(selection) > max_features:
        # Sort indices by their raw gene value (descending)
        sorted_idx = np.argsort(-raw_selection)
        new_selection = np.zeros_like(selection, dtype=bool)
        new_selection[sorted_idx[:max_features]] = True
        selection = new_selection
        
    # Also, if no feature is selected, force select the top one.
    if np.sum(selection) == 0:
        sorted_idx = np.argsort(-raw_selection)
        selection[sorted_idx[0]] = True

    # Clamp thresholds to [0, 1]
    thresholds = np.clip(thresholds, 0.0, 1.0)
    
    # Apply the composite rule: For each selected feature, require X[feature] > threshold.
    mask = np.ones(len(X), dtype=bool)
    for i, sel in enumerate(selection):
        if sel:
            mask &= (X.iloc[:, i] > thresholds[i])
    
    subset = y[mask]
    if len(subset) == 0:
        return -1e6  # penalize if rule selects no samples
    
    fraud_rate = subset.mean()  # fraction of frauds in subset

    # Penalize if fraud rate is below desired threshold
    if fraud_rate < desired_fraud_rate:
        return -1e6

    # Reward: combination of fraud rate and size of subset
    reward = fraud_rate * len(subset)
    return reward

# Genetic Algorithm Configuration
ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=20,
                       fitness_func=fitness_func,
                       sol_per_pop=50,
                       num_genes=chromosome_length,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=10,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random")

start_time = time.time()
ga_instance.run()
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# Retrieve the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
raw_selection = solution[:num_features]
selection = raw_selection > 0.5
# Repair solution if needed:
if np.sum(selection) > max_features:
    sorted_idx = np.argsort(-raw_selection)
    new_selection = np.zeros_like(selection, dtype=bool)
    new_selection[sorted_idx[:max_features]] = True
    selection = new_selection

thresholds = np.clip(solution[num_features:], 0.0, 1.0)
selected_features = X.columns[selection]
rule = {feat: thresholds[i] for i, feat in enumerate(X.columns) if selection[i]}

# Evaluate final rule
mask = np.ones(len(X), dtype=bool)
for feat, thresh in rule.items():
    mask &= (X[feat] > thresh)
subset = y[mask]
final_fraud_rate = subset.mean() if len(subset) > 0 else 0

print("Optimized Composite Rule:")
for feat, thresh in rule.items():
    print(f"  if {feat} > {thresh:.4f}")
print(f"Fraud rate in selected subset: {final_fraud_rate:.4%} over {np.sum(mask)} samples")
if len(subset) > 0:
    print(f"Estimated fraud count in subset: {np.sum(mask)*final_fraud_rate:.0f}")
else:
    print("No samples selected by the rule.")
