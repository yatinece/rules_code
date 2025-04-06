import pandas as pd
import numpy as np
import time

base_dt=pd.read_csv("./creditcardfraud/creditcard.csv")
print(base_dt.columns)

import numpy as np
import pandas as pd
import pygad

# Suppose X is your DataFrame of features and y is your binary target (fraud=1)
# Convert all columns except 'Time' and 'Class' to features (X)

X = base_dt.drop(columns=['Time', 'Class']).astype(float)  # convert features to float
y = base_dt['Class'].astype(int)  # assuming Class is 0/1

print(f"current fraud rate is {y.mean()}")
import numpy as np
import pandas as pd
import pygad

# Assuming X is your DataFrame of features and y is your binary target (fraud=1)

# Define the number of candidate features
num_features = X.shape[1]

# Maximum number of features to include in the rule
max_features = 3

# Desired minimum fraud rate in the selected subset
desired_fraud_rate = 0.001

# Chromosome length: num_features for selection + num_features for thresholds
chromosome_length = num_features * 2
print(f"num_features are {num_features} , max_features are {max_features} , desired_fraud_rate are {desired_fraud_rate}")

def fitness_func(ga_instance, solution, solution_idx):
    # Decode the chromosome
    selection = solution[:num_features] > 0.5  # Binary selection
    thresholds = solution[num_features:]       # Continuous thresholds
    
    # Enforce maximum feature constraint
    if np.sum(selection) > max_features or np.sum(selection) == 0:
        return -1e6  # Heavy penalty
    
    # Apply the rule to filter the dataset
    mask = np.ones(len(X), dtype=bool)
    for i, selected in enumerate(selection):
        if selected:
            mask &= X.iloc[:, i] > thresholds[i]
    
    subset = y[mask]
    if len(subset) == 0:
        return -1e6  # Heavy penalty for empty subset
    
    fraud_rate = subset.mean()  # Calculate fraud rate
    
    # Penalize if fraud rate is below the desired threshold
    if fraud_rate < desired_fraud_rate:
        return -1e6  # Heavy penalty
    
    # Reward based on fraud rate and subset size
    reward = fraud_rate * len(subset)
    return reward

# Genetic Algorithm Configuration
ga_instance = pygad.GA(num_generations=10000,
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

 # Save timestamp
start = time.time()
# Run GA
ga_instance.run()
    # Save timestamp
end = time.time()

print(f"time taken: {end - start}")

# Retrieve the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
selection = solution[:num_features] > 0.5
thresholds = solution[num_features:]
selected_features = X.columns[selection]
rule = {feat: thresholds[i] for i, feat in enumerate(X.columns) if selection[i]}

# Evaluate the final rule
mask = np.ones(len(X), dtype=bool)
for feat, thresh in rule.items():
    mask &= X[feat] > thresh
subset = y[mask]
final_fraud_rate = subset.mean() if len(subset) > 0 else 0

print("Optimized Composite Rule:")
for feat, thresh in rule.items():
    print(f"  {feat} > {thresh:.4f}")
print(f"Fraud rate in selected subset: {final_fraud_rate:.4%} over {np.sum(mask)} samples and total fraud is {np.sum(mask)*final_fraud_rate}")





