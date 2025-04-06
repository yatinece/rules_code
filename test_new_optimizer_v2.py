import pandas as pd
import numpy as np
import time
import pygad

# -----------------------
# Load and prepare the dataset
# -----------------------
base_dt = pd.read_csv("./creditcardfraud/creditcard.csv")
print("Columns:", base_dt.columns)

# Create a working copy of the dataset.
working_df = base_dt.copy()

# Define candidate feature columns (exclude 'Time' and 'Class')
feature_cols = working_df.drop(columns=['Time', 'Class']).columns.tolist()

# -----------------------
# Global parameters for GA-based rule generation
# -----------------------
num_features = len(feature_cols)
max_features = 7  # Maximum number of features allowed in each composite rule
# Set desired fraud amount threshold for a rule to be considered useful.
desired_fraud_amount = 1000.0  # Adjust as needed

# For GA encoding, each chromosome will have:
# - num_features genes for feature selection (binary: >0.5 means selected)
# - num_features genes for thresholds (continuous in [0,1])
chromosome_length = num_features * 2
print(f"Number of candidate features: {num_features}, Max features allowed: {max_features}, Desired fraud amount: {desired_fraud_amount}")

# -----------------------
# Normalization function
# -----------------------
def normalize_features(df):
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    return df_norm

# -----------------------
# Fitness function for GA (evaluates on the current working_df)
# -----------------------
def fitness_func(ga_instance, solution, solution_idx):
    global working_df
    # Normalize candidate features in the current working_df.
    X_current = normalize_features(working_df[feature_cols])
    y_current = working_df['Class']
    
    # Decode chromosome: first part for selection, second part for thresholds.
    raw_selection = solution[:num_features]
    selection = raw_selection > 0.5
    thresholds = solution[num_features:]
    
    # Repair: if more than max_features are selected, keep only the top max_features genes.
    if np.sum(selection) > max_features:
        sorted_idx = np.argsort(-raw_selection)
        new_selection = np.zeros_like(selection, dtype=bool)
        new_selection[sorted_idx[:max_features]] = True
        selection = new_selection
    # If no feature is selected, force select the top one.
    if np.sum(selection) == 0:
        sorted_idx = np.argsort(-raw_selection)
        selection[sorted_idx[0]] = True

    # Clamp thresholds to [0,1]
    thresholds = np.clip(thresholds, 0.0, 1.0)
    
    # Apply composite rule: For each selected feature, require X_current[feature] > threshold.
    mask = np.ones(len(X_current), dtype=bool)
    for i, sel in enumerate(selection):
        if sel:
            mask &= (X_current.iloc[:, i] > thresholds[i])
    
    subset = working_df[mask]
    if len(subset) == 0:
        return -1e6  # Heavy penalty if rule selects no samples
    
    # Compute total fraud amount from the original data in the selected subset.
    fraud_amount = subset.loc[subset['Class'] == 1, "Amount"].sum()
    
    if fraud_amount < desired_fraud_amount:
        return -1e6  # Heavy penalty if fraud amount is too low
    
    # Reward is simply the total fraud amount (you can combine with subset size if desired)
    reward = fraud_amount
    return reward

# -----------------------
# GA configuration function (runs GA for one rule search)
# -----------------------
def run_ga_for_rule():
    ga_inst = pygad.GA(num_generations=100,
                       num_parents_mating=20,
                       fitness_func=fitness_func,
                       sol_per_pop=50,
                       num_genes=chromosome_length,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       mutation_percent_genes=10,
                       parent_selection_type="sss",
                       crossover_type="single_point",
                       mutation_type="random",
                       # Note: Updated stop_criteria with numeric value (as a string, but parsed as number)
                       stop_criteria=["reach_1000000"])  
    ga_inst.run()
    best_solution, best_solution_fitness, best_solution_idx = ga_inst.best_solution()
    return best_solution, best_solution_fitness

# -----------------------
# Loop: Generate rules, remove samples covered, and repeat until no valid rule is found.
# -----------------------
rules = []  # To store discovered rules
iteration = 1

while True:
    print(f"\nIteration {iteration}: Working dataset size: {len(working_df)} samples")
    
    best_solution, best_fitness = run_ga_for_rule()
    print(f"Best fitness in iteration {iteration}: {best_fitness:.2f}")
    
    # Decode the best solution to extract the rule.
    raw_selection = best_solution[:num_features]
    selection = raw_selection > 0.5
    if np.sum(selection) > max_features:
        sorted_idx = np.argsort(-raw_selection)
        new_selection = np.zeros_like(selection, dtype=bool)
        new_selection[sorted_idx[:max_features]] = True
        selection = new_selection
    if np.sum(selection) == 0:
        sorted_idx = np.argsort(-raw_selection)
        selection[sorted_idx[0]] = True

    thresholds = np.clip(best_solution[num_features:], 0.0, 1.0)
    rule = {feat: thresholds[i] for i, feat in enumerate(feature_cols) if selection[i]}
    
    # Evaluate rule on current working_df
    X_current = normalize_features(working_df[feature_cols])
    mask = np.ones(len(X_current), dtype=bool)
    for i, feat in enumerate(feature_cols):
        if selection[i]:
            mask &= (X_current[feat] > thresholds[i])
    
    subset = working_df[mask]
    fraud_amount = subset.loc[subset['Class'] == 1, "Amount"].sum()
    
    print("Derived Composite Rule:")
    for feat, thresh in rule.items():
        print(f"  if {feat} > {thresh:.4f}")
    print(f"Fraud amount in selected subset: {fraud_amount:.2f} over {len(subset)} samples")
    
    # If no valid rule is found, break.
    if len(subset) == 0 or fraud_amount < desired_fraud_amount:
        print("No valid rule found. Stopping iterations.")
        break
    
    # Save discovered rule.
    rules.append((rule, fraud_amount, len(subset)))
    
    # Remove samples covered by this rule.
    working_df = working_df[~mask].reset_index(drop=True)
    
    # If working dataset is too small, stop.
    if len(working_df) < 100:
        print("Remaining dataset too small. Stopping iterations.")
        break
    
    iteration += 1

# Print all discovered rules.
print("\nDiscovered Rules:")
for idx, (r, amt, cnt) in enumerate(rules, start=1):
    print(f"Rule {idx}:")
    for feat, thresh in r.items():
        print(f"  if {feat} > {thresh:.4f}")
    print(f"  --> Fraud amount: {amt:.2f} from {cnt} samples\n")
