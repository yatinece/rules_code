import numpy as np
import pandas as pd
import time

# Load and prepare the dataset
base_dt = pd.read_csv("./creditcardfraud/creditcard.csv")
print("Columns:", base_dt.columns)

# Create a working copy of the dataset
working_df = base_dt.copy()

# Define candidate feature columns (exclude 'Time' and 'Class')
feature_cols = working_df.drop(columns=['Time', 'Class']).columns.tolist()

# Global parameters for MBO-based rule generation
num_features = len(feature_cols)
max_features = 7  # Maximum number of features allowed in each composite rule
desired_fraud_amount = 1000.0  # Adjust as needed

# Normalize features
def normalize_features(df):
    df_norm = df.copy()
    for col in feature_cols:
        df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
    return df_norm

# Fitness function for MBO
def fitness_func(solution):
    global working_df
    X_current = normalize_features(working_df[feature_cols])
    y_current = working_df['Class']
    
    selection = solution[:num_features] > 0.5
    thresholds = np.clip(solution[num_features:], 0.0, 1.0)
    
    if np.sum(selection) > max_features:
        sorted_idx = np.argsort(-solution[:num_features])
        selection = np.zeros_like(selection, dtype=bool)
        selection[sorted_idx[:max_features]] = True
    if np.sum(selection) == 0:
        selection[np.argmax(solution[:num_features])] = True

    mask = np.ones(len(X_current), dtype=bool)
    for i, sel in enumerate(selection):
        if sel:
            mask &= (X_current.iloc[:, i] > thresholds[i])
    
    subset = working_df[mask]
    if len(subset) == 0:
        return -1e6
    
    fraud_amount = subset.loc[subset['Class'] == 1, "Amount"].sum()
    if fraud_amount < desired_fraud_amount:
        return -1e6
    
    return fraud_amount

# MBO algorithm
def migrating_birds_optimization(num_birds, num_iterations):
    # Initialize the flock
    flock = np.random.rand(num_birds, num_features * 2)
    best_solution = None
    best_fitness = -np.inf
    
    for iteration in range(num_iterations):
        for i in range(num_birds):
            current_solution = flock[i]
            current_fitness = fitness_func(current_solution)
            
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution
            
            # Migrate: share information with neighboring birds
            left_neighbor = flock[(i - 1) % num_birds]
            right_neighbor = flock[(i + 1) % num_birds]
            new_solution = (current_solution + left_neighbor + right_neighbor) / 3
            
            # Mutation: introduce some randomness
            mutation = np.random.normal(0, 0.1, size=new_solution.shape)
            new_solution += mutation
            
            flock[i] = np.clip(new_solution, 0.0, 1.0)
        
        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness:.2f}")
    
    return best_solution, best_fitness

# Run the MBO algorithm
best_solution, best_fitness = migrating_birds_optimization(num_birds=50, num_iterations=100)
print(f"Best solution found with fitness: {best_fitness:.2f}")
