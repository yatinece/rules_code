import pandas as pd
import numpy as np
import time
import pygad
import optuna
import os
from datetime import datetime
import csv
import torch 

start_time = time.time()
# Original configuration - will be modified by Optuna
BASE_CONFIG = {
    # Dataset Configuration
    'dataset_path': "../creditcardfraud/creditcard.csv",
    'new_study_only': True,
    # Rule Discovery Parameters
    'max_rules_per_run': 8,          # Maximum number of rules to discover per run
    'min_fraud_rate': 0.01,         # Minimum acceptable fraud rate (1%)
    'min_subset_size_absolute': 50,  # Minimum number of transactions in a subset
    'min_subset_size_percent': 0.005, # Minimum subset size as percent of data (0.5%)
    'max_iterations': 12,            # Maximum iterations per run
    'num_runs': 1,                   # Number of runs with different random seeds
    
    # Genetic Algorithm Parameters
    'ga_num_generations': 150,       # Number of generations for genetic algorithm
    'ga_population_size': 80,        # Population size per generation
    'ga_num_parents': 25,            # Number of parents per generation
    'ga_mutation_percent': 15,       # Mutation percentage
    'max_features_per_rule': 10,     # Maximum number of features allowed in a rule
    
    # Rule Diversity Parameters
    'feature_reuse_penalty': 0.3,    # Penalty factor for reusing features (30% per feature)
    'min_threshold_value': 0.001,    # Minimum threshold to consider (to avoid > 0.0000)
    'LESS_LOOKUP_RULE_POSITION' : 3
}



# Ensemble Rule Parameters
BASE_CONFIG['ensemble_min_fraud_rate'] = BASE_CONFIG['min_fraud_rate'] / 2  # 50% of the individual rule threshold

# Load dataset once
base_dt = pd.read_csv(BASE_CONFIG['dataset_path'])
original_dt = base_dt.copy()


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")    
# Create time-based results directory   
def create_results_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"fraud_detection_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
# --- In discover_rules_with_config: Add dataset shuffling and update GA seed ---
def discover_rules_with_config(config):
    """
    Discover fraud detection rules using genetic algorithm optimization.
    
    Returns:
        tuple: (rules, ensemble_rules, fraud_detection_rate)
    """
    # Use a run-specific seed (if provided) to shuffle the dataset
    run_seed = config.get("run_seed", None)
    base_dt = original_dt.copy()
    if run_seed is not None:
        base_dt = base_dt.sample(frac=1, random_state=run_seed)
    
    rules = []
    remaining_samples = len(base_dt)
    remaining_frauds = base_dt['Class'].sum()
    
    # Track previously used features to encourage diversity
    previous_features = set()
    
    # Keep track of the indices corresponding to the original dataset
    original_index = base_dt.index.copy()

    print(f"Starting with {remaining_samples:,} total samples, {remaining_frauds} frauds")
    print(f"Initial global fraud rate: {remaining_frauds/remaining_samples:.4%}")
    
    for iteration in range(config['max_iterations']):
        print(f"\n===== ITERATION {iteration+1} =====")
        
        if remaining_samples < 1000 or remaining_frauds < 5:
            print(f"Stopping: too few remaining samples ({remaining_samples:,}) or frauds ({remaining_frauds})")
            break
        
        # Prepare features (X) and target (y)
        X = base_dt.drop(columns=['Time', 'Class']).astype(float)
        y = base_dt['Class'].astype(int)
        time_values = base_dt['Time'].values
        
        current_fraud_rate = y.mean()
        print(f"Current fraud rate: {current_fraud_rate:.4%}")
        
        if current_fraud_rate < 0.0001:
            print(f"Stopping: fraud rate extremely low ({current_fraud_rate:.4%})")
            break
        
        # Normalize X to [0,1]
        X_norm = (X - X.min()) / (X.max() - X.min())
        X = X_norm
        
        num_features = X.shape[1]
        max_features = config['max_features_per_rule']
        desired_fraud_rate = config['min_fraud_rate']
        
        chromosome_length = num_features * 2
        
        # Use run_seed in addition to iteration for GA seed
        ga_seed = (iteration * 1000) + (run_seed if run_seed is not None else 0)
        
        # Prepare data for GA using PyTorch tensors on the appropriate device
        X_tensor = torch.tensor(X.values, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32, device=device)

        def fitness_func(ga_instance, solution, solution_idx):
            raw_selection = solution[:num_features]
            selection = raw_selection > 0.5
            thresholds = solution[num_features:]
            
            if np.sum(selection) > max_features:
                sorted_idx = np.argsort(-raw_selection)
                selection = np.zeros_like(selection, dtype=bool)
                selection[sorted_idx[:max_features]] = True

            if np.sum(selection) == 0:
                sorted_idx = np.argsort(-raw_selection)
                selection[sorted_idx[0]] = True

            thresholds = np.clip(thresholds, config['min_threshold_value'], 1.0)
            
            # Convert selection and thresholds to GPU tensors
            selection_tensor = torch.tensor(selection, dtype=torch.bool, device=device)
            thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32, device=device)
            
            mask = torch.ones(X_tensor.shape[0], dtype=torch.bool, device=device)
            
            for i in range(num_features):
                if selection_tensor[i]:
                    col = X_tensor[:, i]
                    if i % config["LESS_LOOKUP_RULE_POSITION"] == 0:
                        mask &= col < thresholds_tensor[i]
                    else:
                        mask &= col > thresholds_tensor[i]
            
            subset_y = y_tensor[mask]
            subset_len = subset_y.shape[0]
            
            if subset_len == 0:
                return -1e6

            fraud_rate = torch.mean(subset_y)
            fraud_count = torch.sum(subset_y)

            min_subset_size = max(config['min_subset_size_absolute'], len(X) * config['min_subset_size_percent'])
            
            if subset_len < min_subset_size or fraud_rate.item() < desired_fraud_rate:
                return -1e6

            feature_reuse_factor = 0
            if previous_features:
                selected_feature_names = set([X.columns[i] for i, sel in enumerate(selection) if sel])
                feature_overlap = selected_feature_names.intersection(previous_features)
                feature_reuse_factor = len(feature_overlap) / len(selected_feature_names) if selected_feature_names else 0

            reward = torch.sqrt(fraud_rate) * fraud_count
            reward = reward * (1 - feature_reuse_factor * config['feature_reuse_penalty'])

            return reward.item()

        ga_instance = pygad.GA(num_generations=config['ga_num_generations'],
                             num_parents_mating=config['ga_num_parents'],
                             fitness_func=fitness_func,
                             sol_per_pop=config['ga_population_size'],
                             num_genes=chromosome_length,
                             init_range_low=0.0,
                             init_range_high=1.0,
                             mutation_percent_genes=config['ga_mutation_percent'],
                             parent_selection_type="sss",
                             crossover_type="single_point",
                             mutation_type="random",
                             random_seed=ga_seed)
        
        start_ga = time.time()
        ga_instance.run()
        end_ga = time.time()
        print(f"GA run time: {end_ga - start_ga:.2f} seconds")
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        if solution_fitness <= 0:
            print("No useful rule found in this iteration. Stopping.")
            break
            
        raw_selection = solution[:num_features]
        selection = raw_selection > 0.5
        if np.sum(selection) > max_features:
            sorted_idx = np.argsort(-raw_selection)
            selection = np.zeros_like(selection, dtype=bool)
            selection[sorted_idx[:max_features]] = True
        
        thresholds = np.clip(solution[num_features:], config['min_threshold_value'], 1.0)
        selected_features = X.columns[selection]
        rule = {}
        
        for i, (feat, sel) in enumerate(zip(X.columns, selection)):
            if sel:
                threshold_value = thresholds[i]
                if threshold_value <= config['min_threshold_value']:
                    continue
                if i % config["LESS_LOOKUP_RULE_POSITION"] == 0:
                    rule[f"{feat} <"] = threshold_value
                else:
                    rule[f"{feat} >"] = threshold_value
        
        mask = np.ones(len(X), dtype=bool)
        for feat, thresh in rule.items():
            if " <" in feat:
                actual_feat = feat.replace(" <", "")
                mask &= (X[actual_feat] < thresh)
            else:
                actual_feat = feat.replace(" >", "")
                mask &= (X[actual_feat] > thresh)
        
        subset = y[mask]
        subset_size = len(subset)
        
        if subset_size == 0:
            print("Rule selects no samples. Skipping this iteration.")
            continue
            
        final_fraud_rate = subset.mean() 
        fraud_count = subset.sum()
        
        if final_fraud_rate < config['min_fraud_rate'] * 0.8 or subset_size < config['min_subset_size_absolute']:
            print(f"Rule doesn't meet criteria: fraud rate = {final_fraud_rate:.4%}, subset size = {subset_size}")
            continue
        
        rule_indices = original_index[mask]
        
        # Update previous features for diversity
        for feat_key in rule.keys():
            feat_name = feat_key.split(" ")[0]
            previous_features.add(feat_name)
        
        rule_stats = {
            'iteration': iteration + 1,
            'rule': rule,
            'subset_size': subset_size,
            'fraud_rate': final_fraud_rate,
            'fraud_count': fraud_count,
            'rule_indices': rule_indices,
            'solution_fitness': solution_fitness
        }
        
        print("\nRule Statistics:")
        print(f"Rule #{iteration + 1}:")
        for feat, thresh in rule.items():
            print(f"  if {feat} {thresh:.4f}")
        print(f"Subset size: {subset_size:,} samples ({subset_size/len(X):.2%} of data)")
        print(f"Fraud rate in subset: {final_fraud_rate:.4%}")
        print(f"Fraud count in subset: {fraud_count}")
        
        rules.append(rule_stats)
        
        # Remove samples covered by this rule to enforce mutual exclusivity
        base_dt = base_dt[~mask].copy()
        original_index = original_index[~mask]
        remaining_samples = len(base_dt)
        remaining_frauds = base_dt['Class'].sum()
        
        print(f"\nRemoved {subset_size} samples, including {fraud_count} frauds")
        print(f"Remaining samples: {remaining_samples:,}, Remaining frauds: {remaining_frauds}")
        
        if len(rules) >= config['max_rules_per_run'] or remaining_frauds == 0:
            break

    # --- Add a final "catch-all" rule for collective exhaustiveness ---
    if len(base_dt) > 0:
        print("\n===== FINAL CATCH-ALL RULE =====")
        catch_all_rule = {
            'iteration': len(rules) + 1,
            'rule': {"Default": "Covers remaining samples"},
            'subset_size': len(base_dt),
            'fraud_rate': base_dt['Class'].mean(),
            'fraud_count': base_dt['Class'].sum(),
            'rule_indices': original_index,
            'solution_fitness': None
        }
        rules.append(catch_all_rule)
        print(f"Final rule covers {len(base_dt)} samples with fraud rate {base_dt['Class'].mean():.4%}")

    # Create ensemble rules (pairwise combinations)
    print("\n===== ENSEMBLE RULES =====")
    ensemble_rules = []
    for i in range(len(rules)):
        for j in range(i+1, len(rules)):
            rule_i_indices = set(rules[i]['rule_indices'])
            rule_j_indices = set(rules[j]['rule_indices'])
            combined_indices = list(rule_i_indices.union(rule_j_indices))
            if combined_indices:
                ensemble_data = original_dt.loc[combined_indices]
                ensemble_fraud_rate = ensemble_data['Class'].mean()
                ensemble_fraud_count = ensemble_data['Class'].sum()
                
                if (ensemble_fraud_count > rules[i]['fraud_count'] and 
                    ensemble_fraud_count > rules[j]['fraud_count'] and
                    ensemble_fraud_rate >= config['ensemble_min_fraud_rate']):
                    
                    ensemble_rules.append({
                        'rule': f"Rule #{i+1} OR Rule #{j+1}",
                        'rule_components': [i+1, j+1],
                        'subset_size': len(combined_indices),
                        'fraud_rate': ensemble_fraud_rate,
                        'fraud_count': ensemble_fraud_count,
                        'rule_indices': combined_indices
                    })
    
    ensemble_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
    
    for i, ensemble in enumerate(ensemble_rules[:5]):
        print(f"\nEnsemble Rule #{i+1}: {ensemble['rule']}")
        print(f"Subset size: {ensemble['subset_size']:,}")
        print(f"Fraud rate: {ensemble['fraud_rate']:.4%}")
        print(f"Fraud count: {ensemble['fraud_count']}")
    
    total_fraud_count = sum(rule['fraud_count'] for rule in rules if rule['fraud_count'] is not None)
    total_dataset_fraud = original_dt['Class'].sum()
    fraud_detection_rate = total_fraud_count / total_dataset_fraud if total_dataset_fraud > 0 else 0
    
    return rules, ensemble_rules, fraud_detection_rate

# Export ensemble rules to CSV (new function)
def export_ensemble_rules_to_csv(ensemble_rules, all_rules, results_dir, filename="ensemble_rules.csv"):
    filepath = os.path.join(results_dir, filename)
    
    # Prepare data for CSV
    csv_data = []
    for i, ensemble in enumerate(ensemble_rules):
        # Get component rule details
        component_rules_str = []
        for rule_num in ensemble['rule_components']:
            # Find the actual rule in all_rules that matches this number
            component_rule = None
            for rule in all_rules:
                if rule['iteration'] == rule_num and rule.get('run', 1) == ensemble.get('run', 1):
                    component_rule = rule
                    break
            
            if component_rule:
                rule_str = "; ".join([f"{feat} {thresh:.4f}" for feat, thresh in component_rule['rule'].items()])
                component_rules_str.append(f"Rule {rule_num}: {rule_str}")
        
        component_details = " | ".join(component_rules_str)
        
        csv_data.append({
            'Ensemble_Rule_Number': i+1,
            'Component_Rules': ensemble['rule'],
            'Component_Details': component_details,
            'Subset_Size': ensemble['subset_size'],
            'Fraud_Rate': ensemble['fraud_rate'],
            'Fraud_Count': ensemble['fraud_count']
        })
    
    # Write to CSV
    if csv_data:
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"Ensemble rules exported to {filepath}")
    else:
        print("No ensemble rules to export")
        
    return filepath

# Define objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    config = {
        # Dataset Configuration - not changing
        'dataset_path': BASE_CONFIG['dataset_path'],
        'min_fraud_rate': BASE_CONFIG['min_fraud_rate'],
        # Ensemble Rule Parameters
        'ensemble_min_fraud_rate': BASE_CONFIG['ensemble_min_fraud_rate'],
        # Rule Discovery Parameters
        'max_rules_per_run': trial.suggest_int('max_rules_per_run', 3, 15),
        
        'min_subset_size_absolute': trial.suggest_int('min_subset_size_absolute', 30, 200),
        'min_subset_size_percent': trial.suggest_float('min_subset_size_percent', 0.001, 0.02),
        'max_iterations': trial.suggest_int('max_iterations', 5, 20),
        'num_runs':  trial.suggest_int('num_runs',1, 1),
        
        # Genetic Algorithm Parameters
        'ga_num_generations': trial.suggest_int('ga_num_generations', 50, 300),
        'ga_population_size': trial.suggest_int('ga_population_size', 40, 200),
        'ga_num_parents': trial.suggest_int('ga_num_parents', 10, 40),
        'ga_mutation_percent': trial.suggest_int('ga_mutation_percent', 5, 40),
        'max_features_per_rule': trial.suggest_int('max_features_per_rule', 3, 15),
        
        # Rule Diversity Parameters
        'feature_reuse_penalty': trial.suggest_float('feature_reuse_penalty', 0.1, 0.8),
        'min_threshold_value': trial.suggest_float('min_threshold_value', 0.0001, 0.01),
        'LESS_LOOKUP_RULE_POSITION' : trial.suggest_int('LESS_LOOKUP_RULE_POSITION', 3, 9)
    }
    
    print(f"\n=== Starting trial {trial.number} ===")
    print(f"Parameters: {config}")
    
    try:
        # Run discovery with this configuration
        rules, ensemble_rules, fraud_detection_rate = discover_rules_with_config(config)
        
        # Also consider rule quality and diversity
        avg_fraud_rate = sum(rule['fraud_rate'] for rule in rules) / len(rules) if rules else 0
        num_rules_found = len(rules)
        
        # Consider ensemble rules in the objective
        best_ensemble_score = 0
        if ensemble_rules:
            # Use the best ensemble rule's fraud count as part of the score
            best_ensemble = ensemble_rules[0]
            best_ensemble_score = best_ensemble['fraud_count'] / original_dt['Class'].sum()
        
        # The objective is to maximize fraud detection rate, number of rules found, and ensemble effectiveness
        score = fraud_detection_rate * (1 + 0.2 * min(num_rules_found, 10)/10 + 0.3 * best_ensemble_score)
        
        # Store additional metrics
        trial.set_user_attr("num_rules", num_rules_found)
        trial.set_user_attr("avg_fraud_rate", avg_fraud_rate)
        trial.set_user_attr("fraud_detection_rate", fraud_detection_rate)
        trial.set_user_attr("num_ensemble_rules", len(ensemble_rules))
        trial.set_user_attr("best_ensemble_score", best_ensemble_score)
        print(f"--------------------------------"*10)
        print(f"Score: {score}")
        print(f"--------------------------------"*10)
        return score
    
    except Exception as e:
        print(f"Error in trial: {e}")
        return -1  # Return a bad score on error

def run_optimization(n_trials=50, study_name="fraud_detection_optimization"):
    # Create a new study

    if BASE_CONFIG['new_study_only'] == True:
        try :
            optuna.delete_study(study_name=study_name, storage="sqlite:///fraud_detection_optuna.db")
        except:
            pass

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///fraud_detection_optuna.db",
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=4)
    
    print("\n===== Optimization Complete =====")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_trial.value}")
    print("\nBest parameters:")
    for param, value in study.best_trial.params.items():
        print(f"    {param}: {value}")
    
    return study

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def export_rules_to_csv(rules, results_dir, filename="rules.csv"):
    filepath = os.path.join(results_dir, filename)
    
    csv_data = []
    for rule in rules:
        # Convert rule dict to string representation safely
        rule_str = "; ".join([f"{feat} {safe_float(thresh):.4f}" for feat, thresh in rule['rule'].items()])
        
        csv_data.append({
            'Run': rule.get('run', 1),
            'Iteration': rule['iteration'],
            'Rule': rule_str,
            'Subset_Size': rule['subset_size'],
            'Fraud_Rate': safe_float(rule['fraud_rate']),
            'Fraud_Count': rule['fraud_count'],
            'Fitness_Score': rule['solution_fitness']
        })
    
    # Sort by fraud count (descending)
    csv_data.sort(key=lambda x: x['Fraud_Count'], reverse=True)
    
    # Write to CSV
    with open(filepath, 'w', newline='') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    
    print(f"Rules exported to {filepath}")
    return filepath


import os
import csv
import traceback

def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def safe_int(val):
    try:
        return int(val)
    except:
        return 0

# --- Main function ---
def run_with_best_params(study, results_dir, num_runs=1):
    best_params = study.best_trial.params
    optimized_config = BASE_CONFIG.copy()
    optimized_config.update(best_params)
    optimized_config['num_runs'] = num_runs  # Use multiple runs for final result

    print("\n===== Running with optimized parameters =====")
    print(f"Configuration: {optimized_config}")

    try:
        total_frauds = original_dt['Class'].sum()
        print(f"Total frauds in dataset: {total_frauds}")

        all_rules = []
        all_ensemble_rules = []
        all_detected_indices = set()

        for run in range(num_runs):
            optimized_config["run_seed"] = run + 1
            print(f"\n=== Run {run+1}/{num_runs} with run_seed = {optimized_config['run_seed']} ===")
            rules, ensemble_rules, _ = discover_rules_with_config(optimized_config)

            for r in rules:
                r['run'] = run + 1
                all_rules.append(r)
                all_detected_indices.update(r['rule_indices'])

            for e in ensemble_rules:
                # Ensure numeric fields are correctly typed
                e['subset_size'] = safe_int(e.get('subset_size'))
                e['fraud_rate'] = safe_float(e.get('fraud_rate'))
                e['fraud_count'] = safe_int(e.get('fraud_count'))
                e['run'] = run + 1
                all_ensemble_rules.append(e)

        detected_data = original_dt.loc[list(all_detected_indices)]
        detected_frauds = detected_data['Class'].sum()

        best_ensemble_detection = 0
        if all_ensemble_rules:
            all_ensemble_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
            best_ensemble_detection = all_ensemble_rules[0]['fraud_count']

        print("\n===== Final Results =====")
        print(f"Total individual rules discovered: {len(all_rules)}")
        print(f"Total ensemble rules created: {len(all_ensemble_rules)}")
        print(f"Unique frauds detected by individual rules: {detected_frauds} out of {total_frauds} ({detected_frauds/total_frauds:.4%})")
        try :
            if best_ensemble_detection > 0:
                print(f"Best ensemble rule detects {best_ensemble_detection} frauds ({best_ensemble_detection/total_frauds:.4%} of all frauds)")
        except:
            print(f"No ensemble rules found")

        csv_path = export_rules_to_csv(all_rules, results_dir, "optimized_rules.csv")
        if all_ensemble_rules:
            ensemble_csv_path = export_ensemble_rules_to_csv(all_ensemble_rules, all_rules, results_dir, "optimized_ensemble_rules.csv")

        summary_path = os.path.join(results_dir, "summary.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Total Rules', 'Total Ensemble Rules', 'Total Frauds', 'Detected Frauds (Individual)',
                             'Detection Rate (Individual)', 'Best Ensemble Detection'])
            writer.writerow([len(all_rules), len(all_ensemble_rules), total_frauds, detected_frauds,
                             detected_frauds/total_frauds, best_ensemble_detection])
        print(f"Summary exported to {summary_path}")

        config_path = os.path.join(results_dir, "best_config.csv")
        with open(config_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            for param, value in optimized_config.items():
                writer.writerow([param, value])
        print(f"Configuration exported to {config_path}")

        all_rules.sort(key=lambda x: x['fraud_count'], reverse=True)

        print("\nTop 5 individual rules:")
        for i, rule in enumerate(all_rules[:5]):
            print(f"\nRule #{i+1} (from Run {rule['run']}):")
            for feat, thresh in rule['rule'].items():
                print(f"  if {feat} {float(thresh):.4f}")
            print(f"Fraud count: {rule['fraud_count']} ({rule['fraud_rate']:.4%} precision)")

        if all_ensemble_rules:
            print("\nTop 3 ensemble rules:")
            for i, ensemble in enumerate(all_ensemble_rules[:3]):
                print(f"\nEnsemble Rule #{i+1} (from Run {ensemble['run']}): {ensemble['rule']}")
                print(f"Subset size: {ensemble['subset_size']:,}")
                print(f"Fraud rate: {ensemble[float('fraud_rate')]:.4%}")
                print(f"Fraud count: {ensemble['fraud_count']}")

        return all_rules, all_ensemble_rules, detected_frauds/total_frauds

    except Exception as e:
        print(f"Error running with best parameters: {e}")
        traceback.print_exc()
        return [], [], 0


# Full workflow - updated to include ensemble rules

# --- In run_full_workflow: Remove pickle saving, only export CSVs ---
def run_full_workflow(n_trials=30):
    print("Starting fraud detection hyperparameter optimization...")
    
    results_dir = create_results_directory()
    print(f"Results will be stored in: {results_dir}")
    
    study = run_optimization(n_trials=n_trials)
    end_time = time.time()
    print(f"Time taken to run_optimization: {end_time - start_time:.2f} seconds")
    trials_data = []
    for trial in study.trials:
        if trial.value is not None:
            trial_data = {
                'Trial': trial.number,
                'Score': trial.value,
                'Num_Rules': trial.user_attrs.get('num_rules', 'N/A'),
                'Avg_Fraud_Rate': trial.user_attrs.get('avg_fraud_rate', 'N/A'),
                'Fraud_Detection_Rate': trial.user_attrs.get('fraud_detection_rate', 'N/A')
            }
            trial_data.update(trial.params)
            trials_data.append(trial_data)
    
    trials_data.sort(key=lambda x: x['Score'], reverse=True)
    
    trials_path = os.path.join(results_dir, "optimization_trials.csv")
    if trials_data:
        with open(trials_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trials_data[0].keys())
            writer.writeheader()
            writer.writerows(trials_data)
    
    print(f"Optimization trials exported to {trials_path}")
    
    print("\nRunning with best parameters...")
    best_rules, best_ensemble_rules, detection_rate = run_with_best_params(study, results_dir)
    
    summary_path = os.path.join(results_dir, "rules_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=== FRAUD DETECTION RULES SUMMARY ===\n\n")
        f.write(f"Detection rate: {detection_rate:.4%}\n")
        f.write(f"Individual rules found: {len(best_rules)}\n")  
        f.write(f"Ensemble rules created: {len(best_ensemble_rules)}\n\n")
        
        f.write("TOP 5 INDIVIDUAL RULES:\n")
        for i, rule in enumerate(sorted(best_rules, key=lambda x: x['fraud_count'], reverse=True)[:5]):
            f.write(f"\nRule #{i+1} (from Run {rule['run']}):\n")
            for feat, thresh in rule['rule'].items():
                f.write(f"  if {feat} {thresh:.4f}\n")
            f.write(f"Fraud count: {rule['fraud_count']} ({rule['fraud_rate']:.4%} precision)\n")
        
        if best_ensemble_rules:
            f.write("\nTOP 3 ENSEMBLE RULES:\n")
            for i, ensemble in enumerate(sorted(best_ensemble_rules, key=lambda x: x['fraud_count'], reverse=True)[:3]):
                f.write(f"\nEnsemble Rule #{i+1} (from Run {ensemble['run']}): {ensemble['rule']}\n")
                f.write(f"Subset size: {ensemble['subset_size']:,}\n")
                f.write(f"Fraud rate: {ensemble['fraud_rate']:.4%}\n")
                f.write(f"Fraud count: {ensemble['fraud_count']}\n")
    
    print(f"Rules summary saved to {summary_path}")
    print(f"\nFinal fraud detection rate: {detection_rate:.4%}")
    print(f"All results saved to {results_dir}")
    print("Optimization and evaluation complete!")
    
    return study, best_rules, best_ensemble_rules, results_dir

if __name__ == "__main__":
    run_full_workflow(n_trials=1)
    end_time = time.time()
    print(f"Time taken to load run_full_workflow: {end_time - start_time:.2f} seconds")

