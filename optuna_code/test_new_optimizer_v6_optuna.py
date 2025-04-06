import pandas as pd
import numpy as np
import time
import pygad
import optuna
import os
from datetime import datetime
import csv

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
    'num_runs': 3,                   # Number of runs with different random seeds
    
    # Genetic Algorithm Parameters
    'ga_num_generations': 150,       # Number of generations for genetic algorithm
    'ga_population_size': 80,        # Population size per generation
    'ga_num_parents': 25,            # Number of parents per generation
    'ga_mutation_percent': 15,       # Mutation percentage
    'max_features_per_rule': 10,     # Maximum number of features allowed in a rule
    
    # Rule Diversity Parameters
    'feature_reuse_penalty': 0.3,    # Penalty factor for reusing features (30% per feature)
    'min_threshold_value': 0.001,    # Minimum threshold to consider (to avoid > 0.0000)
}

# Ensemble Rule Parameters
BASE_CONFIG['ensemble_min_fraud_rate'] = BASE_CONFIG['min_fraud_rate'] / 2  # 50% of the individual rule threshold

# Load dataset once
base_dt = pd.read_csv(BASE_CONFIG['dataset_path'])
original_dt = base_dt.copy()

# Create time-based results directory
def create_results_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"fraud_detection_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Function to discover rules with a given configuration
def discover_rules_with_config(config):
    """
    Discover fraud detection rules using genetic algorithm optimization.
    
    This function takes a configuration dictionary and discovers rules that identify 
    subsets of transactions with high fraud rates.
    
    Returns:
        tuple: (rules, ensemble_rules, fraud_detection_rate)
    """
    # Reset dataset for new rule discovery 
    base_dt = original_dt.copy()
    rules = []
    remaining_samples = len(base_dt)
    remaining_frauds = base_dt['Class'].sum()
    
    # Track previously used features to encourage diversity
    previous_features = set()
    
    # Keep track of all detected indices
    original_index = base_dt.index.copy()

    print(f"Starting with {remaining_samples:,} total samples, {remaining_frauds} frauds")
    print(f"Initial global fraud rate: {remaining_frauds/remaining_samples:.4%}")
    
    for iteration in range(config['max_iterations']):
        print(f"\n===== ITERATION {iteration+1} =====")
        
        # Skip if too few samples remain
        if remaining_samples < 1000 or remaining_frauds < 5:
            print(f"Stopping: too few remaining samples ({remaining_samples:,}) or frauds ({remaining_frauds})")
            break
            
        # Prepare features (X) and target (y)
        X = base_dt.drop(columns=['Time', 'Class']).astype(float)
        y = base_dt['Class'].astype(int)
        time_values = base_dt['Time'].values
        
        current_fraud_rate = y.mean()
        print(f"Current fraud rate: {current_fraud_rate:.4%}")
        
        # If fraud rate is too low, we might still find patterns but lower our expectations
        if current_fraud_rate < 0.0001:  # Less than 0.01% - extremely low
            print(f"Stopping: fraud rate extremely low ({current_fraud_rate:.4%})")
            break
            
        # Normalize X so that each feature is scaled to [0,1]
        X_norm = (X - X.min()) / (X.max() - X.min())
        X = X_norm
        
        # Genetic algorithm parameters
        num_features = X.shape[1]  # candidate features count
        max_features = config['max_features_per_rule']  # maximum number of features allowed in composite rule
        desired_fraud_rate = config['min_fraud_rate']  # Minimum acceptable fraud rate
        
        # Chromosome encoding: first num_features for selection (binary), then num_features for thresholds.
        chromosome_length = num_features * 2
        
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
        
            # Clamp thresholds to [0, 1] and ensure minimum threshold value
            thresholds = np.clip(thresholds, config['min_threshold_value'], 1.0)
            
            # Apply the composite rule: For each selected feature, require X[feature] > threshold.
            mask = np.ones(len(X), dtype=bool)
            for i, sel in enumerate(selection):
                if sel:
                    # Use feature index to determine rule type (adds variety)
                    if i % 3 == 0:  # Every 3rd feature uses < comparison
                        mask &= (X.iloc[:, i] < thresholds[i])
                    else:  # Most features use > comparison
                        mask &= (X.iloc[:, i] > thresholds[i])
            
            subset = y[mask]
            if len(subset) == 0:
                return -1e6  # penalize if rule selects no samples
            
            # Calculate subset statistics  
            fraud_rate = subset.mean()  # fraction of frauds in subset
            fraud_count = subset.sum()  # number of frauds in subset
            
            # Minimum subset size to avoid overfitting
            min_subset_size = max(config['min_subset_size_absolute'], 
                                len(X) * config['min_subset_size_percent'])
            
            if len(subset) < min_subset_size:
                return -1e6  # Penalize small subsets
        
            # Penalize if fraud rate is below desired threshold
            if fraud_rate < desired_fraud_rate:
                return -1e6
                
            # Calculate penalty for feature reuse from previous rules
            feature_reuse_factor = 0
            if previous_features:
                selected_feature_names = set([X.columns[i] for i, sel in enumerate(selection) if sel])
                feature_overlap = selected_feature_names.intersection(previous_features)
                feature_reuse_factor = len(feature_overlap) / len(selected_feature_names) if selected_feature_names else 0
        
            # Modified reward: Balance between fraud rate and absolute number of frauds found
            # Square root of fraud_rate gives diminishing returns for very high rates
            # Multiplying by fraud_count emphasizes finding more fraud cases
            reward = np.sqrt(fraud_rate) * fraud_count
            
            # Apply penalty for feature reuse
            reward = reward * (1 - feature_reuse_factor * config['feature_reuse_penalty'])
            
            return reward
        
        # Genetic Algorithm Configuration
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
                             random_seed=iteration*1000)  # Different seed per iteration
        
        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Get the best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        # Check if solution is worth keeping
        if solution_fitness <= 0:
            print("No useful rule found in this iteration. Stopping.")
            break
            
        raw_selection = solution[:num_features]
        selection = raw_selection > 0.5
        # Repair solution if needed:
        if np.sum(selection) > max_features:
            sorted_idx = np.argsort(-raw_selection)
            new_selection = np.zeros_like(selection, dtype=bool)
            new_selection[sorted_idx[:max_features]] = True
            selection = new_selection
        
        thresholds = np.clip(solution[num_features:], config['min_threshold_value'], 1.0)
        selected_features = X.columns[selection]
        rule = {}
        
        # Create rule with appropriate comparison operators
        for i, (feat, sel) in enumerate(zip(X.columns, selection)):
            if sel:
                threshold_value = thresholds[i]
                # Skip rules with thresholds very close to 0 (they're not meaningful)
                if threshold_value <= config['min_threshold_value']:
                    continue
                    
                if i % 3 == 0:  # Every 3rd feature uses < comparison
                    rule[f"{feat} <"] = threshold_value
                else:
                    rule[f"{feat} >"] = threshold_value
        
        # Evaluate final rule
        mask = np.ones(len(X), dtype=bool)
        for feat, thresh in rule.items():
            if " <" in feat:
                actual_feat = feat.replace(" <", "")
                mask &= (X[actual_feat] < thresh)
            else:
                actual_feat = feat.replace(" >", "")
                mask &= (X[actual_feat] > thresh)
        
        # Rule statistics
        subset = y[mask]
        subset_times = time_values[mask]
        subset_size = len(subset)
        
        if subset_size == 0:
            print("Rule selects no samples. Skipping this iteration.")
            continue
            
        final_fraud_rate = subset.mean() 
        fraud_count = subset.sum()
        
        # Check if rule meets our criteria (now less strict)
        if final_fraud_rate < config['min_fraud_rate'] * 0.8 or subset_size < config['min_subset_size_absolute']:
            print(f"Rule doesn't meet criteria: fraud rate = {final_fraud_rate:.4%}, subset size = {subset_size}")
            continue
        
        # Save the original indices of the records that match this rule
        rule_indices = original_index[mask]
        
        # Add matched features to previous_features set to encourage diversity
        for feat_key in rule.keys():
            feat_name = feat_key.split(" ")[0]  # Extract feature name without operator
            previous_features.add(feat_name)
        
        # Calculate rule statistics
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
        
        # Analyze Time distribution (assuming Time is in seconds) - from v4
        if len(subset_times) > 0:
            # Convert time to hours for better visualization
            hours = subset_times / 3600
            
            # Calculate time-based statistics
            time_min = hours.min()
            time_max = hours.max()
            span_hours = time_max - time_min
            
            print(f"\nTime distribution:")
            print(f"Time span: {span_hours:.2f} hours ({span_hours/24:.2f} days)")
            
            # Weekly distribution (assuming Time starts at 0)
            max_days = int(np.ceil(time_max / 24))
            weeks = int(np.ceil(max_days / 7))
            
            if weeks > 0:
                print("\nWeekly fraud distribution:")
                week_counts = [0] * weeks
                week_totals = [0] * weeks
                
                for i, t in enumerate(hours):
                    day = int(t / 24)
                    week = min(int(day / 7), weeks - 1)
                    week_totals[week] += 1
                    if subset.iloc[i] == 1:
                        week_counts[week] += 1
                
                for week in range(weeks):
                    week_rate = week_counts[week] / week_totals[week] if week_totals[week] > 0 else 0
                    print(f"Week {week+1}: {week_counts[week]} frauds out of {week_totals[week]} transactions ({week_rate:.4%})")
        
        # Store this rule
        rules.append(rule_stats)
        
        # Remove detected samples for next iteration
        base_dt = base_dt[~mask].copy()
        original_index = original_index[~mask]  # Update the original indices
        remaining_samples = len(base_dt)
        remaining_frauds = base_dt['Class'].sum()
        
        print(f"\nRemoved {subset_size} samples, including {fraud_count} frauds")
        print(f"Remaining samples: {remaining_samples:,}, Remaining frauds: {remaining_frauds}")
        
        # Stop if we've reached the max rules or no more frauds to find
        if len(rules) >= config['max_rules_per_run'] or remaining_frauds == 0:
            break
    
    # Create ensemble rules (combinations of individual rules) - from v4
    print("\n===== ENSEMBLE RULES =====")
    ensemble_rules = []
    
    # Try all pairwise combinations
    for i in range(len(rules)):
        for j in range(i+1, len(rules)):
            # Combine rules i and j (Union of their detections)
            rule_i_indices = set(rules[i]['rule_indices'])
            rule_j_indices = set(rules[j]['rule_indices'])
            combined_indices = list(rule_i_indices.union(rule_j_indices))
            
            # Calculate ensemble rule statistics
            if combined_indices:
                ensemble_data = original_dt.loc[combined_indices]
                ensemble_fraud_rate = ensemble_data['Class'].mean()
                ensemble_fraud_count = ensemble_data['Class'].sum()
                
                # If the ensemble rule is good, store it
                if (ensemble_fraud_count > rules[i]['fraud_count'] and 
                    ensemble_fraud_count > rules[j]['fraud_count'] and
                    ensemble_fraud_rate >= config['ensemble_min_fraud_rate']):
                    
                    ensemble_rules.append({
                        'rule': f"Rule #{i+1} OR Rule #{j+1}",
                        'rule_components': [i+1, j+1],  # Store the component rule numbers
                        'subset_size': len(combined_indices),
                        'fraud_rate': ensemble_fraud_rate,
                        'fraud_count': ensemble_fraud_count,
                        'rule_indices': combined_indices
                    })
    
    # Sort ensemble rules by fraud count (descending)
    ensemble_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
    
    # Display top ensemble rules
    for i, ensemble in enumerate(ensemble_rules[:5]):  # Top 5 ensemble rules
        print(f"\nEnsemble Rule #{i+1}: {ensemble['rule']}")
        print(f"Subset size: {ensemble['subset_size']:,}")
        print(f"Fraud rate: {ensemble['fraud_rate']:.4%}")
        print(f"Fraud count: {ensemble['fraud_count']}")
    
    # Total stats
    total_fraud_count = sum(rule['fraud_count'] for rule in rules)
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
        'ensemble_min_fraud_rate': trial.suggest_float('ensemble_min_fraud_rate', 0.001, 0.01),
        # Rule Discovery Parameters
        'max_rules_per_run': trial.suggest_int('max_rules_per_run', 3, 15),
        
        'min_subset_size_absolute': trial.suggest_int('min_subset_size_absolute', 30, 200),
        'min_subset_size_percent': trial.suggest_float('min_subset_size_percent', 0.001, 0.02),
        'max_iterations': trial.suggest_int('max_iterations', 5, 20),
        'num_runs': 1,  # Keep as 1 for optimization speed
        
        # Genetic Algorithm Parameters
        'ga_num_generations': trial.suggest_int('ga_num_generations', 50, 300),
        'ga_population_size': trial.suggest_int('ga_population_size', 40, 200),
        'ga_num_parents': trial.suggest_int('ga_num_parents', 10, 40),
        'ga_mutation_percent': trial.suggest_int('ga_mutation_percent', 5, 40),
        'max_features_per_rule': trial.suggest_int('max_features_per_rule', 3, 15),
        
        # Rule Diversity Parameters
        'feature_reuse_penalty': trial.suggest_float('feature_reuse_penalty', 0.1, 0.8),
        'min_threshold_value': trial.suggest_float('min_threshold_value', 0.0001, 0.01),
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
        
        return score
    
    except Exception as e:
        print(f"Error in trial: {e}")
        return -1  # Return a bad score on error

def run_optimization(n_trials=50, study_name="fraud_detection_optimization"):
    # Create a new study
    if BASE_CONFIG['new_study_only'] == True:
        try :
            optuna.delete_study(study_name="fraud_detection_optuna", storage="sqlite:///fraud_detection_optuna.db")
        except:
            pass
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage="sqlite:///fraud_detection_optuna.db",
        load_if_exists=True
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=n_trials)
    
    print("\n===== Optimization Complete =====")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best score: {study.best_trial.value}")
    print("\nBest parameters:")
    for param, value in study.best_trial.params.items():
        print(f"    {param}: {value}")
    
    return study

# Export rules to CSV - modified to handle ensemble rules
def export_rules_to_csv(rules, results_dir, filename="rules.csv"):
    filepath = os.path.join(results_dir, filename)
    
    # Prepare data for CSV
    csv_data = []
    for rule in rules:
        # Convert rule dict to string representation
        rule_str = "; ".join([f"{feat} {thresh:.4f}" for feat, thresh in rule['rule'].items()])
        
        csv_data.append({
            'Run': rule.get('run', 1),
            'Iteration': rule['iteration'],
            'Rule': rule_str,
            'Subset_Size': rule['subset_size'],
            'Fraud_Rate': rule['fraud_rate'],
            'Fraud_Count': rule['fraud_count'],
            'Fitness_Score': rule['solution_fitness']
        })
    
    # Sort by fraud count (descending)
    csv_data.sort(key=lambda x: x['Fraud_Count'], reverse=True)
    
    # Write to CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys() if csv_data else [])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Rules exported to {filepath}")
    return filepath

# Run with best parameters - modified to include ensemble rules
def run_with_best_params(study, results_dir, num_runs=3):
    # Get best parameters
    best_params = study.best_trial.params
    
    # Update the configuration with the best parameters
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
            print(f"\n=== Run {run+1}/{num_runs} ===")
            rules, ensemble_rules, _ = discover_rules_with_config(optimized_config)
            
            # Add run identifier
            for r in rules:
                r['run'] = run + 1
                all_rules.append(r)
                
                # Track unique fraud detections
                all_detected_indices.update(r['rule_indices'])
            
            # Add run identifier to ensemble rules
            for e in ensemble_rules:
                e['run'] = run + 1
                all_ensemble_rules.append(e)
        
        # Calculate overall detection rate for individual rules
        detected_data = original_dt.loc[list(all_detected_indices)]
        detected_frauds = detected_data['Class'].sum()
        
        # Calculate best ensemble rule detection
        best_ensemble_detection = 0
        if all_ensemble_rules:
            all_ensemble_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
            best_ensemble_detection = all_ensemble_rules[0]['fraud_count']
        
        print("\n===== Final Results =====")
        print(f"Total individual rules discovered: {len(all_rules)}")
        print(f"Total ensemble rules created: {len(all_ensemble_rules)}")
        print(f"Unique frauds detected by individual rules: {detected_frauds} out of {total_frauds} ({detected_frauds/total_frauds:.4%})")
        
        if best_ensemble_detection > 0:
            print(f"Best ensemble rule detects {best_ensemble_detection} frauds ({best_ensemble_detection/total_frauds:.4%} of all frauds)")
        
        # Export rules to CSV
        csv_path = export_rules_to_csv(all_rules, results_dir, "optimized_rules.csv")
        
        # Export ensemble rules to CSV
        if all_ensemble_rules:
            ensemble_csv_path = export_ensemble_rules_to_csv(all_ensemble_rules, all_rules, results_dir, "optimized_ensemble_rules.csv")
        
        # Export summary to CSV
        summary_path = os.path.join(results_dir, "summary.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Total Rules', 'Total Ensemble Rules', 'Total Frauds', 'Detected Frauds (Individual)', 
                          'Detection Rate (Individual)', 'Best Ensemble Detection'])
            writer.writerow([len(all_rules), len(all_ensemble_rules), total_frauds, detected_frauds, 
                          detected_frauds/total_frauds, best_ensemble_detection])
        
        print(f"Summary exported to {summary_path}")
        
        # Export configuration
        config_path = os.path.join(results_dir, "best_config.csv")
        with open(config_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            for param, value in optimized_config.items():
                writer.writerow([param, value])
        
        print(f"Configuration exported to {config_path}")
        
        # Sort rules by fraud count
        all_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
        
        print("\nTop 5 individual rules:")
        for i, rule in enumerate(all_rules[:5]):
            print(f"\nRule #{i+1} (from Run {rule['run']}):")
            for feat, thresh in rule['rule'].items():
                print(f"  if {feat} {thresh:.4f}")
            print(f"Fraud count: {rule['fraud_count']} ({rule['fraud_rate']:.4%} precision)")
        
        # Display top ensemble rules
        if all_ensemble_rules:
            print("\nTop 3 ensemble rules:")
            for i, ensemble in enumerate(all_ensemble_rules[:3]):
                print(f"\nEnsemble Rule #{i+1} (from Run {ensemble['run']}): {ensemble['rule']}")
                print(f"Subset size: {ensemble['subset_size']:,}")
                print(f"Fraud rate: {ensemble['fraud_rate']:.4%}")
                print(f"Fraud count: {ensemble['fraud_count']}")
        
        return all_rules, all_ensemble_rules, detected_frauds/total_frauds
        
    except Exception as e:
        import traceback
        print(f"Error running with best parameters: {e}")
        traceback.print_exc()
        return [], [], 0

# Full workflow - updated to include ensemble rules

def run_full_workflow(n_trials=30):
    print("Starting fraud detection hyperparameter optimization...")
    
    # Create results directory
    results_dir = create_results_directory()
    print(f"Results will be stored in: {results_dir}")
    
    # Run the optimization
    study = run_optimization(n_trials=n_trials)
    
    # Export optimization results
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
    
    # Sort by score (descending)
    trials_data.sort(key=lambda x: x['Score'], reverse=True)
    
    # Export to CSV
    trials_path = os.path.join(results_dir, "optimization_trials.csv")
    if trials_data:
        with open(trials_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trials_data[0].keys())
            writer.writeheader()
            writer.writerows(trials_data)
    
    print(f"Optimization trials exported to {trials_path}")
    
    # Run with best parameters
    print("\nRunning with best parameters...")
    best_rules, best_ensemble_rules, detection_rate = run_with_best_params(study, results_dir)
    
    # Save the rules objects as pickle files for future use
    import pickle
    
    # Save best individual rules as pickle
    rules_pickle_path = os.path.join(results_dir, "best_rules.pkl")
    with open(rules_pickle_path, 'wb') as f:
        pickle.dump(best_rules, f)
    print(f"Best individual rules saved to {rules_pickle_path}")
    
    # Save ensemble rules as pickle
    ensemble_pickle_path = os.path.join(results_dir, "best_ensemble_rules.pkl")
    with open(ensemble_pickle_path, 'wb') as f:
        pickle.dump(best_ensemble_rules, f)
    print(f"Best ensemble rules saved to {ensemble_pickle_path}")
    
    # Create a summary file for easy access to rule information
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