import pandas as pd
import numpy as np
import time
import pygad

# Load dataset
base_dt = pd.read_csv("./creditcardfraud/creditcard.csv")

for i in range(100):
    base_dt = base_dt.append(base_dt)
print("Columns:", base_dt.columns)

# Create a copy of the original dataset to preserve it
original_dt = base_dt.copy()

# Function to discover rules with multiple enhancements
def discover_rules(max_rules=10, min_fraud_rate=0.002, max_iterations=15, random_seed=None):
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
    
    for iteration in range(max_iterations):
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
        max_features = 10          # maximum number of features allowed in composite rule (increased)
        desired_fraud_rate = min_fraud_rate  # Minimum acceptable fraud rate
        
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
        
            # Clamp thresholds to [0, 1]
            thresholds = np.clip(thresholds, 0.0, 1.0)
            
            # Apply the composite rule: For each selected feature, require X[feature] > threshold.
            mask = np.ones(len(X), dtype=bool)
            for i, sel in enumerate(selection):
                if sel:
                    # Use feature index to determine rule type (adds variety)
                    if i % 3 == 0:  # Every 3rd feature uses < comparison for variety
                        mask &= (X.iloc[:, i] < thresholds[i])
                    else:  # Most features use > comparison
                        mask &= (X.iloc[:, i] > thresholds[i])
            
            subset = y[mask]
            if len(subset) == 0:
                return -1e6  # penalize if rule selects no samples
            
            # Calculate subset statistics  
            fraud_rate = subset.mean()  # fraction of frauds in subset
            fraud_count = subset.sum()  # number of frauds in subset
            
            # Minimum subset size to avoid overfitting (but less restrictive now)
            min_subset_size = max(50, len(X) * 0.005)  # At least 0.5% of data or 50 samples
            
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
            
            # Apply penalty for feature reuse (30% reduction per reused feature)
            reward = reward * (1 - feature_reuse_factor * 0.3)
            
            return reward
        
        # Genetic Algorithm Configuration - enhanced parameters
        ga_instance = pygad.GA(num_generations=150,  # More generations
                               num_parents_mating=40,  # More parents
                               fitness_func=fitness_func,
                               sol_per_pop=80,  # Larger population
                               num_genes=chromosome_length,
                               init_range_low=0.0,
                               init_range_high=1.0,
                               mutation_percent_genes=15,  # Higher mutation rate
                               parent_selection_type="sss",
                               crossover_type="single_point",
                               mutation_type="random",
                               random_seed=random_seed)
        
        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        # Retrieve the best solution
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
        
        thresholds = np.clip(solution[num_features:], 0.0, 1.0)
        selected_features = X.columns[selection]
        rule = {}
        
        # Create rule with appropriate comparison operators
        for i, (feat, sel) in enumerate(zip(X.columns, selection)):
            if sel:
                if i % 3 == 0:  # Every 3rd feature uses < comparison
                    rule[f"{feat} <"] = thresholds[i]
                else:
                    rule[f"{feat} >"] = thresholds[i]
        
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
        if final_fraud_rate < min_fraud_rate * 0.8 or subset_size < 50:  # 20% lower threshold
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
        
        # Analyze Time distribution (assuming Time is in seconds)
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
        if len(rules) >= max_rules or remaining_frauds == 0:
            break
    
    # Final summary of all rules
    print("\n===== RULE DISCOVERY SUMMARY =====")
    print(f"Found {len(rules)} rules")
    
    total_detected = 0
    total_frauds_detected = 0
    
    for i, rule_info in enumerate(rules):
        print(f"\nRule #{i+1}:")
        for feat, thresh in rule_info['rule'].items():
            print(f"  if {feat} {thresh:.4f}")
        print(f"Subset size: {rule_info['subset_size']:,}")
        print(f"Fraud rate: {rule_info['fraud_rate']:.4%}")
        print(f"Fraud count: {rule_info['fraud_count']}")
        
        total_detected += rule_info['subset_size']
        total_frauds_detected += rule_info['fraud_count']
    
    # Create ensemble rules (combinations of individual rules)
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
                    ensemble_fraud_rate >= min_fraud_rate * 0.5):  # Lower threshold for ensembles
                    
                    ensemble_rules.append({
                        'rule': f"Rule #{i+1} OR Rule #{j+1}",
                        'subset_size': len(combined_indices),
                        'fraud_rate': ensemble_fraud_rate,
                        'fraud_count': ensemble_fraud_count
                    })
    
    # Sort ensemble rules by fraud count (descending)
    ensemble_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
    
    # Display top ensemble rules
    for i, ensemble in enumerate(ensemble_rules[:5]):  # Top 5 ensemble rules
        print(f"\nEnsemble Rule #{i+1}: {ensemble['rule']}")
        print(f"Subset size: {ensemble['subset_size']:,}")
        print(f"Fraud rate: {ensemble['fraud_rate']:.4%}")
        print(f"Fraud count: {ensemble['fraud_count']}")
    
    # Overall stats
    total_samples = len(original_dt)
    total_frauds = original_dt['Class'].sum()
    
    print(f"\nOverall detection stats:")
    print(f"Total samples in dataset: {total_samples:,}")
    print(f"Total frauds in dataset: {total_frauds}")
    print(f"Total samples detected by all individual rules: {total_detected:,} ({total_detected/total_samples:.2%})")
    print(f"Total frauds detected by individual rules: {total_frauds_detected} ({total_frauds_detected/total_frauds:.2%} of all frauds)")
    
    if ensemble_rules:
        best_ensemble = ensemble_rules[0]
        print(f"Best ensemble rule detects {best_ensemble['fraud_count']} frauds ({best_ensemble['fraud_count']/total_frauds:.2%} of all frauds)")
    
    return rules, ensemble_rules

# Run multiple times with different random seeds for more diversity
def multi_run_discovery(num_runs=3):
    all_rules = []
    all_ensembles = []
    
    for run in range(num_runs):
        print(f"\n\n=========== STARTING RUN {run+1} OF {num_runs} ===========")
        import time
        start = time.time()
        rules, ensembles = discover_rules(max_rules=8, min_fraud_rate=0.01, max_iterations=12, 
                                        random_seed=run*1000)
        end = time.time()
        print(f"time taken: {end - start}")
        
        # Add run identifier to rules
        for r in rules:
            r['run'] = run + 1
        all_rules.extend(rules)
        all_ensembles.extend(ensembles)
    
    # Final summary across all runs
    print("\n\n=========== FINAL SUMMARY ACROSS ALL RUNS ===========")
    print(f"Total individual rules found: {len(all_rules)}")
    
    # Get unique fraud cases detected across all rules
    all_detected_indices = set()
    all_detected_frauds = 0
    
    for rule in all_rules:
        rule_indices = set(rule['rule_indices'])
        # Count new frauds not previously detected
        new_indices = rule_indices - all_detected_indices
        if new_indices:
            new_frauds = original_dt.loc[list(new_indices), 'Class'].sum()
            all_detected_frauds += new_frauds
            all_detected_indices.update(rule_indices)
    
    total_frauds = original_dt['Class'].sum()
    print(f"Combined unique frauds detected: {all_detected_frauds} ({all_detected_frauds/total_frauds:.2%} of all frauds)")
    
    # Sort rules by fraud count for final display
    all_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
    
    print("\nTop 10 rules across all runs:")
    for i, rule in enumerate(all_rules[:10]):
        print(f"\nRule #{i+1} (from Run {rule['run']}):")
        for feat, thresh in rule['rule'].items():
            print(f"  if {feat} {thresh:.4f}")
        print(f"Fraud count: {rule['fraud_count']} ({rule['fraud_rate']:.4%} precision)")
    
    return all_rules, all_ensembles

# Run with multiple seeds for better coverage
try:
    all_discovered_rules, all_ensemble_rules = multi_run_discovery(num_runs=3)
except Exception as e:
    import traceback
    print(f"Error during rule discovery: {e}")
    traceback.print_exc()