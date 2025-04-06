import pandas as pd
import numpy as np
import time
import pygad
import optuna

# ==========================================================================
# CONFIGURATION SECTION
# ==========================================================================
CONFIG = {
    # Dataset Configuration
    'dataset_path': "./creditcardfraud/creditcard.csv",
    
    # Rule Discovery Parameters
    'max_rules_per_run': 8,          # Maximum number of rules to discover per run
    'min_fraud_rate': 0.002,         # Minimum acceptable fraud rate (0.2%)
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
    
    # Ensemble Rule Parameters
    'ensemble_min_fraud_rate': 0.001 # 50% of the individual rule threshold
}

class RuleDiscovery:
    def __init__(self, config, dataset, comparison_mod=3, random_seed=None):
        """
        Initialize the rule discovery process.
        
        Parameters:
            config (dict): Configuration dictionary.
            dataset (pd.DataFrame): Loaded dataset.
            comparison_mod (int): Modulo value used to determine the comparison operator.
                                  For example, if set to 3, then if i % 3 == 0 use '<', else use '>'.
            random_seed (int): Optional random seed for reproducibility.
        """
        self.config = config
        self.original_dt = dataset.copy()
        self.comparison_mod = comparison_mod
        self.random_seed = random_seed

    def _initialize_data(self, dt):
        """
        Prepare and normalize the dataset.
        """
        # Drop unused columns and normalize features to [0,1]
        X = dt.drop(columns=['Time', 'Class']).astype(float)
        y = dt['Class'].astype(int)
        X_norm = (X - X.min()) / (X.max() - X.min())
        return X_norm, y, dt['Time'].values

    def _fitness_func(self, X, y, previous_features):
        """
        Returns the fitness function for the genetic algorithm.
        """
        num_features = X.shape[1]
        max_features = self.config['max_features_per_rule']
        desired_fraud_rate = self.config['min_fraud_rate']

        def fitness_func(ga_instance, solution, solution_idx):
            raw_selection = solution[:num_features]
            selection = raw_selection > 0.5  
            thresholds = solution[num_features:]
            
            # Repair if too many features selected
            if np.sum(selection) > max_features:
                sorted_idx = np.argsort(-raw_selection)
                new_selection = np.zeros_like(selection, dtype=bool)
                new_selection[sorted_idx[:max_features]] = True
                selection = new_selection
                
            if np.sum(selection) == 0:
                sorted_idx = np.argsort(-raw_selection)
                selection[sorted_idx[0]] = True
            
            thresholds = np.clip(thresholds, self.config['min_threshold_value'], 1.0)
            
            # Apply the composite rule using parameterized comparison
            mask = np.ones(len(X), dtype=bool)
            for i, sel in enumerate(selection):
                if sel:
                    if i % self.comparison_mod == 0:
                        mask &= (X.iloc[:, i] < thresholds[i])
                    else:
                        mask &= (X.iloc[:, i] > thresholds[i])
            
            subset = y[mask]
            if len(subset) == 0:
                return -1e6  # penalize if rule selects no samples
            
            fraud_rate = subset.mean()
            fraud_count = subset.sum()
            min_subset_size = max(self.config['min_subset_size_absolute'], 
                                  len(X) * self.config['min_subset_size_percent'])
            
            if len(subset) < min_subset_size or fraud_rate < desired_fraud_rate:
                return -1e6
            
            feature_reuse_factor = 0
            if previous_features:
                selected_feature_names = set([X.columns[i] for i, sel in enumerate(selection) if sel])
                feature_overlap = selected_feature_names.intersection(previous_features)
                if selected_feature_names:
                    feature_reuse_factor = len(feature_overlap) / len(selected_feature_names)
            
            reward = np.sqrt(fraud_rate) * fraud_count
            reward *= (1 - feature_reuse_factor * self.config['feature_reuse_penalty'])
            
            return reward
        
        return fitness_func

    def _run_genetic_algorithm(self, X, y, previous_features):
        """
        Configure and run the genetic algorithm.
        """
        num_features = X.shape[1]
        chromosome_length = num_features * 2  # first half for feature selection, second half for thresholds
        
        ga_instance = pygad.GA(
            num_generations=self.config['ga_num_generations'],
            num_parents_mating=self.config['ga_num_parents'],
            fitness_func=self._fitness_func(X, y, previous_features),
            sol_per_pop=self.config['ga_population_size'],
            num_genes=chromosome_length,
            init_range_low=0.0,
            init_range_high=1.0,
            mutation_percent_genes=self.config['ga_mutation_percent'],
            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            random_seed=self.random_seed
        )
        start_time = time.time()
        ga_instance.run()
        end_time = time.time()
        print(f"GA completed in {end_time - start_time:.2f} seconds")
        return ga_instance

    def _decode_solution(self, solution, X):
        """
        Decode the GA solution to extract the rule.
        """
        num_features = X.shape[1]
        raw_selection = solution[:num_features]
        selection = raw_selection > 0.5  
        if np.sum(selection) > self.config['max_features_per_rule']:
            sorted_idx = np.argsort(-raw_selection)
            new_selection = np.zeros_like(selection, dtype=bool)
            new_selection[sorted_idx[:self.config['max_features_per_rule']]] = True
            selection = new_selection
        
        thresholds = np.clip(solution[num_features:], self.config['min_threshold_value'], 1.0)
        rule = {}
        for i, (feat, sel) in enumerate(zip(X.columns, selection)):
            if sel and thresholds[i] > self.config['min_threshold_value']:
                # Use the parameterized modulo for operator selection
                if i % self.comparison_mod == 0:
                    rule[f"{feat} <"] = thresholds[i]
                else:
                    rule[f"{feat} >"] = thresholds[i]
        return rule

    def run_rule_discovery(self):
        """
        Run the rule discovery process for a single run.
        """
        dt = self.original_dt.copy()
        rules = []
        previous_features = set()
        original_index = dt.index.copy()
        remaining_samples = len(dt)
        remaining_frauds = dt['Class'].sum()
        
        print(f"Starting with {remaining_samples:,} samples, {remaining_frauds} frauds")
        print(f"Initial fraud rate: {remaining_frauds/remaining_samples:.4%}")
        
        for iteration in range(self.config['max_iterations']):
            print(f"\n===== ITERATION {iteration+1} =====")
            if remaining_samples < 1000 or remaining_frauds < 5:
                print(f"Stopping: too few samples ({remaining_samples:,}) or frauds ({remaining_frauds})")
                break
            
            X, y, time_values = self._initialize_data(dt)
            current_fraud_rate = y.mean()
            print(f"Current fraud rate: {current_fraud_rate:.4%}")
            
            if current_fraud_rate < 0.0001:
                print(f"Stopping: fraud rate extremely low ({current_fraud_rate:.4%})")
                break
            
            ga_instance = self._run_genetic_algorithm(X, y, previous_features)
            solution, solution_fitness, _ = ga_instance.best_solution()
            
            if solution_fitness <= 0:
                print("No useful rule found in this iteration. Stopping.")
                break

            rule = self._decode_solution(solution, X)
            
            # Apply the decoded rule
            mask = np.ones(len(X), dtype=bool)
            for feat_op, thresh in rule.items():
                feat = feat_op.split(" ")[0]
                op = feat_op.split(" ")[1]
                if op == "<":
                    mask &= (X[feat] < thresh)
                else:
                    mask &= (X[feat] > thresh)
            
            subset = y[mask]
            if len(subset) == 0:
                print("Rule selects no samples. Skipping this iteration.")
                continue

            final_fraud_rate = subset.mean()
            fraud_count = subset.sum()
            if final_fraud_rate < self.config['min_fraud_rate'] * 0.8 or len(subset) < self.config['min_subset_size_absolute']:
                print(f"Rule doesn't meet criteria: fraud rate = {final_fraud_rate:.4%}, subset size = {len(subset)}")
                continue
            
            # Save indices and update feature diversity
            rule_indices = original_index[mask]
            for feat_key in rule.keys():
                feat_name = feat_key.split(" ")[0]
                previous_features.add(feat_name)
            
            # Display rule stats
            print("\nRule Statistics:")
            print(f"Rule #{iteration+1}:")
            for feat, thresh in rule.items():
                print(f"  if {feat} {thresh:.4f}")
            print(f"Subset size: {len(subset):,} samples ({len(subset)/len(X):.2%} of data)")
            print(f"Fraud rate: {final_fraud_rate:.4%}")
            print(f"Fraud count: {fraud_count}")
            
            # Optionally, include time-based analysis here if needed...
            
            # Store discovered rule details
            rules.append({
                'iteration': iteration + 1,
                'rule': rule,
                'subset_size': len(subset),
                'fraud_rate': final_fraud_rate,
                'fraud_count': fraud_count,
                'rule_indices': rule_indices,
                'solution_fitness': solution_fitness
            })
            
            # Remove detected samples and update counters
            dt = dt[~mask].copy()
            original_index = original_index[~mask]
            remaining_samples = len(dt)
            remaining_frauds = dt['Class'].sum()
            print(f"Removed {len(subset)} samples (including {fraud_count} frauds)")
            print(f"Remaining samples: {remaining_samples:,}, Remaining frauds: {remaining_frauds}")
            
            if len(rules) >= self.config['max_rules_per_run'] or remaining_frauds == 0:
                break
        
        print("\n===== RULE DISCOVERY SUMMARY =====")
        print(f"Found {len(rules)} rules")
        for rule_info in rules:
            print(f"\nRule #{rule_info['iteration']}:")
            for feat, thresh in rule_info['rule'].items():
                print(f"  if {feat} {thresh:.4f}")
            print(f"Subset size: {rule_info['subset_size']:,}")
            print(f"Fraud count: {rule_info['fraud_count']}")
        return rules

def multi_run_discovery(config, dataset, comparison_mod=3):
    """
    Run multiple discovery processes (with different random seeds) and combine rules.
    """
    all_rules = []
    for run in range(config['num_runs']):
        print(f"\n\n=========== STARTING RUN {run+1} OF {config['num_runs']} ===========")
        rd = RuleDiscovery(config, dataset, comparison_mod=comparison_mod, random_seed=run*1000)
        rules = rd.run_rule_discovery()
        # Tag rules with run number
        for r in rules:
            r['run'] = run + 1
        all_rules.extend(rules)
    # Final summary can be added here (e.g., unique fraud detection, sorting, etc.)
    print("\n\n=========== FINAL SUMMARY ACROSS ALL RUNS ===========")
    print(f"Total individual rules found: {len(all_rules)}")
    return all_rules

# =============================================================================
# OPTUNA INTEGRATION FOR HYPERPARAMETER TUNING
# =============================================================================
def objective(trial):
    """
    Objective function for Optuna to tune the comparison modulo parameter.
    Here, we search for an integer value between 2 and 5.
    """
    comparison_mod = trial.suggest_int("comparison_mod", 2, 5)
    
    print(f"Trial with comparison_mod = {comparison_mod}")
    
    # Run the multi-run discovery with the current hyperparameter
    all_rules = multi_run_discovery(CONFIG, pd.read_csv(CONFIG['dataset_path']), comparison_mod=comparison_mod)
    
    # Define an objective metric: for instance, total fraud count detected
    total_fraud_detected = sum(rule['fraud_count'] for rule in all_rules)
    
    # The study will maximize total fraud count
    return total_fraud_detected

if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv(CONFIG['dataset_path'])
    print("Dataset columns:", dataset.columns)
    
    # Option 1: Run rule discovery with a fixed comparison modulo parameter
    discovered_rules = multi_run_discovery(CONFIG, dataset, comparison_mod=3)
    
    # Option 2: Use Optuna to tune the comparison_mod parameter
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
