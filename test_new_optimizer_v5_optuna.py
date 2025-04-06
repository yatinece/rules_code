import pandas as pd
import numpy as np
import time
import pygad
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
import plotly.io as pio
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import logging

# Original configuration - will be modified by Optuna
BASE_CONFIG = {
    # Dataset Configuration
    'dataset_path': "./creditcardfraud/creditcard.csv",
    
    # Rule Discovery Parameters
    'max_rules_per_run': 8,          # Maximum number of rules to discover per run
    'min_fraud_rate': 0.01,         # Minimum acceptable fraud rate (0.2%)
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

# Load dataset once
base_dt = pd.read_csv(BASE_CONFIG['dataset_path'])
original_dt = base_dt.copy()

# Function to discover rules with a given configuration
def discover_rules_with_config(config):
    # Reset dataset for new rule discovery 
    """
    Discover fraud detection rules using genetic algorithm optimization.
    
    This function takes a configuration dictionary and discovers rules that identify 
    subsets of transactions with high fraud rates. It uses an iterative process where:
    
    1. For each iteration:
        - Normalizes features to [0,1] range
        - Uses genetic algorithm to find optimal feature thresholds
        - Evaluates rules based on fraud rate and subset size
        - Removes detected samples for next iteration
        
    2. The genetic algorithm optimizes:
        - Feature selection (which features to include in rule)
        - Threshold values for selected features
        - Rule quality based on fraud rate and subset size
        
    3. Includes diversity mechanisms:
        - Penalizes reuse of previously used features
        - Maintains minimum subset sizes
        - Enforces minimum fraud rates
        
    Args:
        config (dict): Configuration parameters including:
            - Dataset parameters (path, etc)
            - Rule discovery parameters (min fraud rate, subset sizes)
            - Genetic algorithm parameters (generations, population size)
            - Rule diversity parameters (feature reuse penalties)
            
    Returns:
        tuple: (rules, fraud_detection_rate) where:
            rules (list): List of discovered rules and their statistics
            fraud_detection_rate (float): Overall fraction of frauds detected
    """
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
        
        # Analyze Time distribution
        if len(subset_times) > 0:
            # Convert time to hours
            hours = subset_times / 3600
            
            # Calculate time-based statistics
            time_min = hours.min()
            time_max = hours.max()
            span_hours = time_max - time_min
            
            print(f"\nTime distribution:")
            print(f"Time span: {span_hours:.2f} hours ({span_hours/24:.2f} days)")
        
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
    
    # Total stats
    total_fraud_count = sum(rule['fraud_count'] for rule in rules)
    total_dataset_fraud = original_dt['Class'].sum()
    fraud_detection_rate = total_fraud_count / total_dataset_fraud if total_dataset_fraud > 0 else 0
    
    return rules, fraud_detection_rate

# Define objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    config = {
        # Dataset Configuration - not changing
        'dataset_path': BASE_CONFIG['dataset_path'],
        
        # Rule Discovery Parameters
        'max_rules_per_run': trial.suggest_int('max_rules_per_run', 3, 15),
        #'min_fraud_rate': trial.suggest_float('min_fraud_rate', 0.001, 0.01),
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
        
        # Ensemble Rule Parameters
        'ensemble_min_fraud_rate': trial.suggest_float('ensemble_min_fraud_rate', 0.0005, 0.002)
    }
    
    print(f"\n=== Starting trial {trial.number} ===")
    print(f"Parameters: {config}")
    
    try:
        # Run discovery with this configuration
        rules, fraud_detection_rate = discover_rules_with_config(config)
        
        # Also consider rule quality and diversity
        avg_fraud_rate = sum(rule['fraud_rate'] for rule in rules) / len(rules) if rules else 0
        num_rules_found = len(rules)
        
        # The objective is to maximize fraud detection rate and number of rules found
        score = fraud_detection_rate * (1 + 0.2 * min(num_rules_found, 10)/10)
        
        # Store additional metrics
        trial.set_user_attr("num_rules", num_rules_found)
        trial.set_user_attr("avg_fraud_rate", avg_fraud_rate)
        trial.set_user_attr("fraud_detection_rate", fraud_detection_rate)
        
        return score
    
    except Exception as e:
        print(f"Error in trial: {e}")
        return -1  # Return a bad score on error

def run_optimization(n_trials=50, study_name="fraud_detection_optimization"):
    # Create a new study
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

# Dashboard for visualizing optimization results
def create_dashboard(study):
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Fraud Detection Hyperparameter Optimization Dashboard"),
        
        html.Div([
            html.Div([
                html.H3("Optimization History"),
                dcc.Graph(id='optimization-history')
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                html.H3("Parameter Importance"),
                dcc.Graph(id='param-importance')
            ], style={'width': '50%', 'display': 'inline-block'})
        ]),
        
        html.Div([
            html.H3("Parameter Relationships"),
            dcc.Dropdown(
                id='param-x',
                options=[{'label': param, 'value': param} for param in study.best_trial.params.keys()],
                value=list(study.best_trial.params.keys())[0]
            ),
            dcc.Dropdown(
                id='param-y',
                options=[{'label': param, 'value': param} for param in study.best_trial.params.keys()],
                value=list(study.best_trial.params.keys())[1] if len(study.best_trial.params.keys()) > 1 else list(study.best_trial.params.keys())[0]
            ),
            dcc.Graph(id='param-relationship')
        ]),
        
        html.Div([
            html.H3("Parameter Analysis"),
            dcc.Dropdown(
                id='param-analysis',
                options=[{'label': param, 'value': param} for param in study.best_trial.params.keys()],
                value=list(study.best_trial.params.keys())[0]
            ),
            dcc.Graph(id='param-slice')
        ]),
        
        html.Div([
            html.H3("Trial Details"),
            dcc.Dropdown(
                id='trial-selector',
                options=[{'label': f"Trial #{trial.number} (Score: {trial.value:.4f})", 
                          'value': trial.number} 
                         for trial in study.trials if trial.value is not None],
                value=study.best_trial.number
            ),
            html.Div(id='trial-details')
        ])
    ])
    
    @app.callback(
        Output('optimization-history', 'figure'),
        Input('trial-selector', 'value')  # Not actually used, but triggers refresh
    )
    def update_optimization_history(_):
        fig = plot_optimization_history(study)
        return fig
    
    @app.callback(
        Output('param-importance', 'figure'),
        Input('trial-selector', 'value')  # Not actually used, but triggers refresh
    )
    def update_param_importance(_):
        fig = plot_param_importances(study)
        return fig
    
    @app.callback(
        Output('param-relationship', 'figure'),
        [Input('param-x', 'value'),
         Input('param-y', 'value')]
    )
    def update_param_relationship(param_x, param_y):
        fig = plot_contour(study, params=[param_x, param_y])
        return fig
    
    @app.callback(
        Output('param-slice', 'figure'),
        Input('param-analysis', 'value')
    )
    def update_param_slice(param):
        fig = plot_slice(study, params=[param])
        return fig
    
    @app.callback(
        Output('trial-details', 'children'),
        Input('trial-selector', 'value')
    )
    def update_trial_details(trial_number):
        trial = study.trials[trial_number]
        
        return html.Div([
            html.H4(f"Trial #{trial.number}"),
            html.P(f"Score: {trial.value:.4f}"),
            html.P(f"Num Rules Found: {trial.user_attrs.get('num_rules', 'N/A')}"),
            html.P(f"Avg Fraud Rate: {trial.user_attrs.get('avg_fraud_rate', 'N/A'):.4%}"),
            html.P(f"Fraud Detection Rate: {trial.user_attrs.get('fraud_detection_rate', 'N/A'):.4%}"),
            html.H5("Parameters:"),
            html.Ul([html.Li(f"{param}: {value}") for param, value in trial.params.items()]),
        ])
    
    return app

# Run with best parameters
def run_with_best_params(study, num_runs=3):
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
        all_detected_indices = set()
        
        for run in range(num_runs):
            print(f"\n=== Run {run+1}/{num_runs} ===")
            rules, _ = discover_rules_with_config(optimized_config)
            
            # Add run identifier
            for r in rules:
                r['run'] = run + 1
                all_rules.append(r)
                
                # Track unique fraud detections
                all_detected_indices.update(r['rule_indices'])
        
        # Calculate overall detection rate
        detected_data = original_dt.loc[list(all_detected_indices)]
        detected_frauds = detected_data['Class'].sum()
        
        print("\n===== Final Results =====")
        print(f"Total rules discovered: {len(all_rules)}")
        print(f"Unique frauds detected: {detected_frauds} out of {total_frauds} ({detected_frauds/total_frauds:.4%})")
        
        # Sort rules by fraud count
        all_rules.sort(key=lambda x: x['fraud_count'], reverse=True)
        
        print("\nTop 5 rules:")
        for i, rule in enumerate(all_rules[:5]):
            print(f"\nRule #{i+1} (from Run {rule['run']}):")
            for feat, thresh in rule['rule'].items():
                print(f"  if {feat} {thresh:.4f}")
            print(f"Fraud count: {rule['fraud_count']} ({rule['fraud_rate']:.4%} precision)")
        
        return all_rules, detected_frauds/total_frauds
        
    except Exception as e:
        import traceback
        print(f"Error running with best parameters: {e}")
        traceback.print_exc()
        return [], 0

# Full workflow
def run_full_workflow(n_trials=30, dashboard_port=8050):
    print("Starting fraud detection hyperparameter optimization...")
    
    # Run the optimization
    study = run_optimization(n_trials=n_trials)
    
    # Create and run the dashboard
    app = create_dashboard(study)
    print(f"\nStarting dashboard on port {dashboard_port}...")
    print(f"Open http://localhost:{dashboard_port} in your browser to view the results.")
    print("Press Ctrl+C to stop the dashboard and continue...")
    
    try:
        app.run(debug=False, port=dashboard_port)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    
    # Run with best parameters
    print("\nRunning with best parameters...")
    best_rules, detection_rate = run_with_best_params(study)
    
    print(f"\nFinal fraud detection rate: {detection_rate:.4%}")
    print("Optimization and evaluation complete!")
    
    return study, best_rules

if __name__ == "__main__":
    run_full_workflow(n_trials=10)