import pandas as pd
import numpy as np
import time
import pygad
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
import plotly.io as pio
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import json

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
    'max_features_per_rule': 10,     # Maximum features allowed in a rule
    
    # Rule Diversity Parameters
    'feature_reuse_penalty': 0.3,    # Penalty factor for reusing features (30% per feature)
    'min_threshold_value': 0.001,    # Minimum threshold to consider (to avoid > 0.0000)
    
    # Ensemble Rule Parameters
    'ensemble_min_fraud_rate': 0.001, # 50% of the individual rule threshold
    
    # Comparison Operator Parameters
    'comparison_operator_pattern': 3  # Every Nth feature uses < comparison, others use >
}

# Load dataset once
try:
    base_dt = pd.read_csv(BASE_CONFIG['dataset_path'])
    original_dt = base_dt.copy()
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Using dummy dataset for development purposes")
    # Create a dummy dataset for development if real data is not available
    original_dt = pd.DataFrame({
        'Time': np.random.randint(0, 172800, 1000),
        'V1': np.random.randn(1000),
        'V2': np.random.randn(1000),
        'V3': np.random.randn(1000),
        'V4': np.random.randn(1000),
        'Amount': np.random.exponential(100, 1000),
        'Class': np.random.choice([0, 1], 1000, p=[0.98, 0.02])
    })
    base_dt = original_dt.copy()

# Global variables to store rules and results
all_discovered_rules = []
best_rules = []
trial_details = {}

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
        X_norm = (X - X.min()) / (X.max() - X.min() + 1e-10)  # Added small epsilon to avoid division by zero
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
                    if i % BASE_CONFIG['comparison_operator_pattern'] == 0:  # Every Nth feature uses < comparison for variety
                        mask &= (X.iloc[:, i] < thresholds[i])
                    else:  # Most features use > comparison
                        mask &= (X.iloc[:, i] > thresholds[i])
            
            subset = y[mask]
            subset_size = len(subset)
            if subset_size == 0:
                return -1e6  # penalize if rule selects no samples
            
            # Calculate subset statistics  
            fraud_rate = subset.mean()  # fraction of frauds in subset
            fraud_count = subset.sum()  # number of frauds in subset
            
            # Minimum subset size to avoid overfitting
            min_subset_size = max(config['min_subset_size_absolute'], 
                                len(X) * config['min_subset_size_percent'])
            
            if subset_size < min_subset_size:
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
                    
                if i % BASE_CONFIG['comparison_operator_pattern'] == 0:  # Every Nth feature uses < comparison
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
            'rule_indices': rule_indices.tolist(),  # Convert to list for JSON serialization
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
            
            # Add time stats to rule_stats
            rule_stats['time_span_hours'] = span_hours
            rule_stats['time_min_hours'] = time_min
            rule_stats['time_max_hours'] = time_max
        
        # Store this rule
        rules.append(rule_stats)
        
        # Also store in global list for dashboard
        global all_discovered_rules
        rule_stats_copy = rule_stats.copy()
        rule_stats_copy['trial_id'] = f"Run-{len(all_discovered_rules)}"  # Add unique identifier
        all_discovered_rules.append(rule_stats_copy)
        
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
        'min_fraud_rate': BASE_CONFIG['min_fraud_rate'],  # Fixed for consistency
        'min_subset_size_absolute': trial.suggest_int('min_subset_size_absolute', 30, 200),
        'min_subset_size_percent': trial.suggest_float('min_subset_size_percent', 0.001, 0.02),
        'max_iterations': trial.suggest_int('max_iterations', 5, 20),
        'num_runs': 3,  # Keep as 1 for optimization speed
        
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
        'ensemble_min_fraud_rate': trial.suggest_float('ensemble_min_fraud_rate', 0.0005, 0.002),
        
        # Comparison Operator Parameters
        'comparison_operator_pattern': trial.suggest_int('comparison_operator_pattern', 2, 9)
    }
    
    print(f"\n=== Starting trial {trial.number} ===")
    print(f"Parameters: {config}")
    
    global trial_details
    
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
        
        # Store rules for this trial
        trial_details[trial.number] = {
            'rules': rules,
            'avg_fraud_rate': avg_fraud_rate,
            'num_rules': num_rules_found,
            'fraud_detection_rate': fraud_detection_rate,
            'score': score,
            'params': config
        }
        
        return score
    
    except Exception as e:
        import traceback
        print(f"Error in trial: {e}")
        traceback.print_exc()
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

# Dashboard for visualizing optimization results and rules
def create_dashboard(study):
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    app.layout = html.Div([
        html.H1("Fraud Detection Rule System Dashboard", style={'textAlign': 'center'}),
        
        dcc.Tabs([
            # Tab for hyperparameter optimization
            dcc.Tab(label='Hyperparameter Optimization', children=[
                html.Div([
                    html.Div([
                        html.H3("Optimization History"),
                        dcc.Graph(id='optimization-history')
                    ], className='six columns'),
                    
                    html.Div([
                        html.H3("Parameter Importance"),
                        dcc.Graph(id='param-importance')
                    ], className='six columns')
                ], className='row'),
                
                html.Div([
                    html.H3("Parameter Relationships"),
                    html.Div([
                        html.Div([
                            html.Label("X Parameter:"),
                            dcc.Dropdown(
                                id='param-x',
                                options=[{'label': param, 'value': param} for param in study.best_trial.params.keys()],
                                value=list(study.best_trial.params.keys())[0]
                            )
                        ], className='six columns'),
                        
                        html.Div([
                            html.Label("Y Parameter:"),
                            dcc.Dropdown(
                                id='param-y',
                                options=[{'label': param, 'value': param} for param in study.best_trial.params.keys()],
                                value=list(study.best_trial.params.keys())[1] if len(study.best_trial.params.keys()) > 1 else list(study.best_trial.params.keys())[0]
                            )
                        ], className='six columns')
                    ], className='row'),
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
            ]),
            
            # New tab for rules visualization
            dcc.Tab(label='Rules Visualization', children=[
                html.Div([
                    html.H3("Rule Performance Overview", style={'textAlign': 'center'}),
                    
                    # Controls section
                    html.Div([
                        html.Div([
                            html.Label("Sort By:"),
                            dcc.Dropdown(
                                id='rule-sort-by',
                                options=[
                                    {'label': 'Fraud Rate (high to low)', 'value': 'fraud_rate-desc'},
                                    {'label': 'Fraud Count (high to low)', 'value': 'fraud_count-desc'},
                                    {'label': 'Subset Size (high to low)', 'value': 'subset_size-desc'},
                                    {'label': 'Fitness Score (high to low)', 'value': 'solution_fitness-desc'}
                                ],
                                value='fraud_count-desc'
                            )
                        ], className='six columns'),
                        
                        html.Div([
                            html.Label("Show Top Rules:"),
                            dcc.Slider(
                                id='top-n-rules',
                                min=5,
                                max=50,
                                step=5,
                                value=10,
                                marks={i: str(i) for i in range(5, 51, 5)}
                            )
                        ], className='six columns')
                    ], className='row', style={'marginBottom': '20px'}),
                    
                    # Visualization section
                    html.Div([
                        html.Div([
                            html.H4("Rule Performance Metrics"),
                            dcc.Graph(id='rule-performance-chart')
                        ], className='six columns'),
                        
                        html.Div([
                            html.H4("Feature Usage in Rules"),
                            dcc.Graph(id='feature-usage-chart')
                        ], className='six columns')
                    ], className='row'),
                    
                    # Rule details section
                    html.Div([
                        html.H3("Rule Details", style={'textAlign': 'center'}),
                        html.Div([
                            html.Label("Select Rule:"),
                            dcc.Dropdown(id='rule-selector')
                        ]),
                        html.Div(id='rule-details', style={'marginTop': '20px'})
                    ], style={'marginTop': '40px'})
                ])
            ]),
            
            # New tab for best rules and ensemble
            dcc.Tab(label='Best Rules & Ensemble', children=[
                html.Div([
                    html.H3("Best Rules Summary", style={'textAlign': 'center'}),
                    html.Div(id='best-rules-summary'),
                    
                    html.Div([
                        html.H4("Fraud Detection Coverage", style={'textAlign': 'center'}),
                        dcc.Graph(id='fraud-detection-coverage')
                    ], style={'marginTop': '30px'}),
                    
                    html.Div([
                        html.H4("Best Rules Comparison", style={'textAlign': 'center'}),
                        dcc.Graph(id='best-rules-comparison')
                    ], style={'marginTop': '30px'}),
                    
                    html.Div([
                        html.Button(
                            'Run Best Rules Ensemble', 
                            id='run-ensemble-button',
                            style={
                                'backgroundColor': '#4CAF50',
                                'color': 'white',
                                'padding': '10px 20px',
                                'textAlign': 'center',
                                'fontSize': '16px',
                                'margin': '20px auto',
                                'display': 'block'
                            }
                        ),
                        html.Div(id='ensemble-results', style={'marginTop': '20px'})
                    ])
                ])
            ])
        ])
    ], style={'fontFamily': 'Arial, sans-serif', 'margin': '0 auto', 'maxWidth': '1200px', 'padding': '20px'})
    
    # Define callbacks for optimization tab
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
        global trial_details
        
        if trial_number not in trial_details:
            return html.Div([
                html.H4(f"Trial #{trial_number}"),
                html.P("Detailed information not available for this trial.")
            ])
        
        trial_info = trial_details[trial_number]
        rules = trial_info['rules']
        
        # Create rules table
        rules_table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Rule #"),
                    html.Th("Fraud Rate"),
                    html.Th("Fraud Count"),
                    html.Th("Subset Size"),
                    html.Th("Fitness Score")
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(f"{rule['iteration']}"),
                    html.Td(f"{rule['fraud_rate']:.4%}"),
                    html.Td(f"{rule['fraud_count']}"),
                    html.Td(f"{rule['subset_size']:,}"),
                    html.Td(f"{rule['solution_fitness']:.2f}")
                ]) for rule in rules
            ])
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
        
        return html.Div([
            html.H4(f"Trial #{trial_number}"),
            html.Div([
                html.Div([
                    html.P(f"Score: {trial_info['score']:.4f}"),
                    html.P(f"Rules Found: {trial_info['num_rules']}"),
                    html.P(f"Avg Fraud Rate: {trial_info['avg_fraud_rate']:.4%}"),
                    html.P(f"Fraud Detection Rate: {trial_info['fraud_detection_rate']:.4%}")
                    # Additional callbacks for rules visualization tab
    @app.callback(
        Output('rule-selector', 'options'),
        [Input('rule-sort-by', 'value'),
         Input('top-n-rules', 'value')]
    )
    def update_rule_selector(sort_by, top_n):
        global all_discovered_rules
        
        if not all_discovered_rules:
            return []
            
        # Parse sort criteria
        sort_field, sort_direction = sort_by.split('-')
        reverse = sort_direction == 'desc'
        
        # Sort rules
        sorted_rules = sorted(all_discovered_rules, 
                            key=lambda x: x.get(sort_field, 0),
                            reverse=reverse)
        
        # Take top N
        top_rules = sorted_rules[:top_n]
        
        # Create options
        options = []
        for i, rule in enumerate(top_rules):
            fraud_rate = rule.get('fraud_rate', 0)
            fraud_count = rule.get('fraud_count', 0)
            
            # Format label
            label = f"Rule {i+1}: {fraud_rate:.2%} fraud rate, {fraud_count} frauds"
            options.append({'label': label, 'value': json.dumps(rule)})
            
        return options
    
    @app.callback(
        Output('rule-selector', 'value'),
        [Input('rule-selector', 'options')]
    )
    def set_default_rule(options):
        if options and len(options) > 0:
            return options[0]['value']
        return None
    
    @app.callback(
        Output('rule-details', 'children'),
        Input('rule-selector', 'value')
    )
    def update_rule_details(rule_json):
        if not rule_json:
            return html.Div("No rule selected")
            
        rule = json.loads(rule_json)
        
        # Create condition list
        conditions = []
        for feat, thresh in rule.get('rule', {}).items():
            conditions.append(html.Li(f"{feat} {thresh:.4f}"))
        
        # Format details
        return html.Div([
            html.H4(f"Rule Details"),
            
            # Stats card
            html.Div([
                html.Div([
                    html.H5("Performance Metrics:"),
                    html.P(f"Fraud Rate: {rule.get('fraud_rate', 0):.4%}"),
                    html.P(f"Fraud Count: {rule.get('fraud_count', 0)}"),
                    html.P(f"Subset Size: {rule.get('subset_size', 0):,} transactions"),
                    html.P(f"Fitness Score: {rule.get('solution_fitness', 0):.2f}")
                ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'marginBottom': '15px'})
            ]),
            
            # Rule conditions
            html.Div([
                html.H5("Rule Conditions:"),
                html.Ul(conditions)
            ])
        ])
    
    @app.callback(
        Output('rule-performance-chart', 'figure'),
        [Input('rule-sort-by', 'value'),
         Input('top-n-rules', 'value')]
    )
    def update_rule_performance_chart(sort_by, top_n):
        global all_discovered_rules
        
        if not all_discovered_rules:
            return go.Figure()
            
        # Parse sort criteria
        sort_field, sort_direction = sort_by.split('-')
        reverse = sort_direction == 'desc'
        
        # Sort rules
        sorted_rules = sorted(all_discovered_rules, 
                            key=lambda x: x.get(sort_field, 0),
                            reverse=reverse)
        
        # Take top N
        top_rules = sorted_rules[:top_n]
        
        # Create data for chart
        rule_ids = [f"Rule {i+1}" for i in range(len(top_rules))]
        fraud_rates = [rule.get('fraud_rate', 0) * 100 for rule in top_rules]
        subset_sizes = [rule.get('subset_size', 0) for rule in top_rules]
        fraud_counts = [rule.get('fraud_count', 0) for rule in top_rules]
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar charts
        fig.add_trace(
            go.Bar(x=rule_ids, y=fraud_rates, name="Fraud Rate (%)", marker_color='crimson'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=rule_ids, y=fraud_counts, name="Fraud Count", marker_color='royalblue'),
            secondary_y=True
        )
        
        # Add scatter plot for subset size
        fig.add_trace(
            go.Scatter(x=rule_ids, y=subset_sizes, mode='markers', 
                     name='Subset Size', marker=dict(size=12, color='green')),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title_text="Rule Performance Metrics",
            barmode='group',
            height=500
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Count", secondary_y=True)
        
        return fig
    
    @app.callback(
        Output('feature-usage-chart', 'figure'),
        [Input('rule-sort-by', 'value'),
         Input('top-n-rules', 'value')]
    )
    def update_feature_usage_chart(sort_by, top_n):
        global all_discovered_rules
        
        if not all_discovered_rules:
            return go.Figure()
            
        # Parse sort criteria
        sort_field, sort_direction = sort_by.split('-')
        reverse = sort_direction == 'desc'
        
        # Sort rules
        sorted_rules = sorted(all_discovered_rules, 
                            key=lambda x: x.get(sort_field, 0),
                            reverse=reverse)
        
        # Take top N
        top_rules = sorted_rules[:top_n]
        
        # Count feature usage
        feature_counts = {}
        for rule in top_rules:
            for feat_key in rule.get('rule', {}).keys():
                # Extract base feature name
                feat_name = feat_key.split(" ")[0]
                if feat_name in feature_counts:
                    feature_counts[feat_name] += 1
                else:
                    feature_counts[feat_name] = 1
        
        # Sort features by usage count
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f[0] for f in sorted_features]
        feature_usage = [f[1] for f in sorted_features]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=feature_names, y=feature_usage, marker_color='forestgreen')
        ])
        
        fig.update_layout(
            title_text="Feature Usage in Rules",
            xaxis_title="Feature",
            yaxis_title="Frequency",
            height=500
        )
        
        return fig
    
    # Define callbacks for Best Rules tab
    @app.callback(
        Output('best-rules-summary', 'children'),
        Input('run-ensemble-button', 'n_clicks')  # Not actually used, but triggers refresh
    )
    def update_best_rules_summary(_):
        global best_rules
        
        if not best_rules:
            # If best_rules is empty, populate it from all_discovered_rules
            global all_discovered_rules
            if all_discovered_rules:
                # Sort by fraud_count (most important metric)
                sorted_rules = sorted(all_discovered_rules, key=lambda x: x.get('fraud_count', 0), reverse=True)
                best_rules = sorted_rules[:10]  # Take top 10
            else:
                return html.Div("No rules discovered yet. Run optimization to discover rules.")
        
        # Create table
        rows = []
        for i, rule in enumerate(best_rules):
            # Format rule conditions as string
            rule_conditions = ", ".join([f"{feat} {thresh:.4f}" for feat, thresh in rule.get('rule', {}).items()])
            
            rows.append(html.Tr([
                html.Td(i+1),
                html.Td(f"{rule.get('fraud_rate', 0):.4%}"),
                html.Td(rule.get('fraud_count', 0)),
                html.Td(f"{rule.get('subset_size', 0):,}"),
                html.Td(rule_conditions)
            ]))
        
        table = html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Rank"),
                    html.Th("Fraud Rate"),
                    html.Th("Fraud Count"),
                    html.Th("Subset Size"),
                    html.Th("Rule Conditions")
                ])
            ),
            html.Tbody(rows)
        ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
        
        return html.Div([
            html.P("Top rules ranked by fraud detection performance:"),
            table
        ])
    
    @app.callback(
        Output('fraud-detection-coverage', 'figure'),
        Input('run-ensemble-button', 'n_clicks')  # Not actually used, but triggers refresh
    )
    def update_fraud_coverage(_):
        global best_rules
        
        if not best_rules:
            return go.Figure()
            
        # Calculate cumulative fraud coverage
        fraud_counts = [rule.get('fraud_count', 0) for rule in best_rules]
        rule_ids = [f"Rule {i+1}" for i in range(len(best_rules))]
        
        # Calculate cumulative sums
        cum_fraud = np.cumsum(fraud_counts)
        
        # Total fraud in dataset
        total_fraud = original_dt['Class'].sum()
        cum_fraud_percent = cum_fraud / total_fraud * 100
        
        # Create figure
        fig = go.Figure()
        
        # Add bar chart for individual counts
        fig.add_trace(
            go.Bar(
                x=rule_ids,
                y=fraud_counts,
                name="Individual Fraud Count"
            )
        )
        
        # Add line chart for cumulative percentage
        fig.add_trace(
            go.Scatter(
                x=rule_ids,
                y=cum_fraud_percent,
                mode='lines+markers',
                name='Cumulative Detection %',
                yaxis='y2',
                line=dict(color='firebrick', width=3)
            )
        )
        
        # Update layout for dual axes
        fig.update_layout(
            title_text="Fraud Detection Coverage",
            yaxis=dict(title="Fraud Count"),
            yaxis2=dict(
                title="Cumulative Detection %",
                overlaying='y',
                side='right',
                ticksuffix='%'
            ),
            height=500
        )
        
        return fig
    
    @app.callback(
        Output('best-rules-comparison', 'figure'),
        Input('run-ensemble-button', 'n_clicks')  # Not actually used, but triggers refresh
    )
    def update_rules_comparison(_):
        global best_rules
        
        if not best_rules:
            return go.Figure()
            
        # Extract metrics
        rule_ids = [f"Rule {i+1}" for i in range(len(best_rules))]
        fraud_rates = [rule.get('fraud_rate', 0) * 100 for rule in best_rules]
        subset_percents = [rule.get('subset_size', 0) / len(original_dt) * 100 for rule in best_rules]
        
        # Create a parallel coordinates plot
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(color=fraud_rates, colorscale='Jet', showscale=True, cmin=0, cmax=max(fraud_rates)),
                dimensions=[
                    dict(label='Rule', values=list(range(1, len(best_rules)+1)), tickvals=list(range(1, len(best_rules)+1))),
                    dict(label='Fraud Rate (%)', values=fraud_rates),
                    dict(label='Subset Size (% of data)', values=subset_percents)
                ]
            )
        )
        
        fig.update_layout(
            title="Comparison of Top Rules",
            height=600
        )
        
        return fig
    
    @app.callback(
        Output('ensemble-results', 'children'),
        Input('run-ensemble-button', 'n_clicks')
    )
    def run_ensemble(n_clicks):
        if n_clicks is None:
            return html.Div("Click the button to run ensemble analysis")
            
        global best_rules
        if not best_rules:
            return html.Div("No rules available for ensemble analysis")
        
        # Initialize metrics
        total_transactions = len(original_dt)
        total_fraud = original_dt['Class'].sum()
        detected_indices = set()
        rule_coverage = []
        
        # Track which transactions are detected by each rule
        for rule in best_rules:
            # Get indices covered by this rule
            rule_indices = set(rule.get('rule_indices', []))
            
            # Calculate new coverage (indices not previously detected)
            new_coverage = rule_indices - detected_indices
            
            # Add to total detected
            detected_indices.update(new_coverage)
            
            # Store coverage stats
            rule_coverage.append({
                'total_coverage': len(rule_indices),
                'new_coverage': len(new_coverage),
                'detected_fraud': sum(original_dt.loc[list(rule_indices), 'Class'])
            })
        
        # Calculate overall ensemble metrics
        total_detected = len(detected_indices)
        
        # Create a mask for all detected transactions
        detection_mask = original_dt.index.isin(list(detected_indices))
        detected_fraud = original_dt.loc[detection_mask, 'Class'].sum()
        
        # Create a results card
        results_card = html.Div([
            html.H4("Ensemble Results", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.H5("Overall Detection Metrics:"),
                    html.P(f"Transactions Analyzed: {total_transactions:,}"),
                    html.P(f"Total Fraud Cases: {total_fraud:,}"),
                    html.P(f"Detected Transactions: {total_detected:,} ({total_detected/total_transactions:.2%})"),
                    html.P([
                        "Detected Fraud Cases: ",
                        html.Span(f"{detected_fraud:,} ({detected_fraud/total_fraud:.2%})", 
                                style={'color': 'green', 'fontWeight': 'bold'})
                    ]),
                    html.P(f"False Positive Rate: {(total_detected - detected_fraud)/total_detected:.4%}")
                ], style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px', 'marginBottom': '20px'})
            ]),
            
            html.H5("Rule Contribution Analysis:"),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Rule"),
                        html.Th("Detected Fraud"),
                        html.Th("Total Coverage"),
                        html.Th("Unique Coverage")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(f"Rule {i+1}"),
                        html.Td(f"{stats['detected_fraud']}"),
                        html.Td(f"{stats['total_coverage']:,}"),
                        html.Td(f"{stats['new_coverage']:,}")
                    ]) for i, stats in enumerate(rule_coverage)
                ])
            ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
        ])
        
        return results_card
    
    return app

# Function to create optimal configuration from best trial
def get_optimal_config(study):
    best_trial = study.best_trial
    optimal_config = BASE_CONFIG.copy()
    
    for param, value in best_trial.params.items():
        optimal_config[param] = value
        
    # Adjust some parameters for production
    optimal_config['num_runs'] = 3  # Run multiple times for better results
    
    return optimal_config

# Function to run with optimal configuration and save results
def run_optimal_configuration(study, output_path="optimal_rules.json"):
    optimal_config = get_optimal_config(study)
    
    print("Running with optimal configuration:")
    for param, value in optimal_config.items():
        print(f"    {param}: {value}")
    
    # Run multiple times and collect the best rules
    global best_rules
    best_rules = []
    
    for run in range(optimal_config['num_runs']):
        print(f"\n=== Production Run {run+1}/{optimal_config['num_runs']} ===")
        
        rules, fraud_detection_rate = discover_rules_with_config(optimal_config)
        
        print(f"Run {run+1} complete:")
        print(f"    Rules discovered: {len(rules)}")
        print(f"    Fraud detection rate: {fraud_detection_rate:.4%}")
        
        # Add to best rules
        for rule in rules:
            rule['run'] = run + 1
            best_rules.append(rule)
    
    # Sort by fraud count (most important metric)
    best_rules = sorted(best_rules, key=lambda x: x['fraud_count'], reverse=True)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(best_rules, f, indent=4)
    
    print(f"\nResults saved to {output_path}")
    
    return best_rules

# Helper functions for visualization
def make_subplots(**kwargs):
    """Create a figure with multiple subplots."""
    from plotly.subplots import make_subplots as plotly_make_subplots
    return plotly_make_subplots(**kwargs)

# Main execution function
def main(optimization_trials=50):
    print("Fraud Detection Rule System")
    print("===========================")
    
    # First run hyperparameter optimization
    print("\nRunning hyperparameter optimization...")
    study = run_optimization(n_trials=optimization_trials)
    
    # Then run with optimal configuration
    print("\nRunning with optimal configuration...")
    rules = run_optimal_configuration(study)
    
    # Create and run dashboard
    print("\nStarting dashboard...")
    app = create_dashboard(study)
    
    # Return everything needed for analysis
    return {
        'study': study,
        'rules': rules,
        'app': app
    }

# Run the main function if script is executed directly
if __name__ == "__main__":
    results = main(optimization_trials=50)
    
    # Start the dashboard
    print("\nStarting dashboard. Press Ctrl+C to exit.")
    results['app'].run_server(debug=True)