import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import re
import itertools
from collections import defaultdict

fraud_detection_approaches_map =  {
    "Exact Greedy Algorithm": "exact",
    "Approximate Greedy Algorithm": "approx",
    "Histogram-based Algorithm": "hist"
}


# Mapping for evaluation criteria
eval_metrics_map = {
    "Discrimination Ability (AUC)": "auc",
    "Error Rate": "error",
    "Prediction Confidence (Log Loss)": "logloss"
}


fraud_detection_approaches_map_rev =  {
     "exact" : "Exact Greedy Algorithm",
     "approx" : "Approximate Greedy Algorithm",
     "hist" : "Histogram-based Algorithm"
}


# Mapping for evaluation criteria
eval_metrics_map_rev = {
   "auc" : "Discrimination Ability (AUC)",
    "error" : "Error Rate",
    "logloss" : "Prediction Confidence (Log Loss)"
}

class RuleExtractor:
    def __init__(self, df, target_col='Class', feature_cols=None, 
                 tree_methods=['hist', 'exact', 'approx'],
                 eval_metrics=['auc', 'error', 'logloss'],
                 rule_direction='lt',
                 config=None):
        """
        df: DataFrame with data.
        target_col: Name of target column (binary: 0/1).
        feature_cols: List of feature columns. If None, uses all except target.
        tree_methods: List of XGBoost tree methods to test.
        eval_metrics: List of evaluation metrics to test.
        rule_direction: 'lt' for "<" rules, 'gt' for ">" rules, or 'both'.
        config: Dictionary for extra configuration.
        """
        self.df = df.copy()
        self.target_col = target_col
        if feature_cols is None:
            self.feature_cols = [col for col in df.columns if col != target_col]
        else:
            self.feature_cols = feature_cols
        self.tree_methods = tree_methods if tree_methods else ['hist', 'approx']
        self.eval_metrics = eval_metrics if eval_metrics else ['auc']
        self.rule_direction = rule_direction.lower()
        
        # Default configuration
        default_config = {
            'feature_reuse_penalty': 0.0,
            'max_rule_conditions': 3
        }
        
        # Update with user-provided config
        if config is not None:
            default_config.update(config)
        self.config = default_config
        
        # Data used for grid search extraction
        self.X = self.df[self.feature_cols]
        self.y = self.df[self.target_col]
        
        # Define multiple regex patterns to handle different model dump formats
        self.patterns = [
            re.compile(r"(\d+):\[(V\d+)<([\-0-9.e]+)\] yes=(\d+),no=(\d+),missing=\d+,gain=([\-0-9.e]+),cover=([\-0-9.e]+)"),
            re.compile(r"(\d+):\[(\w+)<([\-0-9.e]+)\] yes=(\d+),no=(\d+),missing=\d+,gain=([\-0-9.e]+),cover=([\-0-9.e]+)")
        ]
    
    def _extract_split(self, line):
        """Try multiple regex patterns and return the first successful match groups."""
        for pat in self.patterns:
            match = pat.match(line)
            if match:
                return match.groups()
        return None

    def apply_individual_rule(self, rule_text, X):
        """
        Apply a single rule to the given data
        Returns a boolean mask of rows matching the rule
        
        Example rule text: "transaction_amount > 1000"
        """
        # Simple rule parsing - this might need to be extended based on your rule format
        parts = rule_text.split()
        if len(parts) >= 3:
            feature = parts[0]
            operator = parts[1]
            threshold = float(parts[2])
            
            if feature in X.columns:
                if operator == '>':
                    return X[feature] > threshold
                elif operator == '>=':
                    return X[feature] >= threshold
                elif operator == '<':
                    return X[feature] < threshold
                elif operator == '<=':
                    return X[feature] <= threshold
                elif operator == '==':
                    return X[feature] == threshold
                elif operator == '!=':
                    return X[feature] != threshold
        
        # If rule cannot be parsed, return all False
        return pd.Series([False] * len(X))

    def _train_model(self, X, y, num_boost_round, max_depth, tree_method, eval_metric, extra_params=None):
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set default extra parameters if none provided
        if extra_params is None:
            extra_params = {}
        
        params = {
            'max_depth': max_depth,
            'objective': 'binary:logistic',
            'tree_method': tree_method,
            'eval_metric': eval_metric,
            'seed': 42,  # For reproducibility
            # Use provided extra parameters or default values
            'gamma': extra_params.get('gamma', 0.0),
            'min_child_weight': extra_params.get('min_child_weight', 1),
            'subsample': extra_params.get('subsample', 1.0),
            'colsample_bytree': extra_params.get('colsample_bytree', 1.0)
        }
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round)
        return bst.get_dump(with_stats=True)

    
    def _apply_rule(self, X, rule_text):
        """
        Applies a rule to a DataFrame and returns a boolean mask of matching records.
        Rule text is in the format "feature operator value"
        """
        parts = rule_text.strip().split()
        if len(parts) != 3:
            return pd.Series([False] * len(X), index=X.index)
                
        feature, operator, threshold = parts
        threshold = float(threshold)
            
        if operator == '<':
            return X[feature] < threshold
        elif operator == '>':
            return X[feature] > threshold
        else:
            return pd.Series([False] * len(X), index=X.index)
    

    def apply_compound_rule(self, X, rule_conditions, join_type='AND'):
        """
        Apply a compound rule with multiple conditions.
        
        Args:
            X: DataFrame to apply the rule to
            rule_conditions: List of individual rule strings ["feature1 < value1", "feature2 > value2", ...]
            join_type: 'AND' or 'OR' to determine how conditions are combined
            
        Returns:
            Boolean mask of records matching the compound rule
        """
        if not rule_conditions:
            return pd.Series([False] * len(X), index=X.index)
        
        # Start with all True (for AND) or all False (for OR)
        if join_type.upper() == 'AND':
            result_mask = pd.Series([True] * len(X), index=X.index)
            for condition in rule_conditions:
                result_mask &= self._apply_rule(X, condition)
        else:  # OR
            result_mask = pd.Series([False] * len(X), index=X.index)
            for condition in rule_conditions:
                result_mask |= self._apply_rule(X, condition)
                
        return result_mask

    def apply_rule(self, extractor, X):
        """
        Apply the compound rule to the given data
        Returns a boolean mask of rows matching the rule
        """
        if not self.conditions:
            return pd.Series([False] * len(X))
        
        masks = []
        for condition in self.conditions:
            # Parse the condition and apply it
            # This is a simplified approach - you might need more complex parsing
            # depending on your rule format
            mask = extractor.apply_individual_rule(condition, X)
            masks.append(mask)
        
        # Combine masks based on join type
        if self.join_type == 'AND':
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask & mask
        else:  # OR
            final_mask = masks[0]
            for mask in masks[1:]:
                final_mask = final_mask | mask
                
        return final_mask
        
    def extract_rules_grid(self, num_boost_round=10, max_depth=3, min_support=20):
        """
        Loop over all parameter combinations, extract candidate splits,
        compute a reward, and return a DataFrame of unique candidate rules sorted by reward.
        """
        all_splits = []
        progress_bar = st.progress(0)

        # Count total combinations for progress tracking
        total_combinations = len(self.tree_methods) * len(self.eval_metrics)
        combination_count = 0

        for tm, em in itertools.product(self.tree_methods, self.eval_metrics):
            combination_count += 1
            progress_bar.progress(combination_count / total_combinations)
            st.write(f"Training with method: {fraud_detection_approaches_map_rev[tm]} and metric: {eval_metrics_map_rev[em]}")

            dtrain = xgb.DMatrix(self.X, label=self.y)
            model_dump = self._train_model(self.X, self.y, num_boost_round, max_depth, tm, em)
            
            for tree_idx, tree in enumerate(model_dump):
                for line in tree.split('\n'):
                    groups = self._extract_split(line)
                    if groups and len(groups) >= 5:
                        node_id, feature, thresh = groups[0], groups[1], groups[2]
                        gain_val = groups[5] if len(groups) > 5 else "0"
                        cover_val = groups[6] if len(groups) > 6 else "0"
                        
                        threshold = float(thresh)
                        gain = float(gain_val)
                        cover = float(cover_val)
                        
                        if self.rule_direction == 'gt':
                            mask = self.X[feature] > threshold
                            operator = '>'
                        elif self.rule_direction == 'lt':
                            mask = self.X[feature] < threshold
                            operator = '<'
                        else:  # both
                            mask_lt = self.X[feature] < threshold
                            lt_total = mask_lt.sum()
                            if lt_total >= min_support:
                                all_splits.append({
                                    'Tree Method': tm,
                                    'Eval Metric': em,
                                    'Tree Number': tree_idx,
                                    'Node': int(node_id),
                                    'Feature': feature,
                                    'Threshold': threshold,
                                    'Operator': '<',
                                    'Gain': gain,
                                    'Coverage': cover,
                                    'Total': lt_total,
                                    'Frauds': self.y[mask_lt].sum(),
                                    'Fraud Rate': (self.y[mask_lt].sum() / lt_total) if lt_total > 0 else 0
                                })
                            
                            mask_gt = self.X[feature] > threshold
                            gt_total = mask_gt.sum()
                            if gt_total >= min_support:
                                all_splits.append({
                                    'Tree Method': tm,
                                    'Eval Metric': em,
                                    'Tree Number': tree_idx,
                                    'Node': int(node_id),
                                    'Feature': feature,
                                    'Threshold': threshold,
                                    'Operator': '>',
                                    'Gain': gain,
                                    'Coverage': cover,
                                    'Total': gt_total,
                                    'Frauds': self.y[mask_gt].sum(),
                                    'Fraud Rate': (self.y[mask_gt].sum() / gt_total) if gt_total > 0 else 0
                                })
                            continue
                        
                        total = mask.sum()
                        if total >= min_support:
                            all_splits.append({
                                'Tree Method': tm,
                                'Eval Metric': em,
                                'Tree Number': tree_idx,
                                'Node': int(node_id),
                                'Feature': feature,
                                'Threshold': threshold,
                                'Operator': operator,
                                'Gain': gain,
                                'Coverage': cover,
                                'Total': total,
                                'Frauds': self.y[mask].sum(),
                                'Fraud Rate': (self.y[mask].sum() / total) if total > 0 else 0
                            })
        
        progress_bar.progress(1.0)
        
        splits_df = pd.DataFrame(all_splits)
        if splits_df.empty:
            st.warning("No splits were extracted. Check model dump or parameters.")
            return splits_df
        else:
            # Drop duplicates and compute reward
            unique_splits = splits_df.drop_duplicates(subset=["Feature", "Threshold", "Operator"])
            
            # Compute reward: here reward = sqrt(fraud_rate) * fraud_count
            def compute_reward(row):
                return np.sqrt(row['Fraud Rate']) * row['Frauds']
                
            unique_splits['Reward'] = unique_splits.apply(compute_reward, axis=1)
            unique_splits = unique_splits.sort_values(by=['Reward', 'Gain'], ascending=False).reset_index(drop=True)
            
            # Add rule text
            unique_splits['Rule Text'] = unique_splits.apply(
                lambda row: f"{row['Feature']} {row['Operator']} {row['Threshold']}", axis=1
            )
            unique_splits['Suggested Rule'] = unique_splits.apply(
                lambda row: f"IF {row['Feature']} {row['Operator']} {row['Threshold']} THEN FLAG", axis=1
            )
            
            return unique_splits

class CompoundRuleBuilder:
    def __init__(self):
        """Initialize a compound rule builder to manage multi-condition rules"""
        self.conditions = []
        self.join_type = 'AND'
        
    def add_condition(self, condition):
        """Add a condition to the rule"""
        self.conditions.append(condition)
        
    def remove_condition(self, index):
        """Remove a condition by index"""
        if 0 <= index < len(self.conditions):
            del self.conditions[index]
            
    def clear_conditions(self):
        """Clear all conditions"""
        self.conditions = []
        
    def set_join_type(self, join_type):
        """Set the logical operator (AND/OR) for combining conditions"""
        self.join_type = join_type
        
    def get_rule_text(self):
        """Get the full rule text with all conditions"""
        if not self.conditions:
            return "No conditions defined"
        
        conditions_text = f" {self.join_type} ".join([f"({cond})" for cond in self.conditions])
        return f"IF {conditions_text} THEN FLAG"
    
    def apply_rule(self, extractor, X):
        """Apply this compound rule to the data X using the extractor's helper methods"""
        return extractor.apply_compound_rule(X, self.conditions, self.join_type)

# Initialize session state variables
def init_session_state():
    if 'applied_rules' not in st.session_state:
        st.session_state['applied_rules'] = []
    if 'current_iteration' not in st.session_state:
        st.session_state['current_iteration'] = 0
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = None
    if 'current_df' not in st.session_state:
        st.session_state['current_df'] = None
    if 'rule_builder' not in st.session_state:
        st.session_state['rule_builder'] = CompoundRuleBuilder()
    if 'matched_data_history' not in st.session_state:
        st.session_state['matched_data_history'] = []

# Main Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="XGBoost Rule Extraction")
    st.title("XGBoost Rule Extraction Dashboard")
    st.write("""
    This dashboard allows you to extract rules from XGBoost models and apply them iteratively to your data.
    Upload your data, configure parameters, extract rules, and apply them one by one to find patterns.
    You can now create compound rules with multiple conditions!
    """)
    
    # Initialize session state
    init_session_state()
    
    # Sidebar setup
    st.sidebar.header("Rule Extraction Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data or use previously loaded data
        if st.session_state['original_df'] is None or st.sidebar.button("Reload Data"):
            df = pd.read_csv(uploaded_file)
            st.session_state['original_df'] = df.copy()
            st.session_state['current_df'] = df.copy()
            st.session_state['applied_rules'] = []
            st.session_state['current_iteration'] = 0
            st.session_state['matched_data_history'] = []
            st.success("Data loaded successfully!")
        
        df = st.session_state['current_df']
        
        # Display data summary
        st.subheader("Current Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Original Records", st.session_state['original_df'].shape[0])
        with col3:
            if 'target_col' in st.session_state and st.session_state['target_col'] in df.columns:
                positive_rate = df[st.session_state['target_col']].mean()
                st.metric("Target Rate", f"{positive_rate:.2%}")
        
        # Data preview with expander
        with st.expander("Data Preview"):
            st.dataframe(df.head(10))
        
        # Target column selection
        target_col = st.sidebar.selectbox(
            "Target Column", 
            df.columns.tolist(),
            key="target_col_select"
        )
        
        if target_col:
            st.session_state['target_col'] = target_col
            
            # Feature columns selection (multi-select with all columns except target)
            potential_features = [col for col in df.columns if col != target_col]
            feature_cols = st.sidebar.multiselect(
                "Feature Columns", 
                potential_features,
                default=potential_features[:min(len(potential_features), 10)],  # Default to first 10 features
                key="feature_cols_select"
            )
            
            # # Tree methods selection
            # tree_methods = st.sidebar.multiselect(
            #     "Methods",
            #     ["hist", "exact", "approx"],
            #     default=["hist"],
            #     key="tree_methods_select"
            # )
            
            # # Evaluation metrics selection
            # eval_metrics = st.sidebar.multiselect(
            #     "Metrics",
            #     ["auc", "error", "logloss"],
            #     default=["auc"],
            #     key="eval_metrics_select"
            # )
            st.sidebar.header("Fraud Detection Settings")
            # Mapping for fraud detection approaches


            # Business-friendly settings for fraud detection approaches
            selected_approaches = st.sidebar.multiselect(
                "Fraud Detection Approaches",
                options=list(fraud_detection_approaches_map.keys()),
                default=["Exact Greedy Algorithm"],
                key="tree_methods_select"
            )
            # Translate to XGBoost parameter values
            tree_methods = [fraud_detection_approaches_map[approach] for approach in selected_approaches]

            st.sidebar.markdown(
                "Select the analysis approach to simulate different strategies in fraud detection. "
                "Historical Analysis leverages past patterns, Exact Analysis focuses on precision, "
                "and Approximate Analysis offers a faster, high-level overview."
            )

            # Business-friendly settings for evaluation criteria
            selected_metrics = st.sidebar.multiselect(
                "Evaluation Criteria",
                options=list(eval_metrics_map.keys()),
                default=["Discrimination Ability (AUC)"],
                key="eval_metrics_select"
            )
            # Translate to XGBoost parameter values
            eval_metrics = [eval_metrics_map[metric] for metric in selected_metrics]

            st.sidebar.markdown(
                "Choose the evaluation criteria to assess how well the system distinguishes between normal and suspicious behavior. "
                "AUC measures overall alert efficiency, Misclassification Rate indicates error, and Log Loss quantifies prediction uncertainty."
            )

            # Rule direction selection
            rule_direction = st.sidebar.radio(
                "Rule Direction",
                ["lt", "gt", "both"],
                index=2,  # Default to 'both'
                key="rule_direction_select"
            )
            
            # XGBoost parameters
            # st.sidebar.subheader("XGBoost Parameters")
            # num_boost_round = st.sidebar.slider("Number of Boosting Rounds", 1, 50, 10, key="num_boost_round_slider")
            # max_depth = st.sidebar.slider("Maximum Tree Depth", 1, 10, 3, key="max_depth_slider")
            # min_support = st.sidebar.slider("Minimum Support (samples)", 5, int(df.shape[0] * 0.2), 20, key="min_support_slider")
            # gamma = st.sidebar.number_input("Gamma (min_split_loss)", value=0.0, step=0.1, key="gamma_input")
            # min_child_weight = st.sidebar.number_input("Min Child Weight", value=1.0, step=0.1, key="min_child_weight_input")
            # subsample = st.sidebar.slider("Subsample Ratio", 0.1, 1.0, 1.0, step=0.1, key="subsample_slider")
            # colsample_bytree = st.sidebar.slider("Column Subsample Ratio", 0.1, 1.0, 1.0, step=0.1, key="colsample_bytree_slider")

            st.sidebar.header("Fraud Detection Settings")
            # Investigation Rounds
            num_boost_round = st.sidebar.slider(
                "Investigation Rounds", 1, 50, 10, key="num_boost_round_slider"
            )
            st.sidebar.markdown(
                "The number of iterations that simulate rounds of fraud investigation. More rounds can uncover additional patterns in fraudulent behavior."
            )
            # Alert Complexity
            max_depth = st.sidebar.slider(
                "Alert Complexity", 1, 10, 3, key="max_depth_slider"
            )
            st.sidebar.markdown(
                "Controls the complexity of individual fraud alerts. Simpler alerts are easier to interpret, while more complex alerts may capture subtle fraud patterns."
            )
            # Minimum Suspicious Case Volume
            min_support = st.sidebar.slider(
                "Minimum Suspicious Case Volume", 5, int(df.shape[0] * 0.2), 20, key="min_support_slider"
            )
            st.sidebar.markdown(
                "The minimum number of cases required to consider an alert significant. This helps ensure alerts are based on enough evidence of suspicious activity."
            )
            # Detection Sensitivity Threshold
            gamma = st.sidebar.number_input(
                "Detection Sensitivity Threshold", value=0.0, step=0.1, key="gamma_input"
            )
            st.sidebar.markdown(
                "Determines how sensitive the system is to subtle differences in risk. Lower thresholds can trigger alerts for small improvements in risk detection."
            )
            # Minimum Affected Group Size
            min_child_weight = st.sidebar.number_input(
                "Minimum Affected Group Size", value=1.0, step=0.1, key="min_child_weight_input"
            )
            st.sidebar.markdown(
                "The smallest number of records in a subgroup for it to be considered reliable. This prevents alerts from being based on too few cases."
            )
            # Case Sampling Ratio
            subsample = st.sidebar.slider(
                "Case Sampling Ratio", 0.1, 1.0, 1.0, step=0.1, key="subsample_slider"
            )
            st.sidebar.markdown(
                "The fraction of the dataset used in each investigation round. Adjust this to focus on broad patterns or niche segments of potential fraud."
            )
            # Indicator Diversity Ratio
            colsample_bytree = st.sidebar.slider(
                "Indicator Diversity Ratio", 0.1, 1.0, 1.0, step=0.1, key="colsample_bytree_slider"
            )
            st.sidebar.markdown(
                "The proportion of fraud indicators (features) considered during analysis. Including more indicators can reveal complex fraud schemes."
            )

            # Bundle the new parameters into a dictionary
            extra_params = {
                'gamma': gamma,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree
            }
            # Show current iteration information
            if st.session_state['current_iteration'] > 0:
                st.sidebar.subheader("Current Progress")
                st.sidebar.info(f"Iteration: {st.session_state['current_iteration']}")
                st.sidebar.info(f"Rules Applied: {len(st.session_state['applied_rules'])}")
                
                # Option to reset and start over
                if st.sidebar.button("Start Over", key="start_over_button"):
                    st.session_state['current_df'] = st.session_state['original_df'].copy()
                    st.session_state['applied_rules'] = []
                    st.session_state['current_iteration'] = 0
                    st.session_state['matched_data_history'] = []
                    st.session_state['rule_builder'].clear_conditions()
                    st.rerun()
            
            # Extract rules button
            extract_rules_button = st.sidebar.button("Extract Rules", key="extract_rules_button")
            
            # Main content area
            main_area = st.container()
            
            with main_area:
                # Display previously applied rules
                if st.session_state['applied_rules']:
                    st.subheader("Previously Applied Rules")
                    rules_df = pd.DataFrame(st.session_state['applied_rules'])
                    st.dataframe(rules_df[['Iteration', 'Rule Text', 'Fraud Rate', 'Total', 'Frauds']])
                
                # Extract rules when button is clicked
                if extract_rules_button:
                    # Check if target column is binary
                    if df[target_col].nunique() > 2:
                        st.error(f"Target column '{target_col}' is not binary. Please select a binary target column.")
                    elif len(feature_cols) == 0:
                        st.error("Please select at least one feature column.")
                    else:
                        with st.spinner("Extracting rules..."):
                            # Initialize extractor with current data
                            extractor = RuleExtractor(
                                df, 
                                target_col=target_col, 
                                feature_cols=feature_cols,
                                tree_methods=tree_methods,
                                eval_metrics=eval_metrics,
                                rule_direction=rule_direction
                            )
                            
                            # Extract rules
                            rules_df = extractor.extract_rules_grid(
                                num_boost_round=num_boost_round,
                                max_depth=max_depth,
                                min_support=min_support
                            )
                            
                            if not rules_df.empty:
                                st.session_state['rules_df'] = rules_df
                                st.session_state['extractor'] = extractor
                                st.success(f"Successfully extracted {len(rules_df)} rules!")
                            else:
                                st.warning("No rules extracted. Try adjusting the parameters or changing the data.")
                
                # Display extracted rules if available
                if 'rules_df' in st.session_state:
                    st.header("Extracted Rules")
                    
                    # Select columns to display
                    display_cols = ['Suggested Rule', 'Reward', 'Fraud Rate', 'Total', 'Frauds', 'Feature', 'Threshold', 'Operator']
                    rules_to_show = st.session_state['rules_df'][display_cols].copy()
                    
                    # Format percentages
                    rules_to_show['Fraud Rate'] = rules_to_show['Fraud Rate'].map('{:.2%}'.format)
                    
                    # Show rules with pagination
                    page_size = 10
                    total_pages = (len(rules_to_show) - 1) // page_size + 1
                    page_number = st.number_input(
                        f"Page (1-{total_pages})", 
                        min_value=1, 
                        max_value=total_pages, 
                        value=1,
                        key="page_number"
                    )
                    
                    start_idx = (page_number - 1) * page_size
                    end_idx = min(start_idx + page_size, len(rules_to_show))
                    
                    st.dataframe(rules_to_show.iloc[start_idx:end_idx])
                    
                    # Filter options
                    with st.expander("Filter Rules"):
                        min_fraud_rate = st.slider(
                            "Minimum Fraud Rate", 
                            0.0, 1.0, 0.0, 0.05,
                            key="min_fraud_rate_slider"
                        )
                        
                        min_total = st.slider(
                            "Minimum Sample Size", 
                            0, int(df.shape[0]), 10,
                            key="min_total_slider"
                        )
                        
                        feature_filter = st.multiselect(
                            "Filter by Feature",
                            options=st.session_state['rules_df']['Feature'].unique(),
                            key="feature_filter"
                        )
                        
                        filtered_rules = st.session_state['rules_df'].copy()
                        if min_fraud_rate > 0:
                            filtered_rules = filtered_rules[filtered_rules['Fraud Rate'] >= min_fraud_rate]
                        if min_total > 0:
                            filtered_rules = filtered_rules[filtered_rules['Total'] >= min_total]
                        if feature_filter:
                            filtered_rules = filtered_rules[filtered_rules['Feature'].isin(feature_filter)]
                        
                        if len(filtered_rules) > 0:
                            st.write(f"Found {len(filtered_rules)} rules matching filters:")
                            display_filtered = filtered_rules[display_cols].copy()
                            display_filtered['Fraud Rate'] = display_filtered['Fraud Rate'].map('{:.2%}'.format)
                            st.dataframe(display_filtered)
                            
                            # Use filtered rules for selection
                            st.session_state['filtered_rules'] = filtered_rules
                        else:
                            st.warning("No rules match the filter criteria.")
                    
                    # Rule selection and compound rule builder
                    st.header("Build Multi-Condition Rule")
                    
                    # Choose data to get rules from
                    if 'filtered_rules' in st.session_state and not st.session_state['filtered_rules'].empty:
                        rules_for_selection = st.session_state['filtered_rules']
                    else:
                        rules_for_selection = st.session_state['rules_df']
                    
                    # Current compound rule status
                    rule_builder = st.session_state['rule_builder']
                    
                    if rule_builder.conditions:
                        st.subheader("Current Rule Conditions")
                        for i, condition in enumerate(rule_builder.conditions):
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                st.text(f"{i+1}. {condition}")
                            with col2:
                                if st.button("Remove", key=f"remove_condition_{i}"):
                                    rule_builder.remove_condition(i)
                                    st.rerun()
                        
                        st.info(f"Current Rule: {rule_builder.get_rule_text()}")
                        
                    # Add new condition
                    st.subheader("Add New Condition")

                    selected_rule_idx = st.selectbox(
                        "Select rule to add as condition",
                        range(len(rules_for_selection)),
                        format_func=lambda i: rules_for_selection['Rule Text'].iloc[i],
                        key="rule_selection"
                    )

                    selected_rule = rules_for_selection.iloc[selected_rule_idx]

                    # Add condition to compound rule and re-extract rules
                    if st.button("Add to Compound Rule", key="add_condition_button"):
                        rule_builder.add_condition(selected_rule['Rule Text'])
                        st.success(f"Added condition: {selected_rule['Rule Text']}")
                        
                        # If we have conditions, apply the compound rule to filter data
                        if rule_builder.conditions:
                            extractor = st.session_state['extractor']
                            
                            # Apply compound rule to get filtered data
                            current_mask = rule_builder.apply_rule(extractor, df[extractor.feature_cols])
                            filtered_df = df[current_mask]
                            
                            # Display current compound rule info
                            st.subheader("Current Compound Rule")
                            st.info(f"Rule: {rule_builder.get_rule_text()}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Matched Records", len(filtered_df))
                            with col2:
                                fraud_rate = filtered_df[target_col].mean() if len(filtered_df) > 0 else 0
                                st.metric("Fraud Rate", f"{fraud_rate:.2%}")
                            with col3:
                                st.metric("% of Data", f"{len(filtered_df)/len(df):.2%}")
                            
                            # Check if enough data remains for rule extraction
                            if len(filtered_df) >= min_support:
                                with st.spinner("Extracting new rules on filtered data..."):
                                    # Create new extractor with filtered data
                                    new_extractor = RuleExtractor(
                                        filtered_df,
                                        target_col=target_col,
                                        feature_cols=feature_cols,
                                        tree_methods=tree_methods,
                                        eval_metrics=eval_metrics,
                                        rule_direction=rule_direction
                                    )
                                    
                                    # Extract new rules
                                    new_rules_df = new_extractor.extract_rules_grid(
                                        num_boost_round=num_boost_round,
                                        max_depth=max_depth,
                                        min_support=min_support
                                    )
                                    
                                    if not new_rules_df.empty:
                                        # Display new extracted rules immediately
                                        st.subheader("Extracted Rules from Filtered Data")
                                        
                                        # Select columns to display
                                        display_cols = ['Suggested Rule', 'Reward', 'Fraud Rate', 'Total', 'Frauds', 'Feature', 'Threshold', 'Operator']
                                        display_rules = new_rules_df[display_cols].copy()
                                        
                                        # Format percentages
                                        display_rules['Fraud Rate'] = display_rules['Fraud Rate'].map('{:.2%}'.format)
                                        
                                        # Show top 10 rules
                                        st.dataframe(display_rules.head(10))
                                        
                                        # Update session state
                                        st.session_state['rules_df'] = new_rules_df
                                        st.session_state['extractor'] = new_extractor
                                        st.success(f"Extracted {len(new_rules_df)} new rules from filtered data!")
                                    else:
                                        st.warning("No rules could be extracted from filtered data.")
                            else:
                                st.warning(f"Filtered data has fewer than {min_support} samples. Cannot generate new rules.")
                        
                        # We don't immediately do a full page refresh so the user can see the results
                        if st.button("Continue", key="continue_after_add"):
                            st.rerun()

                                      
                    # Apply compound rule
                    if rule_builder.conditions:
                        st.header("Apply Compound Rule")
                        
                        if st.button("Apply Compound Rule", key="apply_compound_rule"):
                            # Apply the rule
                            extractor = st.session_state['extractor']
                            
                            # Apply compound rule
                            mask = rule_builder.apply_rule(extractor, df[extractor.feature_cols])
                            matched_df = df[mask]
                            remaining_df = df[~mask]
                            
                            # Display results
                            st.subheader("Results after applying compound rule")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Matched Records", len(matched_df))
                                if len(matched_df) > 0:
                                    matched_fraud_rate = matched_df[target_col].mean()
                                    st.metric("Matched Fraud Rate", f"{matched_fraud_rate:.2%}")
                                else:
                                    matched_fraud_rate = 0
                                    st.metric("Matched Fraud Rate", "0.00%")
                                
                            with col2:
                                st.metric("Remaining Records", len(remaining_df))
                                if len(remaining_df) > 0:
                                    remaining_fraud_rate = remaining_df[target_col].mean()
                                    st.metric("Remaining Fraud Rate", f"{remaining_fraud_rate:.2%}")
                                else:
                                    remaining_fraud_rate = 0
                                    st.metric("Remaining Fraud Rate", "0.00%")
                            
                            # Show matched and remaining data
                            tab1, tab2 = st.tabs(["Matched Data", "Remaining Data"])
                            with tab1:
                                st.write(f"Data matching rule: **{rule_builder.get_rule_text()}**")
                                st.dataframe(matched_df.head(10))
                                
                            with tab2:
                                st.write("Remaining data (not matching rule)")
                                st.dataframe(remaining_df.head(10))
                            
                            # Confirm application of rule
                            st.subheader("Confirm Rule Application")
                            confirm_text = (
                                f"Are you sure you want to apply this compound rule and continue with the remaining {len(remaining_df)} records? "
                                f"This will remove {len(matched_df)} records ({len(matched_df)/len(df):.1%} of current data) "
                                f"with fraud rate of {matched_fraud_rate:.2%}."
                            )
                            st.write(confirm_text)
                            
                            # Store these temporarily for the confirmation step
                            st.session_state['temp_matched_df'] = matched_df
                            st.session_state['temp_remaining_df'] = remaining_df
                            
                            # Option to save matched data for further analysis
                            if st.button("Save Matched Data Without Removing", key="save_matched_button"):
                                # Save matched data to history without removing from current dataset
                                matched_data_entry = {
                                    "Iteration": st.session_state['current_iteration'],
                                    "Rule": rule_builder.get_rule_text(),
                                    "Data": matched_df.copy(),
                                    "Fraud Rate": matched_fraud_rate,
                                    "Count": len(matched_df)
                                }
                                st.session_state['matched_data_history'].append(matched_data_entry)
                                st.success(f"Saved {len(matched_df)} matched records for analysis without removing from dataset")
                            
                            if st.button("Confirm and Continue", key="confirm_button"):
                                # Update the current data to the remaining subset
                                st.session_state['current_df'] = remaining_df.copy()
                                
                                # Add rule to applied rules
                                rule_dict = {
                                    'Iteration': st.session_state['current_iteration'] + 1,
                                    'Rule Text': rule_builder.get_rule_text(),
                                    'Fraud Rate': matched_fraud_rate,
                                    'Total': len(matched_df),
                                    'Frauds': int(matched_df[target_col].sum()),
                                    'Samples Removed': len(matched_df),
                                    'Percent Removed': len(matched_df) / len(df)
                                }
                                st.session_state['applied_rules'].append(rule_dict)
                                
                                # Save matched data to history
                                matched_data_entry = {
                                    "Iteration": st.session_state['current_iteration'] + 1,
                                    "Rule": rule_builder.get_rule_text(),
                                    "Data": matched_df.copy(),
                                    "Fraud Rate": matched_fraud_rate,
                                    "Count": len(matched_df)
                                }
                                st.session_state['matched_data_history'].append(matched_data_entry)
                                
                                # Increment iteration counter
                                st.session_state['current_iteration'] += 1
                                
                                # Clear the rule builder for the next iteration
                                rule_builder.clear_conditions()
                                
                                # Clear rules to force re-extraction on the new subset
                                #if 'rules_df' in st.session_state:
                                # Clear rules to force re-extraction on the new subset
                                    # Clear rules to force re-extraction on the new subset
                                if 'rules_df' in st.session_state:
                                    del st.session_state['rules_df']
                                if 'filtered_rules' in st.session_state:
                                    del st.session_state['filtered_rules']
                                
                                st.success(f"Rule applied! Removed {len(matched_df)} records. {len(remaining_df)} records remaining.")
                                st.rerun()
                    
                    # View matched data history
                    if st.session_state['matched_data_history']:
                        st.header("View Previous Matched Segments")
                        
                        # Create selectbox with rules and their metrics
                        history_items = [
                            f"Iteration {item['Iteration']}: {item['Rule']} ({item['Count']} records, {item['Fraud Rate']:.2%} fraud rate)"
                            for item in st.session_state['matched_data_history']
                        ]
                        
                        selected_history_index = st.selectbox(
                            "Select a previously saved segment to view",
                            range(len(history_items)),
                            format_func=lambda i: history_items[i],
                            key="history_select"
                        )
                        
                        selected_history = st.session_state['matched_data_history'][selected_history_index]
                        
                        st.subheader(f"Data from Iteration {selected_history['Iteration']}")
                        st.write(f"Rule: {selected_history['Rule']}")
                        st.metric("Records", selected_history['Count'])
                        st.metric("Fraud Rate", f"{selected_history['Fraud Rate']:.2%}")
                        
                        st.dataframe(selected_history['Data'].head(10))
                        
                        # Export button
                        if st.button("Export this segment to CSV", key="export_segment"):
                            segment_csv = selected_history['Data'].to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=segment_csv,
                                file_name=f"segment_iteration_{selected_history['Iteration']}.csv",
                                mime="text/csv"
                            )
    else:
        st.info("Please upload a CSV file to get started.")
        
        # Sample dataset info
        st.header("How to use this tool")
        st.write("""
        1. Upload a CSV file containing your dataset
        2. Select the target column (should be binary: 0/1)
        3. Choose feature columns to use for rule extraction
        4. Configure XGBoost parameters
        5. Extract rules
        6. Build compound rules by combining multiple conditions
        7. Apply rules iteratively to segment your data
        
        This tool helps you identify important patterns in your data using XGBoost and convert them into human-readable rules.
        It's especially useful for fraud detection, risk assessment, and other classification tasks where explainability is important.
        """)
        
        # Example of expected data format
        st.subheader("Expected Data Format")
        st.write("""
        Your CSV file should have:
        - A binary target column (e.g., 0 for normal, 1 for fraud)
        - Multiple feature columns with numerical values
        
        Example:
        """)
        
        example_data = {
            "transaction_amount": [120.50, 500.00, 45.25, 1200.75, 85.30],
            "transaction_hour": [14, 2, 12, 23, 9],
            "customer_age": [32, 45, 19, 27, 53],
            "days_since_last_purchase": [5, 120, 2, 15, 30],
            "is_fraud": [0, 1, 0, 1, 0]
        }
        st.dataframe(pd.DataFrame(example_data))

if __name__ == "__main__":
    main()