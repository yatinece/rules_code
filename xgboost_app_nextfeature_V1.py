import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import re
import itertools
from collections import defaultdict

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

    def _train_model(self, X, y, num_boost_round, max_depth, tree_method, eval_metric):
        """Helper method that trains an XGBoost model and returns the model dump."""
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'max_depth': max_depth,
            'objective': 'binary:logistic',
            'tree_method': tree_method,
            'eval_metric': eval_metric,
            'seed': 42  # For reproducibility
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
            st.write(f"Training with tree_method: {tm} and eval_metric: {em}")

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

# Main Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="XGBoost Rule Extraction")
    st.title("XGBoost Rule Extraction Dashboard")
    st.write("""
    This dashboard allows you to extract rules from XGBoost models and apply them iteratively to your data.
    Upload your data, configure parameters, extract rules, and apply them one by one to find patterns.
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
            
            # Tree methods selection
            tree_methods = st.sidebar.multiselect(
                "Tree Methods",
                ["hist", "exact", "approx", "gpu_hist"],
                default=["hist"],
                key="tree_methods_select"
            )
            
            # Evaluation metrics selection
            eval_metrics = st.sidebar.multiselect(
                "Evaluation Metrics",
                ["auc", "error", "logloss"],
                default=["auc"],
                key="eval_metrics_select"
            )
            
            # Rule direction selection
            rule_direction = st.sidebar.radio(
                "Rule Direction",
                ["lt", "gt", "both"],
                index=2,  # Default to 'both'
                key="rule_direction_select"
            )
            
            # XGBoost parameters
            st.sidebar.subheader("XGBoost Parameters")
            num_boost_round = st.sidebar.slider("Number of Boosting Rounds", 1, 50, 10, key="num_boost_round_slider")
            max_depth = st.sidebar.slider("Maximum Tree Depth", 1, 10, 3, key="max_depth_slider")
            min_support = st.sidebar.slider("Minimum Support (samples)", 5, int(df.shape[0] * 0.2), 20, key="min_support_slider")
            
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
                    st.experimental_rerun()
            
            # Extract rules button
            extract_rules_button = st.sidebar.button("Extract Rules", key="extract_rules_button")
            
            # Main content area
            main_area = st.container()
            
            with main_area:
                # Display previously applied rules
                if st.session_state['applied_rules']:
                    st.subheader("Previously Applied Rules")
                    rules_df = pd.DataFrame(st.session_state['applied_rules'])
                    st.dataframe(rules_df[['Iteration', 'Suggested Rule', 'Fraud Rate', 'Total', 'Frauds']])
                
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
                    
                    # Rule selection
                    st.header("Apply Rule to Data")
                    
                    if 'filtered_rules' in st.session_state and not st.session_state['filtered_rules'].empty:
                        rules_for_selection = st.session_state['filtered_rules']
                    else:
                        rules_for_selection = st.session_state['rules_df']
                    
                    selected_rule_idx = st.selectbox(
                        "Select rule to apply",
                        range(len(rules_for_selection)),
                        format_func=lambda i: rules_for_selection['Suggested Rule'].iloc[i],
                        key="rule_selection"
                    )
                    
                    selected_rule = rules_for_selection.iloc[selected_rule_idx]
                    
                    if st.button("Apply Selected Rule", key="apply_rule_button"):
                        # Apply the rule
                        extractor = st.session_state['extractor']
                        rule_text = selected_rule['Rule Text']
                        
                        mask = extractor._apply_rule(extractor.X, rule_text)
                        matched_df = df[mask]
                        remaining_df = df[~mask]
                        
                        # Display results
                        st.subheader("Results after applying rule")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Matched Records", len(matched_df))
                            matched_fraud_rate = matched_df[target_col].mean()
                            st.metric("Matched Fraud Rate", f"{matched_fraud_rate:.2%}")
                            
                        with col2:
                            st.metric("Remaining Records", len(remaining_df))
                            remaining_fraud_rate = remaining_df[target_col].mean()
                            st.metric("Remaining Fraud Rate", f"{remaining_fraud_rate:.2%}")
                        
                        # Show matched and remaining data
                        tab1, tab2 = st.tabs(["Matched Data", "Remaining Data"])
                        with tab1:
                            st.write(f"Data matching rule: **{selected_rule['Suggested Rule']}**")
                            st.dataframe(matched_df.head(10))
                            
                        with tab2:
                            st.write("Remaining data (not matching rule)")
                            st.dataframe(remaining_df.head(10))
                        
                        # Confirm application of rule
                        st.subheader("Confirm Rule Application")
                        confirm_text = (
                            f"Are you sure you want to apply this rule and continue with the remaining {len(remaining_df)} records? "
                            f"This will remove {len(matched_df)} records ({len(matched_df)/len(df):.1%} of current data) "
                            f"with fraud rate of {matched_fraud_rate:.2%}."
                        )
                        
                        # Store these temporarily for the confirmation step
                        st.session_state['temp_matched_df'] = matched_df
                        st.session_state['temp_remaining_df'] = remaining_df
                        st.session_state['temp_selected_rule'] = selected_rule
                        
                        if st.button("Confirm and Continue", key="confirm_button"):
                            # Update the current data to the remaining subset
                            st.session_state['current_df'] = st.session_state['temp_remaining_df'].copy()
                            
                            # Add rule to applied rules
                            rule_dict = st.session_state['temp_selected_rule'].to_dict()
                            rule_dict['Iteration'] = st.session_state['current_iteration'] + 1
                            rule_dict['Samples Removed'] = len(st.session_state['temp_matched_df'])
                            rule_dict['Percent Removed'] = len(st.session_state['temp_matched_df']) / len(df)
                            st.session_state['applied_rules'].append(rule_dict)
                            
                            # Increment iteration counter
                            st.session_state['current_iteration'] += 1
                            
                            # Clear rules to force re-extraction on the new subset
                            if 'rules_df' in st.session_state:
                                del st.session_state['rules_df']
                            if 'filtered_rules' in st.session_state:
                                del st.session_state['filtered_rules']
                                
                            st.success(f"Rule applied! Continuing with {len(st.session_state['current_df'])} remaining records.")
                            
                            # Allow downloading filtered data
                            st.download_button(
                                label="Download Matched Data",
                                data=st.session_state['temp_matched_df'].to_csv(index=False).encode('utf-8'),
                                file_name=f'matched_data_rule_{st.session_state["current_iteration"]}.csv',
                                mime='text/csv'
                            )
                            
                            # Rerun to refresh the UI with the new data
                            st.experimental_rerun()
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()