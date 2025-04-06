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
        dtrain = xgb.DMatrix(self.X, label=self.y)
        all_splits = []

        for tm, em in itertools.product(self.tree_methods, self.eval_metrics):
            st.write(f"Training with tree_method: {tm} and eval_metric: {em}")
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

# Main Streamlit app
def main():
    st.title("XGBoost Rule Extraction Dashboard")
    st.write("""
    This dashboard allows you to extract rules from XGBoost models and apply them to your data.
    Upload your data, configure parameters, and extract rules to identify patterns.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Configuration sidebar
        st.sidebar.header("Rule Extraction Configuration")
        
        # Target column selection
        target_col = st.sidebar.selectbox("Target Column", df.columns.tolist())
        
        # Feature columns selection (multi-select with all columns except target)
        potential_features = [col for col in df.columns if col != target_col]
        feature_cols = st.sidebar.multiselect(
            "Feature Columns", 
            potential_features,
            default=potential_features
        )
        
        # Tree methods selection
        tree_methods = st.sidebar.multiselect(
            "Tree Methods",
            ["hist", "exact", "approx", "gpu_hist"],
            default=["hist", "approx"]
        )
        
        # Evaluation metrics selection
        eval_metrics = st.sidebar.multiselect(
            "Evaluation Metrics",
            ["auc", "error", "logloss"],
            default=["auc"]
        )
        
        # Rule direction selection
        rule_direction = st.sidebar.radio(
            "Rule Direction",
            ["lt", "gt", "both"],
            index=2  # Default to 'both'
        )
        
        # XGBoost parameters
        st.sidebar.subheader("XGBoost Parameters")
        num_boost_round = st.sidebar.slider("Number of Boosting Rounds", 1, 50, 10)
        max_depth = st.sidebar.slider("Maximum Tree Depth", 1, 10, 3)
        min_support = st.sidebar.slider("Minimum Support (samples)", 5, 100, 20)
        
        # Initialize extractor when user clicks button
        if st.sidebar.button("Extract Rules"):
            # Check if target column is binary
            if df[target_col].nunique() > 2:
                st.error(f"Target column '{target_col}' is not binary. Please select a binary target column.")
            elif len(feature_cols) == 0:
                st.error("Please select at least one feature column.")
            else:
                with st.spinner("Extracting rules..."):
                    # Initialize extractor
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
                        st.session_state['df'] = df
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
            
            # Show rules
            st.dataframe(rules_to_show)
            
            # Rule selection
            st.header("Apply Rule to Data")
            selected_rule_idx = st.selectbox(
                "Select rule to apply",
                range(len(st.session_state['rules_df'])),
                format_func=lambda i: st.session_state['rules_df']['Suggested Rule'].iloc[i]
            )
            
            selected_rule = st.session_state['rules_df'].iloc[selected_rule_idx]
            
            if st.button("Apply Selected Rule"):
                # Apply the rule
                extractor = st.session_state['extractor']
                rule_text = selected_rule['Rule Text']
                
                mask = extractor._apply_rule(extractor.X, rule_text)
                matched_df = st.session_state['df'][mask]
                remaining_df = st.session_state['df'][~mask]
                
                # Display results
                st.subheader("Results after applying rule")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Matched Records", len(matched_df))
                    st.metric("Matched Fraud Rate", f"{matched_df[target_col].mean():.2%}")
                    
                with col2:
                    st.metric("Remaining Records", len(remaining_df))
                    st.metric("Remaining Fraud Rate", f"{remaining_df[target_col].mean():.2%}")
                
                # Show matched and remaining data
                tab1, tab2 = st.tabs(["Matched Data", "Remaining Data"])
                with tab1:
                    st.write(f"Data matching rule: **{selected_rule['Suggested Rule']}**")
                    st.dataframe(matched_df)
                    
                with tab2:
                    st.write("Remaining data (not matching rule)")
                    st.dataframe(remaining_df)
                
                # Allow downloading results
                st.download_button(
                    label="Download Matched Data",
                    data=matched_df.to_csv(index=False).encode('utf-8'),
                    file_name='matched_data.csv',
                    mime='text/csv'
                )
                
                st.download_button(
                    label="Download Remaining Data",
                    data=remaining_df.to_csv(index=False).encode('utf-8'),
                    file_name='remaining_data.csv',
                    mime='text/csv'
                )
                
                # Continue with the remaining data
                if st.button("Continue with Remaining Data"):
                    st.session_state['df'] = remaining_df
                    # Clear rules to force re-extraction on the new subset
                    del st.session_state['rules_df']
                    st.success("Data updated to the remaining subset. Please extract rules again.")
                    st.experimental_rerun()

if __name__ == "__main__":
    main()