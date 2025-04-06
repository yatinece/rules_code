import pandas as pd
import numpy as np
import time

base_dt=pd.read_csv("./creditcardfraud/creditcard.csv")
print(base_dt.columns)


X = base_dt.drop(columns=['Time', 'Class']).astype(float)  # convert features to float
y = base_dt['Class'].astype(int)  # assuming Class is 0/1


def find_composite_rule(X, y, max_features=3, fraud_threshold=0.25, forced_features=None):
    """
    Greedy search for a composite rule of the form:
       if (feature_1 > t1) AND (feature_2 > t2) AND ... then fraud rate >= fraud_threshold.
       
    Parameters:
      X (pd.DataFrame): feature DataFrame.
      y (pd.Series): binary target series (fraud=1, non-fraud=0).
      max_features (int): maximum number of features to include in the composite rule.
      fraud_threshold (float): minimum fraud ratio required on the selected subset.
      forced_features (dict): a dictionary of forced features with fixed thresholds,
                              e.g. {'V3': 0.5}. These conditions are always included.
    
    Returns:
      rule (dict): dictionary mapping selected feature names to their thresholds.
      final_fraud_rate (float): fraud ratio on the subset defined by the composite rule.
      n_selected (int): number of rows in the subset.
    """
    forced_features = forced_features or {}
    rule = {}  # dictionary to hold feature: threshold pairs
    # Start with all data points (use boolean mask)
    candidate_mask = np.ones(len(X), dtype=bool)
    
    # Apply forced features first:
    for feat, thresh in forced_features.items():
        if feat in X.columns:
            rule[feat] = thresh
            candidate_mask = candidate_mask & (X[feat] > thresh)
    
    remaining_features = [f for f in X.columns if f not in rule]
    
    # Greedy addition of features:
    while len(rule) < max_features and remaining_features:
        best_candidate = None
        best_candidate_thresh = None
        best_candidate_fraud_rate = 0
        best_candidate_mask = None
        
        # For each remaining feature, try to find a threshold that when added to current rule,
        # yields a fraud rate above fraud_threshold, and pick the candidate that maximizes the fraud rate.
        for feat in remaining_features:
            df_temp = pd.DataFrame({'feat': X.loc[candidate_mask, feat], 'target': y.loc[candidate_mask]})
            df_temp = df_temp.sort_values(by='feat')
            
            candidate_thresh = None
            candidate_fraud_rate = 0
            candidate_mask_temp = None
            
            # Check each unique value as potential threshold:
            for val in df_temp['feat'].unique():
                temp_mask = candidate_mask & (X[feat] > val)
                if temp_mask.sum() == 0:
                    continue
                temp_fraud_rate = y[temp_mask].mean()
                # Only consider if fraud rate meets our threshold and improves over current candidate:
                if temp_fraud_rate >= fraud_threshold and temp_fraud_rate > candidate_fraud_rate:
                    candidate_thresh = val
                    candidate_fraud_rate = temp_fraud_rate
                    candidate_mask_temp = temp_mask
            
            # Select the best candidate across features:
            if candidate_thresh is not None and candidate_fraud_rate > best_candidate_fraud_rate:
                best_candidate = feat
                best_candidate_thresh = candidate_thresh
                best_candidate_fraud_rate = candidate_fraud_rate
                best_candidate_mask = candidate_mask_temp
        
        # If no candidate improves the fraud rate, stop
        if best_candidate is None:
            break
        
        # Update rule and candidate mask:
        rule[best_candidate] = best_candidate_thresh
        candidate_mask = best_candidate_mask
        remaining_features.remove(best_candidate)
    
    final_fraud_rate = y[candidate_mask].mean() if candidate_mask.sum() > 0 else 0
    n_selected = candidate_mask.sum()
    
    return rule, final_fraud_rate, n_selected

# Example usage:
# Assume df is your DataFrame with features, and y is your target series.
# For instance, force feature 'V3' with threshold 0.5, and allow a maximum of 3 features in the rule.

 # Save timestamp
start = time.time()
rule, fraud_rate, count = find_composite_rule(X, y, max_features=3, fraud_threshold=0.15, forced_features={})
    # Save timestamp
end = time.time()
print(f"time taken: {end - start}")
print("Composite Rule:", rule)
print("Fraud Rate on selected subset: {:.2%}".format(fraud_rate))
print("Number of rows selected:", count)

