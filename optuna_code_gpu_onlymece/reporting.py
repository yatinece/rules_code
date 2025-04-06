import pandas as pd
import numpy as np
import time
import pygad
import optuna
import os
from datetime import datetime
import csv
import torch
import logging

# logging.basicConfig(
#     filename='myapp.log',
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
logging.basicConfig(level=logging.INFO, format='%(message)s')

class ReportingResults:
    def __init__(self, config):
        self.config = config
        self.original_dt = pd.read_csv(config['dataset_path'])
        
        # Set up device (GPU or CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create time-based results directory
        self.results_dir = self.create_results_directory()
        self.rule_location = config['read_rule_location']

    def create_results_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"./reporting/reporting_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        return results_dir
    
    def read_rules(self , rule_filename= "final_mece_rules.csv"):
        rules = pd.read_csv(os.path.join(self.rule_location, rule_filename))
        logging.info("\n%s", rules.head().to_string(index=False))

        rules.sort_values(by='Fraud_Count', ascending=False, inplace=True)

        rules = rules[~(rules['Rule'].str.contains("Default: Catch-all"))]

        rules['Rule'] = rules['Rule'].apply(lambda x: x.replace( ";" ," and"))
        logging.info("[rules sorted by fraud count]\n%s", rules.head().to_string(index=False))
        return rules

    def read_original_data(self):
        original_data = pd.read_csv(self.config['dataset_path'])
        logging.info("\n%s", original_data.head().to_string(index=False))
        original_data['Class'] = original_data['Class'].astype(int)
        X = original_data.drop(columns=['Time', 'Class']).astype(float)
        y = original_data['Class'].astype(int)

        X_norm = (X - X.min()) / (X.max() - X.min())
        X_norm[['Class', 'Amount']] = original_data[['Class', 'Amount']]
        logging.info("\n%s", original_data.head().to_string(index=False))
        return X_norm
    
    def apply_rules(self):
        # Apply rules to original data
        rules = self.read_rules()
        original_data = self.read_original_data()
        dataset_rules_columns = []
        for index, row in rules.iterrows():
            logging.info("\n%s", row['Rule'])
            logging.info("\n%s", original_data.head().to_string(index=False))
            logging.info("\n%s", rules.columns)
            original_data['Rule_' + str(row['Run']) + "_" + str(row['Iteration'])] = original_data.eval(row['Rule'])
            dataset_rules_columns.append('Rule_' + str(row['Run']) + "_" + str(row['Iteration']))

        logging.info("\n%s", rules.head().to_string(index=False))
        return rules, original_data, dataset_rules_columns
    
    def reporting_results(self):

        rules, original_data, dataset_rules_columns = self.apply_rules()

        # Group by rule match combinations
        grouped = original_data.groupby(dataset_rules_columns).agg({
            'Class': 'sum',
            'Amount': 'sum'
        }).reset_index()

        # Calculate fraud rate and fraud amount percentage
        grouped['Total_Rows'] = grouped.groupby(dataset_rules_columns)['Class'].transform('count')  



        # Create DataFrame from summary
        combined_kpi_df = pd.DataFrame(grouped)
        combined_kpi_df.to_csv(os.path.join(self.results_dir, "combined_kpi_df.csv"), index=False)
        logging.info("[COMBINED_KPI_DF] ", "saved file to ", os.path.join(self.results_dir, "combined_kpi_df.csv"))
        grouped.to_csv(os.path.join(self.results_dir, "grouped.csv"), index=False)
        logging.info("[GROUPED] ", "saved file to ", os.path.join(self.results_dir, "grouped.csv"))
        # Optional:
        logging.info("[COMBINED_KPI_DF] \n combined_kpi_df %s", combined_kpi_df.to_string(index=False))
        # Reporting results
        logging.info("[RULES] \n%s", rules.head().to_string(index=False))
        logging.info("[ORIGINAL_DATA] \n%s", original_data.head().to_string(index=False))
        logging.info("[DATASET_RULE_COLUMNS] \n%s", dataset_rules_columns)

        logging.info("[new shape]\n%s ", original_data.shape)
        for column in dataset_rules_columns:
            logging.info("[COLUMN] \n%s", column)
            grouped = original_data.groupby(column).agg({
                'Class': 'sum',
                'Amount': 'sum'
            }).reset_index()

            logging.info("[column iterated on]\n%s", f"{str(column)}\n{grouped.head().to_string(index=False)}")

            original_data=original_data[~original_data[column]]
            logging.info("[new shape]\n%s ", original_data.shape)










# Main script
if __name__ == "__main__":
    start_time = time.time()
    
    # Original configuration
    BASE_CONFIG = {
        # Dataset Configuration
        'dataset_path': "./creditcardfraud/creditcard.csv",
        'read_rule_location': "./optuna_code_gpu_onlymece/fraud_detection_results_20250406_012207",

    }
    
    optimizer = ReportingResults(BASE_CONFIG)
    optimizer.reporting_results()
    
    end_time = time.time()
    print(f"Time taken for complete execution: {end_time - start_time:.2f} seconds")