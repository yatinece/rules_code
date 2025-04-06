import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ruleopt import RUGClassifier
from ruleopt.rule_cost import Gini, Length
from ruleopt.solver import ORToolsSolver

# Configuration Dictionary
CONFIG = {
    'dataset_path': "./creditcardfraud/creditcard.csv",
    'test_size': 0.2,
    'random_state': 42,
    'optuna_trials': 100
}

# Load the dataset
def load_data(config):
    df = pd.read_csv(config['dataset_path'])
    X = df.drop(columns=['Time', 'Class']).astype(float)
    y = df['Class'].astype(int)
    return train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

# Objective function for Optuna
def objective(trial):
    X_train, X_test, y_train, y_test = load_data(CONFIG)

    # Suggest hyperparameters
    penalty = trial.suggest_float("penalty", 1, 10.0)
    max_rmp_calls = trial.suggest_int("max_rmp_calls", 0, 20)
    max_depth = trial.suggest_int("max_depth", 1, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    class_weight_option = trial.suggest_categorical("class_weight", ["balanced", None])
    cost_function_choice = trial.suggest_categorical("cost_function", ["Gini", "Length"])
    cost_function = Gini() if cost_function_choice == "Gini" else Length()

    tree_parameters = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight_option,
    }

    # Initialize solver and classifier
    solver = ORToolsSolver(penalty=penalty)
    rug = RUGClassifier(
        solver=solver,
        random_state=CONFIG['random_state'],
        max_rmp_calls=max_rmp_calls,
        rule_cost=cost_function,
        **tree_parameters
    )

    # Train and evaluate
    rug.fit(X_train, y_train)
    y_pred = rug.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main function to execute the optimization
def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=CONFIG['optuna_trials'])

    # Output best parameters and accuracy
    print("Best parameters:", study.best_params)
    print(f"Best accuracy: {study.best_value:.2f}")

    # Optional: Train final model with best parameters and evaluate
    X_train, X_test, y_train, y_test = load_data(CONFIG)
    best_params = study.best_params
    solver = ORToolsSolver(penalty=best_params['penalty'])
    cost_function = Gini() if best_params['cost_function'] == "Gini" else Length()
    tree_parameters = {
        "max_depth": best_params['max_depth'],
        "min_samples_leaf": best_params['min_samples_leaf'],
        "class_weight": best_params['class_weight'],
    }

    final_model = RUGClassifier(
        solver=solver,
        random_state=CONFIG['random_state'],
        max_rmp_calls=best_params['max_rmp_calls'],
        rule_cost=cost_function,
        **tree_parameters
    )

    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
