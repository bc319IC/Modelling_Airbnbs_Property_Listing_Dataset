import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from tabular_data import load_airbnb
import numpy as np
from itertools import product
from sklearn.model_selection import GridSearchCV
import os
import joblib
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from joblib import load

def train_linear_regression():
    """
    Trains a linear regression model using SGDRegressor to predict 'Price_Night'.
    The data is split into training and test sets, and the model's performance is evaluated.
    """
    # Load features and labels
    features, labels = load_airbnb(label='Price_Night')
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    # Initialize the SGDRegressor model
    model = SGDRegressor()
    # Train the model on the training data
    model.fit(X_train, y_train)
    # Make predictions on the training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    # Calculate performance metrics for the training set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    # Calculate performance metrics for the test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    # Output the results
    print("Training set performance:")
    print(f"RMSE: {train_rmse}")
    print(f"R^2: {train_r2}")
    print("\nTest set performance:")
    print(f"RMSE: {test_rmse}")
    print(f"R^2: {test_r2}")

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams):
    """
    Perform a grid search over a range of hyperparameter values for a given regression model class.
    
    Parameters:
        model_class (class): The regression model class (e.g., SGDRegressor).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        hyperparams (dict): Dictionary mapping hyperparameter names to lists of values to try.
        
    Returns:
        best_model (object): The model instance with the best hyperparameters.
        best_hyperparams (dict): The best hyperparameter values.
        best_metrics (dict): Performance metrics of the best model.
    """
    # Store the best model and hyperparameters
    best_model = None
    best_hyperparams = None
    best_validation_rmse = np.inf  # Start with a very high RMSE for comparison
    # Get the keys (parameter names) and values (lists of parameter values) from the hyperparams dictionary
    param_names = list(hyperparams.keys())
    param_values = list(hyperparams.values())
    # Iterate over all combinations of hyperparameter values
    for param_combination in product(*param_values):
        # Create a dictionary of the current combination of hyperparameters
        current_params = dict(zip(param_names, param_combination))
        # Initialize the model with the current hyperparameters
        model = model_class(**current_params)
        # Train the model on the training data
        model.fit(X_train, y_train)
        # Predict on the validation set
        y_val_pred = model.predict(X_val)
        # Calculate validation RMSE
        validation_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        # If this is the best model so far (lowest validation RMSE), save it
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_model = model
            best_hyperparams = current_params
    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    # Collect performance metrics
    best_metrics = {
        'validation_RMSE': best_validation_rmse,
        'test_RMSE': test_rmse,
        'test_R^2': test_r2
    }
    return best_model, best_hyperparams, best_metrics

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    """
    Tunes the hyperparameters of the given model class using the provided training, validation, and test data.
    
    Parameters:
    - model_class: The regression model class to be tuned.
    - X_train, y_train: Training features and labels.
    - X_val, y_val: Validation features and labels.
    - X_test, y_test: Test features and labels.
    - param_grid: Dictionary of hyperparameter names mapping to a list of values to be tried.
    
    Returns:
    - best_model: The best model found during tuning.
    - best_params: Dictionary of the best hyperparameter values.
    - best_metrics: Dictionary of performance metrics (validation and test RMSE and R²).
    """
    best_model = None
    best_params = None
    best_val_rmse = float('inf')
    best_metrics = {}
    # Iterate through all combinations of hyperparameter values
    from itertools import product
    for param_values in product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), param_values))
        model = model_class(**params)  # Initialize the model with the current hyperparameters
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Validation performance
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        # Check if this model is better (lower validation RMSE)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model
            best_params = params
            # Test set performance
            y_test_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_r2 = r2_score(y_test, y_test_pred)
            # Store the best metrics
            best_metrics = {
                'validation_RMSE': best_val_rmse,
                'test_RMSE': test_rmse,
                'test_R2': test_r2
            }
    return best_model, best_params, best_metrics

def save_model(best_model, best_params, best_metrics, folder="models/regression/linear_regression"):
    """
    Saves the model, its hyperparameters, and performance metrics.

    Parameters:
    - folder: The path where the files should be saved.
    """
    # Ensure the directory exists
    os.makedirs(folder, exist_ok=True)
    model = best_model  # The trained model from your tuning function
    best_params = best_params  # The best hyperparameters from your tuning function
    best_metrics = best_metrics  # The best metrics from your tuning function
    # Save the model
    model_file_path = os.path.join(folder, "model.joblib")
    joblib.dump(model, model_file_path)
    # Save hyperparameters
    hyperparameters_file_path = os.path.join(folder, "hyperparameters.json")
    with open(hyperparameters_file_path, 'w') as f:
        json.dump(best_params, f)
    # Save metrics
    metrics_file_path = os.path.join(folder, "metrics.json")
    with open(metrics_file_path, 'w') as f:
        json.dump(best_metrics, f)
    print(f"Model, hyperparameters, and metrics saved in {folder}")

def evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluates multiple regression models, tunes their hyperparameters, and saves the best models.
    
    Models evaluated:
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    
    Parameters:
    - X_train, y_train: Training features and labels.
    - X_val, y_val: Validation features and labels.
    - X_test, y_test: Test features and labels.
    """
    # Define model classes and their hyperparameter grids
    model_configs = {
        'decision_tree': {
            'model_class': DecisionTreeRegressor,
            'param_grid': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'random_forest': {
            'model_class': RandomForestRegressor,
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'gradient_boosting': {
            'model_class': GradientBoostingRegressor,
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        }
    }
    # Loop through each model and evaluate
    for model_name, config in model_configs.items():
        print(f"Evaluating {model_name.replace('_', ' ').title()}...")
        # Tune the model's hyperparameters using the custom function
        best_model, best_params, best_metrics = tune_regression_model_hyperparameters(
            config['model_class'], X_train, y_train, X_val, y_val, X_test, y_test, config['param_grid']
        )
        # Save the model, hyperparameters, and metrics in the respective folder
        folder_path = os.path.join('models', 'regression', model_name)
        save_model(best_model, best_params, best_metrics, folder=folder_path)
        print(f"Best {model_name.replace('_', ' ').title()} saved in {folder_path}.\n")

def find_best_model(base_folder='models/regression'):
    """
    Finds the best regression model by comparing test RMSE of all saved models.
    
    Parameters:
    - base_folder: The folder containing subfolders for each model.
    
    Returns:
    - best_model: The best performing model based on test RMSE.
    - best_hyperparameters: Dictionary of hyperparameters of the best model.
    - best_metrics: Dictionary of performance metrics of the best model.
    """
    best_model = None
    best_hyperparameters = None
    best_metrics = None
    best_test_rmse = float('inf')  # Start with a large value for comparison
    # Loop through each model subfolder
    for model_folder in os.listdir(base_folder):
        model_path = os.path.join(base_folder, model_folder)
        if os.path.isdir(model_path):
            # Load the model, hyperparameters, and metrics
            model_file = os.path.join(model_path, 'model.joblib')
            hyperparameters_file = os.path.join(model_path, 'hyperparameters.json')
            metrics_file = os.path.join(model_path, 'metrics.json')
            if os.path.exists(model_file) and os.path.exists(hyperparameters_file) and os.path.exists(metrics_file):
                # Load the model
                model = load(model_file)
                # Load the hyperparameters
                with open(hyperparameters_file, 'r') as f:
                    hyperparameters = json.load(f)
                # Load the metrics
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                # Compare based on test RMSE
                test_rmse = metrics.get('test_RMSE', float('inf'))
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_model = model
                    best_hyperparameters = hyperparameters
                    best_metrics = metrics
    return best_model, best_hyperparameters, best_metrics



if __name__ == "__main__":
    # Load the data
    features, labels = load_airbnb(label='Price_Night')
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # Evalaute linear model
    '''
    # Define the parameter grid for tuning
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'max_iter': [1000, 5000, 10000],
        'penalty': ['l2', 'l1', 'elasticnet']
    }
    # Tune the model hyperparameters
    best_model, best_params, best_metrics = tune_regression_model_hyperparameters(
        SGDRegressor, X_train, y_train, X_val, y_val, X_test, y_test, param_grid
    )
    save_model(best_model, best_params, best_metrics)
    '''
    # Evaluate all models
    evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test)
    # Find the best model
    best_model, best_hyperparameters, best_metrics = find_best_model()
    print(f"Best model found: {best_model}")
    print(f"Hyperparameters: {best_hyperparameters}")
    print(f"Performance metrics: {best_metrics}")