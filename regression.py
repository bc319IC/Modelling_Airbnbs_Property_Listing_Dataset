from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
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

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    """
    Tunes the hyperparameters of the given model class using the provided training, validation, and test data.
    
    Parameters
    ----------
    model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams

    Returns
    -------
    best_model, best_hyperparams, best_metrics
    """
    # Instantiate the model
    model = model_class()
    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    # Fit the model
    grid_search.fit(X_train, y_train)
    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on training set
    train_rmse, train_r2 = compute_metrics(best_model, X_train, y_train)
    # Predict on the validation set
    val_rmse, val_r2 = compute_metrics(best_model, X_val, y_val)
    # Evaluate the best model on the test set
    test_rmse, test_r2 = compute_metrics(best_model, X_test, y_test)

    # Save best hyperparameters and metrics
    best_params = grid_search.best_params_
    best_metrics = {
        'train_RMSE': train_rmse, 
        'train_R2': train_r2,
        'validation_RMSE': val_rmse, 
        'validation_R2': val_r2,
        'test_RMSE': test_rmse, 
        'test_R2': test_r2
    }
    return best_model, best_params, best_metrics

def compute_metrics(best_model, X_set, y_set):
    """
    Computes rmse and r2 for the best model on the provided data set.

    Parameters
    ----------
    best_model, X_set, y_set

    Returns
    -------
    rmse, r2
    """
    y_pred = best_model.predict(X_set)
    rmse = root_mean_squared_error(y_set, y_pred)
    r2 = r2_score(y_set, y_pred)
    return rmse, r2

def save_model(model, hyperparams, metrics, folder="models/regression"):
    """
    Saves the model, hyperparameters, and metrics to the specified folder.

    Parameters
    ----------
    model, hyperparams, metrics, folder="models/classification"

    Returns
    -------
    None
    """
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    # Save the model as model.joblib
    model_path = os.path.join(folder, "model.joblib")
    joblib.dump(model, model_path)
    # Save the hyperparameters as hyperparameters.json
    hyperparams_path = os.path.join(folder, "hyperparameters.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f)
    # Save the metrics as metrics.json
    metrics_path = os.path.join(folder, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Model, hyperparameters, and metrics saved to {folder}")

def evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test, task_folder):
    """
    Evaluates multiple regression models, tunes their hyperparameters, and saves the best models.
    
    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test, task_folder

    Returns
    -------
    None
    """
    # Model classes and their respective folder names
    model_classes = [
        (SGDRegressor, 'linear_regression'),
        (DecisionTreeRegressor, 'decision_tree'),
        (RandomForestRegressor, 'random_forest'),
        (GradientBoostingRegressor, 'gradient_boosting')
    ]
    # Define hyperparameters to tune for each model
    hyperparams_dict = {
        SGDRegressor: {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [10000, 20000],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'eta0': [0.01, 0.1, 1, 10]
        },
        DecisionTreeRegressor: {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 2, 4]
        },
        RandomForestRegressor: {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 2, 4]
        },
        GradientBoostingRegressor: {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [1, 3, 5, 7],
            'subsample': [0.6, 0.8, 1.0]
        }
    }

    # Loop through each model class, tune it, evaluate, and save results
    for model_class, folder_name in model_classes:
        print(f"Evaluating model: {model_class.__name__}")
        # Tune the model's hyperparameters
        best_model, best_hyperparams, best_metrics = tune_regression_model_hyperparameters(
            model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams_dict[model_class]
        )
        # Define folder to save model, hyperparameters, and metrics
        model_folder = os.path.join(task_folder, folder_name)
        save_model(best_model, best_hyperparams, best_metrics, folder=model_folder)
        print(f"Model, hyperparameters, and metrics saved for {model_class.__name__}")

def find_best_model(X_train, y_train, X_val, y_val, X_test, y_test, task_folder):
    """
    Finds and loads the best model based on validation RMSE.

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test, task_folder

    Returns
    -------
    best_model, best_hyperparams, best_metrics

    """
    evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test, task_folder=task_folder)
    best_model = None
    best_hyperparams = None
    best_metrics = None
    best_validation_rmse = float('inf')  # Start with a large value for comparison
    # Iterate through each model folder in the task folder
    for model_folder in os.listdir(task_folder):
        model_path = os.path.join(task_folder, model_folder)
        if os.path.isdir(model_path):
            try:
                # Load the metrics
                metrics_file = os.path.join(model_path, 'metrics.json')
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                # Compare based on validation rmse
                validation_rmse = metrics.get('validation_RMSE', 0)
                if validation_rmse < best_validation_rmse:
                    best_validation_rmse = validation_rmse
                    # Load the best model
                    best_model = load(os.path.join(model_path, 'model.joblib'))
                    # Load the hyperparameters
                    hyperparams_file = os.path.join(model_path, 'hyperparameters.json')
                    with open(hyperparams_file, 'r') as f:
                        best_hyperparams = json.load(f)
                    # Store the best metrics
                    best_metrics = metrics
            except Exception as e:
                print(f"Error processing model in {model_folder}: {e}")
    return best_model, best_hyperparams, best_metrics

def normalise(df):
    """
    Normalises the DataFrame using Z-score normalisation.

    Parameters
    ----------
    df

    Returns
    -------
    df_normalised
    """
    # Apply Z-score normalisation
    df_normalised = df.copy()
    df_normalised = (df - df.mean()) / df.std()
    return df_normalised


if __name__ == "__main__":
    # Load the data
    label = 'Price_Night'
    features, labels = load_airbnb(label)
    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Evaluate all models and find the best model
    best_model, best_hyperparameters, best_metrics = find_best_model(X_train, y_train, X_val, y_val, X_test, y_test, task_folder=f"models/regression/{label}")
    print(f"Best model found: {best_model}")
    print(f"Hyperparameters: {best_hyperparameters}")
    print(f"Performance metrics: {best_metrics}")

    # Evaluate with normalised features, especially for SGD
    normalised_features = normalise(features)
    nX_train, nX_temp, ny_train, ny_temp = train_test_split(normalised_features, labels, test_size=0.2, random_state=42)
    nX_val, nX_test, ny_val, ny_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    best_model, best_hyperparameters, best_metrics = find_best_model(nX_train, ny_train, nX_val, ny_val, nX_test, ny_test, task_folder=f"models/regression/normalised/{label}")
    print(f"Best model found: {best_model}")
    print(f"Hyperparameters: {best_hyperparameters}")
    print(f"Performance metrics: {best_metrics}")