import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from tabular_data import load_airbnb
import joblib
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from joblib import load
from sklearn.model_selection import GridSearchCV

def train_logistic_regression():
    """
    Trains a logistic regression model and evaluates performance.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Load the dataset with "Category" as the label
    features, labels = load_airbnb(label='Category')
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    # Train a Logistic Regression model
    logistic_model = LogisticRegression(max_iter=10000)  # Increase max_iter if convergence issues arise
    logistic_model.fit(X_train, y_train)
    # Make predictions on both training and test sets
    y_train_pred = logistic_model.predict(X_train)
    y_test_pred = logistic_model.predict(X_test)
    # Evaluate metrics for the training set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    print(f"Training Set Performance:")
    print(f"Accuracy: {train_accuracy:.2f}")
    print(f"Precision: {train_precision:.2f}")
    print(f"Recall: {train_recall:.2f}")
    print(f"F1 Score: {train_f1:.2f}")
    print("\n")
    # Evaluate metrics for the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"Test Set Performance:")
    print(f"Accuracy: {test_accuracy:.2f}")
    print(f"Precision: {test_precision:.2f}")
    print(f"Recall: {test_recall:.2f}")
    print(f"F1 Score: {test_f1:.2f}")
    print("\n")
    # Classification report
    print("Training Set Classification Report:")
    print(classification_report(y_train, y_train_pred))
    print("Test Set Classification Report:")
    print(classification_report(y_test, y_test_pred))

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, X_test, y_test, param_grid):
    """
    Tunes the hyperparameters of the given classification model class using the provided training, validation, and test data.
    
    Parameters
    ----------
    model_class, X_train, y_train, X_val, y_val, X_test, y_test, param_grid

    Returns
    -------
    best_model, best_hyperparams, best_metrics
    """
    # Instantiate the model
    model = model_class()
    # Perform grid search with validation accuracy as the scoring metric
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # Fit the model
    grid_search.fit(X_train, y_train)
    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on training set
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_precision = precision_score(y_train, y_train_pred, average='weighted')

    # Predict on the validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    val_recall = recall_score(y_val, y_val_pred, average='weighted')
    val_precision = precision_score(y_val, y_val_pred, average='weighted')

    # Evaluate the best model on the test set
    y_test_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_precision = precision_score(y_test, y_test_pred, average='weighted')

    # Save best hyperparameters and metrics
    best_params = grid_search.best_params_
    best_metrics = {
        'train_accuracy': train_accuracy, 
        'train_f1': train_f1,
        'train_recall': train_recall,
        'train_precision': train_precision,
        'validation_accuracy': val_accuracy, 
        'validation_f1': val_f1,
        'validation_recall': val_recall,
        'validation_precision': val_precision,
        'test_accuracy': test_accuracy, 
        'test_f1': test_f1,
        'test_recall': test_recall,
        'test_precision': test_precision
    }

    return best_model, best_params, best_metrics

def save_model(model, hyperparams, metrics, folder="models/classification"):
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
    Evaluates multiple models, tunes their hyperparameters, and saves them.

    Parameters
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test, task_folder

    Returns
    -------
    None
    """
    # Ensure the task folder exists
    os.makedirs(task_folder, exist_ok=True)
    # Model classes and their respective folder names
    model_classes = [
        (LogisticRegression, 'logistic_regression'),
        (DecisionTreeClassifier, 'decision_tree'),
        (RandomForestClassifier, 'random_forest'),
        (GradientBoostingClassifier, 'gradient_boosting')
    ]
    # Define hyperparameters to tune for each model
    hyperparams_dict = {
        LogisticRegression: {
            'C': [0.1, 1.0, 10],
            'max_iter': [10000, 20000, 30000],
            'solver': ['sag', 'saga', 'liblinear']
        },
        DecisionTreeClassifier: {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20, 40],
            'min_samples_leaf': [1, 2, 4]
        },
        RandomForestClassifier: {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20, 40, 60, 80, 100, 120],
            'min_samples_leaf': [1, 2, 4]
        },
        GradientBoostingClassifier: {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [1, 3, 5],
            'subsample': [0.6, 0.8]
        }
    }

    # Loop through each model class, tune it, evaluate, and save results
    for model_class, folder_name in model_classes:
        print(f"Evaluating model: {model_class.__name__}")
        # Tune the model's hyperparameters
        best_model, best_hyperparams, best_metrics = tune_classification_model_hyperparameters(
            model_class, X_train, y_train, X_val, y_val, X_test, y_test, hyperparams_dict[model_class]
        )
        # Define folder to save model, hyperparameters, and metrics
        model_folder = os.path.join(task_folder, folder_name)
        save_model(best_model, best_hyperparams, best_metrics, folder=model_folder)
        print(f"Model, hyperparameters, and metrics saved for {model_class.__name__}")

def find_best_model(X_train, y_train, X_val, y_val, X_test, y_test, task_folder):
    """
    Finds and loads the best model based on validation accuracy.

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
    best_validation_accuracy = float('-inf')  # Initialise with a very low value
    # Iterate through each model folder in the task folder
    for model_folder in os.listdir(task_folder):
        model_path = os.path.join(task_folder, model_folder)
        if os.path.isdir(model_path):
            try:
                # Load the metrics
                metrics_file = os.path.join(model_path, 'metrics.json')
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                # Compare based on validation accuracy
                validation_accuracy = metrics.get('validation_accuracy', 0)
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
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


if __name__ == "__main__":
    # Load the dataset with "Category" as the label
    label = 'Category'
    features, labels = load_airbnb(label)
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Evaluate all models and find the best model
    best_model, best_hyperparams, best_metrics = find_best_model(X_train, y_train, X_val, y_val, X_test, y_test, task_folder=f"models/classification/{label}")
    print(f"Best Model: {best_model}")
    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best Metrics: {best_metrics}")