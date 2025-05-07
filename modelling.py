from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
import regression
import classification
import regression_nn
import classification_nn

if __name__ == "__main__":
    # Set the type of problem, model and the label
    problem_type = 'regression' 
    label = 'bedrooms'

    # Load the data
    features, labels = load_airbnb(label)
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Regression
    if problem_type == 'regression':
        # Evaluate all models and find the best model
        best_model, best_hyperparameters, best_metrics = regression.find_best_model(X_train, y_train, X_val, y_val, X_test, y_test, task_folder=f"models/regression/{label}")
        print(f"Best model found: {best_model}")
        print(f"Hyperparameters: {best_hyperparameters}")
        print(f"Performance metrics: {best_metrics}")
        # Evaluate with normalised features, especially for SGD
        normalised_features = regression.normalise(features)
        nX_train, nX_temp, ny_train, ny_temp = train_test_split(normalised_features, labels, test_size=0.2, random_state=42)
        nX_val, nX_test, ny_val, ny_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        best_model, best_hyperparameters, best_metrics = regression.find_best_model(nX_train, ny_train, nX_val, ny_val, nX_test, ny_test, task_folder=f"models/regression/normalised/{label}")
        print(f"Best model found: {best_model}")
        print(f"Hyperparameters: {best_hyperparameters}")
        print(f"Performance metrics: {best_metrics}")

    # Classification
    if problem_type == 'classification':
        # Evaluate all models and find the best model
        best_model, best_hyperparams, best_metrics = classification.find_best_model(X_train, y_train, X_val, y_val, X_test, y_test, task_folder=f"models/classification/{label}")
        print(f"Best Model: {best_model}")
        print(f"Best Hyperparameters: {best_hyperparams}")
        print(f"Best Metrics: {best_metrics}")

    # nn Regression
    if problem_type == 'regression':
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = regression_nn.prepare_datasets(label)
        # Create DataLoaders
        train_loader, val_loader, test_loader = regression_nn.create_dataloaders(train_dataset, val_dataset, test_dataset)
        # Find the best neural network model
        best_model, best_metrics, best_hyperparams = regression_nn.find_best_nn(train_loader, val_loader, test_loader, folder=f"models/neural_networks/regression/{label}", num_epochs=100)
        print(f"Best Model Hyperparameters: {best_hyperparams}")
        print(f"Best Model Metrics: {best_metrics}")

    # nn Classification
    if problem_type == 'classification':
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = classification_nn.prepare_datasets(label)
        # Create DataLoaders
        train_loader, val_loader, test_loader = classification_nn.create_dataloaders(train_dataset, val_dataset, test_dataset)
        # Find the best neural network model
        best_model, best_metrics, best_hyperparams = classification_nn.find_best_nn(train_loader, val_loader, test_loader, folder=f"models/neural_networks/classification/{label}", num_epochs=100)
        print(f"Best Model Hyperparameters: {best_hyperparams}")
        print(f"Best Model Metrics: {best_metrics}")