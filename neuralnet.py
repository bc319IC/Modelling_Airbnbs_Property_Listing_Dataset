import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import json
import joblib
from datetime import datetime
import os
import time
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import itertools

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialise the dataset with features and labels.

        Parameters
        ----------
        features, lables

        Returns
        -------
        none
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        """
        Returns the size of the dataset.

        Parameters
        ----------
        none

        Returns
        -------
        len(self.features)
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get a single data point (features, label).

        Parameters
        ----------
        idx

        Returns
        -------
        features, label
        """
        features = self.features[idx]
        label = self.labels[idx]
        return features, label
    
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        Initialise the neural network with the specified hyperparamters.

        Parameters
        ----------
        input_size, output_size
        config - YAML file containing the hyperparamters of the neural network

        Returns
        -------
        none
        """
        super().__init__()
        self.layers = nn.ModuleList()
        hidden_layer_width = config['hidden_layer_width']
        depth = config['depth']
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_layer_width))
        # Hidden layers
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
        # Output layer
        self.layers.append(nn.Linear(hidden_layer_width, output_size))
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Define the forward pass of the network.

        Parameters
        ----------
        x

        Returns
        -------
        x
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
    

def get_nn_config(yaml_file="nn_config.yaml"):
    """
    Reads the YAML file and returns the configuration as a dictionary.

    Parameters
    ----------
    yaml_file - default is nn_config.yaml

    Returns
    -------
    config
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prepare_datasets():
    """
    Splits the data into training, validation, and test sets.

    Parameters
    ----------
    none

    Returns
    -------
    train_dataset, val_dataset, test_dataset
    """
    features, labels = load_airbnb(label="Price_Night")
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # Create datasets
    train_dataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
    val_dataset = AirbnbNightlyPriceRegressionDataset(X_val, y_val)
    test_dataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Create DataLoaders for training, validation, and test sets.

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset, batch_size=32

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, test_loader, num_epochs=10, config=None, log_dir="runs/nn_training"):
    """
    Trains, validates, and tests the model and saves the model along with its metrics.

    Parameters
    ----------
    model - instance of the neural network
    train_loader, val_loader, test_loader, num_epochs=10, config=None
    log_dir="runs/nn_training" - default directory for the SummaryWriter

    Returns
    -------
    metrics
    """
    # Initialise TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)
    # Start timing the entire training process
    training_start_time = time.time()
    # Get the optimiser from the config
    learning_rate = config["learning_rate"]
    optimiser_name = config["optimiser"]
    if optimiser_name == "adam":
        optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimiser_name == "sgd":
        optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimiser: {optimiser_name}")
    # Loss function for regression MSE
    loss_function = nn.MSELoss()

    # Loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        # Training phase
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimiser.zero_grad()
            # Forward pass: compute predicted output by passing features through the model
            outputs = model(features)
            loss = loss_function(outputs, labels.unsqueeze(1))
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Perform optimisation: update model parameters
            optimiser.step()
            # Accumulate loss over batches
            running_loss += loss.item()
        # Calculate average loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        # Compute training rmse and r2  
        train_rmse, train_r2 = compute_metrics(model, train_loader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0 # To compute loss
        with torch.no_grad():  # No need to track gradients for validation
            for features, labels in val_loader:
                outputs = model(features)
                # Validation loss
                loss = loss_function(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        # Compute validation rmse and r2  
        val_rmse, val_r2 = compute_metrics(model, val_loader)
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        # Print losses at the end of each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    # Calculate the total training time
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    # Compute inference latency
    inference_start_time = time.time()
    with torch.no_grad():
        for features, _ in val_loader:
            model(features)  # Forward pass for inference
    inference_end_time = time.time()
    inference_latency = (inference_end_time - inference_start_time) / len(val_loader.dataset)
    
    # Loop over epochs
    for epoch in range(num_epochs):
        # Test phase
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0 # To compute loss
        with torch.no_grad():  # No need to track gradients for test
            for features, labels in test_loader:
                outputs = model(features)
                # Test loss
                loss = loss_function(outputs, labels.unsqueeze(1))
                test_loss += loss.item()
        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        # Compute test rmse and r2  
        test_rmse, test_r2 = compute_metrics(model, test_loader)
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Test', avg_test_loss, epoch)
        # Print losses at the end of each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")
    
    # Log metrics to a dictionary
    metrics = {
        "training_duration": training_duration,
        "inference_latency": inference_latency,
        "training_loss": avg_train_loss,
        "training_rmse": train_rmse,
        "training_r2": train_r2,
        "validation_loss": avg_val_loss,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2,
        "test_loss": avg_test_loss,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }
    # Save model, hyperparameters, and metrics       
    save_model(model, config, metrics)
    # Close TensorBoard writer
    writer.close()
    return metrics

def compute_metrics(model, data_loader):
    """
    Computes RMSE and R2 score for the given model on the provided data_loader.

    Parameters
    ----------
    model, data_loader

    Returns
    -------
    rsme, r2

    """
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            # Forward pass
            predictions = model(features)
            # Store predictions and true labels for metric computation
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
    # Calculate RMSE and R^2 score
    rmse = root_mean_squared_error(all_labels, all_predictions)
    r2 = r2_score(all_labels, all_predictions)
    return rmse, r2

def save_model(model, hyperparameters, metrics, folder="models/neural_networks/regression"):
    """
    Saves the model with the timestamp, metrics, and hyperparameters.

    Parameters
    ----------
    model, hyperparameters, metrics
    folder="models/neural_networks/regression" - default directory for saving models

    Returns
    -------
    none

    """
    # Create timestamp folder name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(folder, timestamp)
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    # Save model
    if isinstance(model, torch.nn.Module):
        # PyTorch model
        model_path = os.path.join(save_path, "model.pt")
        torch.save(model.state_dict(), model_path)
    else:
        # scikit-learn model (or other compatible)
        model_path = os.path.join(save_path, "model.joblib")
        joblib.dump(model, model_path)
    # Save hyperparameters
    hyperparameters_path = os.path.join(save_path, "hyperparameters.json")
    with open(hyperparameters_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    # Save performance metrics
    metrics_path = os.path.join(save_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(convert_to_serialisable(metrics), f, indent=4)
    print(f"Model, hyperparameters, and metrics saved to {save_path}")

def convert_to_serialisable(obj):
    """
    Recursively converts non-serialisable data types 
    to Python native types for JSON serialisation.

    Parameters
    ----------
    obj

    Returns
    -------
    obj
    """
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serialisable(item) for item in obj]
    return obj

def generate_nn_configs():
    """
    Generates different combination of hyperparameters.

    Parameters
    ----------
    none

    Returns
    -------
    configs
    """
    # Define hyperparameter options
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layer_widths = [64, 128, 256]
    depths = [2, 3, 4]  # Number of hidden layers
    optimisers = ['adam']  # Different optimisers
    # Generate all possible combinations 
    combinations = list(itertools.product(learning_rates, hidden_layer_widths, depths, optimisers))
    configs = []
    for (lr, width, depth, optimiser) in combinations:
        config = {
            'learning_rate': lr,
            'hidden_layer_width': width,
            'depth': depth,
            'optimiser': optimiser
        }
        configs.append(config)
    return configs

def find_best_nn(train_loader, val_loader, test_loader, num_epochs=10, folder="models/neural_networks/regression"):
    """
    Finds the best neural network model using the different hyperparameter combinations.

    Parameters
    ----------
    train_loader, val_loader, test_loader, num_epochs=10, folder="models/neural_networks/regression"

    Returns
    -------
    best_model, best_metrics, best_hyperparameters
    """
    configs = generate_nn_configs()
    best_model = None
    best_metrics = None
    best_hyperparameters = None
    best_validation_rmse = float('inf')  # Initialise with a very high value for RMSE
    # Loop through all configurations
    for i, config in enumerate(configs):
        print(f"Training model {i + 1}/{len(configs)} with config: {config}")
        # Initialise the model with the current config
        input_size = train_dataset[0][0].shape[0]  # Number of input features
        output_size = 1  # Predicting a single value (nightly price)
        model = FullyConnectedNN(input_size, output_size, config=config)
        # Train and save the models and test
        metrics = train(model, train_loader, val_loader, test_loader, num_epochs=num_epochs, config=config)
        # Check if this model is the best one based on validation RMSE
        if metrics['validation_rmse'] < best_validation_rmse:
            best_validation_rmse = metrics['validation_rmse']
            best_model = model
            best_metrics = metrics
            best_hyperparameters = config
    # Return the best model, its metrics, and hyperparameters
    return best_model, best_metrics, best_hyperparameters


if __name__ == "__main__":
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    # Find the best neural network model
    best_model, best_metrics, best_hyperparameters = find_best_nn(train_loader, val_loader, test_loader, num_epochs=100)
    print(f"Best Model Hyperparameters: {best_hyperparameters}")
    print(f"Best Model Metrics: {best_metrics}")

    # Run a particular model from the YAML config file
    '''
    # Load the config from the YAML file
    config = get_nn_config("nn_config.yaml")
    # Initialise the model with the config
    input_size = train_dataset[0][0].shape[0]  # Number of input features
    output_size = 1  # Predicting a single value (nightly price)
    model = FullyConnectedNN(input_size, output_size, config)
    # Train the model
    train(model, train_loader, val_loader, test_loader, num_epochs=10, config=config)
    '''