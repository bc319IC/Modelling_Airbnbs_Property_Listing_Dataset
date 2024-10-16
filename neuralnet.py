import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize the dataset with features and labels.
        :param features: A numpy array or pandas DataFrame of the tabular numerical features.
        :param labels: A numpy array or pandas Series of the nightly prices (labels).
        """
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Get a single data point (features, label).
        :param idx: Index of the data point to retrieve.
        :return: Tuple of (features, label).
        """
        features = self.features[idx]
        label = self.labels[idx]
        return features, label
    
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network.
        :param input_size: Number of input features.
        :param hidden_size: Number of units in the hidden layer.
        :param output_size: Number of output units (for regression, it's 1).
        """
        super(FullyConnectedNN, self).__init__()
        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Define the forward pass.
        :param x: Input features tensor.
        :return: Output tensor.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def prepare_datasets():
    features, labels = load_airbnb(label="Price_Night")
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # Create datasets
    train_dataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
    val_dataset = AirbnbNightlyPriceRegressionDataset(X_val, y_val)
    test_dataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test)
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Create DataLoaders for train, validation, and test sets.
    :param train_dataset: The training dataset
    :param val_dataset: The validation dataset
    :param test_dataset: The test dataset
    :param batch_size: Number of samples per batch.
    :return: Dataloaders for train, validation, and test sets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, log_dir="runs/nightly_price_regression"):
    """
    Complete training loop to optimize the model's parameters.
    :param model: The PyTorch model.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param num_epochs: Number of epochs to train for.
    :param learning_rate: Learning rate for the optimizer.
    """
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)
    # Optimizer: Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Loss function for regression MSE
    loss_function = nn.MSELoss()
    # Loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        # Training phase
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass: compute predicted output by passing features through the model
            outputs = model(features)
            loss = loss_function(outputs, labels.unsqueeze(1))  # MSELoss expects (N, 1) shape for outputs and labels
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Perform optimization: update model parameters
            optimizer.step()
            # Accumulate loss over batches
            running_loss += loss.item()
            # Calculate training accuracy (for regression, we'll count prediction within a margin as correct)
            predicted = outputs.round()  # Rounding as a proxy for classification (modify as needed)
            correct_train += (predicted == labels.unsqueeze(1)).sum().item()
            total_train += labels.size(0)
        # Calculate average loss and accuracy for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # No need to track gradients for validation
            for features, labels in val_loader:
                outputs = model(features)
                loss = loss_function(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
                # Validation accuracy
                predicted = outputs.round()
                correct_val += (predicted == labels.unsqueeze(1)).sum().item()
                total_val += labels.size(0)
        # Calculate average validation loss and accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        # Print losses and accuracy at the end of each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets()
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset)
    input_size = train_dataset[0][0].shape[0]  # Number of input features
    hidden_size = 128  # Arbitrary choice for now, can be tuned
    output_size = 1  # Since we're predicting the nightly price, which is a single value
    model = FullyConnectedNN(input_size, hidden_size, output_size)
    # Call the train function (but just perform a forward pass for now)
    train(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001)