import logging
from typing import Dict, List, Tuple
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPIFTConfig:
    """
    Configuration class for CPI-FT framework.
    
    Attributes:
    - num_tasks (int): Number of tasks.
    - num_parameters (int): Number of parameters.
    - learning_rate (float): Learning rate for fine-tuning.
    - batch_size (int): Batch size for training.
    - num_epochs (int): Number of epochs for training.
    - velocity_threshold (float): Velocity threshold for parameter updates.
    """
    def __init__(self, num_tasks: int, num_parameters: int, learning_rate: float, batch_size: int, num_epochs: int, velocity_threshold: float):
        self.num_tasks = num_tasks
        self.num_parameters = num_parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.velocity_threshold = velocity_threshold

class CPIFTModel:
    """
    CPI-FT model class.
    
    Attributes:
    - config (CPIFTConfig): Configuration for CPI-FT framework.
    - model (torch.nn.Module): PyTorch model for fine-tuning.
    """
    def __init__(self, config: CPIFTConfig, model: torch.nn.Module):
        self.config = config
        self.model = model

    def train(self, train_data: List[Tuple[np.ndarray, np.ndarray]], validation_data: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Train the CPI-FT model.
        
        Args:
        - train_data (List[Tuple[np.ndarray, np.ndarray]]): Training data.
        - validation_data (List[Tuple[np.ndarray, np.ndarray]]): Validation data.
        """
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_data, batch_size=self.config.batch_size, shuffle=False)

        # Initialize optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train the model
        for epoch in range(self.config.num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

            # Evaluate the model on validation data
            self.model.eval()
            total_correct = 0
            with torch.no_grad():
                for batch in validation_loader:
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
            accuracy = total_correct / len(validation_data)
            logger.info(f'Validation Accuracy: {accuracy:.4f}')

    def evaluate(self, test_data: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Evaluate the CPI-FT model on test data.
        
        Args:
        - test_data (List[Tuple[np.ndarray, np.ndarray]]): Test data.
        
        Returns:
        - accuracy (float): Accuracy on test data.
        - precision (float): Precision on test data.
        - recall (float): Recall on test data.
        - f1_score (float): F1 score on test data.
        """
        # Create data loader
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False)

        # Evaluate the model on test data
        self.model.eval()
        total_correct = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_precision += precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                total_recall += recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
                total_f1 += f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        accuracy = total_correct / len(test_data)
        precision = total_precision / len(test_loader)
        recall = total_recall / len(test_loader)
        f1 = total_f1 / len(test_loader)
        return accuracy, precision, recall, f1

class CPIFTDataset(Dataset):
    """
    CPI-FT dataset class.
    
    Attributes:
    - data (List[Tuple[np.ndarray, np.ndarray]]): Dataset.
    """
    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

def create_dataset(data: List[Tuple[np.ndarray, np.ndarray]]) -> CPIFTDataset:
    """
    Create a CPI-FT dataset.
    
    Args:
    - data (List[Tuple[np.ndarray, np.ndarray]]): Dataset.
    
    Returns:
    - dataset (CPIFTDataset): CPI-FT dataset.
    """
    return CPIFTDataset(data)

def train_test_split_data(data: List[Tuple[np.ndarray, np.ndarray]], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Split data into training and test sets.
    
    Args:
    - data (List[Tuple[np.ndarray, np.ndarray]]): Dataset.
    - test_size (float, optional): Proportion of data for test set. Defaults to 0.2.
    - random_state (int, optional): Random seed. Defaults to 42.
    
    Returns:
    - train_data (List[Tuple[np.ndarray, np.ndarray]]): Training data.
    - test_data (List[Tuple[np.ndarray, np.ndarray]]): Test data.
    """
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

def main():
    # Set up configuration
    config = CPIFTConfig(num_tasks=10, num_parameters=100, learning_rate=0.001, batch_size=32, num_epochs=10, velocity_threshold=0.5)

    # Create dataset
    data = [(np.random.rand(10), np.random.randint(0, 2, 10)) for _ in range(1000)]
    train_data, test_data = train_test_split_data(data)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create CPI-FT model
    cpift_model = CPIFTModel(config, model)

    # Train the model
    cpift_model.train(train_data, test_data)

    # Evaluate the model
    accuracy, precision, recall, f1 = cpift_model.evaluate(test_data)
    logger.info(f'Test Accuracy: {accuracy:.4f}')
    logger.info(f'Test Precision: {precision:.4f}')
    logger.info(f'Test Recall: {recall:.4f}')
    logger.info(f'Test F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()