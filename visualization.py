import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualizationConfig:
    """Configuration for visualization."""
    def __init__(self, 
                 data_path: str, 
                 model_path: str, 
                 num_classes: int, 
                 num_features: int, 
                 batch_size: int, 
                 num_epochs: int, 
                 learning_rate: float, 
                 threshold: float):
        """
        Args:
        - data_path (str): Path to the dataset.
        - model_path (str): Path to the model.
        - num_classes (int): Number of classes.
        - num_features (int): Number of features.
        - batch_size (int): Batch size.
        - num_epochs (int): Number of epochs.
        - learning_rate (float): Learning rate.
        - threshold (float): Threshold for velocity-threshold algorithm.
        """
        self.data_path = data_path
        self.model_path = model_path
        self.num_classes = num_classes
        self.num_features = num_features
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.threshold = threshold

class VisualizationDataset(Dataset):
    """Dataset for visualization."""
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
        - data (np.ndarray): Data.
        - labels (np.ndarray): Labels.
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.data[index], self.labels[index]

class Visualization:
    """Results visualization."""
    def __init__(self, config: VisualizationConfig):
        """
        Args:
        - config (VisualizationConfig): Configuration.
        """
        self.config = config
        self.data = None
        self.labels = None
        self.model = None

    def load_data(self) -> None:
        """Load data."""
        try:
            self.data = np.load(self.config.data_path)
            self.labels = np.load(self.config.data_path.replace('data', 'labels'))
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    def load_model(self) -> None:
        """Load model."""
        try:
            self.model = torch.load(self.config.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def preprocess_data(self) -> None:
        """Preprocess data."""
        try:
            scaler = StandardScaler()
            self.data = scaler.fit_transform(self.data)
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")

    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to split data: {e}")

    def reduce_dimensionality(self, data: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        try:
            pca = PCA(n_components=2)
            return pca.fit_transform(data)
        except Exception as e:
            logger.error(f"Failed to reduce dimensionality: {e}")

    def visualize_data(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Visualize data."""
        try:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis')
            plt.title('Data Visualization')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
        except Exception as e:
            logger.error(f"Failed to visualize data: {e}")

    def visualize_model(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Visualize model."""
        try:
            plt.figure(figsize=(10, 8))
            predictions = self.model.predict(data)
            sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predictions, palette='viridis')
            plt.title('Model Visualization')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.show()
        except Exception as e:
            logger.error(f"Failed to visualize model: {e}")

    def evaluate_model(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Evaluate model."""
        try:
            predictions = self.model.predict(data)
            accuracy = accuracy_score(labels, predictions)
            report = classification_report(labels, predictions)
            matrix = confusion_matrix(labels, predictions)
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Classification Report:\n{report}")
            logger.info(f"Confusion Matrix:\n{matrix}")
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")

    def run(self) -> None:
        """Run visualization."""
        try:
            self.load_data()
            self.load_model()
            self.preprocess_data()
            X_train, X_test, y_train, y_test = self.split_data()
            X_train_reduced = self.reduce_dimensionality(X_train)
            X_test_reduced = self.reduce_dimensionality(X_test)
            self.visualize_data(X_train_reduced, y_train)
            self.visualize_model(X_train_reduced, y_train)
            self.evaluate_model(X_test, y_test)
        except Exception as e:
            logger.error(f"Failed to run visualization: {e}")

if __name__ == "__main__":
    config = VisualizationConfig(
        data_path='data.npy',
        model_path='model.pth',
        num_classes=10,
        num_features=100,
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        threshold=0.5
    )
    visualization = Visualization(config)
    visualization.run()