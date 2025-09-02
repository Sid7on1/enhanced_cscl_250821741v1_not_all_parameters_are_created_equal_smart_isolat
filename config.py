import logging
import os
import yaml
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

# Constants
CONFIG_FILE = "optimization_config.yaml"

# Default configuration values
_C = {
    "optimizer": {
        "type": "SGD",
        "lr": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0001,
    },
    "loss": {
        "function": "crossentropy",
        "label_smoothing": 0.1,
        "reduction": "mean",
    },
    "metrics": ["accuracy", "f1_score"],
    "dataset": {
        "type": "CIFAR10",
        "root": "datasets/cifar10",
        "train_batch_size": 64,
        "test_batch_size": 128,
        "num_workers": 4,
    },
    "model": {
        "type": "ResNet50",
        "pretrained": True,
        "num_classes": 10,
        "feature_extraction": False,
    },
    "training": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 100,
        "checkpoint_dir": "checkpoints",
        "log_interval": 10,
        "validation_freq": 1,
    },
}


def load_config(config_file: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load configuration file and update default values.

    Parameters:
    - config_file (str): Path to the configuration file.

    Returns:
    - Dict[str, Any]: Loaded configuration dictionary.
    """
    if not os.path.isfile(config_file):
        logger.warning(f"Config file '{config_file}' not found. Using default configuration.")
        return _C

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Update default config with loaded values
    _C.update(config)
    logger.info(f"Configuration loaded from '{config_file}'.")

    return _C


def get_config(section: str, option: str, default: Any = None) -> Any:
    """
    Retrieve a specific option from the configuration.

    Parameters:
    - section (str): Section of the configuration.
    - option (str): Option to retrieve.
    - default (Any, optional): Default value if option is not found. Defaults to None.

    Returns:
    - Any: Value of the specified option.
    """
    if section not in _C:
        raise KeyError(f"Section '{section}' not found in the configuration.")

    if option not in _C[section]:
        if default is None:
            raise KeyError(f"Option '{option}' not found in section '{section}'.")
        else:
            logger.warning(f"Option '{option}' not found in section '{section}'. Using default value.")
            return default

    return _C[section][option]


def set_config(section: str, option: str, value: Any) -> None:
    """
    Set a specific option in the configuration.

    Parameters:
    - section (str): Section of the configuration.
    - option (str): Option to set.
    - value (Any): New value for the option.

    Raises:
    - KeyError: If the specified section does not exist.
    """
    if section not in _C:
        raise KeyError(f"Section '{section}' not found in the configuration.")

    _C[section][option] = value
    logger.info(f"Updated configuration: Section '{section}', Option '{option}' set to '{value}'.")


def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Create an optimizer based on the configuration.

    Parameters:
    - model (nn.Module): Model to optimize.

    Returns:
    - torch.optim.Optimizer: Optimizer instance.
    """
    optimizer_config = get_config("optimizer", None)
    optimizer_type = optimizer_config["type"]
    optimizer_params = {
        k: get_config("optimizer", k) for k in optimizer_config if k != "type"
    }

    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def get_loss_function() -> nn.Module:
    """
    Create a loss function based on the configuration.

    Returns:
    - nn.Module: Loss function.
    """
    loss_config = get_config("loss", None)
    loss_function_name = loss_config["function"]

    if loss_function_name == "crossentropy":
        loss_function = nn.CrossEntropyLoss(**{k: v for k, v in loss_config.items() if k != "function"})
    elif loss_function_name == "mse":
        loss_function = nn.MSELoss(**{k: v for k, v in loss_config.items() if k != "function"})
    else:
        raise ValueError(f"Unsupported loss function: {loss_function_name}")

    return loss_function


def get_metrics() -> List[str]:
    """
    Retrieve the list of metrics to evaluate during training.

    Returns:
    - List[str]: List of metric names.
    """
    return get_config("metrics", None)


def get_dataset(dataset_type: Optional[str] = None) -> Any:
    """
    Load and return a dataset based on the configuration.

    Parameters:
    - dataset_type (str, optional): Type of the dataset. If not provided, uses config value.

    Returns:
    - Any: Dataset instance.
    """
    dataset_config = get_config("dataset", None)
    if dataset_type is not None:
        dataset_config["type"] = dataset_type

    dataset_type = dataset_config["type"]

    if dataset_type == "CIFAR10":
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import ToTensor

        transform = ToTensor()
        root = dataset_config["root"]
        train_batch_size = dataset_config["train_batch_size"]
        test_batch_size = dataset_config["test_batch_size"]
        num_workers = dataset_config["num_workers"]

        train_dataset = CIFAR10(root, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root, train=False, download=True, transform=transform)

        dataset = {
            "train": train_dataset,
            "test": test_dataset,
            "train_loader": torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers
            ),
            "test_loader": torch.utils.data.DataLoader(
                test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
            ),
        }
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return dataset


def get_model(model_type: Optional[str] = None, num_classes: Optional[int] = None) -> nn.Module:
    """
    Load and return a model based on the configuration.

    Parameters:
    - model_type (str, optional): Type of the model. If not provided, uses config value.
    - num_classes (int, optional): Number of output classes. If not provided, uses config value.

    Returns:
    - nn.Module: Model instance.
    """
    model_config = get_config("model", None)
    if model_type is not None:
        model_config["type"] = model_type
    if num_classes is not None:
        model_config["num_classes"] = num_classes

    model_type = model_config["type"]
    num_classes = model_config["num_classes"]
    pretrained = model_config["pretrained"]
    feature_extraction = model_config["feature_extraction"]

    if model_type == "ResNet50":
        from torchvision.models import resnet50

        model = resnet50(pretrained=pretrained)
        if not feature_extraction:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "VGG16":
        from torchvision.models import vgg16

        model = vgg16(pretrained=pretrained)
        if not feature_extraction:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def get_device() -> torch.device:
    """
    Get the device to use for training based on the configuration.

    Returns:
    - torch.device: Device to use for training.
    """
    return torch.device(get_config("training", "device"))


def get_checkpoint_dir() -> str:
    """
    Get the directory for saving model checkpoints.

    Returns:
    - str: Path to the checkpoint directory.
    """
    checkpoint_dir = get_config("training", "checkpoint_dir")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_log_interval() -> int:
    """
    Get the interval for logging training progress.

    Returns:
    - int: Log interval (number of batches).
    """
    return get_config("training", "log_interval")


def get_validation_freq() -> int:
    """
    Get the frequency of validation during training.

    Returns:
    - int: Number of epochs between each validation.
    """
    return get_config("training", "validation_freq")


def get_num_epochs() -> int:
    """
    Get the total number of training epochs.

    Returns:
    - int: Number of training epochs.
    """
    return get_config("training", "epochs")


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy for a batch of predictions.

    Parameters:
    - output (torch.Tensor): Model output tensor.
    - target (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: Accuracy value.
    """
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return correct.view(-1).float().sum(0) / batch_size


def f1_score(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the F1 score for a batch of predictions.

    Parameters:
    - output (torch.Tensor): Model output tensor.
    - target (torch.Tensor): Ground truth labels.

    Returns:
    - torch.Tensor: F1 score.
    """
    smooth = 1e-6

    true_positive = (output * target).sum(dim=1)
    true_negative = ((1 - output) * (1 - target)).sum(dim=1)
    false_positive = ((1 - output) * target).sum(dim=1)
    false_negative = (output * (1 - target)).sum(dim=1)

    precision = torch.div(true_positive, true_positive + false_positive + smooth)
    recall = torch.div(true_positive, true_positive + false_negative + smooth)

    f1 = 2 * (precision * recall) / (precision + recall + smooth)

    return f1.mean()


def validate(model: nn.Module, loss_function: nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """
    Validate the model on the given dataset.

    Parameters:
    - model (nn.Module): Model to validate.
    - loss_function (nn.Module): Loss function.
    - dataloader (torch.utils.data.DataLoader): Data loader for the validation set.

    Returns:
    - Dict[str, float]: Validation metrics.
    """
    model.eval()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    f1_meter = AverageMeter()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(get_device()), targets.to(get_device())

            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            loss_meter.update(loss.item(), inputs.size(0))
            accuracy_meter.update(accuracy(outputs, targets), inputs.size(0))
            f1_meter.update(f1_score(outputs, targets), inputs.size(0))

    return {"loss": loss_meter.avg, "accuracy": accuracy_meter.avg, "f1": f1_meter.avg}


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_best: bool = False
) -> None:
    """
    Save a model checkpoint.

    Parameters:
    - model (nn.Module): Model to save.
    - optimizer (torch.optim.Optimizer): Optimizer to save.
    - epoch (int): Current epoch number.
    - save_best (bool): Whether to save only if it's the best model so far.
    """
    checkpoint_dir = get_checkpoint_dir()
    model_state = {"epoch": epoch, "state_dict": model.state_dict()}
    optimizer_state = optimizer.state_dict()

    model_filename = os.path.join(checkpoint_dir, "model_checkpoint.pth")
    optimizer_filename = os.path.join(checkpoint_dir, "optimizer_checkpoint.pth")

    torch.save(model_state, model_filename)
    torch.save(optimizer_state, optimizer_filename)

    if save_best:
        best_filename = os.path.join(checkpoint_dir, "model_best.pth")
        torch.save(model_state, best_filename)
        logger.info("Saved best model checkpoint.")


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    log_interval: int,
    validation_freq: int,
    save_best: bool = False,
) -> None:
    """
    Train the model for a specified number of epochs.

    Parameters:
    - model (nn.Module): Model to train.
    - optimizer (torch.optim.Optimizer): Optimizer to use.
    - loss_function (nn.Module): Loss function.
    - train_loader (torch.utils.data.DataLoader): Data loader for the training set.
    - epochs (int): Number of training epochs.
    - log_interval (int): Number of batches between log updates.
    - validation_freq (int): Number of epochs between each validation.
    - save_best (bool): Whether to save only if it's the best model so far.
    """
    model = model.to(get_device())
    loss_function = loss_function.to(get_device())

    for epoch in range(epochs):
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(get_device()), targets.to(get_device())

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )

        if (epoch + 1) % validation_freq == 0:
            logger.info(f"Validation at epoch {epoch+1}")
            val_metrics = validate(model, loss_function, train_loader)
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - Loss: {val_metrics['loss']:.4f} - "
                f"Accuracy: {val_metrics['accuracy']:.4f} - F1 Score: {val_metrics['f1']:.4f}"
            )

            if save_best and val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, epoch + 1, save_best=True)

        save_checkpoint(model, optimizer, epoch + 1)

    logger.info("Training finished.")


def main():
    # Load configuration
    load_config()

    # Get required configurations
    device = get_device()
    num_epochs = get