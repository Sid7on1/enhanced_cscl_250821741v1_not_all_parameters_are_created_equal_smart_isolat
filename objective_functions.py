import logging
import numpy as np
import torch
from torch import nn
from typing import Callable, Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectiveFunction:
    """
    Base class for objective functions.
    """

    def __init__(self):
        self.name = None

    def get_name(self) -> str:
        """
        Get the name of the objective function.

        Returns:
            str: The name of the objective function.
        """
        return self.name

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the given inputs and targets.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: The computed loss.
        """
        raise NotImplementedError("Subclasses must implement the 'compute_loss' method.")

    def get_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the gradients of the loss function with respect to the inputs.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the gradients of the loss function
                with respect to the inputs. The keys are the parameter names, and the values
                are the corresponding gradients.
        """
        raise NotImplementedError("Subclasses must implement the 'get_gradients' method.")


class MeanSquaredError(ObjectiveFunction):
    """
    Mean Squared Error (MSE) objective function.
    """

    def __init__(self):
        super().__init__()
        self.name = "mean_squared_error"

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Mean Squared Error loss.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: The computed MSE loss.
        """
        return torch.mean((inputs - targets) ** 2)

    def get_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the gradients of the MSE loss function with respect to the inputs.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the gradients of the MSE loss function
                with respect to the inputs.
        """
        gradients = {
            "inputs": 2 * (inputs - targets) / inputs.shape[0]
        }
        return gradients


class BinaryCrossEntropy(ObjectiveFunction):
    """
    Binary Cross-Entropy (BCE) objective function.
    """

    def __init__(self):
        super().__init__()
        self.name = "binary_cross_entropy"

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the Binary Cross-Entropy loss.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            torch.Tensor: The computed BCE loss.
        """
        eps = 1e-15
        bs = inputs.size(0)
        inputs = inputs.view(bs, -1)
        targets = targets.view(bs, -1)

        bce = -targets * torch.log(inputs + eps) - (1.0 - targets) * torch.log(1.0 - inputs + eps)
        loss = torch.sum(bce) / (bs * inputs.size(1))

        return loss

    def get_gradients(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the gradients of the BCE loss function with respect to the inputs.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the gradients of the BCE loss function
                with respect to the inputs.
        """
        eps = 1e-15
        bs = inputs.size(0)
        inputs = inputs.view(bs, -1)
        targets = targets.view(bs, -1)

        gradients = {
            "inputs": (-targets / (inputs + eps)) + (1.0 - targets) / (1.0 - inputs + eps)
        }

        return gradients


class CPIFTObjective(ObjectiveFunction):
    """
    Core Parameter Isolation Fine-Tuning (CPI-FT) objective function.
    """

    def __init__(self, lambda_val: float = 0.5, gamma: float = 0.1):
        super().__init__()
        self.name = "cpift_objective"
        self.lambda_val = lambda_val
        self.gamma = gamma

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor, core_params: torch.Tensor) -> torch.Tensor:
        """
        Compute the CPI-FT loss.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.
            core_params (torch.Tensor): Tensor indicating core parameter regions.

        Returns:
            torch.Tensor: The computed CPI-FT loss.
        """
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        core_loss = torch.mean((core_params - inputs) ** 2)

        return ce_loss + self.lambda_val * core_loss

    def get_gradients(self, inputs: torch.Tensor, targets: torch.Tensor, core_params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the gradients of the CPI-FT loss function with respect to the inputs.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.
            core_params (torch.Tensor): Tensor indicating core parameter regions.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the gradients of the CPI-FT loss function
                with respect to the inputs.
        """
        ce_loss = nn.CrossEntropyLoss()
        ce_grad = torch.autograd.grad(ce_loss(inputs, targets), inputs)[0]

        core_grad = 2 * (inputs - core_params) / inputs.shape[0]

        gradients = {
            "inputs": ce_grad + self.lambda_val * core_grad
        }

        return gradients


class CompositeObjective:
    """
    Class to manage composite objective functions.
    """

    def __init__(self, objectives: List[ObjectiveFunction]):
        """
        Initialize the composite objective function.

        Args:
            objectives (List[ObjectiveFunction]): List of objective functions to combine.
        """
        self.objectives = objectives

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss by aggregating losses from individual objectives.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.
            **kwargs: Additional keyword arguments specific to each objective function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the computed losses for each objective.
        """
        losses = {}
        for objective in self.objectives:
            loss = objective.compute_loss(inputs, targets, **kwargs)
            losses[objective.get_name()] = loss

        return losses

    def get_gradients(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute the gradients of the composite loss function with respect to the inputs.

        Args:
            inputs (torch.Tensor): Model predictions or outputs.
            targets (torch.Tensor): Ground truth target values.
            **kwargs: Additional keyword arguments specific to each objective function.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: A nested dictionary containing the gradients of each
                objective function with respect to the inputs. The outer dictionary is indexed by the
                objective name, and the inner dictionary contains the parameter name and gradient.
        """
        gradients = {}
        for objective in self.objectives:
            grad = objective.get_gradients(inputs, targets, **kwargs)
            gradients[objective.get_name()] = grad

        return gradients


# Example usage
if __name__ == "__main__":
    # Create objective functions
    mse_loss = MeanSquaredError()
    bce_loss = BinaryCrossEntropy()
    cpift_loss = CPIFTObjective(lambda_val=0.3, gamma=0.2)

    # Composite objective function
    composite_obj = CompositeObjective(objectives=[mse_loss, bce_loss, cpift_loss])

    # Mock inputs and targets
    inputs = torch.randn(100, 1, requires_grad=True)
    targets = torch.randn(100, 1)

    # Compute composite loss
    composite_loss = composite_obj.compute_loss(inputs, targets)
    for name, loss in composite_loss.items():
        logger.info(f"{name} loss: {loss.item():.4f}")

    # Get composite gradients
    composite_grads = composite_obj.get_gradients(inputs, targets)
    for name, grads in composite_grads.items():
        for param, grad in grads.items():
            logger.info(f"{name} gradient for parameter '{param}': {grad.detach().numpy()}")