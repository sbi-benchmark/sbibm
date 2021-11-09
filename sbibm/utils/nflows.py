from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from nflows import distributions as distributions_
from nflows import flows, transforms
from nflows.distributions.base import Distribution
from nflows.nn import nets
from nflows.utils import torchutils
from sbi.utils.torchutils import create_alternating_binary_mask
from torch import distributions as dist
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm  # noqa

from sbibm.utils.torch import get_default_device


def get_flow(
    model: str,
    dim_distribution: int,
    dim_context: Optional[int] = None,
    embedding: Optional[torch.nn.Module] = None,
    hidden_features: int = 50,
    made_num_mixture_components: int = 10,
    made_num_blocks: int = 4,
    flow_num_transforms: int = 5,
    mean=0.0,
    std=1.0,
) -> torch.nn.Module:
    """Density estimator

    Args:
        model: Model, one of maf / made / nsf
        dim_distribution: Dim of distribution
        dim_context: Dim of context
        embedding: Embedding network
        hidden_features: For all, number of hidden features
        made_num_mixture_components: For MADEs only, number of mixture components
        made_num_blocks: For MADEs only, number of blocks
        flow_num_transforms: For flows only, number of transforms
        mean: For normalization
        std: For normalization

    Returns:
        Neural network
    """
    standardizing_transform = transforms.AffineTransform(
        shift=-mean / std, scale=1 / std
    )

    features = dim_distribution
    context_features = dim_context

    if model == "made":
        transform = standardizing_transform
        distribution = distributions_.MADEMoG(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=made_num_blocks,
            num_mixture_components=made_num_mixture_components,
            use_residual_blocks=True,
            random_mask=False,
            activation=torch.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            custom_initialization=True,
        )
        neural_net = flows.Flow(transform, distribution, embedding)

    elif model == "maf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.MaskedAffineAutoregressiveTransform(
                            features=features,
                            hidden_features=hidden_features,
                            context_features=context_features,
                            num_blocks=2,
                            use_residual_blocks=False,
                            random_mask=False,
                            activation=torch.tanh,
                            dropout_probability=0.0,
                            use_batch_norm=True,
                        ),
                        transforms.RandomPermutation(features=features),
                    ]
                )
                for _ in range(flow_num_transforms)
            ]
        )

        transform = transforms.CompositeTransform([standardizing_transform, transform])

        distribution = distributions_.StandardNormal((features,))
        neural_net = flows.Flow(transform, distribution, embedding)

    elif model == "nsf":
        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.PiecewiseRationalQuadraticCouplingTransform(
                            mask=create_alternating_binary_mask(
                                features=features, even=(i % 2 == 0)
                            ),
                            transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=hidden_features,
                                context_features=context_features,
                                num_blocks=2,
                                activation=torch.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                            ),
                            num_bins=10,
                            tails="linear",
                            tail_bound=3.0,
                            apply_unconditional_transform=False,
                        ),
                        transforms.LULinear(features, identity_init=True),
                    ]
                )
                for i in range(flow_num_transforms)
            ]
        )

        transform = transforms.CompositeTransform([standardizing_transform, transform])

        distribution = distributions_.StandardNormal((features,))
        neural_net = flows.Flow(transform, distribution, embedding)

    elif model == "nsf_bounded":

        transform = transforms.CompositeTransform(
            [
                transforms.CompositeTransform(
                    [
                        transforms.PiecewiseRationalQuadraticCouplingTransform(
                            mask=create_alternating_binary_mask(
                                features=dim_distribution, even=(i % 2 == 0)
                            ),
                            transform_net_create_fn=lambda in_features, out_features: nets.ResidualNet(
                                in_features=in_features,
                                out_features=out_features,
                                hidden_features=hidden_features,
                                context_features=context_features,
                                num_blocks=2,
                                activation=F.relu,
                                dropout_probability=0.0,
                                use_batch_norm=False,
                            ),
                            num_bins=10,
                            tails="linear",
                            tail_bound=np.sqrt(
                                3
                            ),  # uniform with sqrt(3) bounds has unit-variance
                            apply_unconditional_transform=False,
                        ),
                        transforms.RandomPermutation(features=dim_distribution),
                    ]
                )
                for i in range(flow_num_transforms)
            ]
        )

        transform = transforms.CompositeTransform([standardizing_transform, transform])

        distribution = StandardUniform(shape=(dim_distribution,))
        neural_net = flows.Flow(transform, distribution, embedding)

    else:
        raise ValueError

    return neural_net


def train_flow(
    flow,
    dataset,
    batch_size=100,
    learning_rate=5e-4,
    validation_fraction=0.1,
    stop_after_epochs=20,
    clip_grad_norm=True,
    transform=False,
):
    """
    Train a normalizing flow with maximum likelihood

    Args:
        flow: nflows.flows.Flow
        dataset: torch.tensor()
        batch_size: size of the minibatch
        learning_rate: learning rate
        validation_fraction: fraction of datapoints to be used for validation
        stop_after_epochs: stop training after validation loss has not decreased for this many epochs
        clip_grad_norm: whether to clip the norm of the gradient
        transform: Optional transformation added to output of flow

    Returns:
        Trained flow
    """
    if transform is None or not transform:
        transform = dist.transforms.identity_transform
    dataset = transform(dataset)

    # Get total number of training examples
    num_examples = dataset.shape[0]

    # Select random test and validation splits from (parameter, observation) pairs
    permuted_indices = torch.randperm(num_examples)
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    device = get_default_device()

    dataset = dataset.to(device)
    # Dataset is shared for training and validation loaders.
    dataset = data.TensorDataset(dataset)
    flow = flow.to(device)

    # Create neural_net and validation loaders using a subset sampler.
    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=SubsetRandomSampler(train_indices),
    )
    val_loader = data.DataLoader(
        dataset,
        batch_size=min(batch_size, num_validation_examples),
        shuffle=False,
        drop_last=True,
        sampler=SubsetRandomSampler(val_indices),
    )

    optimizer = optim.Adam(list(flow.parameters()), lr=learning_rate)
    # Keep track of best_validation log_prob seen so far.
    best_validation_log_prob = -1e100
    # Keep track of number of epochs since last improvement.
    epochs_since_last_improvement = 0
    # Keep track of model with best validation performance.
    best_model_state_dict = None

    # Each run also has a dictionary of summary statistics which are populated
    # over the course of training.
    summary = {
        "epochs": [],
        "best-validation-log-probs": [],
    }

    epochs = 0
    converged = False
    while not converged:

        # Train for a single epoch.
        flow.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = (batch[0],)  # .to(device),

            # just do maximum likelihood
            log_prob = flow.log_prob(inputs[0])
            loss = -torch.mean(log_prob)
            loss.backward()
            if clip_grad_norm:
                clip_grad_norm_(flow.parameters(), max_norm=5.0)
            optimizer.step()

        epochs += 1

        # Calculate validation performance.
        flow.eval()
        log_prob_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = (batch[0].to(device),)
                # just do maximum likelihood in the first round
                log_prob = flow.log_prob(inputs[0])
                log_prob_sum += log_prob.sum().item()
        validation_log_prob = log_prob_sum / num_validation_examples

        print("Epoch:", epochs, " --  validation loss", -validation_log_prob)

        # Check for improvement in validation performance over previous epochs.
        if validation_log_prob > best_validation_log_prob:
            best_validation_log_prob = validation_log_prob
            epochs_since_last_improvement = 0
            best_model_state_dict = deepcopy(flow.state_dict())
        else:
            epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if epochs_since_last_improvement > stop_after_epochs - 1:
            flow.load_state_dict(best_model_state_dict)
            converged = True

    # Update summary.
    summary["epochs"].append(epochs)
    summary["best-validation-log-probs"].append(best_validation_log_prob)

    # Transforms
    flow = FlowWrapper(flow, transform)

    return flow


class FlowWrapper:
    def __init__(self, flow, transform):
        self.flow = flow
        self.transform = transform

    def sample(self, *args, **kwargs):
        Y = self.flow.sample(*args, **kwargs)
        return self.transform.inv(Y)

    def log_prob(self, parameters_constrained):
        parameters_unconstrained = self.transform(parameters_constrained)
        log_probs = self.flow.log_prob(parameters_unconstrained)
        log_probs += self.transform.log_abs_det_jacobian(
            parameters_constrained, parameters_unconstrained
        )
        return log_probs


class StandardUniform(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self._log_z = np.log(1 / (2 * np.sqrt(3)) ** shape[0])

    def _log_prob(self, inputs, context):
        all_probs = torch.tensor([])
        for i in inputs:
            if torch.all(torch.abs(i) < torch.sqrt(torch.tensor([3.0]))):
                all_probs = torch.cat((all_probs, torch.tensor([self._log_z])), 0)
            else:
                all_probs = torch.cat((all_probs, torch.tensor([-1e10])), 0)
        return all_probs

    def _sample(self, num_samples, context):
        if context is None:
            return (torch.rand(num_samples, *self._shape) - 0.5) * 2 * np.sqrt(3)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = (
                (torch.rand(context_size * num_samples, *self._shape) - 0.5)
                * 2
                * np.sqrt(3)
            )
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)
