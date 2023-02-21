import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import logging
from torch import Tensor, nn
from sbi.utils.torchutils import atleast_2d, ensure_theta_batched
from sbi.samplers.mcmc import SliceSamplerVectorized, sir_init
from sbi.utils import tensor2numpy
from torch.distributions import Bernoulli, Distribution, TransformedDistribution
from torch import Tensor, optim

from torch.utils import data

from julia import Julia
from warnings import warn

JULIA_PROJECT = str(Path(__file__).parent / "julia")
os.environ["JULIA_PROJECT"] = JULIA_PROJECT


def find_sysimage():
    """Find sysimage for DiffModels.jl"""

    if "JULIA_SYSIMAGE_DIFFMODELS" in os.environ:
        environ_path = Path(os.environ["JULIA_SYSIMAGE_DIFFMODELS"])
        if environ_path.exists():
            return str(environ_path)
        else:
            warn("JULIA_SYSIMAGE_DIFFMODELS is set but image does not exist")
            return None
    else:
        warn("JULIA_SYSIMAGE_DIFFMODELS not set")
        default_path = Path("~/.julia_sysimage_diffmodels.so").expanduser()
        if default_path.exists():
            warn(f"Defaulting to {default_path}")
            return str(default_path)
        else:
            return None

# initialize Julia
jl = Julia(
    compiled_modules=False,
    sysimage=find_sysimage(),
    runtime="julia",
)
from julia import Main


class DDMJulia:
    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
        dim_parameters: int = 2,
        seed: int = -1,
    ) -> None:
        """Wrapping DDM simulation and likelihood computation from Julia.

        Based on Julia package DiffModels.jl

        https://github.com/DrugowitschLab/DiffModels.jl

        Calculates likelihoods via Navarro and Fuss 2009.
        """

        self.dt = dt
        self.num_trials = num_trials
        self.seed = seed

        Main.eval("using DiffModels")
        Main.eval("using Random")

        # forward model and likelihood for two-param case, symmetric bounds.
        if dim_parameters == 2:
            self.simulate = Main.eval(
                f"""
                    function simulate(vs, as; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                        num_parameters = size(vs)[1]
                        rt = fill(NaN, (num_parameters, num_trials))
                        c = fill(NaN, (num_parameters, num_trials))

                        # seeding
                        if seed > 0
                            Random.seed!(seed)
                        end
                        for i=1:num_parameters
                            drift = ConstDrift(vs[i], dt)
                            # Pass 0.5a to get bound from boundary separation.
                            bound = ConstSymBounds(0.5 * as[i], dt)
                            s = sampler(drift, bound)
                        
                            for j=1:num_trials
                                rt[i, j], cj = rand(s)
                                c[i, j] = cj ? 1.0 : 0.0
                            end
                        end
                        return rt, c
                    end
                """
            )
            self.log_likelihood = Main.eval(
                f"""
                    function log_likelihood(vs, as, rts, cs; dt={self.dt}, l_lower_bound=1e-29)
                        batch_size = size(vs)[1]
                        num_trials = size(rts)[1]

                        logl = zeros(batch_size)

                        for i=1:batch_size
                            drift = ConstDrift(vs[i], dt)
                            # Pass 0.5a to get bound from boundary separation.
                            bound = ConstSymBounds(0.5 * as[i], dt)

                            for j=1:num_trials
                                if cs[j] == 1.0
                                    logl[i] += log(max(l_lower_bound, pdfu(drift, bound, rts[j])))
                                else
                                    logl[i] += log(max(l_lower_bound, pdfl(drift, bound, rts[j])))
                                end
                            end
                        end
                        return logl
                    end
                """
            )
        # forward model and likelihood for four-param case via asymmetric bounds
        # as in LAN paper, "simpleDDM".
        else:
            self.simulate_simpleDDM = Main.eval(
                f"""
                    function simulate_simpleDDM(v, bl, bu; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                        num_parameters = size(v)[1]
                        rt = fill(NaN, (num_parameters, num_trials))
                        c = fill(NaN, (num_parameters, num_trials))

                        # seeding
                        if seed > 0
                            Random.seed!(seed)
                        end

                        for i=1:num_parameters
                            drift = ConstDrift(v[i], dt)
                            bound = ConstAsymBounds(bu[i], bl[i], dt)
                            s = sampler(drift, bound)

                            for j=1:num_trials
                                # Simulate DDM.
                                rt[i, j], cj = rand(s)
                                c[i, j] = cj ? 1.0 : 0.0
                            end

                        end
                        return rt, c
                    end
                """
            )
            self.log_likelihood_simpleDDM = Main.eval(
                f"""
                    function log_likelihood_simpleDDM(v, bl, bu, rts, cs; ndt=0, dt={self.dt}, l_lower_bound=1e-29)
                        # eps is the numerical lower bound for the likelihood used in HDDM.
                        parameter_batch_size = size(v)[1]
                        num_trials = size(rts)[1]
                        # If no ndt is passed, use zeros without effect.
                        if ndt == 0
                            ndt = zeros(parameter_batch_size)
                        end

                        logl = zeros(parameter_batch_size)

                        for i=1:parameter_batch_size
                            drift = ConstDrift(v[i], dt)
                            bound = ConstAsymBounds(bu[i], bl[i], dt)

                            for j=1:num_trials
                                # Subtract the current ndt from rt to get correct likelihood.
                                rt = rts[j] - ndt[i]
                                # If rt negative (too high ndt) likelihood is 0.
                                if rt < 0
                                    # 1e-29 is the lower bound for negative rts used in HDDM.
                                    logl[i] += log(l_lower_bound)
                                else
                                    if cs[j] == 1.0
                                        logl[i] += log(max(l_lower_bound, pdfu(drift, bound, rt)))
                                    else
                                        logl[i] += log(max(l_lower_bound, pdfl(drift, bound, rt)))
                                    end
                                end
                            end
                        end
                        return logl
                    end
                """
            )


class WeibullDDM:
    """Implementation of the DDM with collapsing decision boundaries as used
    in Fengler et al. 2021.
    """

    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
        dim_parameters: int = 5,
        seed: int = -1,
        tmax: int = 20,
    ) -> None:
        """Wrapping DDM simulation and likelihood computation from Julia.

        Based on Julia package DiffModels.jl

        https://github.com/DrugowitschLab/DiffModels.jl

        Calculates likelihoods via Navarro and Fuss 2009.
        """

        self.dt = dt
        self.num_trials = num_trials
        self.seed = seed
        self.tmax = tmax

        self.jl = Julia(
            compiled_modules=False,
            sysimage=find_sysimage(),
            runtime="julia",
        )
        self.jl.eval("using DiffModels")
        self.jl.eval("using Random")

        self.simulate = self.jl.eval(
            f"""
                function simulate_fullDDM(v, bu, bl, ndt, alpha, beta; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                    num_parameters = size(v)[1]
                    rt = fill(NaN, (num_parameters, num_trials))
                    c = fill(NaN, (num_parameters, num_trials))
                    t = range(0, {self.tmax}, step=dt)

                    # seeding
                    if seed > 0
                        Random.seed!(seed)
                    end

                    for i=1:num_parameters
                        for j=1:num_trials
                            # Weibull fun for collapsing bound.
                            b = bu[i] * exp.(- t.^alpha[i] / beta[i])
                            # time derivative of bound.
                            bg = -bu[i] * exp.(-t.^alpha[i] / beta[i]) .* alpha[i] .* t.^(alpha[i]-1) ./ beta[i]
                            upper = VarBound(b, bg, dt)
                            # lower bound
                            b = bl[i] * exp.(- t.^alpha[i] / beta[i])
                            # time derivative of bound.
                            bg = -bl[i] * exp.(-t.^alpha[i] / beta[i]) .* alpha[i] .* t.^(alpha[i]-1) ./ beta[i]
                            lower = VarBound(b, bg, dt)
                            bound = VarAsymBounds(upper, lower)
                            drift = ConstDrift(v[i], dt)
                            s = sampler(drift, bound)

                            # Simulate DDM.
                            rt[i, j], cj = rand(s)
                            c[i, j] = cj ? 1.0 : 0.0
                        end
                    end
                    return rt, c
                end
            """
        )


class LinearCollapseDDM:
    """Implementation of the DDM with linearly collapsing decision boundaries as used
    in Fengler et al. 2021.
    """

    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
        dim_parameters: int = 4,
        seed: int = -1,
        tmax: int = 20,
    ) -> None:
        """Wrapping DDM simulation and likelihood computation from Julia.

        Based on Julia package DiffModels.jl

        https://github.com/DrugowitschLab/DiffModels.jl

        Calculates likelihoods via Navarro and Fuss 2009.
        """

        self.dt = dt
        self.num_trials = num_trials
        self.seed = seed
        self.tmax = tmax

        self.jl = Julia(
            compiled_modules=False,
            sysimage=find_sysimage(),
            runtime="julia",
        )
        self.jl.eval("using DiffModels")
        self.jl.eval("using Random")

        self.simulate = self.jl.eval(
            f"""
                function simulate(v, bu, bl, gamma; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                    num_parameters = size(v)[1]
                    rt = fill(NaN, (num_parameters, num_trials))
                    c = fill(NaN, (num_parameters, num_trials))
                    t = range(0, {self.tmax}, step=dt)

                    # seeding
                    if seed > 0
                        Random.seed!(seed)
                    end

                    for i=1:num_parameters
                        for j=1:num_trials
                            # Linear collapsing bound.
                            b = bu[i] .+ t .* gamma[i]
                            # b = bu[i] .- (t .* sin(gamma[i]) / cos(gamma[i]))
                            # time derivative of bound.
                            bg = fill(gamma[i], length(t))
                            # bg = fill(-sin(gamma[i]) / cos(gamma[i]), length(t))
                            upper = VarBound(b, bg, dt)

                            # lower bound
                            b = bl[i] .+ t .* gamma[i]
                            # b = bl[i] .- (t .* sin(gamma[i]) / cos(gamma[i]))
                            lower = VarBound(b, bg, dt)
                            bound = VarAsymBounds(upper, lower)
                            drift = ConstDrift(v[i], dt)
                            s = sampler(drift, bound)

                            # Simulate DDM.
                            rt[i, j], cj = rand(s)
                            c[i, j] = cj ? 1.0 : 0.0
                        end
                    end
                    return rt, c
                end
            """
        )


class FullDDMJulia:
    """Implementation of the full DDM model as used in Fengler et al. 2021.

    Difference to simple DDM: drift v, offset w and non-decision time change over trials.

    This results in three additional parameters for that change.

    """

    def __init__(
        self,
        dt: float = 0.001,
        num_trials: int = 1,
        dim_parameters: int = 7,
        seed: int = -1,
    ) -> None:
        """Wrapping DDM simulation and likelihood computation from Julia.

        Based on Julia package DiffModels.jl

        https://github.com/DrugowitschLab/DiffModels.jl

        Calculates likelihoods via Navarro and Fuss 2009.
        """

        self.dt = dt
        self.num_trials = num_trials
        self.seed = seed

        self.jl = Julia(
            compiled_modules=False,
            sysimage=find_sysimage(),
            runtime="julia",
        )
        self.jl.eval("using DiffModels")
        self.jl.eval("using Random")

        self.simulate = self.jl.eval(
            f"""
                function simulate_fullDDM(v, a, w, tau, sv, eps_w, eps_tau; dt={self.dt}, num_trials={self.num_trials}, seed={self.seed})
                    num_parameters = size(v)[1]
                    rt = fill(NaN, (num_parameters, num_trials))
                    c = fill(NaN, (num_parameters, num_trials))

                    # seeding
                    if seed > 0
                        Random.seed!(seed)
                    end

                    for i=1:num_parameters
                        for j=1:num_trials
                            # Perturb for current trial.
                            v_j = v[i] + randn() * sv[i]
                            w_j = w[i] - eps_w[i] + rand() * 2 * eps_w[i]
                            tau_j = tau[i] - eps_tau[i] + rand() * 2 * eps_tau[i]
                            bl = -w_j * a[i]
                            bu = (1 - w_j) * a[i]

                            drift = ConstDrift(v_j, dt)
                            bound = ConstAsymBounds(bu, bl, dt)
                            s = sampler(drift, bound)

                            # Simulate DDM.
                            rt[i, j], cj = rand(s)
                            c[i, j] = cj ? 1.0 : 0.0
                        end
                    end
                    return rt, c
                end
            """
        )


# Refactored from
# https://github.com/mackelab/sbi/blob/main/sbi/inference/posteriors/likelihood_based_posterior.py
class LANPotentialFunctionProvider:
    """
    This class is initialized without arguments during the initialization of the
     Posterior class. When called, it specializes to the potential function appropriate
     to the requested mcmc_method.

    Returns:
        Potential function for use by either numpy or pyro sampler.
    """

    def __init__(self, transforms, lan_net, l_lower_bound: float = 1e-7) -> None:
        self.transforms = transforms
        self.lan_net = lan_net
        self.l_lower_bound = l_lower_bound

    def __call__(self, prior, sbi_net: nn.Module, x: Tensor, mcmc_method: str):
        r"""Return potential function for posterior $p(\theta|x)$.

        Switch on numpy or pyro potential function based on mcmc_method.

        Args:
            prior: Prior distribution that can be evaluated, untransformed.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$. Can be a batch of iid
                x.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.likelihood_nn = self.lan_net
        assert not isinstance(prior, TransformedDistribution)
        self.prior = prior
        self.device = "cpu"
        self.x = atleast_2d(x).to(self.device)
        return self.posterior_potential

    def posterior_potential(self, theta: np.array):
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1, possibly in transformed space.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        theta = ensure_theta_batched(torch.as_tensor(theta, dtype=torch.float32))
        theta_untransformed = self.transforms.inv(theta)

        ll_transformed = self._log_likelihoods_over_trials(
            self.x,
            theta,
            ll_lower_bound=np.log(self.l_lower_bound),
        )
        # Because ladj is subtracted from ll already we can just add the untransformed
        # prior log prob.
        potential_transformed = ll_transformed + self.prior.log_prob(
            theta_untransformed
        )

        return potential_transformed

    def _log_likelihoods_over_trials(
        self, observation, theta_unconstrained, ll_lower_bound: float = -16.11809
    ):
        # lower bound for likelihood set to 1e-7 as in
        # https://github.com/lnccbrown/lans/blob/f2636958bbdb6cb891393a137d1d353be5aa69cd/al-mlp/method_comparison_sim.py#L374

        # move to parameters to constrained space.
        parameters_constrained = self.transforms.inv(theta_unconstrained)
        # turn boundary separation into symmetric boundary for LAN.
        parameters_constrained[:, 1] *= 0.5

        # convert RTs on real line to positive RTs.
        rts = abs(observation)
        num_trials = rts.numel()
        num_parameters = parameters_constrained.shape[0]
        assert rts.shape == torch.Size([num_trials, 1])

        # Code down choices as -1 and up choices as +1.
        cs = torch.ones_like(rts)
        cs[observation < 0] *= -1

        # Repeat theta trial times
        theta_repeated = parameters_constrained.repeat(num_trials, 1)
        # repeat trial data theta times.
        rts_repeated = torch.repeat_interleave(rts, num_parameters, dim=0)
        cs_repeated = torch.repeat_interleave(cs, num_parameters, dim=0)

        # stack everything for the LAN net.
        theta_x_stack = torch.cat((theta_repeated, rts_repeated, cs_repeated), dim=1)
        ll_each_trial = torch.tensor(
            self.lan_net.predict_on_batch(theta_x_stack.numpy()),
            dtype=torch.float32,
        ).reshape(num_trials, num_parameters)

        # Lower bound on each trial ll.
        # Sum across trials.
        llsum = torch.where(
            torch.logical_and(
                # Apply lower bound
                ll_each_trial >= ll_lower_bound,
                # Set to lower bound value when rt<=tau.
                # rts need shape of ll_each_trial.
                rts.repeat(1, num_parameters) > parameters_constrained[:, 3],
            ),
            ll_each_trial,
            ll_lower_bound * torch.ones_like(ll_each_trial),
        ).sum(0)

        # But we need log probs in unconstrained space. Get log abs det jacobian
        log_abs_det = self.transforms.log_abs_det_jacobian(
            self.transforms.inv(theta_unconstrained), theta_unconstrained
        )
        # With identity transform, logabsdet returns second dimension, so sum over it.
        if log_abs_det.ndim > 1:
            log_abs_det = log_abs_det.sum(-1)
        # Double check.
        assert llsum.numel() == num_parameters

        return llsum - log_abs_det


def run_mcmc(
    prior: Distribution,
    potential_fn: Callable,
    mcmc_parameters: dict,
    num_samples: int,
) -> Tensor:
    """Run slice sampling MCMC given prior and potential function, return samples.

    Args:
        prior: prior distribution, usually a TransformedDistribution in unconstrained space.
        potential_fn: Callable returning the negative log posterior potential.
        mcmc_parameters: MCMC hyperparameters.
        num_samples: number of samples to obtain.

    Returns:
        Tensor: [description]
    """

    num_chains = mcmc_parameters["num_chains"]
    num_warmup = mcmc_parameters["warmup_steps"]
    thin = mcmc_parameters["thin"]
    init_strategy = mcmc_parameters["init_strategy"]

    # Obtain initial parameters for each chain using sequential importantce reweighting.
    if init_strategy == "sir":
        initial_params = torch.cat(
            [
                sir_init(prior, potential_fn, **mcmc_parameters)
                for _ in range(num_chains)
            ]
        )
    else:
        initial_params = prior.sample((num_chains,))
    dim_samples = initial_params.shape[1]

    # Use vectorized slice sampling.
    posterior_sampler = SliceSamplerVectorized(
        init_params=tensor2numpy(initial_params),
        log_prob_fn=potential_fn,
        num_chains=num_chains,
        verbose=False,
    )
    # Extract relevant samples.
    warmup_ = num_warmup * thin
    num_samples_ = np.ceil((num_samples * thin) / num_chains)
    samples = posterior_sampler.run(warmup_ + num_samples_)
    samples = samples[:, warmup_:, :]  # discard warmup steps
    samples = samples[:, ::thin, :]  # thin chains
    samples = torch.from_numpy(samples)  # chains x samples x dim

    samples = samples.reshape(-1, dim_samples)[:num_samples, :]
    return samples


# Mixed model utils
class BernoulliMN(nn.Module):
    """Net for learning a conditional Bernoulli mass function over choices given parameters.

    Takes as input parameters theta and learns the parameter p of a Bernoulli.

    Defines log prob and sample functions.
    """

    def __init__(self, n_input=4, n_output=1, n_hidden_units=20, n_hidden_layers=2):
        """Initialize Bernoulli mass network.

        Args:
            n_input: number of input features
            n_output: number of output features, default 1 for a single Bernoulli variable.
            n_hidden_units: number of hidden units per hidden layer.
            n_hidden_layers: number of hidden layers.
        """
        super(BernoulliMN, self).__init__()

        self.n_hidden_layers = n_hidden_layers

        self.activation_fun = nn.Sigmoid()

        self.input_layer = nn.Linear(n_input, n_hidden_units)

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        self.output_layer = nn.Linear(n_hidden_units, n_output)

    def forward(self, theta):
        """Return Bernoulli probability predicted from a batch of parameters.

        Args:
            theta: batch of input parameters for the net.

        Returns:
            Tensor: batch of predicted Bernoulli probabilities.
        """
        assert theta.dim() == 2, "theta needs to have a batch dimension."

        # forward path
        theta = self.activation_fun(self.input_layer(theta))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            theta = self.activation_fun(layer(theta))

        p_hat = self.activation_fun(self.output_layer(theta))

        return p_hat

    def log_prob(self, theta, x):
        """Return Bernoulli log probability of choices x, given parameters theta.

        Args:
            theta: parameters for input to the BernoulliMN.
            x: choices to evaluate.

        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict Bernoulli p and evaluate.
        p = self.forward(theta=theta)
        return Bernoulli(probs=p).log_prob(x)

    def sample(self, theta, num_samples):
        """Returns samples from Bernoulli RV with p predicted via net.

        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Bernoulli samples with shape (batch, num_samples, 1)
        """

        # Predict Bernoulli p and sample.
        p = self.forward(theta)
        return Bernoulli(probs=p).sample((num_samples,))


def get_data_loaders(theta, choices, batch_size, validation_fraction):
    """Return train and test data loaders given data.

    Args:
        theta: DDM parameters.
        choices: Corresponding DDM choices.
        batch_size: training batch size.
        validation_fraction: fraction of test data set.

    Returns:
        Dataloader, Dataloader: train and test dataloaders.
    """
    num_examples = theta.shape[0]
    num_training_examples = int((1 - validation_fraction) * num_examples)

    dataset = data.TensorDataset(theta, choices)
    permuted_indices = torch.randperm(num_examples)
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        sampler=data.sampler.SubsetRandomSampler(train_indices),
    )

    val_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=False,
        sampler=data.sampler.SubsetRandomSampler(val_indices),
    )

    return train_loader, val_loader


def train_choice_net(
    theta,
    choices,
    net: BernoulliMN,
    batch_size: int = 1000,
    max_num_epochs: int = 1000,
    learning_rate=5e-4,
    validation_fraction=0.1,
    stop_after_epochs=20,
):
    """Return trained BernoulliMN given data and training hyperparameters.

    Args:
        theta: DDM parameters
        choices: corresponding DDM choices.
        net: initialized BernoulliMN
        batch_size: training batch size
        max_num_epochs: maximum number of epochs to train.
        learning_rate: learning rate.
        validation_fraction: fraction of validation data.
        stop_after_epochs: number of epochs to wait without validation loss reduction
            before stopping training.

    Returns:
        nn.Module, Tensor: Trained net, validation log probs.
    """
    optimizer = optim.Adam(
        list(net.parameters()),
        lr=learning_rate,
    )

    train_loader, val_loader = get_data_loaders(
        theta, choices, batch_size, validation_fraction
    )

    vallp = []
    converged = False
    num_epochs_trained = 0
    largest_vallp = -float("inf")
    last_vallp_change = 0

    log = logging.getLogger(__name__)
    while num_epochs_trained < max_num_epochs and not converged:
        net.train()
        for batch in train_loader:
            optimizer.zero_grad()
            theta_batch, x_batch = (
                batch[0],
                batch[1],
            )
            # Evaluate on x with theta as context.
            log_prob = net.log_prob(x=x_batch, theta=theta_batch)
            loss = -torch.mean(log_prob)
            loss.backward()
            optimizer.step()

        # Calculate validation performance.
        net.eval()
        log_prob_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                theta_batch, x_batch = (
                    batch[0],
                    batch[1],
                )
                # Evaluate on x with theta as context.
                log_prob = net.log_prob(x=x_batch, theta=theta_batch)
                log_prob_sum += log_prob.sum().item()
        # Take mean over all validation samples.
        _val_log_prob = log_prob_sum / (len(val_loader) * val_loader.batch_size)
        vallp.append(_val_log_prob)

        if largest_vallp < _val_log_prob:
            last_vallp_change = 0
            largest_vallp = _val_log_prob
        else:
            last_vallp_change += 1

        converged = last_vallp_change > stop_after_epochs
        num_epochs_trained += 1

    return net, vallp


class MixedModelSyntheticDDM(nn.Module):
    """Class for combining a Bernoulli choice net and a flow over reaction times for a
    joint DDM synthetic likelihood."""

    def __init__(
        self, choice_net: nn.Module, rt_net: nn.Module, use_log_rts: bool = False
    ):
        """Initializa synthetic likelihood class from a choice net and reaction time
        flow.

        Args:
            choice_net: BernoulliMN net trained to predict choices from DDM parameters.
            rt_net: generative model of reaction time given DDM parameters and choices.
            use_log_rts: whether the rt_net was trained with reaction times transformed
                to log space.
        """
        super(MixedModelSyntheticDDM, self).__init__()

        self.choice_net = choice_net
        self.rt_net = rt_net
        self.use_log_rts = use_log_rts

    def sample(self, theta, num_samples: int = 1, track_gradients=False):
        """Return choices and reaction times given DDM parameters.

        Args:
            theta: DDM parameters, shape (batch, 4)
            num_samples: number of samples to generate.

        Returns:
            Tensor: samples (rt, choice) with shape (num_samples, 2)
        """
        assert theta.shape[0] == 1, "for samples, no batching in theta is possible yet."

        with torch.set_grad_enabled(track_gradients):
            # Sample choices given parameters, from BernoulliMN.
            choices = self.choice_net.sample(theta, num_samples).reshape(num_samples, 1)
            # Pass num_samples=1 because the choices in the context contains num_samples elements already.
            samples = self.rt_net.sample(
                num_samples=1,
                # repeat the single theta to match number of sampled choices.
                context=torch.cat((theta.repeat(num_samples, 1), choices), dim=1),
            ).reshape(num_samples, 1)
        return samples.exp() if self.use_log_rts else samples, choices

    def log_prob(
        self, rts, choices, theta, ll_lower_bound=np.log(1e-7), track_gradients=False
    ):
        """Return joint log likelihood of a batch rts and choices,
        for each entry in a batch of parameters theta.

        Note that we calculate the joint log likelihood over the batch of iid trials.
        Therefore, only theta can be batched and the data is fixed (or a batch of data
        is interpreted as iid trials)
        """
        num_parameters = theta.shape[0]
        num_trials = rts.shape[0]
        assert rts.ndim > 1
        assert rts.shape == choices.shape

        # Repeat parameters for each trial.
        theta_repeated = theta.repeat(num_trials, 1)
        # Repeat choices and rts for each parameter in batch.
        choices_repeated = torch.repeat_interleave(choices, num_parameters, dim=0)
        rts_repeated = torch.repeat_interleave(
            torch.log(rts) if self.use_log_rts else rts, num_parameters, dim=0
        )

        with torch.set_grad_enabled(track_gradients):
            # Get choice log probs from choice net.
            # There are only two choices, thus we only have to get the log probs of those.
            # (We could even just calculate one and then use the complement.)
            zero_choice = torch.zeros(1, 1)
            zero_choice_lp = self.choice_net.log_prob(
                theta,
                torch.repeat_interleave(zero_choice, num_parameters, dim=0),
            ).reshape(1, -1)

            # Calculate complement one-choice log prob.
            one_choice_lp = torch.log(1 - zero_choice_lp.exp())
            zero_one_lps = torch.cat((zero_choice_lp, one_choice_lp), dim=0)

            lp_choices = zero_one_lps[
                choices.type_as(torch.zeros(1, dtype=np.int)).squeeze()
            ].reshape(-1)

            # Get rt log probs from rt net.
            lp_rts = self.rt_net.log_prob(
                rts_repeated,
                context=torch.cat((theta_repeated, choices_repeated), dim=1),
            )

        # Combine into joint lp with first dim over trials.
        lp_combined = (lp_choices + lp_rts).reshape(num_trials, num_parameters)

        # Maybe add log abs det jacobian of RTs: log(1/rt) = - log(rt)
        if self.use_log_rts:
            lp_combined -= torch.log(rts)

        # Set to lower bound where reaction happend before non-decision time tau.
        lp = torch.where(
            torch.logical_and(
                # If rt < tau the likelihood should be zero (or at lower bound).
                rts.repeat(1, num_parameters) > theta[:, -1],
                # Apply lower bound.
                lp_combined > ll_lower_bound,
            ),
            lp_combined,
            ll_lower_bound * torch.ones_like(lp_combined),
        )

        # Return sum over iid trial log likelihoods.
        return lp.sum(0)

    def get_potential_fn(
        self,
        data: Tensor,
        transforms,
        prior: Distribution,
        ll_lower_bound: float,
    ):
        """Return potential function for DDM synthetic likelihood.

        Args:
            data: data to condition on, batch of iid trials of (rt, choice)s.
            transforms: applied transforms
            prior: prior in untransformed space.
            ll_lower_bound: lower bound on the log likelihood.
        """

        # Encode rts and choices.
        rts = abs(data)
        choices = torch.ones_like(data)
        choices[data < 0] = 0

        def pf(theta_transformed):
            """Return log posterior potential for parameters theta.

            Args:
                theta_transformed: parameters in unconstrained space.

            Returns:
                Tensor: potential.
            """
            theta_transformed = ensure_theta_batched(
                torch.as_tensor(theta_transformed, dtype=torch.float32)
            )
            # Go to constrained space to get the likelihood.
            theta = transforms.inv(theta_transformed)
            # Get the log abs det jacobian of the transforms.
            ladj = transforms.log_abs_det_jacobian(theta, theta_transformed)

            # Without transforms, logabsdet returns second dimension.
            if ladj.ndim > 1:
                ladj = ladj.sum(-1)

            # Get synthetic log likelihood in constrained space.
            ll = self.log_prob(rts, choices, theta, ll_lower_bound)

            # Get potential in untransformed space.
            potential = ll + prior.log_prob(theta)

            # Return potential in transformed space.
            return potential - ladj

        return pf


def map_x_to_two_D(x: Tensor) -> Tensor:
    """Return DDM data encoded as (rts, choices)."""
    x = x.squeeze()
    x_2d = torch.zeros(x.shape[0], 2)
    x_2d[:, 0] = x.abs()
    x_2d[x >= 0, 1] = 1

    return x_2d
