import os
from pathlib import Path

import numpy as np
import torch

from numba import jit
from torch import Tensor, nn
from sbi.utils.torchutils import ScalarFloat, atleast_2d, ensure_theta_batched
from sbi.mcmc import sir, SliceSamplerVectorized
from sbi.utils import tensor2numpy
from torch.distributions import Bernoulli


from julia import Julia
from warnings import warn

JULIA_PROJECT = str(Path(__file__).parent / "julia")
os.environ["JULIA_PROJECT"] = JULIA_PROJECT


def find_sysimage():
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

        self.jl = Julia(
            compiled_modules=False,
            sysimage=find_sysimage(),
            runtime="julia",
        )
        self.jl.eval("using DiffModels")
        self.jl.eval("using Random")

        # forward model and likelihood for two-param case, symmetric bounds.
        if dim_parameters == 2:
            self.simulate = self.jl.eval(
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
            self.log_likelihood = self.jl.eval(
                f"""
                    function log_likelihood(vs, as, rts, cs; dt={self.dt}, eps=1e-29)
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
            self.simulate_simpleDDM = self.jl.eval(
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
            self.log_likelihood_simpleDDM = self.jl.eval(
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


########
## python simulator and likelihoods
@jit(nopython=True)
def fptd_large(t, w, k):
    """
    Likelihood based on large t approximation from Navarro-Fuss.
    """
    # vectorized, about 4 times speedup
    ks = np.arange(1, k + 1)
    return (
        ks * np.exp(-((ks ** 2) * (np.pi ** 2) * t) / 2) * np.sin(ks * np.pi * w)
    ).sum() * np.pi


@jit(nopython=True)
def fptd_small(t, w, k):
    """
    Likelihood based on small t approximation from Navarro-Fuss.
    """
    ks = np.arange(-np.ceil((k - 1) / 2), np.floor((k - 1) / 2) + 1)
    return ((w + (2 * ks)) * np.exp(-((w + (2 * ks)) ** 2) / (2 * t))).sum() * (
        1 / np.sqrt(2 * np.pi * (t ** 3))
    )


@jit(nopython=True)
def calculate_leading_term(t, v, a, w):
    """
    Compute the leading term that was factored out in Navarro-Fuss.
    """
    # where did the pi go from the general a to a=1, v=0 case? (L92 in Navarro & Fuss 2009)
    return 1 / (a ** 2) * np.exp(-(v * a * w) - (((v ** 2) * t) / 2))


@jit(nopython=True)
def choice_function(t, eps):
    """
    Algorithm to determine whether small t or large t approximation should be used given
    error tolerance.
    """
    eps_l = min(eps, 1 / (t * np.pi))
    eps_s = min(eps, 1 / (2 * np.sqrt(2 * np.pi * t)))
    k_l = int(
        np.ceil(
            max(
                np.sqrt(-(2 * np.log(np.pi * t * eps_l)) / (np.pi ** 2 * t)),
                1 / (np.pi * np.sqrt(t)),
            )
        )
    )
    k_s = int(
        np.ceil(
            max(
                2 + np.sqrt(-2 * t * np.log(2 * eps_s * np.sqrt(2 * np.pi * t))),
                1 + np.sqrt(t),
            )
        )
    )
    return k_s - k_l, k_l, k_s


@jit(nopython=True)
def fptd(t=0, v=0, a=1, w=0.5, tau=0, eps=1e-29):
    """
    Compute first passage time distributions using Navarro-Fuss approximation.

    Args
        eps: lower bound on likelihood evaluations.
    """
    if t < 0:
        # negative reaction times signify upper boundary crossing
        v = -v
        w = 1 - w
        t = np.abs(t) - tau
    else:
        t = t - tau

    if t != 0:
        # compute leading term and which approximation to use based on t
        leading_term = calculate_leading_term(t, v, a, w)
        t_adj = t / (a ** 2)
        sgn_lambda, k_l, k_s = choice_function(t_adj, eps)
        if sgn_lambda >= 0:
            return max(eps, leading_term * fptd_large(t_adj, w, k_l))
        else:
            return max(eps, leading_term * fptd_small(t_adj, w, k_s))
    else:
        return eps


def logfptd_batch_python(v, a, w, tau, rt, c, eps=1e-29):
    """
    Compute a batch of log likelihoods for the 4-param model,
    roughly following the same call signature as the Julia counterpart.

    Note here rt is always positive, and choices are passed in
    separately. Choice = 1 when crossing upper bound, but is denoted by
    negative rt. So the two variables are merged as signed reaction time via:

    rt * -sign(c)
    """
    signed_rt = rt * -np.sign(c)  # negative time is crossing upperbound
    batch_size = np.shape(v)[0]  # number of param configs
    logprob = np.zeros(batch_size)
    for i_b in range(batch_size):
        # loop over batches and aggregate across trials by adding loglikelihood
        logprob[i_b] = np.log(
            np.array(
                [fptd(t, v[i_b], a[i_b], w[i_b], tau[i_b], eps) for t in signed_rt]
            )
        ).sum()

    return logprob


# simulators
@jit(nopython=True)
def ddm(v=0, a=1, w=0.5, tau=0, s=1, dt=0.001, t_max=20):
    """
    Simulate a single trial of the 4-param drift diffusion process.
    """
    s_sqrtdt = s * np.sqrt(dt)
    n_steps = int(t_max / dt + 1)
    noise = np.random.normal(0, 1, size=n_steps) * s_sqrtdt
    x = w * a
    t, i = 0, 0
    while (t <= t_max) and (x >= 0) and (x <= a):
        # drift + diffusion, increment time
        x += v * dt + noise[i]
        t += dt
        i += 1
    return t + tau, np.sign(x)


def ddm_batch_python(v, a, w, tau, dt, t_max, num_trials, seed):
    """
    Simulate multiple trials of DDM per param configuration.
    """
    num_params = np.shape(v)[0]
    rt = np.zeros((num_params, num_trials))
    c = np.zeros((num_params, num_trials))

    # set seed
    np.random.seed(seed)
    for i_p in range(num_params):
        for i_t in range(num_trials):
            # loop simulator
            rt[i_p, i_t], c[i_p, i_t] = ddm(
                v[i_p], a[i_p], w[i_p], tau[i_p], dt=dt, t_max=t_max
            )

    return rt, c


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
            prior: Prior distribution that can be evaluated.
            likelihood_nn: Neural likelihood estimator that can be evaluated.
            x: Conditioning variable for posterior $p(\theta|x)$. Can be a batch of iid
                x.
            mcmc_method: One of `slice_np`, `slice`, `hmc` or `nuts`.

        Returns:
            Potential function for sampler.
        """
        self.likelihood_nn = self.lan_net
        self.prior = prior
        self.device = "cpu"
        self.x = atleast_2d(x).to(self.device)
        return self.np_potential

    def log_likelihood(self, theta: Tensor, track_gradients: bool = False) -> Tensor:
        """Return log likelihood of fixed data given a batch of parameters."""

        log_likelihoods = self._log_likelihoods_over_trials(
            self.x,
            ensure_theta_batched(theta).to(self.device),
        )

        return log_likelihoods

    def np_potential(self, theta: np.array):
        r"""Return posterior log prob. of theta $p(\theta|x)$"

        Args:
            theta: Parameters $\theta$, batch dimension 1.

        Returns:
            Posterior log probability of the theta, $-\infty$ if impossible under prior.
        """
        theta = ensure_theta_batched(torch.as_tensor(theta, dtype=torch.float32))

        # Notice opposite sign to pyro potential.
        return self._log_likelihoods_over_trials(
            self.x,
            theta,
            ll_lower_bound=np.log(self.l_lower_bound),
            # the prior is assumend to live in unconstrained space.
        ) + self.prior.log_prob(theta)

    def _log_likelihoods_over_trials(
        self, observation, theta_unconstrained, ll_lower_bound: float = -16.11809
    ):
        # lower bound for likelihood set to 1e-7 as in
        # https://github.com/lnccbrown/lans/blob/f2636958bbdb6cb891393a137d1d353be5aa69cd/al-mlp/method_comparison_sim.py#L374

        # move to parameters to constrained space.
        parameters_constrained = self.transforms.inv(theta_unconstrained)
        # turn boundary separation into symmetric boundary for LAN.
        parameters_constrained[:, 1] *= 0.5

        rts = abs(observation)
        num_trials = rts.numel()
        num_parameters = parameters_constrained.shape[0]
        assert rts.shape == torch.Size([num_trials, 1])

        # Code down -1 up +1.
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

        # But we need log probs in unconstrained space. Get log abs det jac
        log_abs_det = self.transforms.log_abs_det_jacobian(
            self.transforms.inv(theta_unconstrained), theta_unconstrained
        )
        # Without transforms, logabsdet returns second dimension.
        if log_abs_det.ndim > 1:
            log_abs_det = log_abs_det.sum(-1)

        assert llsum.numel() == num_parameters

        return llsum - log_abs_det


def run_mcmc(prior, potential_fn, mcmc_parameters, num_samples):

    num_chains = mcmc_parameters["num_chains"]
    num_warmup = mcmc_parameters["warmup_steps"]
    thin = mcmc_parameters["thin"]

    initial_params = torch.cat(
        [sir(prior, potential_fn, **mcmc_parameters) for _ in range(num_chains)]
    )
    dim_samples = initial_params.shape[1]

    posterior_sampler = SliceSamplerVectorized(
        init_params=tensor2numpy(initial_params),
        log_prob_fn=potential_fn,
        num_chains=num_chains,
        verbose=False,
    )
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
        super(BernoulliMN, self).__init__()

        self.n_hidden_layers = n_hidden_layers

        self.activation_fun = nn.Sigmoid()

        self.input_layer = nn.Linear(n_input, n_hidden_units)

        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(n_hidden_units, n_hidden_units))

        self.output_layer = nn.Linear(n_hidden_units, n_output)

    def forward(self, theta):
        assert theta.dim() == 2

        # forward path
        theta = self.activation_fun(self.input_layer(theta))

        # iterate n hidden layers, input x and calculate tanh activation
        for layer in self.hidden_layers:
            theta = self.activation_fun(layer(theta))

        p_hat = self.activation_fun(self.output_layer(theta))

        return p_hat

    def log_prob(self, theta, x):
        p = self.forward(theta=theta)
        return Bernoulli(probs=p).log_prob(x)

    def sample(self, theta, num_samples):

        p = self.forward(theta)

        return Bernoulli(probs=p).sample((num_samples,))


from torch import optim
from torch.utils import data


def get_data_loaders(theta, choices, batch_size, validation_fraction):
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


def train(
    theta,
    choices,
    net: BernoulliMN,
    batch_size: int = 100,
    max_num_epochs: int = 1000,
    learning_rate=5e-4,
    validation_fraction=0.1,
    stop_after_epochs=20,
):
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

        return net, vallp


class MixedModelSyntheticDDM(nn.Module):
    def __init__(self, choice_net: nn.Module, rt_net: nn.Module):
        super(MixedModelSyntheticDDM, self).__init__()

        self.choice_net = choice_net
        self.rt_net = rt_net

    def sample(self, theta, num_samples: int = 1):
        assert theta.shape[0] == 1

        choices = (
            self.choice_net.sample(theta, num_samples).reshape(num_samples, 1).detach()
        )
        # Pass num_samples=1 because the choices in the context contains num_samples elements already.
        rts = (
            self.rt_net.sample(
                num_samples=1,
                context=torch.cat((theta.repeat(num_samples, 1), choices), dim=1),
            )
            .reshape(num_samples, 1)
            .detach()
        )
        return rts, choices

    def log_prob(self, rts, choices, theta, ll_lower_bound=np.log(1e-7)):
        """Return joint log likelihood of a batch rts and choices,
        for each entry in a batch of parameters theta.

        Note that we take the joint likelihood over the batch of iid trials.

        I.e., only theta can be batched.
        """
        num_parameters = theta.shape[0]
        num_trials = rts.shape[0]
        assert rts.ndim > 1
        assert rts.shape == choices.shape

        theta_repeated = theta.repeat(num_trials, 1)
        choices_repeated = torch.repeat_interleave(choices, num_parameters, dim=0)
        rts_repeated = torch.repeat_interleave(rts, num_parameters, dim=0)

        lp_choices = (
            self.choice_net.log_prob(theta_repeated, choices_repeated)
            .detach()
            .reshape(-1)
        )

        lp_rts = self.rt_net.log_prob(
            rts_repeated, context=torch.cat((theta_repeated, choices_repeated), dim=1)
        ).detach()

        lp_combined = (lp_choices + lp_rts).reshape(num_trials, num_parameters)

        # Set to lower bound where reaction happend before non-decision time tau.
        lp = torch.where(
            torch.logical_and(
                rts.repeat(1, num_parameters) > theta[:, -1],
                lp_combined > ll_lower_bound,
            ),
            lp_combined,
            ll_lower_bound * torch.ones_like(lp_combined),
        )

        # Return sum over iid trial likelihoods.
        return lp.sum(0)

    def get_potential_fn(self, data, transforms, prior_transformed, ll_lower_bound):
        def pf(theta_transformed):
            theta_transformed = ensure_theta_batched(
                torch.as_tensor(theta_transformed, dtype=torch.float32)
            )
            theta = transforms.inv(theta_transformed)
            ladj = transforms.log_abs_det_jacobian(theta, theta_transformed)
            # Without transforms, logabsdet returns second dimension.
            if ladj.ndim > 1:
                ladj = ladj.sum(-1)

            rts = abs(data)
            choices = torch.ones_like(data)
            choices[data < 0] = 0

            ll = self.log_prob(rts, choices, theta, ll_lower_bound)

            return ll - ladj + prior_transformed.log_prob(theta_transformed)

        return pf
