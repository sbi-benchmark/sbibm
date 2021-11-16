# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict

import pyro
import torch
from opt_einsum import shared_intermediates
from pyro import distributions as dist
from pyro import poutine as poutine
from pyro.infer import config_enumerate
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.util import is_validation_enabled
from pyro.ops.contract import contract_to_tensor
from pyro.poutine.subsample_messenger import _Subsample
from pyro.util import check_site_shape, ignore_jit_warnings
from torch.autograd import grad
from torch.distributions import biject_to
from torch.distributions.transforms import IndependentTransform


def get_log_prob_fn(
    model,
    model_args=(),
    model_kwargs={},
    implementation="pyro",
    automatic_transform_enabled=False,
    transforms=None,
    max_plate_nesting=None,
    jit_compile=False,
    jit_options=None,
    skip_jit_warnings=False,
    **kwargs,
) -> (Callable, Dict[str, Any]):
    """
    Given a Python callable with Pyro primitives, generates the following model-specific
    functions:
    - a log prob function whose input are parameters and whose output
      is the log prob of the model
    - transforms to transform latent sites of `model` to unconstrained space

    Args:
        model: a Pyro model which contains Pyro primitives.
        model_args: optional args taken by `model`.
        model_kwargs: optional kwargs taken by `model`.
        implementation: Switches between implementations
        automatic_transform_enabled: Whether or not should try to infer transforms
            to unconstrained space
        transforms: Optional dictionary that specifies a transform
            for a sample site with constrained support to unconstrained space. The
            transform should be invertible, and implement `log_abs_det_jacobian`.
            If not specified and the model has sites with constrained support,
            automatic transformations will be applied, as specified in
            `torch.distributions.constraint_registry`.
        max_plate_nesting: Optional bound on max number of nested
            `pyro.plate` contexts. This is required if model contains
            discrete sample sites that can be enumerated over in parallel. Will
            try to infer automatically if not provided
        jit_compile: Optional parameter denoting whether to use
            the PyTorch JIT to trace the log density computation, and use this
            optimized executable trace in the integrator.
        jit_options: A dictionary contains optional arguments for
            `torch.jit.trace` function.
        ignore_jit_warnings: Flag to ignore warnings from the JIT
            tracer when `jit_compile=True`. Default is False.

    Returns:
        `log_prob_fn` and `transforms`
    """
    if transforms is None:
        transforms = {}

    if max_plate_nesting is None:
        max_plate_nesting = _guess_max_plate_nesting(model, model_args, model_kwargs)

    model = poutine.enum(
        config_enumerate(model), first_available_dim=-1 - max_plate_nesting
    )
    model_trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)

    has_enumerable_sites = False
    for name, node in model_trace.iter_stochastic_nodes():
        fn = node["fn"]

        if isinstance(fn, _Subsample):
            if fn.subsample_size is not None and fn.subsample_size < fn.size:
                raise NotImplementedError(
                    "Model with subsample sites are not supported."
                )
            continue

        if fn.has_enumerate_support:
            has_enumerable_sites = True
            continue

        if automatic_transform_enabled:
            transforms[name] = biject_to(fn.support).inv
        else:
            transforms[name] = dist.transforms.IndependentTransform(dist.transforms.identity_transform, 1)

    if implementation == "pyro":
        trace_prob_evaluator = TraceEinsumEvaluator(
            model_trace, has_enumerable_sites, max_plate_nesting
        )

        lp_maker = _LPMaker(
            model, model_args, model_kwargs, trace_prob_evaluator, transforms
        )

        lp_fn = lp_maker.get_lp_fn(jit_compile, skip_jit_warnings, jit_options)

    elif implementation == "experimental":
        assert automatic_transform_enabled is False

        if jit_compile:
            warnings.warn("Will not JIT compile, unsupported for now.")

        def lp_fn(input_dict):
            excluded_nodes = set(["_INPUT", "_RETURN"])

            for key, value in input_dict.items():
                model_trace.nodes[key]["value"] = value

            replayed_model = pyro.poutine.replay(model, model_trace)

            log_p = 0
            for trace_enum in iter_discrete_traces("flat", fn=replayed_model):
                trace_enum.compute_log_prob()

                for node_name, node in trace_enum.nodes.items():
                    if node_name in excluded_nodes:
                        continue

                    if node["log_prob"].ndim == 1:
                        log_p += trace_enum.nodes[node_name]["log_prob"]
                    else:
                        log_p += trace_enum.nodes[node_name]["log_prob"].sum(dim=1)

            return log_p

    else:
        raise NotImplementedError

    return lp_fn, transforms


def get_log_prob_grad_fn(
    model,
    model_args=(),
    model_kwargs={},
    implementation="pyro",
    automatic_transform_enabled=False,
    transforms=None,
    max_plate_nesting=None,
    jit_compile=False,
    jit_options=None,
    skip_jit_warnings=False,
    **kwargs,
) -> (Callable, Dict[str, Any]):
    """
    Given a Python callable with Pyro primitives, generates the following model-specific
    functions:
    - a log prob grad function whose input are parameters and whose
      output is the grd of log prob of the model wrt parameters
    - transforms to transform latent sites of `model` to
      unconstrained space

    Args:
        See `get_log_prob_fn`

    Returns:
        `log_prob_grad_fn` and `transforms`
    """
    lp_fn, transforms = get_log_prob_fn(
        model,
        model_args,
        model_kwargs,
        implementation,
        automatic_transform_enabled,
        transforms,
        max_plate_nesting,
        jit_compile,
        jit_options,
        skip_jit_warnings,
    )
    lp_grad_fn = make_log_prob_grad_fn(lp_fn)
    return lp_grad_fn, transforms


class _LPMaker:
    def __init__(
        self, model, model_args, model_kwargs, trace_prob_evaluator, transforms
    ):
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.trace_prob_evaluator = trace_prob_evaluator
        self.transforms = transforms
        self._compiled_fn = None

    def _lp_fn(self, params):
        params_constrained = {k: self.transforms[k].inv(v) for k, v in params.items()}
        cond_model = poutine.condition(self.model, params_constrained)
        model_trace = poutine.trace(cond_model).get_trace(
            *self.model_args, **self.model_kwargs
        )
        log_joint = torch.atleast_1d(self.trace_prob_evaluator.log_prob(model_trace))
        for name, t in self.transforms.items():
            log_joint -= t.log_abs_det_jacobian(params_constrained[name], params[name])
        return log_joint

    def _lp_fn_jit(self, skip_jit_warnings, jit_options, params):
        if not params:
            return self._lp_fn(params)
        names, vals = zip(*sorted(params.items()))

        if self._compiled_fn:
            return self._compiled_fn(*vals)

        with pyro.validation_enabled(False):
            tmp = []
            for _, v in pyro.get_param_store().named_parameters():
                if v.requires_grad:
                    v.requires_grad_(False)
                    tmp.append(v)

            def _lp_jit(*zi):
                params = dict(zip(names, zi))
                return self._lp_fn(params)

            if skip_jit_warnings:
                _lp_jit = ignore_jit_warnings()(_lp_jit)
            self._compiled_fn = torch.jit.trace(_lp_jit, vals, **jit_options)

            for v in tmp:
                v.requires_grad_(True)
            return self._compiled_fn(*vals)

    def get_lp_fn(self, jit_compile=False, skip_jit_warnings=True, jit_options=None):
        if jit_compile:
            jit_options = {"check_trace": False} if jit_options is None else jit_options
            return partial(self._lp_fn_jit, skip_jit_warnings, jit_options)
        return self._lp_fn


def make_log_prob_grad_fn(log_prob_fn):
    """Makes `log_prob_grad_fn`

    Args:
        log_prob_fn: python callable that takes in a dictionary of parameters
        and returns the log prob.

    Returns:
        `log_prob_grad_fn`

    :param dict z: dictionary of parameter values keyed by site name.
    :return: tuple of `(z_grads, log_prob)`, where `z_grads` is a dictionary
        with the same keys as `z` containing gradients and log prob is a
        torch scalar.
    """

    def log_prob_grad_fn(z):
        z_keys, z_nodes = zip(*z.items())
        for node in z_nodes:
            node.requires_grad_(True)
        try:
            log_prob = log_prob_fn(z)

        # deal with singular matrices
        except RuntimeError as e:
            if "singular U" in str(e):
                grads = {k: v.new_zeros(v.shape) for k, v in z.items()}
                return grads, z_nodes[0].new_tensor(float("nan"))
            else:
                raise e

        grads = grad(log_prob, z_nodes)
        for node in z_nodes:
            node.requires_grad_(False)
        return dict(zip(z_keys, grads)), log_prob.detach()

    return log_prob_grad_fn


class TraceEinsumEvaluator:
    """
    Computes the log probability density of a trace (of a model with
    tree structure) that possibly contains discrete sample sites
    enumerated in parallel. This uses optimized `einsum` operations
    to marginalize out the the enumerated dimensions in the trace
    via :class:`~pyro.ops.contract.contract_to_tensor`.

    :param model_trace: execution trace from a static model.
    :param bool has_enumerable_sites: whether the trace contains any
        discrete enumerable sites.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts.
    """

    def __init__(self, model_trace, has_enumerable_sites=False, max_plate_nesting=None):
        self.has_enumerable_sites = has_enumerable_sites
        self.max_plate_nesting = max_plate_nesting
        # To be populated using the model trace once.
        self._enum_dims = set()
        self.ordering = {}
        self._populate_cache(model_trace)

    def _populate_cache(self, model_trace):
        """
        Populate the ordinals (set of ``CondIndepStack`` frames)
        and enum_dims for each sample site.
        """
        if not self.has_enumerable_sites:
            return
        if self.max_plate_nesting is None:
            raise ValueError(
                "Finite value required for `max_plate_nesting` when model "
                "has discrete (enumerable) sites."
            )
        model_trace.compute_log_prob()
        model_trace.pack_tensors()
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                self.ordering[name] = frozenset(
                    model_trace.plate_to_symbol[f.name]
                    for f in site["cond_indep_stack"]
                    if f.vectorized
                )
        self._enum_dims = set(model_trace.symbol_to_dim) - set(
            model_trace.plate_to_symbol.values()
        )

    def _get_log_factors(self, model_trace):
        """
        Aggregates the `log_prob` terms into a list for each
        ordinal.
        """
        model_trace.compute_log_prob()
        model_trace.pack_tensors()
        log_probs = OrderedDict()
        # Collect log prob terms per independence context.
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                log_probs.setdefault(self.ordering[name], []).append(
                    site["packed"]["log_prob"]
                )
        return log_probs

    def log_prob(self, model_trace):
        """
        Returns the log pdf of `model_trace` by appropriately handling
        enumerated log prob factors.
        :return: log pdf of the trace.
        """
        if not self.has_enumerable_sites:
            return model_trace.log_prob_sum()
        log_probs = self._get_log_factors(model_trace)
        with shared_intermediates() as cache:
            return contract_to_tensor(log_probs, self._enum_dims, cache=cache)


def _guess_max_plate_nesting(model, args, kwargs):
    """
    Guesses max_plate_nesting by running the model once
    without enumeration. This optimistically assumes static model
    structure.
    """
    with poutine.block():
        model_trace = poutine.trace(model).get_trace(*args, **kwargs)
    sites = [site for site in model_trace.nodes.values() if site["type"] == "sample"]

    dims = [
        frame.dim
        for site in sites
        for frame in site["cond_indep_stack"]
        if frame.vectorized
    ]
    max_plate_nesting = -min(dims) if dims else 0
    return max_plate_nesting
