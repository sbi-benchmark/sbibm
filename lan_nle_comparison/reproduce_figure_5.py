import keras
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import sbibm
import torch
import time
from joblib import Parallel, delayed

from sbibm.tasks.ddm.utils import run_mcmc, LANPotentialFunctionProvider
from sbibm.algorithms.sbi.utils import wrap_prior_dist


# network trained on KDE likelihood for 4-param ddm
lan_kde_path = "../sbibm/algorithms/lan/lan_pretrained/model_final_ddm.h5"
lan_ana_path = "../sbibm/algorithms/lan/lan_pretrained/model_final_ddm_analytic.h5"
lan_kde = keras.models.load_model(lan_kde_path, compile=False)
lan_ana = keras.models.load_model(lan_ana_path, compile=False)

# Load pretrained NLE model
with open("../sbibm/algorithms/lan/nle_pretrained/mm_688_4.p", "rb") as fh:
    nle = pickle.load(fh)

num_workers = 80
m = num_workers
n = 1024
l_lower_bound = 1e-7
num_samples = 10000


task = sbibm.get_task("ddm")
prior = task.get_prior_dist()
simulator = task.get_simulator(num_trials=n)  # Passing the seed to Julia.

thos = prior.sample((m,))
xos = task.get_simulator()(thos)

mcmc_parameters = {
    "num_chains": 100,
    "thin": 10,
    "warmup_steps": 100,
    "init_strategy": "sir",
    "sir_batch_size": 100,
    "sir_num_batches": 1000,
}

with open("ddm_transforms.p", "rb") as fh:
    transforms = pickle.load(fh)["transforms"]
prior_transformed = wrap_prior_dist(prior, transforms)


def local_run(xi):

    tic = time.time()
    # Get potential function for mixed model.
    potential_fn_mm = nle.get_potential_fn(
        xi.reshape(-1, 1),
        transforms,
        # Pass untransformed prior and correct internally with ladj.
        prior=prior,
        ll_lower_bound=np.log(l_lower_bound),
    )

    # Run MCMC in transformed space.
    transformed_samples = run_mcmc(
        prior=prior_transformed,
        potential_fn=potential_fn_mm,
        mcmc_parameters=mcmc_parameters,
        num_samples=num_samples,
    )

    nle_samples = transforms.inv(transformed_samples)
    nle_time = time.time() - tic

    tic = time.time()
    # Use potential function provided refactored from SBI toolbox for LAN.
    potential_fn_lan = LANPotentialFunctionProvider(transforms, lan_kde, l_lower_bound)

    lan_transformed_samples = run_mcmc(
        prior=prior_transformed,
        # Pass original prior to pf and correct potential with ladj.
        potential_fn=potential_fn_lan(
            prior=prior,
            sbi_net=None,
            x=xi.reshape(-1, 1),
            mcmc_method="slice_np_vectorized",
        ),
        mcmc_parameters=mcmc_parameters,
        num_samples=num_samples,
    )

    lan_samples = transforms.inv(lan_transformed_samples)
    lan_time = time.time() - tic

    return nle_samples, lan_samples, nle_time, lan_time


# run in parallel
results = Parallel(n_jobs=num_workers)(delayed(local_run)(_) for _ in xos)

with open("figure_5_results.p", "wb") as fh:
    pickle.dump(dict(thos=thos, xos=xos, results=results), fh)

print("Done")
