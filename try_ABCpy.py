import sbibm
task_name = "gaussian_linear"
task_name = "two_moons"

task = sbibm.get_task(task_name)  # See sbibm.get_available_tasks() for all tasks
prior = task.get_prior()
simulator = task.get_simulator()
observation = task.get_observation(num_observation=1)  # 10 per task

# Alternatively, we can import existing algorithms, e.g:
from sbibm.algorithms.abcpy.rejection_abc import run as rej_abc  # See help(rej_abc) for keywords
num_samples = 10000
posterior_samples, _, _ = rej_abc(task=task, num_samples=num_samples, num_observation=1, num_simulations=num_samples, quantile=0.03)
# posterior_samples, _, _ = rej_abc(task=task, num_samples=num_samples, num_observation=1, num_simulations=num_samples, num_top_samples=100)

print("Posterior samples shape", posterior_samples.shape)

# Once we got samples from an approximate posterior, compare them to the reference:
# from sbibm.metrics import c2st
# reference_samples = task.get_reference_posterior_samples(num_observation=1)
# c2st_accuracy = c2st(reference_samples, posterior_samples)

# Visualise both posteriors:
from sbibm.visualisation import fig_posterior
fig = fig_posterior(task_name=task_name, observation=1, samples_tensor=posterior_samples, num_samples=100, prior=False)
fig.show()
# Note: Use fig.show() or fig.save() to show or save the figure

# Get results from other algorithms for comparison:
# from sbibm.visualisation import fig_metric
# results_df = sbibm.get_results(dataset="main_paper.csv")
# fig = fig_metric(results_df.query("task == 'two_moons'"), metric="C2ST")