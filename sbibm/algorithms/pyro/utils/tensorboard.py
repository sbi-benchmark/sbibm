from matplotlib import pyplot as plt

from sbibm.utils.tensorboard import tb_plot_posterior


def tb_ess(writer, mcmc, site_name="parameters"):
    n_eff = mcmc.diagnostics()[site_name]["n_eff"].squeeze()
    for p in range(n_eff.shape[0]):
        writer.add_scalar(f"efficient sample size/{site_name}", n_eff[p], p + 1)


def tb_r_hat(writer, mcmc, site_name="parameters"):
    r_hat = mcmc.diagnostics()[site_name]["r_hat"].squeeze()
    for p in range(r_hat.shape[0]):
        writer.add_scalar(f"r hat/{site_name}", r_hat[p], p + 1)


def tb_acf(writer, mcmc, site_name="parameters", num_samples=1000, maxlags=50):
    samples = mcmc.get_samples(num_samples=num_samples, group_by_chain=True)[site_name]
    for c in range(samples.shape[0]):
        for p in range(samples.shape[-1]):
            fig = plt.figure()
            plt.gca().acorr(samples[c, :].squeeze()[:, p].numpy(), maxlags=maxlags)
            writer.add_figure(
                f"acf/chain {c+1}/parameter {p+1}",
                fig,
                close=True,
            )


def tb_posteriors(writer, mcmc, site_name="parameters", num_samples=1000):
    samples = mcmc.get_samples(num_samples=num_samples, group_by_chain=True)[site_name]
    for c in range(samples.shape[0]):
        tb_plot_posterior(
            writer=writer, samples=samples[c, :], tag=f"posterior/chain {c+1}"
        )


def tb_marginals(writer, mcmc, site_name="parameters", num_samples=1000):
    samples = mcmc.get_samples(num_samples=num_samples, group_by_chain=True)[site_name]
    for c in range(samples.shape[0]):
        for p in range(samples.shape[-1]):
            writer.add_histogram(
                f"marginal/{site_name}/{p+1}",
                samples[c, :].squeeze()[:, p],
                c,
            )


def tb_make_hook_fn(writer, site_name="parameters"):
    """Builds hook function for runtime logging"""

    def hook_fn(kernel, samples, stage, i):
        """Logging during run

        Args:
            kernel: MCMC kernel
            samples: Current samples
            stage: Current stage, either `sample` or `warmup`
            i: i'th sample for the given stage
        """
        try:
            num_chain = int(stage.split("[")[1][:-1])
        except:
            num_chain = 0

        stage_prefix = stage.split(" ")[0].lower()

        # Backtransform sample
        samples_inv = kernel.transforms[site_name].inv(samples[site_name])

        for p in range(len(samples[site_name].squeeze())):
            # Trace
            writer.add_scalar(
                f"{stage_prefix}/chain/{num_chain+1}/trace/{site_name}/{p+1}",
                samples_inv.squeeze()[p],
                i,
            )

            # Slice kernel bracket width
            try:
                if stage_prefix == "warmup":
                    kernel_width = kernel._width[p]
                    writer.add_scalar(
                        f"{stage_prefix}/chain/{num_chain+1}/bracket width/{site_name}/{p+1}",
                        kernel_width,
                        i,
                    )
            except:
                pass

    return hook_fn
