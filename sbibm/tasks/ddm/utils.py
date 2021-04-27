import os
from pathlib import Path

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
    def __init__(self, dt: float = 0.001, num_trials: int = 1) -> None:
        """Wrapping DDM simulation and likelihood computation from Julia.

        Based on Julia package DiffModels.jl

        https://github.com/DrugowitschLab/DiffModels.jl

        Calculates likelihoods via Navarro and Fuss 2009.
        """

        self.dt = dt
        self.num_trials = num_trials

        self.jl = Julia(
            compiled_modules=False,
            sysimage=find_sysimage(),
            runtime="julia",
        )
        self.jl.eval("using DiffModels")

        self.simulate = self.jl.eval(
            f"""
                function f(vs, as; dt={self.dt}, num_trials={self.num_trials})
                    num_parameters = size(vs)[1]
                    rt = fill(NaN, (num_parameters, num_trials))
                    c = fill(NaN, (num_parameters, num_trials))
                                        
                    for i=1:num_parameters
                        drift = ConstDrift(vs[i], dt)
                        bound = ConstSymBounds(as[i], dt)
                        s = sampler(drift, bound)
                    
                        for j=1:num_trials
                            rt[i, j], ci = rand(s)
                            c[i, j] = ci ? 1.0 : 0.0
                        end
                    end
                    return rt, c
                end
            """
        )
        self.likelihood = self.jl.eval(
            f"""
                function f(v, a, rts, cs; dt={self.dt})
                    drift = ConstDrift(v, dt)
                    bound = ConstSymBounds(a, dt)
                    
                    loglsum = 0
                    for (rt, c) in zip(rts, cs)
                        if c > 0
                            loglsum += log(pdfu(drift, bound, rt))
                        else
                            loglsum += log(pdfu(drift, bound, rt))
                        end
                    end
                    return exp(loglsum)
                end
            """
        )
