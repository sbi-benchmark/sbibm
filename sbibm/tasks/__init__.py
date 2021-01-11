from pathlib import Path
from typing import Any, List

from sbibm.tasks.task import Task


def get_task(task_name: str, *args: Any, **kwargs: Any) -> Task:
    """Get task

    Args:
        task_name: Name of task

    Returns:
        Task instance
    """
    if task_name == "lotka_volterra":
        from sbibm.tasks.lotka_volterra.task import LotkaVolterra

        return LotkaVolterra(*args, **kwargs)

    elif task_name == "bernoulli_glm":
        from sbibm.tasks.bernoulli_glm.task import BernoulliGLM

        return BernoulliGLM(*args, **kwargs)

    elif task_name == "bernoulli_glm_raw":
        from sbibm.tasks.bernoulli_glm.task import BernoulliGLM

        return BernoulliGLM(*args, summary="raw", **kwargs)

    elif task_name == "gaussian_linear":
        from sbibm.tasks.gaussian_linear.task import GaussianLinear

        return GaussianLinear(*args, **kwargs)

    elif task_name == "gaussian_linear_uniform":
        from sbibm.tasks.gaussian_linear_uniform.task import GaussianLinearUniform

        return GaussianLinearUniform(*args, **kwargs)

    elif task_name == "gaussian_mixture":
        from sbibm.tasks.gaussian_mixture.task import GaussianMixture

        return GaussianMixture(*args, **kwargs)

    elif task_name == "slcp" or task_name == "gaussian_nonlinear":
        from sbibm.tasks.slcp.task import SLCP

        return SLCP(*args, **kwargs)

    elif task_name == "slcp_distractors":
        from sbibm.tasks.slcp.task import SLCP

        return SLCP(*args, distractors=True, **kwargs)

    if task_name == "sir":
        from sbibm.tasks.sir.task import SIR

        return SIR(*args, **kwargs)

    elif task_name == "two_moons":
        from sbibm.tasks.two_moons.task import TwoMoons

        return TwoMoons(*args, **kwargs)

    else:
        raise NotImplementedError()


def get_task_name_display(task_name: str, *args: Any, **kwargs: Any) -> str:
    return get_task(task_name).name_display


def get_available_tasks() -> List[str]:
    """Get available tasks

    Returns:
        List of tasks
    """
    task_dir = Path(__file__).parent.absolute()
    tasks = [f.name for f in task_dir.glob("*") if f.is_dir() and f.name[0] != "_"]
    tasks_extra = ["slcp_distractors", "bernoulli_glm_raw"]
    return tasks + tasks_extra
