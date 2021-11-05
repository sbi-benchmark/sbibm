from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch

from sbibm.utils.torch import get_default_device


def get_float_from_csv(
    path: Union[str, Path],
    dtype: type = np.float32,
):
    """Get a single float from a csv file"""
    with open(path, "r") as fh:
        return np.loadtxt(fh).astype(dtype)


def get_results(
    dataset: str = "main_paper.csv", subfolder: str = "benchmarking_sbi/results/"
) -> pd.DataFrame:
    """Get results from https://github.com/sbi-benchmark/results/

    Args:
        dataset: Filename for dataset
        subfolder: Subfolder in repo

    Returns:
        Dataframe
    """
    df = pd.read_csv(
        f"https://raw.githubusercontent.com/sbi-benchmark/results/main/{subfolder}{dataset}"
    )
    return df


def get_tensor_from_csv(
    path: Union[str, Path], dtype: type = np.float32, atleast_2d: bool = True
) -> torch.Tensor:
    """Get `torch.Tensor` from csv at given path"""
    device = get_default_device()

    if atleast_2d:
        return torch.from_numpy(np.atleast_2d(pd.read_csv(path)).astype(dtype)).to(
            device
        )
    else:
        return torch.from_numpy(pd.read_csv(path).astype(dtype)).to(device)


def get_ndarray_from_csv(
    path: Union[str, Path], dtype: type = np.float32, atleast_2d: bool = True
) -> np.ndarray:
    """Get `np.ndarray` from csv at given path"""
    if atleast_2d:
        return np.atleast_2d(pd.read_csv(path)).astype(dtype)
    else:
        return pd.read_csv(path).astype(dtype)


def save_float_to_csv(
    path: Union[str, Path],
    data: float,
    dtype: type = np.float32,
):
    """Save a single float to a csv file"""
    np.savetxt(
        path,
        np.asarray(data).reshape(-1).astype(np.float32),
        delimiter=",",
    )


def save_tensor_to_csv(
    path: Union[str, Path],
    data: torch.Tensor,
    columns: Optional[Iterable[str]] = None,
    dtype: type = np.float32,
    index: bool = False,
):
    """Save torch.Tensor to csv at given path"""
    pd.DataFrame(
        data.cpu().numpy().astype(dtype),
        columns=columns,
    ).to_csv(path, index=index)
