import logging
import shutil
from pathlib import Path
from typing import Any, Callable

import torch
from sbi.utils.plot import pairplot
from torch.utils.tensorboard import SummaryWriter


def tb_plot_posterior(
    writer: SummaryWriter, samples: torch.Tensor, tag: str = "posterior"
):
    if type(samples) == torch.Tensor:
        samples = samples.numpy()
    fig, _ = pairplot(samples.squeeze(), points=[])
    writer.add_figure(f"{tag}", fig, close=True)


def tb_make_writer(
    logger: logging.Logger = None,
    basepath: str = "tensorboard",
) -> (SummaryWriter, Callable):
    """Builds tensorboard summary writers"""
    log_dir = Path(f"{basepath}/summary")
    if log_dir.exists() and log_dir.is_dir():
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    if logger is not None:
        tbh = TensorboardHandler(writer)
        logger.addHandler(tbh)

    def close_fn():
        writer.flush()
        writer.close()
        logger.removeHandler(tbh)

    return writer, close_fn


class TensorboardHandler(logging.Handler):
    def __init__(
        self,
        writer: SummaryWriter,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.writer = writer

    def emit(self, record: logging.LogRecord):
        try:
            self.writer.add_text("logging", str(record.msg))
            self.writer.flush()
        except:
            print(f"Could not log {record} with msg: {record.msg}")
