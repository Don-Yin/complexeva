"""Device selection and time-last conversions shared by the tensor estimators."""

import math

import numpy
import torch
from numpy.typing import ArrayLike, NDArray

from .._validation import as_signals

Device = str | torch.device | None
Signal = ArrayLike | torch.Tensor


def select_device(x: Signal, device: Device) -> torch.device:
    """Use the requested device, the tensor's device, or CUDA, MPS, then CPU."""
    if device is not None:
        return torch.device(device)
    if isinstance(x, torch.Tensor):
        return x.device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _real_tensor(x: torch.Tensor) -> torch.Tensor:
    """Require a dense, real tensor with a time axis, detached from gradients."""
    if x.ndim == 0:
        raise ValueError(f"x must have a time axis, got shape {tuple(x.shape)}")
    if x.layout != torch.strided or x.is_complex() or x.is_quantized:
        raise ValueError(f"x must be a dense real numeric tensor, got {x.dtype} with layout {x.layout}")
    return x.detach()


def tensor_rows(x: Signal, device: Device) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Convert to contiguous (rows, time), with float32 on MPS and float64 elsewhere."""
    target = select_device(x, device)
    mps = target.type == "mps"
    dtype = torch.float32 if mps else torch.float64
    if isinstance(x, torch.Tensor):
        values = _real_tensor(x)
        # Cast on the side of the transfer that supports float64.
        if mps:
            values = values.to(dtype=dtype).to(target)
        elif values.device.type == "mps":
            values = values.to(target).to(dtype=dtype)
        else:
            values = values.to(device=target, dtype=dtype)
    else:
        array = as_signals(x)
        with numpy.errstate(over="ignore"):
            array = numpy.require(array, dtype=numpy.float32 if mps else numpy.float64, requirements=["C", "W"])
        values = torch.from_numpy(array).to(target)
    if mps or isinstance(x, torch.Tensor):
        finite = values.isfinite()
        if not bool(finite.all()):
            index = tuple((~finite).nonzero()[0].tolist())
            raise ValueError(f"x must be finite in {dtype} on {target}, got {values[index].item()} at index {index}")
    shape = tuple(values.shape[:-1])
    return values.reshape(math.prod(shape), values.shape[-1]).contiguous(), shape


def scalar_or_batch(values: NDArray, shape: tuple[int, ...]) -> float | int | NDArray:
    """Return a Python scalar for a single signal, otherwise reshape the batch."""
    return values.reshape(shape) if shape else values.item()
