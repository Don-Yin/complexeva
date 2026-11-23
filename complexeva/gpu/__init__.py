"""Optional PyTorch backend for Higuchi dimension and the spectrum shuffle test."""

try:
    from ._hfd import hfd
    from ._shuffle import SpectrumShuffleResult, spectrum_shuffle_test
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ImportError("complexeva.gpu requires PyTorch; install it with `pip install 'complexeva[gpu]'`") from error

__all__ = ["SpectrumShuffleResult", "hfd", "spectrum_shuffle_test"]
