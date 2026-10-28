import re
import zlib

import numpy
import pytest

from complexeva import compression_complexity


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64, numpy.int16])
def test_preserves_raw_bytes_and_dtype(dtype):
    data = numpy.arange(100, dtype=dtype)[::2]
    raw = data.tobytes()
    assert compression_complexity(data) == len(zlib.compress(raw)) / len(raw)


def test_nonfinite_values_are_compressed_as_bytes():
    data = numpy.array([numpy.nan, numpy.inf, -numpy.inf])
    raw = data.tobytes()
    assert compression_complexity(data) == len(zlib.compress(raw)) / len(raw)


@pytest.mark.parametrize("data", [numpy.array(1.0), numpy.ones((2, 10))])
def test_invalid_shape_names_bad_shape(data):
    with pytest.raises(ValueError, match=re.escape(str(data.shape))):
        compression_complexity(data)


@pytest.mark.parametrize("data", [numpy.array([]), numpy.empty(3, dtype="V0")])
def test_empty_bytes_name_shape_and_dtype(data):
    with pytest.raises(ValueError) as error:
        compression_complexity(data)
    assert str(data.shape) in str(error.value)
    assert str(data.dtype) in str(error.value)
