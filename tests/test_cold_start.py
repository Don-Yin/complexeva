import os
import subprocess
import sys

import numpy
import pytest
from scipy.signal import hilbert

from complexeva._hilbert import hilbert_envelope

# Every public measure, with the permutation order 4 closure and the tree-counting sample entropy path.
_ALL_MEASURES = """
import numpy
import complexeva as c
x = numpy.random.default_rng(0).standard_normal(5000)
print([c.hfd(x), c.hfd_hilbert(x), c.lempel_ziv(x), c.lempel_ziv_hilbert(x), c.sample_entropy(x), c.sample_entropy(x[:500]),
       c.permutation_entropy(x), c.permutation_entropy(x, order=4), c.dfa(x), c.lyapunov_exponent(x[:1000], min_tsep=10),
       c.compression_complexity(x)])
"""


def test_second_process_reads_the_numba_cache(tmp_path):
    # numba cannot reload a cached function that calls itself, and the second process then crashes.
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}
    command = [sys.executable, "-c", _ALL_MEASURES]
    runs = [subprocess.run(command, env=env, capture_output=True, text=True, check=False) for _ in range(2)]
    assert [run.returncode for run in runs] == [0, 0], [run.stderr[-500:] for run in runs]
    assert runs[0].stdout == runs[1].stdout
    assert any(tmp_path.rglob("*.nbc"))


@pytest.mark.parametrize("n", [*range(1, 70), 255, 256, 257, 1000, 1001, 4096, 4099])
def test_hilbert_envelope_is_scipy_bit_for_bit(n):
    x = numpy.random.default_rng(n).standard_normal(n)
    assert numpy.array_equal(hilbert_envelope(x), numpy.abs(hilbert(x)))
