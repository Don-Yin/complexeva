# overview
implementations of complexity measures in 1D sequences

# examples
```python
from complexeva.sequence import lyapunov_exponent, hfd, lempel_ziv, compression_complexity

signal = numpy.random.rand(1000)  # shape: (1000,)
complexity = hfd(signal)
print(complexity)
```

# install
```bash
pip install complexeva
```