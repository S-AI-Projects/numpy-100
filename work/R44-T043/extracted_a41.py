# Author: Evgeni Burovski
# Comparison: np.add.reduce versus np.sum

import timeit
import numpy as np
# Compare both methods on the original exercise fixture.
Z = np.arange(10)

original_reduce = np.add.reduce(Z)
original_sum = np.sum(Z)

np.testing.assert_equal(original_reduce, 45)
np.testing.assert_equal(original_sum, 45)

print("Original arange(10) results:", original_reduce, original_sum)

# Compare both methods on the same integer fixture.
integer_fixture = np.array([1, 2, 3])

integer_reduce = np.add.reduce(integer_fixture)
integer_sum = np.sum(integer_fixture)

np.testing.assert_equal(integer_reduce, 6)
np.testing.assert_equal(integer_sum, 6)

print("Integer results:", integer_reduce, integer_sum)

# Compare both methods on a fixed floating-point fixture.
float_fixture = np.array([0.5, 1.25, 2.75, 4.0])

float_reduce = np.add.reduce(float_fixture)
float_sum = np.sum(float_fixture)

np.testing.assert_allclose(float_reduce, 8.5, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(float_sum, 8.5, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(float_reduce, float_sum, rtol=1e-12, atol=1e-12)

print("Float results:", float_reduce, float_sum)
print("Reduction result type:", type(float_reduce).__name__)
print("Sum result type:", type(float_sum).__name__)

# Benchmark only after verifying that both methods agree.
number = 10000
repeat = 5

reduce_times = timeit.repeat(
    lambda: np.add.reduce(float_fixture),
    number=number,
    repeat=repeat
)

sum_times = timeit.repeat(
    lambda: np.sum(float_fixture),
    number=number,
    repeat=repeat
)

reduce_ns = min(reduce_times) * 1e9 / number
sum_ns = min(sum_times) * 1e9 / number

print(f"np.add.reduce: {reduce_ns:.2f} ns/call (best of {repeat})")
print(f"np.sum:        {sum_ns:.2f} ns/call (best of {repeat})")
print("Timings are measurements for this environment, not a universal ranking.")
