# Author: NumPy exercise contribution

import warnings
import numpy as np


def run_with_warnings(operation):
    """Run an operation and capture its warning categories."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.errstate(all="warn"):
            result = operation()

    categories = [warning.category.__name__ for warning in caught]
    return result, categories


def assert_runtime_warning(categories):
    assert categories == ["RuntimeWarning"], (
        f"Expected one RuntimeWarning, got {categories}"
    )


# Original example 1: true division of integer zero by zero.
true_division, true_warnings = run_with_warnings(
    lambda: np.array(0) / np.array(0)
)

assert np.isnan(true_division)
assert not np.isinf(true_division)
assert_runtime_warning(true_warnings)


# Original example 2: integer floor division of zero by zero.
floor_division, floor_warnings = run_with_warnings(
    lambda: np.array(0) // np.array(0)
)

np.testing.assert_equal(floor_division, 0)
assert not np.isnan(floor_division)
assert not np.isinf(floor_division)
assert_runtime_warning(floor_warnings)


# Original example 3: inspect the intermediate integer value.
def cast_nan_fixture():
    integer_intermediate = np.array([np.nan]).astype(int)
    float_result = integer_intermediate.astype(float)
    return integer_intermediate, float_result


(cast_integer, cast_float), cast_warnings = run_with_warnings(
    cast_nan_fixture
)

assert np.issubdtype(cast_integer.dtype, np.integer)
assert np.issubdtype(cast_float.dtype, np.floating)
assert not np.any(np.isnan(cast_float))
assert not np.any(np.isinf(cast_float))
assert_runtime_warning(cast_warnings)


print("Original true division:", true_division)
print("True division warnings:", true_warnings)
print("Original floor division:", floor_division)
print("Floor division warnings:", floor_warnings)
print("NaN integer intermediate:", cast_integer)
print("NaN conversion final float:", cast_float)
print("NaN conversion warnings:", cast_warnings)


# Five bounded variations.

# Variation 1: floating-point zero divided by negative zero.
v1, w1 = run_with_warnings(
    lambda: np.array(0.0) / np.array(-0.0)
)
assert np.isnan(v1)
assert not np.isinf(v1)
assert_runtime_warning(w1)

# Variation 2: true division with two zero-valued array elements.
v2, w2 = run_with_warnings(
    lambda: np.array([0, 0]) / np.array([0, 0])
)
assert np.all(np.isnan(v2))
assert not np.any(np.isinf(v2))
assert_runtime_warning(w2)

# Variation 3: int32 floor division.
v3, w3 = run_with_warnings(
    lambda: np.array(0, dtype=np.int32) // np.array(0, dtype=np.int32)
)
np.testing.assert_equal(v3, 0)
assert_runtime_warning(w3)

# Variation 4: floor division with two zero-valued array elements.
v4, w4 = run_with_warnings(
    lambda: np.array([0, 0]) // np.array([0, 0])
)
np.testing.assert_array_equal(v4, [0, 0])
assert_runtime_warning(w4)

# Variation 5: convert two NaN values through an integer array.
def cast_two_nans():
    integer_values = np.array([np.nan, np.nan]).astype(np.int64)
    return integer_values, integer_values.astype(float)


(v5_integer, v5_float), w5 = run_with_warnings(cast_two_nans)

assert np.issubdtype(v5_integer.dtype, np.integer)
assert np.issubdtype(v5_float.dtype, np.floating)
assert not np.any(np.isnan(v5_float))
assert not np.any(np.isinf(v5_float))
assert_runtime_warning(w5)


print("\nFive variations:")
for name, result, categories in [
    ("Variation 1", v1, w1),
    ("Variation 2", v2, w2),
    ("Variation 3", v3, w3),
    ("Variation 4", v4, w4),
    ("Variation 5 integer", v5_integer, w5),
    ("Variation 5 float", v5_float, w5),
]:
    print(f"{name}: {result}; warnings: {categories}")

print("\nAll original and variation assertions passed.")
