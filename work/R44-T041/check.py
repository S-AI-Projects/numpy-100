
import re
import io
import contextlib
from pathlib import Path

import numpy as np

# Find project root
ROOT = Path(__file__).resolve().parents[2]

# Read original source
source_file = ROOT / "source" / "exercises100.ktx"
source = source_file.read_text(encoding="utf-8")

# Extract actual Exercise 66 answer
pattern = r"(?ms)^< a66\s*\n(.*?)(?=^< q67\s*$)"
match = re.search(pattern, source)

if not match:
    raise RuntimeError("Exercise 66 answer not found")

code = match.group(1).strip()

# Execute actual published code
namespace = {"np": np}
output = io.StringIO()

with contextlib.redirect_stdout(output):
    exec(compile(code, str(source_file), "exec"), namespace)

# Check output
actual = output.getvalue().splitlines()
expected = ["2", "2"]

assert actual == expected, (
    f"Expected {expected}, received {actual}"
)

# Check image
I = namespace["I"]
colors = namespace["colors"]
I24 = namespace["I24"]

assert I.shape == (1, 3, 3)
assert I.dtype == np.uint8

# Check RGB channel order
np.testing.assert_array_equal(I[0, 0], [255, 0, 0])
np.testing.assert_array_equal(I[0, 1], [0, 0, 255])
np.testing.assert_array_equal(I[0, 2], [255, 0, 0])

# Check unique colours
expected_colors = np.array(
    [[0, 0, 255], [255, 0, 0]],
    dtype=np.uint8
)

np.testing.assert_array_equal(colors, expected_colors)

assert len(colors) == 2
assert len(np.unique(I24)) == 2

print("ALL CHECKS PASSED")
print("Expected:", expected)
print("Actual:", actual)
print("Image dtype:", I.dtype)
print("Image shape:", I.shape)
print("Unique colours:", len(colors))
print("NumPy version:", np.__version__)
