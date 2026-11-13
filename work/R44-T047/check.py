from pathlib import Path
import json
import platform
import re
import subprocess
import sys

import numpy as np

SOURCE = Path("source/exercises100.ktx")
EVIDENCE = Path("work/R44-T047")

source_text = SOURCE.read_text()
match = re.search(r"< a58\n(.*?)\n(?=< q59)", source_text, re.S)

if not match:
    raise SystemExit("Could not extract the published Exercise 58 solution.")

solution = match.group(1)

expected_output = (
    "Input:\n"
    " [[1. 3.]\n"
    " [2. 6.]]\n"
    "Row means:\n"
    " [[2.]\n"
    " [4.]]\n"
    "Centered matrix:\n"
    " [[-1.  1.]\n"
    " [-2.  2.]]\n"
    "Row means after centering: [0. 0.]\n"
)

assertion_code = """
np.testing.assert_allclose(
    X, [[1., 3.], [2., 6.]], rtol=0, atol=1e-12
)
np.testing.assert_allclose(
    row_means, [[2.], [4.]], rtol=0, atol=1e-12
)
np.testing.assert_allclose(
    Y, [[-1., 1.], [-2., 2.]], rtol=0, atol=1e-12
)
np.testing.assert_allclose(
    Y_older, Y, rtol=0, atol=1e-12
)
np.testing.assert_allclose(
    Y.mean(axis=1), [0., 0.], rtol=0, atol=1e-12
)
"""

script = "import numpy as np\n" + solution + "\n" + assertion_code

result = subprocess.run(
    [sys.executable, "-c", script],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
    raise SystemExit("Exercise 58 checker FAILED.")

actual_output = result.stdout

if actual_output != expected_output:
    print("Expected output:\n", expected_output)
    print("Actual output:\n", actual_output)
    raise SystemExit("Printed output does not match the expected result.")

EVIDENCE.mkdir(parents=True, exist_ok=True)

(EVIDENCE / "inputs.json").write_text(
    json.dumps(
        {
            "exercise": 58,
            "input_matrix": [[1.0, 3.0], [2.0, 6.0]],
            "expected_row_means": [[2.0], [4.0]],
            "expected_centered_matrix": [[-1.0, 1.0], [-2.0, 2.0]],
            "rtol": 0,
            "atol": 1e-12,
        },
        indent=2,
    )
    + "\n"
)

(EVIDENCE / "expected_vs_actual.txt").write_text(
    "EXPECTED OUTPUT:\n"
    + expected_output
    + "\nACTUAL OUTPUT:\n"
    + actual_output
)

(EVIDENCE / "actual_results.txt").write_text(
    "Exercise 58 published solution: PASSED\n"
    "Modern keepdims method: PASSED\n"
    "Older reshape method: PASSED\n"
    "Both methods produce the expected centered matrix: PASSED\n"
    "Every resulting row has mean zero: PASSED\n"
    "Printed output matches expected output: PASSED\n"
)

(EVIDENCE / "tool_versions.txt").write_text(
    f"Python: {platform.python_version()}\n"
    f"NumPy: {np.__version__}\n"
    f"Platform: {platform.platform()}\n"
)

print("Exercise 58 published solution: PASSED")
print("Modern keepdims method: PASSED")
print("Older reshape method: PASSED")
print("Expected output comparison: PASSED")
print("Evidence saved in work/R44-T047/")
