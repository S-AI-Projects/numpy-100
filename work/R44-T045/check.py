from pathlib import Path
import json
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "exercises100.ktx"
EVIDENCE = ROOT / "work" / "R44-T045"

text = SOURCE.read_text(encoding="utf-8")

# Extract the actual published solution for Exercise 59.
start_marker = "< a59\n"
end_marker = "< q60\n"

if text.count(start_marker) != 1 or text.count(end_marker) != 1:
    raise SystemExit("Could not uniquely locate Exercise 59 solution.")

solution_start = text.index(start_marker) + len(start_marker)
solution_end = text.index(end_marker, solution_start)
solution = text[solution_start:solution_end].strip()

if "argsort(kind=\"stable\")" not in solution:
    raise SystemExit("Exercise 59 does not use stable sorting.")

# Execute the extracted article code in a separate Python process.
child_script = f"""
import numpy as np

published_solution = {solution!r}
exec(compile(published_solution, "source/exercises100.ktx:a59", "exec"))

# Additional fixture verifies the documented tie ordering.
ties = np.array([[10, 5], [20, 3], [30, 5], [40, 3]])
sorted_ties = ties[ties[:, 1].argsort(kind="stable")]

expected_ties = np.array([[20, 3], [40, 3], [10, 5], [30, 5]])
np.testing.assert_array_equal(sorted_ties, expected_ties)
print("STABLE_TIE_CHECK_PASSED")
"""

result = subprocess.run(
    [sys.executable, "-c", child_script],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
)

expected = """[[2 9]
 [1 8]
 [3 7]]
[[3 7]
 [1 8]
 [2 9]]
STABLE_TIE_CHECK_PASSED
"""

actual = result.stdout

if actual != expected:
    raise AssertionError(
        f"Output mismatch.\nEXPECTED:\n{expected}\nACTUAL:\n{actual}"
    )

inputs = {
    "exercise": 59,
    "main_input": [[2, 9], [1, 8], [3, 7]],
    "sort_column_index": 1,
    "expected_output": [[3, 7], [1, 8], [2, 9]],
    "tie_input": [[10, 5], [20, 3], [30, 5], [40, 3]],
    "tie_ordering": "Stable: equal column values retain original row order.",
}

(EVIDENCE / "inputs.json").write_text(
    json.dumps(inputs, indent=2) + "\n", encoding="utf-8"
)

(EVIDENCE / "expected_vs_actual.txt").write_text(
    "EXPECTED OUTPUT:\n" + expected
    + "\nACTUAL OUTPUT:\n" + actual,
    encoding="utf-8",
)

(EVIDENCE / "actual_results.txt").write_text(actual, encoding="utf-8")

versions = (
    f"Python: {platform.python_version()}\n"
    f"NumPy: {np.__version__}\n"
    f"Platform: {platform.platform()}\n"
)

(EVIDENCE / "tool_versions.txt").write_text(versions, encoding="utf-8")

print("Exercise 59 published example: PASSED")
print("Stable tie-ordering test: PASSED")
print("Evidence saved in work/R44-T045/")
print(versions)
