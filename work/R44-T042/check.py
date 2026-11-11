from pathlib import Path
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "exercises100.ktx"
EVIDENCE = Path(__file__).resolve().parent

# Extract the actual a15 answer from the authoritative source.
lines = SOURCE.read_text(encoding="utf-8").splitlines()
start = lines.index("< a15") + 1
end = next(
    (i for i in range(start, len(lines)) if lines[i].startswith("< ")),
    len(lines),
)
answer_code = "\n".join(lines[start:end]).strip()

if not answer_code:
    raise AssertionError("Exercise a15 code is empty")

(EVIDENCE / "actual_source_snippet.py").write_text(
    answer_code + "\n", encoding="utf-8"
)

inputs = {
    "4x5": {"shape": [4, 5]},
    "1x3": {"shape": [1, 3]},
}
(EVIDENCE / "inputs.json").write_text(
    json.dumps(inputs, indent=2) + "\n", encoding="utf-8"
)

# Execute the extracted published example and assert its actual variables.
runner = f"""
import numpy as np

code = {answer_code!r}
namespace = {{"np": np}}
exec(compile(code, "source/exercises100.ktx:a15", "exec"), namespace)

expected_4x5 = np.ones((4, 5))
expected_4x5[1:-1, 1:-1] = 0

expected_1x3 = np.ones((1, 3))

np.testing.assert_array_equal(namespace["Z_4x5"], expected_4x5)
np.testing.assert_array_equal(namespace["Z_1x3"], expected_1x3)

assert np.count_nonzero(namespace["Z_4x5"] == 1) == 14
assert np.count_nonzero(namespace["Z_4x5"] == 0) == 6

print("4x5 result:")
print(namespace["Z_4x5"])
print("1x3 result:")
print(namespace["Z_1x3"])
print("ALL ASSERTIONS PASSED")
"""

try:
    result = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
except subprocess.CalledProcessError as error:
    (EVIDENCE / "actual_results.txt").write_text(
        f"STDOUT:\n{error.stdout}\nSTDERR:\n{error.stderr}",
        encoding="utf-8",
    )
    raise

(EVIDENCE / "actual_results.txt").write_text(
    result.stdout, encoding="utf-8"
)

# Confirm that the assertion mechanism rejects a deliberately wrong result.
try:
    np.testing.assert_array_equal(
        np.array([[1, 0, 1]]),
        np.ones((1, 3)),
    )
except AssertionError:
    wrong_result_rejected = True
else:
    raise AssertionError("Checker accepted a deliberately wrong result")

versions = (
    f"Python: {sys.version}\n"
    f"NumPy: {np.__version__}\n"
)
(EVIDENCE / "tool_versions.txt").write_text(
    versions, encoding="utf-8"
)

(EVIDENCE / "expected_vs_actual.txt").write_text(
    "Expected 4x5: 14 border ones and 6 interior zeros.\n"
    "Expected 1x3: three ones; no interior exists.\n\n"
    + result.stdout,
    encoding="utf-8",
)

print(result.stdout)
print("Deliberately wrong result: correctly rejected")
print("ALL CHECKS PASSED")
