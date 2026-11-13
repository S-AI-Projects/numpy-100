from pathlib import Path
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/exercises100.ktx"
EVIDENCE = ROOT / "work/R44-T049"
EVIDENCE.mkdir(parents=True, exist_ok=True)

text = SOURCE.read_text()
start = text.index("< a5\n") + len("< a5\n")
end = text.index("\n< q6", start)
published = text[start:end]

lines = published.splitlines()
help_commands = [line[1:].strip() for line in lines if line.startswith("!")]
python_lines = [line for line in lines if not line.startswith("!")]

assert len(help_commands) == 1, "Expected exactly one published help command"
assert "numpy.info(numpy.add)" in help_commands[0], "Help command changed"

# Execute the actual published help command using this checker's Python.
command = shlex.split(help_commands[0])
command[0] = sys.executable
help_run = subprocess.run(
    command, cwd=ROOT, capture_output=True, text=True
)
if help_run.returncode != 0:
    raise RuntimeError(
        f"Published help command failed:\n{help_run.stderr}"
    )

help_output = help_run.stdout + help_run.stderr
assert help_output.strip(), "np.info did not produce help output"
assert "add" in help_output.lower(), "Help output does not identify add"

(EVIDENCE / "help_output.txt").write_text(help_output)

# Execute the Python portion extracted from the actual published a5 block.
program = "\n".join(python_lines) + """

import numpy as np
from numpy.testing import assert_array_equal

# Validate the actual variables created by the published example.
assert np.add.__name__ == "add"
assert isinstance(np.add, np.ufunc)
assert_array_equal(np.add([1, 2], [3, 4]), [4, 6])
assert_array_equal(result, [4, 6])
assert result is not a and result is not b
assert_array_equal(out, [4, 6])
assert returned is out

print("Ufunc name:", np.add.__name__)
print("Ufunc verification: PASSED")
print("Element-wise addition: PASSED")
print("Separate result allocation: PASSED")
print("out= target reuse: PASSED")
"""

run = subprocess.run(
    [sys.executable, "-c", program],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
if run.returncode != 0:
    raise RuntimeError(
        f"Published Python example failed:\n{run.stdout}\n{run.stderr}"
    )

expected = """Result: [4 6]
Separate result: True
Output array: [4 6]
Returned same output array: True
Ufunc name: add
Ufunc verification: PASSED
Element-wise addition: PASSED
Separate result allocation: PASSED
out= target reuse: PASSED
"""

assert run.stdout == expected, (
    f"Output mismatch\nExpected:\n{expected}\nActual:\n{run.stdout}"
)

# Deliberately mutate the published example's result.
mutation = "result = np.add(a, b)"
assert program.count(mutation) == 1, "Mutation target must occur exactly once"

bad_program = program.replace(
    mutation,
    "result = np.array([99, 99])",
    1,
)
assert bad_program != program, "Deliberate mutation was not applied"

bad_run = subprocess.run(
    [sys.executable, "-c", bad_program],
    cwd=ROOT,
    capture_output=True,
    text=True,
)

# A wrong result must fail an assertion or differ from expected output.
bad_rejected = (
    bad_run.returncode != 0
    or bad_run.stdout != expected
)
assert bad_rejected, "Checker did not reject the deliberate wrong result"

(EVIDENCE / "actual_results.txt").write_text(run.stdout)
(EVIDENCE / "expected_vs_actual.txt").write_text(
    "EXPECTED:\n" + expected + "\nACTUAL:\n" + run.stdout
)
(EVIDENCE / "deliberate_failure_test.txt").write_text(
    "Deliberate wrong-result mutation rejected: YES\n"
    f"Exit code: {bad_run.returncode}\n"
    + bad_run.stderr
)
(EVIDENCE / "inputs.json").write_text(json.dumps({
    "a": [1, 2],
    "b": [3, 4],
    "expected_sum": [4, 6],
    "out_argument": "preallocated array with same shape and dtype as a",
}, indent=2) + "\n")

version_run = subprocess.run(
    [sys.executable, "-c",
     "import sys, numpy; print('Python:', sys.version); "
     "print('NumPy:', numpy.__version__)"],
    cwd=ROOT, capture_output=True, text=True, check=True,
)
(EVIDENCE / "tool_versions.txt").write_text(version_run.stdout)

print("Published np.info help command: PASSED")
print("Ufunc name and type: PASSED")
print("Element-wise addition result: PASSED")
print("Separate result allocation: PASSED")
print("out= target reuse: PASSED")
print("Expected output comparison: PASSED")
print("Deliberate wrong-result rejection: PASSED")
print("Evidence saved in work/R44-T049/")
