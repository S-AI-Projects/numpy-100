from pathlib import Path
import json
import platform
import re
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / "source" / "exercises100.ktx"


def get_block(source_text, block_name):
    pattern = rf"(?ms)^< {block_name}\n(.*?)(?=^< [a-z]\d+\s*$|\Z)"
    match = re.search(pattern, source_text)
    if not match:
        raise AssertionError(f"Could not find source block: {block_name}")
    return match.group(1).strip()


def run_python_file(path):
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


source_text = SOURCE.read_text(encoding="utf-8")

# Confirm that the original exercise question has not changed.
q28 = get_block(source_text, "q28")
expected_q28 = """What are the result of the following expressions? (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```"""
assert q28 == expected_q28, "The original q28 question was changed."

# Check that the hint explains the important NumPy behaviour.
h28 = get_block(source_text, "h28")
for required_text in ("np.isnan", "np.isinf", "np.errstate", "RuntimeWarning"):
    assert required_text in h28, f"Missing hint detail: {required_text}"

# Extract the actual Python answer from the source file.
a28 = get_block(source_text, "a28")

# The repository stores answer blocks as raw Python code.
# Also support fenced Python blocks if they are used later.
code_match = re.search(r"```python\s*\n(.*?)\n```", a28, re.DOTALL)

if code_match:
    answer_code = code_match.group(1)
else:
    answer_code = a28.strip()

assert answer_code, "The a28 answer block is empty."
extracted_file = WORK / "extracted_a28.py"
extracted_file.write_text(answer_code + "\n", encoding="utf-8")

# Run the actual answer code.
actual = run_python_file(extracted_file)
(WORK / "actual_results.txt").write_text(
    (actual.stdout + actual.stderr).rstrip() + "\n",
    encoding="utf-8",
)

if actual.returncode != 0:
    raise AssertionError(
        "The extracted a28 answer failed.\n"
        f"stdout:\n{actual.stdout}\nstderr:\n{actual.stderr}"
    )

for expected_output in (
    "Original true division:",
    "Original floor division:",
    "NaN integer intermediate:",
    "Five variations:",
    "All original and variation assertions passed.",
):
    assert expected_output in actual.stdout, (
        f"Expected output was missing: {expected_output}"
    )

# Record the tested expressions and portable expected properties.
inputs_and_expected = {
    "original_examples": [
        {
            "expression": "np.array(0) / np.array(0)",
            "expected": "NaN; one RuntimeWarning",
        },
        {
            "expression": "np.array(0) // np.array(0)",
            "expected": "zero in the tested environment; one RuntimeWarning",
        },
        {
            "expression": "np.array([np.nan]).astype(int).astype(float)",
            "expected": (
                "integer conversion warns; the resulting float does not "
                "preserve NaN; exact integer value is not assumed portable"
            ),
        },
    ],
    "five_variations": [
        "0.0 divided by -0.0 produces NaN",
        "An array of zero values divided by zero produces NaN values",
        "int32 zero floor-divided by int32 zero produces zero in this environment",
        "An array of integer zeros floor-divided by zero produces zeros in this environment",
        "Two NaN values converted through an integer array do not remain NaN",
    ],
    "warning_category": "RuntimeWarning",
    "numpy_version": np.__version__,
    "python_version": sys.version,
    "platform": platform.platform(),
}
(WORK / "inputs_expected.json").write_text(
    json.dumps(inputs_and_expected, indent=2) + "\n",
    encoding="utf-8",
)

# Deliberately reverse one real assertion to confirm that the checker
# can detect an incorrect expected result.
correct_assertion = "assert np.isnan(true_division)"
wrong_assertion = "assert not np.isnan(true_division)"

assert answer_code.count(correct_assertion) == 1, (
    "Could not uniquely locate the assertion for the deliberate-failure test."
)

wrong_code = answer_code.replace(correct_assertion, wrong_assertion, 1)
wrong_file = WORK / "deliberate_failure.py"
wrong_file.write_text(wrong_code + "\n", encoding="utf-8")

deliberate = run_python_file(wrong_file)
failure_text = (deliberate.stdout + deliberate.stderr).rstrip()
(WORK / "deliberate_failure.txt").write_text(
    failure_text + "\n",
    encoding="utf-8",
)

assert deliberate.returncode != 0, (
    "The deliberately incorrect assertion unexpectedly passed."
)
assert "AssertionError" in failure_text, (
    "The deliberate failure did not produce an AssertionError."
)

print("q28 question unchanged.")
print("h28 hint checks passed.")
print("Extracted a28 code executed successfully.")
print("Original examples and five variations passed.")
print("Deliberately incorrect assertion was detected.")
print(f"NumPy version: {np.__version__}")
print("ALL CHECKS PASSED")
