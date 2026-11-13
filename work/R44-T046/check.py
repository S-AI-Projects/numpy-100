from pathlib import Path
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/exercises100.ktx"
EVIDENCE = ROOT / "work/R44-T046"

text = SOURCE.read_text()

if "< a33\n" not in text or "\n< q34" not in text:
    raise SystemExit("ERROR: Exercise 33 solution block not found.")

solution = text.split("< a33\n", 1)[1].split("\n< q34", 1)[0]

expected = (
    "Yesterday: 2024-02-29\n"
    "Today: 2024-03-01\n"
    "Tomorrow: 2024-03-02\n"
)

child_code = (
    "import numpy as np\n"
    + solution
)

result = subprocess.run(
    [sys.executable, "-c", child_code],
    capture_output=True,
    text=True,
    check=True,
)

actual = result.stdout

if actual != expected:
    raise AssertionError(
        f"Output mismatch!\nExpected:\n{expected}\nActual:\n{actual}"
    )

(EVIDENCE / "inputs.json").write_text(json.dumps({
    "today": "2024-03-01",
    "one_day": "1 day",
    "expected_yesterday": "2024-02-29",
    "expected_today": "2024-03-01",
    "expected_tomorrow": "2024-03-02"
}, indent=2) + "\n")

(EVIDENCE / "expected_vs_actual.txt").write_text(
    "EXPECTED OUTPUT:\n" + expected +
    "\nACTUAL OUTPUT:\n" + actual
)

(EVIDENCE / "actual_results.txt").write_text(actual)

version_result = subprocess.run(
    [
        sys.executable,
        "-c",
        "import sys, numpy, platform; "
        "print('Python:', sys.version.split()[0]); "
        "print('NumPy:', numpy.__version__); "
        "print('Platform:', platform.platform())"
    ],
    capture_output=True,
    text=True,
    check=True,
)

(EVIDENCE / "tool_versions.txt").write_text(version_result.stdout)

print("Exercise 33 published example: PASSED")
print("Yesterday: 2024-02-29")
print("Today: 2024-03-01")
print("Tomorrow: 2024-03-02")
print("Evidence saved in work/R44-T046/")
print(version_result.stdout, end="")
