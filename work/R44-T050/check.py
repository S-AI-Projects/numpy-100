from pathlib import Path
import json
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "work" / "R44-T050"
SOURCE = ROOT / "source" / "exercises100.ktx"

text = SOURCE.read_text()
start = text.index("< a73\n") + len("< a73\n")
end = text.index("\n< q74", start)

published = text[start:end]
python_lines = published.splitlines()

sort_line = "F = np.sort(F,axis=1)"
assert python_lines.count(sort_line) == 1, (
    "Expected exactly one endpoint-normalization operation"
)

inputs = [[0, 1, 2], [2, 1, 3]]
expected_occurrences = [
    [0, 1], [1, 2], [0, 2],
    [1, 2], [1, 3], [2, 3],
]
expected_unique = [
    [0, 1], [0, 2], [1, 2], [1, 3], [2, 3],
]

(EVIDENCE / "inputs.json").write_text(
    json.dumps(
        {
            "triangles": inputs,
            "expected_edges_after_normalization": expected_occurrences,
            "expected_unique_undirected_edges": expected_unique,
        },
        indent=2,
    ) + "\n"
)

prefix = """
import numpy as np

# Supply deterministic test triangles to the actual published example.
_test_faces = np.array([[0, 1, 2], [2, 1, 3]])
np.random.randint = lambda low, high, size: _test_faces.copy()
"""

suffix = """
from numpy.testing import assert_array_equal

expected_occurrences = np.array([
    [0, 1], [1, 2], [0, 2],
    [1, 2], [1, 3], [2, 3],
])
expected_unique = np.array([
    [0, 1], [0, 2], [1, 2], [1, 3], [2, 3],
])

actual_edges = np.column_stack((G["p0"], G["p1"]))

assert_array_equal(F, expected_occurrences)
assert_array_equal(actual_edges, expected_unique)
assert len(G) == 5
assert np.count_nonzero(np.all(F == [1, 2], axis=1)) == 2

print("Unique edges:", actual_edges.tolist())
print("Unique edge count:", len(G))
print("Shared edge occurrences after normalization:",
      np.count_nonzero(np.all(F == [1, 2], axis=1)))
print("Edge normalization: PASSED")
"""


def execute(lines):
    program = prefix + "\n".join(lines) + "\n" + suffix
    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


# Run the actual published example.
correct_run = execute(python_lines)

if correct_run.returncode != 0:
    raise RuntimeError(
        "Published example failed:\n"
        + correct_run.stdout
        + correct_run.stderr
    )

expected_tail = (
    "Unique edges: [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]\n"
    "Unique edge count: 5\n"
    "Shared edge occurrences after normalization: 2\n"
    "Edge normalization: PASSED\n"
)

assert correct_run.stdout.endswith(expected_tail), (
    "Published example output did not match expectations:\n"
    + correct_run.stdout
)

(EVIDENCE / "actual_results.txt").write_text(correct_run.stdout)
(EVIDENCE / "expected_vs_actual.txt").write_text(
    "Expected final output:\n"
    + expected_tail
    + "\nActual final output:\n"
    + correct_run.stdout.split("Unique edges:")[-1].join(
        ["Unique edges:", ""]
    )
)

print("Published example: PASSED")
print("Five unique undirected edges: PASSED")
print("Shared edge normalization: PASSED")
print("Expected output comparison: PASSED")


# Create a temporary one-operation mutant.
# It skips endpoint normalization but leaves the rest unchanged.
mutant_lines = [
    "F = F.copy()" if line == sort_line else line
    for line in python_lines
]

mutant_run = execute(mutant_lines)

(EVIDENCE / "deliberate_failure_test.txt").write_text(
    "Temporary mutant: replaced\n"
    + sort_line
    + "\nwith\nF = F.copy()\n\n"
    + "The mutant was expected to fail the five-edge assertion.\n\n"
    + "Return code: "
    + str(mutant_run.returncode)
    + "\n\nSTDOUT:\n"
    + mutant_run.stdout
    + "\nSTDERR:\n"
    + mutant_run.stderr
)

assert mutant_run.returncode != 0, (
    "Mutation test failed: the example passed without endpoint sorting"
)

print("Deliberate missing-sort mutant rejected: PASSED")


(EVIDENCE / "tool_versions.txt").write_text(
    f"Python: {sys.version}\n"
    f"NumPy: {np.__version__}\n"
    f"Platform: {platform.platform()}\n"
)

print("Evidence saved in work/R44-T050/")
