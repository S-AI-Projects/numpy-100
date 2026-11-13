from pathlib import Path
import json
import platform
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "exercises100.ktx"
EVIDENCE = ROOT / "work" / "R44-T048"

EXPECTED_STDOUT = """Legacy unique rows:
[[1 2]
 [2 1]]
Modern unique rows:
[[1 2]
 [2 1]]
Legacy raw-byte order:
[[256]
 [  1]]
Modern numeric order:
[[  1]
 [256]]
"""

EXPECTED_INPUTS = {
    "required_input": [[1, 2], [1, 2], [2, 1]],
    "required_unique_row_count": 2,
    "boundary_input": [[1], [256]],
    "boundary_dtype": "<i8",
    "legacy_boundary_order": [[256], [1]],
    "modern_boundary_order": [[1], [256]],
}


def extract_answer():
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()

    try:
        start = lines.index("< a96") + 1
    except ValueError as exc:
        raise AssertionError("Could not find < a96 block") from exc

    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("< ")),
        len(lines),
    )

    answer = "\n".join(lines[start:end]).strip()
    if not answer:
        raise AssertionError("The a96 source block is empty")

    return answer


def run_example(answer):
    assertions = """
assert Z.tolist() == [[1, 2], [1, 2], [2, 1]]
assert uZ.tolist() == [[1, 2], [2, 1]]
assert len(uZ) == 2
assert Z_bytes.dtype.str == '<i8'
assert legacy_bytes.tolist() == [[256], [1]]
assert modern_numeric.tolist() == [[1], [256]]
"""

    program = "import numpy as np\n\n" + answer + "\n" + assertions

    return subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)

    answer = extract_answer()
    result = run_example(answer)

    (EVIDENCE / "actual_results.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )

    expected_file = EVIDENCE / "expected_vs_actual.txt"
    expected_file.write_text(
        "EXPECTED STDOUT:\n"
        + EXPECTED_STDOUT
        + "\nACTUAL STDOUT:\n"
        + result.stdout
        + "\nSUBPROCESS EXIT CODE:\n"
        + str(result.returncode)
        + "\n",
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise AssertionError(
            "Published example failed:\n" + result.stderr
        )

    if result.stdout != EXPECTED_STDOUT:
        raise AssertionError(
            "Published output differs from expected output"
        )

    (EVIDENCE / "inputs.json").write_text(
        json.dumps(EXPECTED_INPUTS, indent=2) + "\n",
        encoding="utf-8",
    )

    versions = (
        f"Python: {sys.version}\n"
        f"NumPy: {np.__version__}\n"
        f"Platform: {platform.platform()}\n"
    )
    (EVIDENCE / "tool_versions.txt").write_text(
        versions, encoding="utf-8"
    )

    # Deliberately introduce an incorrect modern result.
    wrong_answer = answer.replace(
        "uZ = np.unique(Z, axis=0)",
        "uZ = Z",
        1,
    )
    if wrong_answer == answer:
        raise AssertionError("Could not create deliberate mutation")

    wrong_result = run_example(wrong_answer)

    failure_detected = (
        wrong_result.returncode != 0
        or wrong_result.stdout != EXPECTED_STDOUT
    )

    (EVIDENCE / "deliberate_failure_test.txt").write_text(
        "Mutation: replaced the modern unique-row result with the "
        "original input Z.\n"
        f"Checker rejected mutation: {failure_detected}\n"
        f"Mutated subprocess exit code: {wrong_result.returncode}\n"
        "\nMutated stdout:\n"
        + wrong_result.stdout
        + "\nMutated stderr:\n"
        + wrong_result.stderr,
        encoding="utf-8",
    )

    if not failure_detected:
        raise AssertionError("Deliberate wrong result was not rejected")

    print("Exercise 96 published source execution: PASSED")
    print("Required two-unique-row assertion: PASSED")
    print("Legacy raw-byte ordering assertion: PASSED")
    print("Modern numeric ordering assertion: PASSED")
    print("Expected output comparison: PASSED")
    print("Deliberate wrong-result rejection: PASSED")
    print(f"Evidence saved in {EVIDENCE.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
