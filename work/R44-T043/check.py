
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source" / "exercises100.ktx"
WORK = Path(__file__).resolve().parent

WORK.mkdir(parents=True, exist_ok=True)


def extract_block(source_text, block_name):
    lines = source_text.splitlines()
    start = f"< {block_name}"

    for index, line in enumerate(lines):
        if line.strip() == start:
            content = []
            for next_line in lines[index + 1:]:
                if next_line.startswith("< "):
                    break
                content.append(next_line)

            return "\n".join(content).strip()

    raise ValueError(f"Block {block_name} not found")


def run_code(code, filename):
    script = WORK / filename
    script.write_text(code + "\n", encoding="utf-8")

    return subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True
    )


def main():
    source_text = SOURCE.read_text(encoding="utf-8")

    question = extract_block(source_text, "q41")
    hint = extract_block(source_text, "h41")
    answer = extract_block(source_text, "a41")

    assert "How to sum a small array faster than np.sum?" in question
    assert "np.add.reduce" in hint
    assert "np.sum" in hint
    assert "np.add.reduce" in answer
    assert "np.sum" in answer

    # Save the actual published example extracted from source.
    extracted_file = WORK / "extracted_a41.py"
    extracted_file.write_text(answer + "\n", encoding="utf-8")

    # Save test inputs and expected results.
    evidence = {
        "original_fixture": {
            "input": "np.arange(10)",
            "expected": 45
        },
        "integer_fixture": {
            "input": [1, 2, 3],
            "expected": 6
        },
        "float_fixture": {
            "input": [0.5, 1.25, 2.75, 4.0],
            "expected": 8.5,
            "tolerance": {
                "rtol": 1e-12,
                "atol": 1e-12
            }
        }
    }

    (WORK / "inputs_expected.json").write_text(
        json.dumps(evidence, indent=4) + "\n",
        encoding="utf-8"
    )

    # Record Python and NumPy versions.
    version_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, numpy; "
            "print('Python:', sys.version); "
            "print('NumPy:', numpy.__version__)"
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True
    )

    (WORK / "tool_versions.txt").write_text(
        version_result.stdout,
        encoding="utf-8"
    )

    # Execute the actual answer block.
    result = run_code(answer, "extracted_a41.py")

    actual_text = (
    result.stdout
    + ("\n" + result.stderr if result.stderr else "")
    ).rstrip() + "\n"

    (WORK / "actual_results.txt").write_text(
        actual_text,
        encoding="utf-8"
    )

    # Propagate failures from the actual published example.
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr
        )

    expected_lines = [
        "Original arange(10) results: 45 45",
        "Integer results: 6 6",
        "Float results: 8.5 8.5"
    ]

    for expected in expected_lines:
        assert expected in result.stdout, (
            f"Missing expected output: {expected}"
        )

    assert "np.add.reduce:" in result.stdout
    assert "np.sum:" in result.stdout

    # Deliberately introduce a wrong expected result.
    wrong_code = answer.replace(
        "np.testing.assert_equal(original_reduce, 45)",
        "np.testing.assert_equal(original_reduce, 46)",
        1
    )

    assert wrong_code != answer, "Failure test mutation was not applied"

    failure_result = run_code(
        wrong_code,
        "deliberate_failure.py"
    )

    failure_text = (
        f"Exit code: {failure_result.returncode}\n"
        + failure_result.stdout
        + failure_result.stderr
    )

    failure_text = "\n".join(
        line.rstrip() for line in failure_text.splitlines()
    ).rstrip() + "\n"

    (WORK / "deliberate_failure.txt").write_text(
        failure_text,
        encoding="utf-8"
    )

    assert failure_result.returncode != 0, (
        "Checker failed to detect the deliberate wrong result"
    )

    print("PASS: Actual a41 source extracted and executed")
    print("PASS: Original, integer and float results verified")
    print("PASS: Timing measurements produced")
    print("PASS: Deliberately incorrect result was rejected")
    print("\nActual exercise output:")
    print(result.stdout)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
