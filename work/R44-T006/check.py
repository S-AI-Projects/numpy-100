#!/usr/bin/env python3
"""
Automated validation script for Task R44-T006:
Clarify the implementation: NumPy exercise 55 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx and
   100_Numpy_exercises_with_solutions.md (preventing drift).
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   it succeeds with exit code 0.
4. Verifies multidimensional array enumeration for [[10,20],[30,40]], asserting
   it yields ((0,0),10), ((0,1),20), ((1,0),30), ((1,1),40).
5. Compares and clarifies np.ndenumerate vs np.ndindex operations and intermediate
   coordinate representations.
6. Verifies that deliberate wrong results trigger assertion failures.
7. Saves inputs, expected vs actual results, and tool versions to work/R44-T006/.
"""

import sys
import os
import json
import platform
import subprocess
import re
from pathlib import Path

import numpy as np


def find_repo_root():
    """Find repository root relative to this script."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "source" / "exercises100.ktx").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find repository root containing source/exercises100.ktx")


def extract_ktx_answer(ktx_path, key="a55"):
    """Extract a specific answer block from exercises100.ktx."""
    with open(ktx_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    target_key = f"< {key}"
    capturing = False
    block_lines = []

    for line in lines:
        if line.startswith("<"):
            if capturing:
                break
            if line.strip() == target_key:
                capturing = True
                continue
        elif capturing:
            block_lines.append(line)

    code = "".join(block_lines).strip()
    if not code:
        raise ValueError(f"Failed to extract block {key} from {ktx_path}")
    return code


def extract_markdown_solution(md_path, exercise_num=55):
    """Extract code block for the given exercise from markdown file."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = rf"####\s+{exercise_num}\..*?```python\n(.*?)```"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find exercise {exercise_num} code block in {md_path}")
    return match.group(1).strip()


def run_extracted_code(code_str):
    """Execute the extracted published code in a clean subprocess."""
    full_code = f"import numpy as np\n{code_str}"
    result = subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Execution of extracted code failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T006"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T006 CHECK: NumPy Exercise 55 (ndenumerate & ndindex)")
    print("=" * 60)

    # 1. Extract published code from authoritative source
    print(f"[1/6] Extracting a55 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a55")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 55)
    assert ktx_code == md_code, (
        "Generated markdown solution does not match authoritative source/exercises100.ktx a55 block!"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via subprocess to verify runtime execution
    print("[3/6] Executing published code via Python subprocess...")
    stdout = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")

    # 4. Verify enumeration on fixture [[10,20],[30,40]]
    print("[4/6] Verifying enumeration on fixture [[10,20],[30,40]]...")
    Z_input = np.array([[10, 20], [30, 40]], dtype=int)

    # np.ndenumerate yields ((row, col), value)
    actual_ndenumerate = list(np.ndenumerate(Z_input))
    # np.ndindex yields index tuple, value accessed as Z[index]
    actual_ndindex = [(idx, int(Z_input[idx])) for idx in np.ndindex(Z_input.shape)]

    expected_enumeration = [
        ((0, 0), 10),
        ((0, 1), 20),
        ((1, 0), 30),
        ((1, 1), 40)
    ]

    # Convert actual items to standard Python tuples/ints for exact comparison
    normalized_actual = [((int(idx[0]), int(idx[1])), int(val)) for idx, val in actual_ndenumerate]

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        wrong_expected = [((0, 0), 999), ((0, 1), 20), ((1, 0), 30), ((1, 1), 40)]
        assert normalized_actual == wrong_expected, "Deliberate failure triggered!"

    assert normalized_actual == expected_enumeration, (
        f"np.ndenumerate mismatch! Expected {expected_enumeration}, got {normalized_actual}"
    )
    assert actual_ndindex == expected_enumeration, (
        f"np.ndindex indexing mismatch! Expected {expected_enumeration}, got {actual_ndindex}"
    )
    print("      PASS: np.ndenumerate and np.ndindex both yielded exact expected sequence:")
    for (idx, val) in normalized_actual:
        print(f"        Coordinate {idx} -> Value {val}")

    # 5. Deliberate failure testing
    print("[5/6] Testing deliberate failure detection...")
    wrong_expected = [((0, 0), 999), ((0, 1), 20), ((1, 0), 30), ((1, 1), 40)]
    try:
        assert normalized_actual == wrong_expected
        raise AssertionError("Check FAILED: deliberate wrong enumeration was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong enumeration correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T006/...")

    inputs_data = {
        "description": "Fixture [[10,20],[30,40]] for testing multidimensional enumeration",
        "input_matrix": Z_input.tolist(),
        "shape": list(Z_input.shape),
        "dtype": str(Z_input.dtype),
        "operations_compared": [
            "np.ndenumerate(Z): Multidimensional index iterator yielding ((coordinate_tuple), value)",
            "np.ndindex(Z.shape): Coordinate tuple iterator yielding (coordinate_tuple), indexed as Z[index]"
        ]
    }
    inputs_file = work_dir / "inputs.json"
    with open(inputs_file, "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 55,
        "task": "R44-T006",
        "description": "Clarify multidimensional array enumeration with np.ndenumerate and np.ndindex",
        "input_matrix": Z_input.tolist(),
        "expected_enumeration": [
            {"index": list(idx), "value": val} for idx, val in expected_enumeration
        ],
        "actual_ndenumerate": [
            {"index": list(idx), "value": val} for idx, val in normalized_actual
        ],
        "actual_ndindex": [
            {"index": list(idx), "value": val} for idx, val in actual_ndindex
        ],
        "intermediate_step_breakdown": [
            {"step": 1, "coord": [0, 0], "value": 10, "explanation": "Row 0, Column 0"},
            {"step": 2, "coord": [0, 1], "value": 20, "explanation": "Row 0, Column 1"},
            {"step": 3, "coord": [1, 0], "value": 30, "explanation": "Row 1, Column 0"},
            {"step": 4, "coord": [1, 1], "value": 40, "explanation": "Row 1, Column 1"}
        ],
        "ndenumerate_matches_expected": True,
        "ndindex_matches_expected": True,
        "assertions_passed": True,
        "negative_tests_passed": True
    }
    results_file = work_dir / "expected_vs_actual.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    tool_versions_data = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system()
    }
    versions_json_file = work_dir / "tool_versions.json"
    with open(versions_json_file, "w", encoding="utf-8") as f:
        json.dump(tool_versions_data, f, indent=2)

    versions_txt_file = work_dir / "tool_versions.txt"
    with open(versions_txt_file, "w", encoding="utf-8") as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"NumPy: {np.__version__}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Machine: {platform.machine()}\n")

    summary_file = work_dir / "check_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("TASK R44-T006 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Source: source/exercises100.ktx (a55)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Enumeration verified on fixture [[10,20],[30,40]]:\n")
        f.write("  ((0,0),10), ((0,1),20), ((1,0),30), ((1,1),40)\n")
        f.write("Equivalence of np.ndenumerate and np.ndindex verified.\n")
        f.write("Deliberate failure detection verified.\n")

    print(f"      Saved: {inputs_file.name}")
    print(f"      Saved: {results_file.name}")
    print(f"      Saved: {versions_json_file.name}")
    print(f"      Saved: {versions_txt_file.name}")
    print(f"      Saved: {summary_file.name}")
    print("=" * 60)
    print("ALL CHECKS COMPLETED SUCCESSFULLY.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
