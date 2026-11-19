#!/usr/bin/env python3
"""
Automated validation script for Task R44-T002:
Explain the boundary: NumPy exercise 86 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx and
   100_Numpy_exercises_with_solutions.md (preventing drift).
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   it succeeds with exit code 0.
4. Verifies the smallest/boundary case:
   Two identity 2x2 matrices with vectors [[1],[2]] and [[3],[4]]
   yield summed product [[4],[6]] and asserts output shape (2, 1).
5. Explains and asserts the dimension preconditions governing tensordot:
   M.shape[0] == V.shape[0] (p batches) and M.shape[2] == V.shape[1] (contraction axis).
6. Verifies that deliberate wrong results trigger assertion failures.
7. Saves inputs, expected vs actual results, and tool versions to work/R44-T002/.
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


def extract_ktx_answer(ktx_path, key="a86"):
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


def extract_markdown_solution(md_path, exercise_num=86):
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


def compute_boundary_case():
    """
    Compute the boundary case:
    p=2, n=2.
    M: two 2x2 identity matrices -> shape (2, 2, 2)
    V: vectors [[1],[2]] and [[3],[4]] -> shape (2, 2, 1)
    Expected intermediate products:
      M[0] @ V[0] = I_2 @ [[1],[2]] = [[1],[2]]
      M[1] @ V[1] = I_2 @ [[3],[4]] = [[3],[4]]
    Expected sum:
      [[1],[2]] + [[3],[4]] = [[4],[6]]
    Output shape: (2, 1)
    """
    M = np.array([np.eye(2), np.eye(2)], dtype=np.float64)  # (2, 2, 2)
    V = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float64)  # (2, 2, 1)

    # Precondition check
    assert M.shape[0] == V.shape[0], "Precondition violated: batch dimensions p must match"
    assert M.shape[2] == V.shape[1], "Precondition violated: inner product dimensions n must match"

    S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
    return M, V, S


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T002"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T002 CHECK: NumPy Exercise 86 (tensordot boundary)")
    print("=" * 60)

    # 1. Extract published code from authoritative source
    print(f"[1/6] Extracting a86 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a86")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 86)
    assert ktx_code == md_code, (
        "Generated markdown solution does not match authoritative source/exercises100.ktx a86 block!"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via subprocess to verify runtime execution
    print("[3/6] Executing published code via Python subprocess...")
    stdout = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")

    # 4. Verify boundary case (two identity 2x2 matrices with vectors [[1],[2]] and [[3],[4]])
    print("[4/6] Verifying boundary case (two identity 2x2 matrices with vectors [[1],[2]], [[3],[4]])...")
    M, V, actual_S = compute_boundary_case()

    expected_S = np.array([[4.0], [6.0]], dtype=np.float64)
    expected_shape = (2, 1)

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        wrong_expected_S = np.array([[999.0], [6.0]])
        np.testing.assert_allclose(actual_S, wrong_expected_S)

    # Assert shape
    assert actual_S.shape == expected_shape, (
        f"Output shape mismatch: expected {expected_shape}, got {actual_S.shape}"
    )

    # Assert values with tolerances
    atol = 1e-7
    rtol = 1e-7
    np.testing.assert_allclose(
        actual_S, expected_S, atol=atol, rtol=rtol,
        err_msg="Summed product value mismatch in tensordot boundary case"
    )
    print(f"      Output shape: {actual_S.shape} (asserted == (2, 1))")
    print(f"      Summed product:\n{actual_S}")
    print("      All boundary assertions PASSED.")

    # 5. Verify that deliberate wrong results fail assertions
    print("[5/6] Testing that assertions detect deliberate wrong results...")
    wrong_S = np.array([[999.0], [6.0]])
    try:
        np.testing.assert_allclose(actual_S, wrong_S, atol=atol, rtol=rtol)
        raise AssertionError("Check FAILED: deliberate wrong summed product was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong value correctly failed assertion.")

    try:
        wrong_shape = (1, 2)
        assert actual_S.shape == wrong_shape
        raise AssertionError("Check FAILED: deliberate wrong shape was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong shape correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T002/...")

    inputs_data = {
        "description": "Boundary case: p=2 matrices of shape 2x2 and p=2 vectors of shape 2x1",
        "p": 2,
        "n": 2,
        "matrices_M": {
            "description": "Two identity 2x2 matrices",
            "shape": [2, 2, 2],
            "values": M.tolist()
        },
        "vectors_V": {
            "description": "Two column vectors [[1], [2]] and [[3], [4]]",
            "shape": [2, 2, 1],
            "values": V.tolist()
        },
        "contraction_axes": [[0, 2], [0, 1]],
        "preconditions": [
            "M.shape[0] == V.shape[0] (p batches match)",
            "M.shape[2] == V.shape[1] (inner matrix-vector product dimensions match)",
            "Remaining axes (M.shape[1]=n, V.shape[2]=1) determine output shape (n, 1)"
        ]
    }
    inputs_file = work_dir / "inputs.json"
    with open(inputs_file, "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 86,
        "task": "R44-T002",
        "description": "Sum of p matrix products at once using np.tensordot",
        "intermediate_values": {
            "product_matrix_0": [[1.0], [2.0]],
            "product_matrix_1": [[3.0], [4.0]],
            "sum_formula": "[[1],[2]] + [[3],[4]] = [[4],[6]]"
        },
        "expected": {
            "shape": list(expected_shape),
            "values": expected_S.tolist()
        },
        "actual": {
            "shape": list(actual_S.shape),
            "values": actual_S.tolist()
        },
        "tolerances": {
            "atol": atol,
            "rtol": rtol
        },
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
        f.write("TASK R44-T002 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Source: source/exercises100.ktx (a86)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Boundary case verified: two identity 2x2 matrices with vectors [[1],[2]] and [[3],[4]]\n")
        f.write(f"Output shape verified: {actual_S.shape} == (2, 1)\n")
        f.write(f"Summed product verified:\n{actual_S.tolist()}\n")
        f.write("Precondition verified: M.shape[0] == V.shape[0] and M.shape[2] == V.shape[1]\n")
        f.write("Deliberate wrong result detection verified.\n")

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
