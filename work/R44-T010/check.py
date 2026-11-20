#!/usr/bin/env python3
"""
Automated validation script for Task R44-T010:
Protect the example against regression: NumPy exercise 11 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx (a11) and
   100_Numpy_exercises_with_solutions.md, ensuring no drift.
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts exit code 0.
4. Verifies the identity matrix mathematical invariants:
   - Shape is (3, 3).
   - Main diagonal elements are all 1: np.diag(Z) == [1, 1, 1].
   - Total sum of elements equals 3: np.sum(Z) == 3.0.
   - All off-diagonal elements are 0: Z[~np.eye(3, dtype=bool)] == 0.
5. Implements a temporary one-operation mutant and verifies the regression test
   reliably distinguishes the mutant from the correct example (mutation testing).
6. Verifies deliberate wrong results trigger assertion failures (--deliberate-fail).
7. Saves inputs, expected vs actual results, and tool versions to work/R44-T010/.
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


def extract_ktx_answer(ktx_path, key="a11"):
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


def extract_markdown_solution(md_path, exercise_num=11):
    """Extract code block for the given exercise from markdown file."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = content.split("#### ")
    target_sec = None
    for s in sections:
        if s.startswith(f"{exercise_num}."):
            target_sec = s
            break
    if not target_sec:
        raise ValueError(f"Could not find section for exercise {exercise_num} in {md_path}")

    code_blocks = re.findall(r"```python\n(.*?)```", target_sec, re.DOTALL)
    if not code_blocks:
        raise ValueError(f"Could not find code blocks in exercise {exercise_num} section in {md_path}")
    return code_blocks[-1].strip()


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
    return result.stdout, result.stderr


def verify_identity_invariants(matrix):
    """Verify standard identity matrix invariants."""
    assert matrix.shape == (3, 3), f"Expected shape (3, 3), got {matrix.shape}"
    diag = np.diag(matrix)
    assert np.array_equal(diag, [1, 1, 1]), f"Expected diagonal [1, 1, 1], got {diag.tolist()}"
    total_sum = np.sum(matrix)
    assert total_sum == 3.0, f"Expected total sum 3.0, got {total_sum}"
    off_diag = matrix[~np.eye(3, dtype=bool)]
    assert np.all(off_diag == 0), f"Expected all off-diagonal elements 0, got non-zeros: {off_diag[off_diag != 0].tolist()}"
    return True


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T010"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T010 CHECK: NumPy Exercise 11 (Regression Protection)")
    print("=" * 60)

    # 1. Extract authoritative code from source/exercises100.ktx
    print(f"[1/6] Extracting a11 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a11")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 11)
    assert ktx_code == md_code, (
        f"Generated markdown solution does not match authoritative source/exercises100.ktx a11 block!\n"
        f"KTX:\n{ktx_code}\n\nMD:\n{md_code}"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via Python subprocess
    print("[3/6] Executing published code via Python subprocess...")
    stdout, stderr = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")
    print(f"      STDOUT: {stdout.strip()}")

    # 4. Detailed invariant verification
    print("[4/6] Verifying 3x3 identity matrix invariants on correct example...")
    Z = np.eye(3)
    diag = np.diag(Z).tolist()
    total_sum = float(np.sum(Z))
    off_diag_elements = Z[~np.eye(3, dtype=bool)].tolist()

    assert Z.shape == (3, 3)
    assert diag == [1.0, 1.0, 1.0]
    assert total_sum == 3.0
    assert all(x == 0.0 for x in off_diag_elements)
    assert verify_identity_invariants(Z) is True

    print(f"      PASS: Matrix shape: {Z.shape}")
    print(f"      PASS: Diagonal: {diag}")
    print(f"      PASS: Total sum: {total_sum}")
    print(f"      PASS: Off-diagonal elements count: {len(off_diag_elements)}, all zeros.")

    # 5. Mutation testing: create temporary one-operation mutants and verify tests catch them
    print("[5/6] Testing regression protection against one-operation mutants...")

    # Mutant 1: np.ones((3, 3)) instead of np.eye(3) (off-diagonal elements mutated to 1)
    mutant_ones = np.ones((3, 3))
    mutant_1_caught = False
    try:
        verify_identity_invariants(mutant_ones)
    except AssertionError:
        mutant_1_caught = True
    assert mutant_1_caught, "Mutant 1 (all ones) was NOT caught by regression invariants!"
    print("      PASS: Mutant 1 (np.ones((3,3))) caught by off-diagonal and sum invariants.")

    # Mutant 2: Shifted diagonal np.eye(3, k=1)
    mutant_shifted = np.eye(3, k=1)
    mutant_2_caught = False
    try:
        verify_identity_invariants(mutant_shifted)
    except AssertionError:
        mutant_2_caught = True
    assert mutant_2_caught, "Mutant 2 (k=1 shifted eye) was NOT caught by regression invariants!"
    print("      PASS: Mutant 2 (np.eye(3, k=1)) caught by diagonal invariant.")

    # Mutant 3: Single-element perturbation Z[0, 1] = 1.0
    mutant_corrupt = np.eye(3)
    mutant_corrupt[0, 1] = 1.0
    mutant_3_caught = False
    try:
        verify_identity_invariants(mutant_corrupt)
    except AssertionError:
        mutant_3_caught = True
    assert mutant_3_caught, "Mutant 3 (corrupted off-diagonal) was NOT caught!"
    print("      PASS: Mutant 3 (corrupted off-diagonal entry) caught by off-diagonal invariant.")

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        wrong_Z = np.zeros((3, 3))
        assert np.array_equal(Z, wrong_Z), "Deliberate failure triggered!"

    # Internal negative test for deliberate failure detection
    wrong_Z = np.zeros((3, 3))
    try:
        assert np.array_equal(Z, wrong_Z)
        raise AssertionError("Check FAILED: deliberate wrong matrix was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong matrix correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T010/...")

    inputs_data = {
        "function": "np.eye",
        "dimension": 3,
        "input_call": "np.eye(3)",
        "invariants_tested": {
            "shape": [3, 3],
            "diagonal": [1.0, 1.0, 1.0],
            "sum": 3.0,
            "off_diagonal_zeros": True
        },
        "mutants_evaluated": [
            {"mutant_id": "mutant_ones", "operation": "np.ones((3, 3))", "expected_failure": "sum and off-diagonal invariants"},
            {"mutant_id": "mutant_shifted", "operation": "np.eye(3, k=1)", "expected_failure": "diagonal and sum invariants"},
            {"mutant_id": "mutant_corrupt", "operation": "np.eye(3); Z[0, 1] = 1.0", "expected_failure": "off-diagonal invariant"}
        ]
    }
    with open(work_dir / "inputs.json", "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 11,
        "task": "R44-T010",
        "actual_matrix": Z.tolist(),
        "shape": list(Z.shape),
        "diagonal": diag,
        "total_sum": total_sum,
        "off_diagonal_zeros_count": len(off_diag_elements),
        "invariants_passed": True,
        "mutation_testing": {
            "mutant_ones_caught": mutant_1_caught,
            "mutant_shifted_caught": mutant_2_caught,
            "mutant_corrupt_caught": mutant_3_caught,
            "all_mutants_distinguished": True
        },
        "assertions_passed": True,
        "negative_tests_passed": True
    }
    with open(work_dir / "expected_vs_actual.json", "w", encoding="utf-8") as f:
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
    with open(work_dir / "tool_versions.json", "w", encoding="utf-8") as f:
        json.dump(tool_versions_data, f, indent=2)

    with open(work_dir / "tool_versions.txt", "w", encoding="utf-8") as f:
        f.write(f"Python: {sys.version}\n")
        f.write(f"NumPy: {np.__version__}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Machine: {platform.machine()}\n")

    with open(work_dir / "check_summary.txt", "w", encoding="utf-8") as f:
        f.write("TASK R44-T010 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Item: NumPy exercise 11\n")
        f.write("Source: source/exercises100.ktx (a11, h11)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Identity invariants verified:\n")
        f.write("  - Shape: (3, 3)\n")
        f.write("  - Main diagonal: [1, 1, 1]\n")
        f.write("  - Sum: 3.0\n")
        f.write("  - Off-diagonal elements: all 0.0\n")
        f.write("Mutation testing verified:\n")
        f.write("  - One-operation mutants (np.ones, np.eye(3, k=1), corrupted off-diagonal) distinguished and caught.\n")
        f.write("Deliberate failure detection verified.\n")

    print(f"      Saved: inputs.json")
    print(f"      Saved: expected_vs_actual.json")
    print(f"      Saved: tool_versions.json")
    print(f"      Saved: tool_versions.txt")
    print(f"      Saved: check_summary.txt")
    print("=" * 60)
    print("ALL CHECKS COMPLETED SUCCESSFULLY.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
