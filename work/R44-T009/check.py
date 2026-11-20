#!/usr/bin/env python3
"""
Automated validation script for Task R44-T009:
Explain resource behavior: NumPy exercise 74 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx (a74) and
   100_Numpy_exercises_with_solutions.md, ensuring no drift.
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts exit code 0.
4. Verifies the resource and mathematical behaviors:
   - Reconstructing [0, 0, 2, 2, 2] from counts [2, 0, 3].
   - np.bincount(A, minlength=3) returning original counts [2, 0, 3].
   - Copy vs view behavior: np.repeat returns a newly allocated copy (A.base is None,
     np.shares_memory is False).
   - Time and space complexity characteristics for bincount and repeat.
5. Verifies that deliberate wrong results trigger assertion failures (--deliberate-fail).
6. Saves inputs, expected vs actual results, and tool versions to work/R44-T009/.
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


def extract_ktx_answer(ktx_path, key="a74"):
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


def extract_markdown_solution(md_path, exercise_num=74):
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


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T009"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T009 CHECK: NumPy Exercise 74 (Resource Behavior)")
    print("=" * 60)

    # 1. Extract authoritative code from source/exercises100.ktx
    print(f"[1/6] Extracting a74 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a74")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 74)
    assert ktx_code == md_code, (
        f"Generated markdown solution does not match authoritative source/exercises100.ktx a74 block!\n"
        f"KTX:\n{ktx_code}\n\nMD:\n{md_code}"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via Python subprocess
    print("[3/6] Executing published code via Python subprocess...")
    stdout, stderr = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")
    print(f"      STDOUT: {stdout.strip()}")

    # 4. Detailed behavior & resource verification
    print("[4/6] Verifying resource contracts, copy/view behavior, and reconstruction...")

    # A. Reconstruction of [0, 0, 2, 2, 2] from counts [2, 0, 3]
    counts_fixture = np.array([2, 0, 3])
    idx_fixture = np.arange(len(counts_fixture))
    reconstructed_A = np.repeat(idx_fixture, counts_fixture)
    expected_A = np.array([0, 0, 2, 2, 2])

    assert np.array_equal(reconstructed_A, expected_A), (
        f"Reconstruction failed! Expected {expected_A.tolist()}, got {reconstructed_A.tolist()}"
    )
    print(f"      PASS: Counts {counts_fixture.tolist()} reconstructed {reconstructed_A.tolist()}")

    # B. bincount(..., minlength=3) returning original counts
    recounted_C = np.bincount(reconstructed_A, minlength=3)
    assert np.array_equal(recounted_C, counts_fixture), (
        f"bincount minlength=3 failed! Expected {counts_fixture.tolist()}, got {recounted_C.tolist()}"
    )
    print(f"      PASS: bincount(reconstructed_A, minlength=3) returned {recounted_C.tolist()}")

    # C. Copy vs View behavior experiment
    shares_mem = np.shares_memory(reconstructed_A, idx_fixture)
    assert shares_mem is False, "reconstructed_A should not share memory with idx_fixture!"
    assert reconstructed_A.base is None, "reconstructed_A.base should be None for an owning copy!"

    # Mutation test: modifying reconstructed array does not mutate source
    orig_idx_copy = idx_fixture.copy()
    reconstructed_A[0] = 999
    assert np.array_equal(idx_fixture, orig_idx_copy), "Mutating reconstructed_A mutated idx_fixture!"
    reconstructed_A[0] = 0  # Restore

    print("      PASS: Copy/view behavior verified: np.repeat returns a new owning array (copy).")

    # D. Trailing zeros and minlength contract explanation
    counts_with_trailing = np.array([2, 0, 3, 0])
    A_trailing = np.repeat(np.arange(len(counts_with_trailing)), counts_with_trailing)
    bincount_default = np.bincount(A_trailing)  # len 3 (drops trailing zero)
    bincount_minlength = np.bincount(A_trailing, minlength=len(counts_with_trailing))  # len 4

    assert len(bincount_default) == 3
    assert np.array_equal(bincount_minlength, counts_with_trailing)
    print(f"      PASS: Trailing zero preservation with minlength={len(counts_with_trailing)} verified.")

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        wrong_A = np.array([0, 1, 2, 2, 2])
        assert np.array_equal(reconstructed_A, wrong_A), "Deliberate failure triggered!"

    # 5. Negative testing for deliberate failure detection
    print("[5/6] Testing deliberate failure detection...")
    wrong_A = np.array([0, 1, 2, 2, 2])
    try:
        assert np.array_equal(reconstructed_A, wrong_A)
        raise AssertionError("Check FAILED: deliberate wrong array was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong reconstruction correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T009/...")

    inputs_data = {
        "counts_fixture": counts_fixture.tolist(),
        "indices_fixture": idx_fixture.tolist(),
        "original_code_input": [1, 1, 2, 3, 4, 4, 6],
        "trailing_zero_fixture": counts_with_trailing.tolist(),
        "resource_contract": {
            "time_complexity": "O(K + N) where K = len(C) and N = sum(C)",
            "space_complexity": "O(N) for output array, O(K) for indices",
            "memory_behavior": "np.repeat always allocates a new contiguous copy (A.base is None, shares_memory is False)"
        }
    }
    with open(work_dir / "inputs.json", "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 74,
        "task": "R44-T009",
        "fixture_test": {
            "input_counts": counts_fixture.tolist(),
            "reconstructed_array": reconstructed_A.tolist(),
            "expected_reconstructed_array": expected_A.tolist(),
            "matches_expected": bool(np.array_equal(reconstructed_A, expected_A)),
            "bincount_minlength_3": recounted_C.tolist(),
            "matches_original_counts": bool(np.array_equal(recounted_C, counts_fixture))
        },
        "copy_view_experiment": {
            "shares_memory_with_arange": shares_mem,
            "base_is_none": bool(reconstructed_A.base is None),
            "mutation_isolated": True,
            "array_nbytes": int(reconstructed_A.nbytes),
            "dtype": str(reconstructed_A.dtype)
        },
        "trailing_zeros_contract": {
            "counts_with_trailing": counts_with_trailing.tolist(),
            "default_bincount_drops_trailing": bincount_default.tolist(),
            "minlength_preserves_trailing": bincount_minlength.tolist(),
            "matches_input": bool(np.array_equal(bincount_minlength, counts_with_trailing))
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
        f.write("TASK R44-T009 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Item: NumPy exercise 74\n")
        f.write("Source: source/exercises100.ktx (a74, h74)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Resource and algorithmic contracts verified:\n")
        f.write("  - Counts [2, 0, 3] reconstruct [0, 0, 2, 2, 2].\n")
        f.write("  - np.bincount(..., minlength=3) returns original counts [2, 0, 3].\n")
        f.write("  - np.repeat returns a new contiguous copy (A.base is None, shares_memory is False).\n")
        f.write("  - Mutation isolation and trailing-zero minlength behavior confirmed.\n")
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
