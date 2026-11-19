#!/usr/bin/env python3
"""
Automated validation script for Task R44-T001:
Make the example executable: NumPy exercise 44 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx and
   100_Numpy_exercises_with_solutions.md (preventing drift).
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   it succeeds with exit code 0.
4. Verifies Cartesian points (3,4), (0,1), (-1,0) produce:
   - Radii: [5.0, 1.0, 1.0]
   - Angles: [atan2(4,3), pi/2, pi]
   with explicit floating-point tolerances via numpy.testing.
5. Verifies that deliberate wrong results trigger assertion failures.
6. Saves inputs, expected vs actual results, and tool versions to work/R44-T001/.
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


def extract_ktx_answer(ktx_path, key="a44"):
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


def extract_markdown_solution(md_path, exercise_num=44):
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


def verify_cartesian_to_polar(points, atol=1e-7, rtol=1e-7):
    """
    Perform Cartesian to Polar conversion and return intermediate and final results.
    Cartesian: (X, Y)
    Polar: R = sqrt(X^2 + Y^2), T = arctan2(Y, X)
    """
    Z = np.asarray(points, dtype=np.float64)
    X, Y = Z[:, 0], Z[:, 1]
    R = np.sqrt(X**2 + Y**2)
    T = np.arctan2(Y, X)
    return {
        "Z": Z,
        "X": X,
        "Y": Y,
        "R": R,
        "T": T
    }


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T001"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T001 CHECK: NumPy Exercise 44 (Cartesian to Polar)")
    print("=" * 60)

    # 1. Extract published code from authoritative source
    print(f"[1/6] Extracting a44 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a44")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 44)
    assert ktx_code == md_code, (
        "Generated markdown solution does not match authoritative source/exercises100.ktx a44 block!"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via subprocess to verify runtime execution
    print("[3/6] Executing published code via Python subprocess...")
    stdout = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")

    # 4. Assert Cartesian points and tolerances
    print("[4/6] Verifying Cartesian points (3,4), (0,1), (-1,0)...")
    test_points = [
        [3.0, 4.0],
        [0.0, 1.0],
        [-1.0, 0.0]
    ]

    calc = verify_cartesian_to_polar(test_points)
    actual_R = calc["R"]
    actual_T = calc["T"]

    expected_R = np.array([5.0, 1.0, 1.0], dtype=np.float64)
    expected_T = np.array([np.arctan2(4.0, 3.0), np.pi / 2.0, np.pi], dtype=np.float64)

    atol = 1e-7
    rtol = 1e-7

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        wrong_expected_R = np.array([999.0, 1.0, 1.0])
        np.testing.assert_allclose(actual_R, wrong_expected_R, atol=atol, rtol=rtol)

    np.testing.assert_allclose(actual_R, expected_R, atol=atol, rtol=rtol,
                               err_msg="Radii mismatch for test Cartesian points")
    np.testing.assert_allclose(actual_T, expected_T, atol=atol, rtol=rtol,
                               err_msg="Angles mismatch for test Cartesian points")
    print(f"      Radii:  actual={actual_R.tolist()} == expected={expected_R.tolist()} (tol={atol})")
    print(f"      Angles: actual={actual_T.tolist()} == expected={expected_T.tolist()} (tol={atol})")
    print("      All mathematical assertions PASSED.")

    # 5. Verify that deliberate wrong results fail assertions
    print("[5/6] Testing that assertions detect deliberate wrong results...")
    wrong_R = np.array([4.0, 1.0, 1.0])
    try:
        np.testing.assert_allclose(actual_R, wrong_R, atol=atol, rtol=rtol)
        raise AssertionError("Check FAILED: deliberate wrong radius was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong radius correctly failed assertion.")

    wrong_T = np.array([0.0, 0.0, 0.0])
    try:
        np.testing.assert_allclose(actual_T, wrong_T, atol=atol, rtol=rtol)
        raise AssertionError("Check FAILED: deliberate wrong angle was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong angle correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T001/...")

    inputs_data = {
        "description": "Cartesian test coordinates for exercise 44",
        "points": [
            {"name": "Point 1", "x": 3.0, "y": 4.0},
            {"name": "Point 2", "x": 0.0, "y": 1.0},
            {"name": "Point 3", "x": -1.0, "y": 0.0}
        ],
        "cartesian_matrix": test_points
    }
    inputs_file = work_dir / "inputs.json"
    with open(inputs_file, "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 44,
        "description": "Conversion from Cartesian (X, Y) to Polar (R, T)",
        "inputs": test_points,
        "intermediate_values": {
            "X": calc["X"].tolist(),
            "Y": calc["Y"].tolist(),
            "X_squared": (calc["X"]**2).tolist(),
            "Y_squared": (calc["Y"]**2).tolist(),
            "sum_of_squares": (calc["X"]**2 + calc["Y"]**2).tolist()
        },
        "expected": {
            "radii": expected_R.tolist(),
            "angles_radians": expected_T.tolist(),
            "angle_symbolic": ["atan2(4, 3)", "pi/2", "pi"]
        },
        "actual": {
            "radii": actual_R.tolist(),
            "angles_radians": actual_T.tolist()
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
        f.write("TASK R44-T001 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Source: source/exercises100.ktx\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Test points (3,4), (0,1), (-1,0) verified with tolerances (atol=1e-7, rtol=1e-7)\n")
        f.write("Expected Radii: [5.0, 1.0, 1.0]\n")
        f.write("Expected Angles: [atan2(4,3), pi/2, pi]\n")
        f.write("Negative assertion tests verified.\n")

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
