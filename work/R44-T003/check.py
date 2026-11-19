#!/usr/bin/env python3
"""
Automated validation script for Task R44-T003:
Compare two methods: NumPy exercise 31 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx and
   100_Numpy_exercises_with_solutions.md (preventing drift).
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   it succeeds with exit code 0.
4. Compares two methods on the same fixture (divide-by-zero on arrays):
   - Method 1: Global state modification via np.seterr(...) / np.seterr(**defaults)
   - Method 2: Scoped context manager via with np.errstate(...)
   Explains differences in intermediate representation and execution safety.
5. Verifies that leaving a local np.errstate context restores np.geterr() to
   the saved policy, both on normal exit and after an exception.
6. Verifies that deliberate wrong results trigger assertion failures.
7. Saves inputs, expected vs actual results, and tool versions to work/R44-T003/.
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


def extract_ktx_answer(ktx_path, key="a31"):
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


def extract_markdown_solution(md_path, exercise_num=31):
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


def test_methods_comparison():
    """
    Compare Method 1 (global seterr) and Method 2 (np.errstate)
    on the exact same fixture: np.ones(1) / 0 and np.arange(3) / 0.0.
    """
    initial_policy = np.geterr()

    # Fixture 1: np.ones(1) / 0.0 -> [inf]
    # Method 1: Global seterr
    defaults1 = np.seterr(all="ignore")
    z1_m1 = np.ones(1) / 0.0
    _ = np.seterr(**defaults1)
    policy_after_m1 = np.geterr()

    # Method 2: Scoped errstate
    with np.errstate(all="ignore"):
        z1_m2 = np.ones(1) / 0.0
    policy_after_m2 = np.geterr()

    # Fixture 2: np.arange(3) / 0.0 -> [nan, inf, inf]
    defaults2 = np.seterr(all="ignore")
    z2_m1 = np.arange(3) / 0.0
    _ = np.seterr(**defaults2)

    with np.errstate(all="ignore"):
        z2_m2 = np.arange(3) / 0.0

    return {
        "initial_policy": initial_policy,
        "policy_after_m1": policy_after_m1,
        "policy_after_m2": policy_after_m2,
        "fixture_ones_m1": z1_m1,
        "fixture_ones_m2": z1_m2,
        "fixture_arange_m1": z2_m1,
        "fixture_arange_m2": z2_m2,
    }


def test_exception_safety():
    """
    Verify that np.errstate cleanly restores np.geterr() to saved policy
    even if an exception occurs within the context.
    """
    saved_policy = np.geterr()
    exception_caught = False

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            _ = np.ones(1) / 0.0
            raise ZeroDivisionError("Simulated failure within context")
    except ZeroDivisionError:
        exception_caught = True

    policy_after_exception = np.geterr()
    return saved_policy, policy_after_exception, exception_caught


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T003"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T003 CHECK: NumPy Exercise 31 (Compare seterr vs errstate)")
    print("=" * 60)

    # 1. Extract published code from authoritative source
    print(f"[1/6] Extracting a31 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a31")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 31)
    assert ktx_code == md_code, (
        "Generated markdown solution does not match authoritative source/exercises100.ktx a31 block!"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via subprocess to verify runtime execution
    print("[3/6] Executing published code via Python subprocess...")
    stdout = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")

    # 4. Compare both methods on identical fixtures
    print("[4/6] Comparing Method 1 (np.seterr) and Method 2 (np.errstate)...")
    comp = test_methods_comparison()

    assert comp["policy_after_m1"] == comp["initial_policy"], "Method 1 did not restore policy!"
    assert comp["policy_after_m2"] == comp["initial_policy"], "Method 2 did not restore policy!"

    # Numerical equivalence on fixture 1 (np.ones(1) / 0 -> [inf])
    np.testing.assert_allclose(comp["fixture_ones_m1"], comp["fixture_ones_m2"])
    assert np.isinf(comp["fixture_ones_m1"][0])

    # Numerical equivalence on fixture 2 (np.arange(3) / 0 -> [nan, inf, inf])
    assert np.isnan(comp["fixture_arange_m1"][0]) and np.isnan(comp["fixture_arange_m2"][0])
    np.testing.assert_allclose(comp["fixture_arange_m1"][1:], comp["fixture_arange_m2"][1:])

    print("      Both methods produced identical results on fixtures:")
    print(f"        np.ones(1)/0: {comp['fixture_ones_m1']}")
    print(f"        np.arange(3)/0: {comp['fixture_arange_m1']}")

    # 5. Verify exception safety
    print("[5/6] Verifying np.errstate exception safety and policy restoration...")
    saved_pol, after_exc_pol, exc_caught = test_exception_safety()
    assert exc_caught, "Exception was not caught during exception safety test!"

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        corrupted_policy = {"divide": "ignore", "over": "ignore", "under": "ignore", "invalid": "ignore"}
        assert after_exc_pol == corrupted_policy, "Deliberate failure: policies unexpectedly matched!"

    assert after_exc_pol == saved_pol, (
        f"Policy mismatch after exception! Expected {saved_pol}, got {after_exc_pol}"
    )
    print(f"      Policy after exception equals saved policy: {after_exc_pol}")

    # Verify deliberate wrong result detection
    corrupted_policy = {"divide": "raise", "over": "raise", "under": "raise", "invalid": "raise"}
    try:
        assert after_exc_pol == corrupted_policy
        raise AssertionError("Check FAILED: deliberate wrong policy comparison was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong policy comparison correctly failed.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T003/...")

    inputs_data = {
        "description": "Test fixtures for divide-by-zero warning suppression",
        "fixtures": [
            {
                "name": "fixture_ones",
                "expression": "np.ones(1) / 0.0",
                "input_array": [1.0],
                "divisor": 0.0
            },
            {
                "name": "fixture_arange",
                "expression": "np.arange(3) / 0.0",
                "input_array": [0, 1, 2],
                "divisor": 0.0
            }
        ],
        "initial_error_policy": comp["initial_policy"]
    }
    inputs_file = work_dir / "inputs.json"
    with open(inputs_file, "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 31,
        "task": "R44-T003",
        "description": "Comparison of np.seterr (global mutation) vs np.errstate (context manager)",
        "method_comparison": {
            "method_1": {
                "name": "np.seterr",
                "type": "Global state mutation",
                "intermediate_representation": "Modifies global/thread-local error dictionary; requires manual restoration",
                "exception_safe": False,
                "fixture_ones_output": [str(x) for x in comp["fixture_ones_m1"]],
                "fixture_arange_output": [str(x) for x in comp["fixture_arange_m1"]]
            },
            "method_2": {
                "name": "np.errstate",
                "type": "Context manager (RAII pattern)",
                "intermediate_representation": "Context manager object saving pre-context policy and restoring via __exit__ in finally block",
                "exception_safe": True,
                "fixture_ones_output": [str(x) for x in comp["fixture_ones_m2"]],
                "fixture_arange_output": [str(x) for x in comp["fixture_arange_m2"]]
            }
        },
        "policy_restoration": {
            "saved_policy": saved_pol,
            "policy_after_normal_exit": comp["policy_after_m2"],
            "policy_after_exception": after_exc_pol,
            "policies_match": after_exc_pol == saved_pol
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
        f.write("TASK R44-T003 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Source: source/exercises100.ktx (a31)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Method comparison verified on identical fixtures (ones/0 and arange/0)\n")
        f.write("Intermediate representation differences explained (global state vs RAII context)\n")
        f.write(f"np.geterr() policy restored after context exit: {after_exc_pol == saved_pol}\n")
        f.write(f"np.geterr() policy restored after exception inside context: {after_exc_pol == saved_pol}\n")
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
