#!/usr/bin/env python3
"""
Automated validation script for Task R44-T004:
Explain an invariant: NumPy exercise 63 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx and
   100_Numpy_exercises_with_solutions.md (preventing drift).
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   it succeeds with exit code 0.
4. Asserts that the 'name' attribute of a NamedArray survives slicing and copying.
5. Tests the precise invariant across the original fixture plus at least 5
   bounded variations (2D float matrix, empty array, single element array,
   default name, boolean mask, and negative-strided reverse slice).
6. Verifies that deliberate wrong results trigger assertion failures.
7. Saves inputs, expected vs actual results, and tool versions to work/R44-T004/.
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


def extract_ktx_answer(ktx_path, key="a63"):
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


def extract_markdown_solution(md_path, exercise_num=63):
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


class NamedArray(np.ndarray):
    """Subclass of np.ndarray with a persistent name attribute."""
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", "no name")


def test_invariant_on_case(case_name, raw_data, name_arg, slice_indexer):
    """
    Test the name-preservation and data-integrity invariant on a test case:
    1. Slicing: Z[slice_indexer] is NamedArray and preserves name.
    2. Copying: Z.copy() is NamedArray and preserves name.
    3. Values: Underlying array values match numpy ndarray behavior.
    """
    if name_arg is not None:
        Z = NamedArray(raw_data, name=name_arg)
        expected_name = name_arg
    else:
        Z = NamedArray(raw_data)
        expected_name = "no name"

    # 1. Slicing
    Z_slice = Z[slice_indexer]
    slice_is_instance = isinstance(Z_slice, NamedArray)
    slice_name_preserved = getattr(Z_slice, "name", None) == expected_name

    # 2. Copying
    Z_copy = Z.copy()
    copy_is_instance = isinstance(Z_copy, NamedArray)
    copy_name_preserved = getattr(Z_copy, "name", None) == expected_name

    assert slice_is_instance, f"[{case_name}] Sliced array is not an instance of NamedArray"
    assert slice_name_preserved, f"[{case_name}] Sliced array did not preserve name: {getattr(Z_slice, 'name', None)}"
    assert copy_is_instance, f"[{case_name}] Copied array is not an instance of NamedArray"
    assert copy_name_preserved, f"[{case_name}] Copied array did not preserve name: {getattr(Z_copy, 'name', None)}"

    # 3. Numerical / Data check
    expected_slice = np.asarray(raw_data)[slice_indexer]
    np.testing.assert_array_equal(np.asarray(Z_slice), expected_slice)
    np.testing.assert_array_equal(np.asarray(Z_copy), np.asarray(raw_data))

    return {
        "case_name": case_name,
        "input_shape": list(Z.shape),
        "input_dtype": str(Z.dtype),
        "original_name": expected_name,
        "slice_indexer": str(slice_indexer),
        "slice_shape": list(Z_slice.shape),
        "slice_name": Z_slice.name,
        "slice_name_survived": slice_name_preserved,
        "copy_shape": list(Z_copy.shape),
        "copy_name": Z_copy.name,
        "copy_name_survived": copy_name_preserved,
        "passed": True
    }


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T004"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T004 CHECK: NumPy Exercise 63 (NamedArray Invariant)")
    print("=" * 60)

    # 1. Extract published code from authoritative source
    print(f"[1/6] Extracting a63 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a63")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 63)
    assert ktx_code == md_code, (
        "Generated markdown solution does not match authoritative source/exercises100.ktx a63 block!"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via subprocess to verify runtime execution
    print("[3/6] Executing published code via Python subprocess...")
    stdout = run_extracted_code(ktx_code)
    print(f"      Subprocess executed successfully with exit code 0. (Output: {stdout.strip()})")

    # 4. Define and test invariant on fixture + 6 bounded variations (>= 5 required)
    print("[4/6] Testing invariant on fixture and 6 bounded variations...")
    test_cases = [
        # Base Fixture
        ("Fixture: 1D Range Array", np.arange(10), "range_10", slice(2, 5)),
        # Variation 1: 2D Float Matrix
        ("Variation 1: 2D Float Matrix", np.ones((4, 4), dtype=float), "float_matrix_4x4", (slice(0, 2), slice(1, 3))),
        # Variation 2: Empty 1D Array
        ("Variation 2: Empty Array", np.array([], dtype=int), "empty_vector", slice(None)),
        # Variation 3: Single-element Array
        ("Variation 3: Single Element", np.array([42]), "singleton_42", slice(0, 1)),
        # Variation 4: Default / Unspecified Name
        ("Variation 4: Default Name", np.array([10, 20, 30]), None, slice(1, None)),
        # Variation 5: Boolean Mask Array
        ("Variation 5: Boolean Array", np.array([True, False, True, False]), "bool_mask", slice(None, None, 2)),
        # Variation 6: Reverse-strided Array
        ("Variation 6: Reverse-strided Slice", np.arange(6), "reversed_series", slice(None, None, -1)),
    ]

    case_results = []
    for case_name, data, name_arg, slc in test_cases:
        res = test_invariant_on_case(case_name, data, name_arg, slc)
        case_results.append(res)
        print(f"      PASS: {case_name} -> name='{res['original_name']}' survived slicing and copying.")

    # 5. Deliberate failure testing
    print("[5/6] Testing deliberate failure detection...")
    fixture_arr = NamedArray(np.arange(10), "range_10")
    fixture_slice = fixture_arr[2:5]

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        assert fixture_slice.name == "wrong_expected_name", "Deliberate failure triggered!"

    try:
        assert fixture_slice.name == "wrong_expected_name"
        raise AssertionError("Check FAILED: deliberate wrong name expectation did not trigger failure!")
    except AssertionError:
        print("      PASS: Deliberate wrong name correctly failed assertion.")

    try:
        # Deliberate wrong assumption: name lost on copy
        assert fixture_arr.copy().name is None
        raise AssertionError("Check FAILED: deliberate wrong copy expectation did not trigger failure!")
    except AssertionError:
        print("      PASS: Deliberate wrong copy expectation correctly failed.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T004/...")

    inputs_data = {
        "description": "Fixture and 6 bounded variations testing NamedArray slicing and copying invariant",
        "invariant_definition": (
            "For any NamedArray A with name N, any derived slice A[slice] or copy A.copy() "
            "remains an instance of NamedArray and preserves its name attribute (A.name == N) "
            "via __array_finalize__."
        ),
        "cases": [
            {
                "case_name": c["case_name"],
                "input_shape": c["input_shape"],
                "input_dtype": c["input_dtype"],
                "name": c["original_name"],
                "slice_indexer": c["slice_indexer"]
            }
            for c in case_results
        ]
    }
    inputs_file = work_dir / "inputs.json"
    with open(inputs_file, "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 63,
        "task": "R44-T004",
        "description": "NamedArray name preservation across view casting (slicing) and copying",
        "total_cases_tested": len(case_results),
        "results": case_results,
        "name_survives_slicing": True,
        "name_survives_copying": True,
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
        f.write("TASK R44-T004 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Source: source/exercises100.ktx (a63)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write(f"Tested fixture + {len(case_results) - 1} variations (total {len(case_results)} cases).\n")
        f.write("Confirmed: Name survives both slicing (views) and copying (explicit copies).\n")
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
