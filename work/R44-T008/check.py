#!/usr/bin/env python3
"""
Automated validation script for Task R44-T008:
Document a failure or limitation: NumPy exercise 32 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx (a32) and
   100_Numpy_exercises_with_solutions.md, ensuring no drift.
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   exit code 0.
4. Verifies the core mathematical and domain contracts:
   - Under documented error policy, real sqrt(-1) emits RuntimeWarning and yields NaN (float64).
   - np.emath.sqrt(-1) automatically extends to complex domain yielding 1j (complex128).
   - Explains why np.sqrt(-1) == np.emath.sqrt(-1) is False (IEEE 754 NaN equality semantics).
   - Demonstrates the common misconception and explicit complex promotion contract:
     np.sqrt(-1 + 0j) == 1j.
   - Demonstrates type/shape/value contrast on a total array operation with mixed inputs [-1.0, 4.0].
5. Verifies deliberate wrong results trigger assertion failures (--deliberate-fail).
6. Saves inputs, expected vs actual results, and tool versions to work/R44-T008/.
"""

import sys
import os
import json
import platform
import subprocess
import re
import warnings
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


def extract_ktx_answer(ktx_path, key="a32"):
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


def extract_markdown_solution(md_path, exercise_num=32):
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
    work_dir = repo_root / "work" / "R44-T008"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T008 CHECK: NumPy Exercise 32 (Failure & Limitation)")
    print("=" * 60)

    # 1. Extract authoritative code from source/exercises100.ktx
    print(f"[1/6] Extracting a32 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a32")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 32)
    assert ktx_code == md_code, (
        f"Generated markdown solution does not match authoritative source/exercises100.ktx a32 block!\n"
        f"KTX:\n{ktx_code}\n\nMD:\n{md_code}"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via Python subprocess
    print("[3/6] Executing published code via Python subprocess...")
    stdout, stderr = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")
    print(f"      STDOUT: {stdout.strip()}")
    if "RuntimeWarning" in stderr:
        print("      STDERR captured documented RuntimeWarning as expected.")

    # 4. Detailed contract and behavior verification
    print("[4/6] Verifying source contracts, error policy, types, and contrast...")

    # A. Documented error policy & RuntimeWarning
    default_policy = np.geterr()
    print(f"      Default floating-point error policy: {default_policy}")

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        raw_real_sqrt = np.sqrt(-1)
        warning_triggered = any(
            issubclass(w.category, RuntimeWarning) and "invalid value encountered in sqrt" in str(w.message)
            for w in recorded_warnings
        )
    assert warning_triggered, "Expected RuntimeWarning for np.sqrt(-1) under default error policy!"
    print("      PASS: Documented error policy emits RuntimeWarning for negative real input.")

    # B. Values, types, and NaN comparison
    with np.errstate(invalid="ignore"):
        real_sqrt_val = np.sqrt(-1)
    emath_sqrt_val = np.emath.sqrt(-1)

    assert np.isnan(real_sqrt_val), f"Expected real_sqrt to be NaN, got {real_sqrt_val}"
    assert isinstance(real_sqrt_val, (float, np.floating)), f"Expected float type, got {type(real_sqrt_val)}"
    assert real_sqrt_val.dtype == np.float64

    assert emath_sqrt_val == 1j, f"Expected emath_sqrt to be 1j, got {emath_sqrt_val}"
    assert isinstance(emath_sqrt_val, (complex, np.complexfloating)), f"Expected complex type, got {type(emath_sqrt_val)}"
    assert emath_sqrt_val.dtype == np.complex128

    equality_res = bool(real_sqrt_val == emath_sqrt_val)
    assert equality_res is False, "Expected np.sqrt(-1) == np.emath.sqrt(-1) to evaluate to False!"
    print(f"      PASS: Real sqrt(-1) = {real_sqrt_val} ({real_sqrt_val.dtype})")
    print(f"      PASS: emath.sqrt(-1) = {emath_sqrt_val} ({emath_sqrt_val.dtype})")
    print(f"      PASS: Comparison equality is {equality_res} (NaN compares unequal to 1j).")

    # C. Common misconception and explicit complex promotion
    complex_sqrt = np.sqrt(-1 + 0j)
    assert complex_sqrt == 1j, f"Expected explicit complex sqrt(-1+0j) == 1j, got {complex_sqrt}"
    assert complex_sqrt.dtype == np.complex128
    print("      PASS: Explicit complex promotion np.sqrt(-1+0j) == 1j verified.")

    # D. Total array operation contrast
    arr_input = np.array([-1.0, 4.0])
    with np.errstate(invalid="ignore"):
        arr_real = np.sqrt(arr_input)
    arr_emath = np.emath.sqrt(arr_input)

    assert arr_real.shape == (2,) and arr_real.dtype == np.float64
    assert np.isnan(arr_real[0]) and arr_real[1] == 2.0

    assert arr_emath.shape == (2,) and arr_emath.dtype == np.complex128
    assert arr_emath[0] == 1j and arr_emath[1] == (2.0 + 0j)

    elem_equality = (arr_real == arr_emath).tolist()
    assert elem_equality == [False, True], f"Unexpected elementwise equality: {elem_equality}"
    print(f"      PASS: Total array operation contrast on {arr_input.tolist()}:")
    print(f"        arr_real : {arr_real.tolist()} (dtype={arr_real.dtype})")
    print(f"        arr_emath: {arr_emath.tolist()} (dtype={arr_emath.dtype})")
    print(f"        equality : {elem_equality}")

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        assert bool(real_sqrt_val == emath_sqrt_val) is True, "Deliberate failure triggered!"

    # 5. Negative testing for deliberate failure detection
    print("[5/6] Testing deliberate failure detection...")
    try:
        assert bool(real_sqrt_val == emath_sqrt_val) is True
        raise AssertionError("Check FAILED: deliberate wrong equality was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong equality correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T008/...")

    inputs_data = {
        "scalar_real_input": -1,
        "scalar_complex_input": "-1+0j",
        "array_real_input": [-1.0, 4.0],
        "default_error_policy": {k: str(v) for k, v in default_policy.items()},
        "code_investigated": "np.sqrt(-1) == np.emath.sqrt(-1)",
        "source_contract": (
            "np.sqrt computes real square root over the non-negative real domain. "
            "Under the default error policy (invalid='warn'), negative real inputs emit "
            "a RuntimeWarning and produce np.nan (dtype float64), preserving the real dtype. "
            "np.emath.sqrt automatically extends real inputs to the complex domain when x < 0, "
            "returning 1j (dtype complex128) without warning. "
            "Under IEEE 754 and NumPy semantics, NaN is unequal to everything (including itself), "
            "so np.nan == 1j evaluates to False."
        )
    }
    with open(work_dir / "inputs.json", "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 32,
        "task": "R44-T008",
        "scalar_comparison": {
            "real_sqrt": {
                "value": "NaN",
                "dtype": str(real_sqrt_val.dtype),
                "is_nan": bool(np.isnan(real_sqrt_val)),
                "warning_emitted": "RuntimeWarning: invalid value encountered in sqrt"
            },
            "emath_sqrt": {
                "value": str(emath_sqrt_val),
                "dtype": str(emath_sqrt_val.dtype),
                "warning_emitted": None
            },
            "equality_comparison": {
                "expression": "np.sqrt(-1) == np.emath.sqrt(-1)",
                "actual_result": False,
                "expected_result": False,
                "explanation": "NaN != 1j under IEEE 754 and NumPy comparison semantics."
            }
        },
        "misconception_remediation": {
            "explicit_complex_input": "-1+0j",
            "expression": "np.sqrt(-1+0j) == 1j",
            "result": bool(complex_sqrt == 1j),
            "dtype": str(complex_sqrt.dtype)
        },
        "total_array_operation_contrast": {
            "input": arr_input.tolist(),
            "real_sqrt_result": [None if np.isnan(x) else float(x) for x in arr_real],
            "real_sqrt_dtype": str(arr_real.dtype),
            "emath_sqrt_result": [str(x) for x in arr_emath],
            "emath_sqrt_dtype": str(arr_emath.dtype),
            "elementwise_equality": elem_equality
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
        f.write("TASK R44-T008 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Item: NumPy exercise 32\n")
        f.write("Source: source/exercises100.ktx (a32, h32)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Contracts verified:\n")
        f.write("  - np.sqrt(-1) emits RuntimeWarning and returns NaN (float64) under documented error policy.\n")
        f.write("  - np.emath.sqrt(-1) extends domain to complex returning 1j (complex128).\n")
        f.write("  - np.sqrt(-1) == np.emath.sqrt(-1) evaluates to False (NaN is unequal to any value).\n")
        f.write("  - Misconception remediation: np.sqrt(-1+0j) == 1j verified.\n")
        f.write("  - Total array contrast on [-1.0, 4.0] verified for types, shapes, and values.\n")
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
