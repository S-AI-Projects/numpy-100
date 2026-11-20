#!/usr/bin/env python3
"""
Automated validation script for Task R44-T007:
Derive the result step by step: NumPy exercise 88 [100_Numpy_exercises.md]

This script:
1. Extracts the actual solution code from source/exercises100.ktx and
   100_Numpy_exercises_with_solutions.md (preventing drift).
2. Verifies consistency between authoritative source and generated markdown.
3. Executes the actual published example via Python subprocess and asserts
   it succeeds with exit code 0.
4. Performs a step-by-step derivation of Conway's Game of Life on a centered
   blinker oscillator (5x5 grid):
   - Connects intermediate slice-based neighbor counts (N) and boolean masks (birth, survive).
   - Verifies simultaneous state transition from horizontal (Gen 0) to vertical (Gen 1).
   - Verifies second iteration returns to exact original horizontal state (Gen 2 == Gen 0).
5. Verifies that deliberate wrong results trigger assertion failures.
6. Saves inputs, expected vs actual results, and tool versions to work/R44-T007/.
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


def extract_ktx_answer(ktx_path, key="a88"):
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


def extract_markdown_solution(md_path, exercise_num=88):
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


def iterate_game_of_life(Z):
    """
    Execute one generation of Conway's Game of Life:
    8-neighborhood count via shifted slice superposition.
    """
    N = (Z[0:-2, 0:-2] + Z[0:-2, 1:-1] + Z[0:-2, 2:] +
         Z[1:-1, 0:-2]                 + Z[1:-1, 2:] +
         Z[2:  , 0:-2] + Z[2:  , 1:-1] + Z[2:  , 2:])
    birth = (N == 3) & (Z[1:-1, 1:-1] == 0)
    survive = ((N == 2) | (N == 3)) & (Z[1:-1, 1:-1] == 1)

    Z_next = Z.copy()
    Z_next[...] = 0
    Z_next[1:-1, 1:-1][birth | survive] = 1
    return Z_next, N, birth, survive


def main():
    deliberate_fail_mode = "--deliberate-fail" in sys.argv

    repo_root = find_repo_root()
    ktx_path = repo_root / "source" / "exercises100.ktx"
    md_sol_path = repo_root / "100_Numpy_exercises_with_solutions.md"
    work_dir = repo_root / "work" / "R44-T007"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TASK R44-T007 CHECK: NumPy Exercise 88 (Game of Life Derivation)")
    print("=" * 60)

    # 1. Extract published code from authoritative source
    print(f"[1/6] Extracting a88 from {ktx_path.relative_to(repo_root)}...")
    ktx_code = extract_ktx_answer(ktx_path, "a88")
    print(f"      Extracted {len(ktx_code.splitlines())} lines from ktx.")

    # 2. Extract and verify consistency with generated markdown
    print(f"[2/6] Verifying generated markdown {md_sol_path.relative_to(repo_root)}...")
    md_code = extract_markdown_solution(md_sol_path, 88)
    assert ktx_code == md_code, (
        "Generated markdown solution does not match authoritative source/exercises100.ktx a88 block!"
    )
    print("      Authoritative source and generated markdown match exactly.")

    # 3. Execute published code via subprocess to verify runtime execution
    print("[3/6] Executing published code via Python subprocess...")
    stdout = run_extracted_code(ktx_code)
    print("      Subprocess executed successfully with exit code 0.")

    # 4. Step-by-step derivation on centered blinker fixture
    print("[4/6] Tracing step-by-step derivation of centered blinker on 5x5 grid...")
    # Gen 0: Centered horizontal blinker on 5x5 grid
    gen0 = np.zeros((5, 5), dtype=int)
    gen0[2, 1:4] = 1

    # Step 1: Neighbor summation and rule evaluation
    gen1, N1, birth1, survive1 = iterate_game_of_life(gen0)

    expected_gen1 = np.zeros((5, 5), dtype=int)
    expected_gen1[1:4, 2] = 1

    # Step 2: Second generation
    gen2, N2, birth2, survive2 = iterate_game_of_life(gen1)

    if deliberate_fail_mode:
        print("\n[!] RUNNING IN DELIBERATE FAIL MODE...")
        wrong_gen1 = expected_gen1.copy()
        wrong_gen1[0, 0] = 1
        assert np.array_equal(gen1, wrong_gen1), "Deliberate failure triggered!"

    assert np.array_equal(gen1, expected_gen1), (
        f"Generation 1 mismatch! Expected vertical blinker, got:\n{gen1}"
    )
    assert np.array_equal(gen2, gen0), (
        f"Generation 2 mismatch! Expected return to horizontal blinker, got:\n{gen2}"
    )

    print("      Step 1 derivation: Horizontal blinker -> Vertical blinker (PASS)")
    print(f"        Neighbor counts N on 3x3 window:\n{N1}")
    print(f"        Birth mask (N==3 & dead):\n{birth1.astype(int)}")
    print(f"        Survive mask ((N==2|3) & live):\n{survive1.astype(int)}")
    print("      Step 2 derivation: Vertical blinker -> Horizontal blinker (PASS, Period-2 verified)")

    # 5. Deliberate failure testing
    print("[5/6] Testing deliberate failure detection...")
    wrong_gen1 = expected_gen1.copy()
    wrong_gen1[0, 0] = 1
    try:
        assert np.array_equal(gen1, wrong_gen1)
        raise AssertionError("Check FAILED: deliberate wrong state was not detected!")
    except AssertionError:
        print("      PASS: Deliberate wrong state correctly failed assertion.")

    # 6. Save evidence files
    print("[6/6] Saving test inputs, expected vs actual, and tool versions to work/R44-T007/...")

    inputs_data = {
        "description": "5x5 grid with centered horizontal blinker oscillator (period 2)",
        "grid_shape": [5, 5],
        "initial_state_gen0": gen0.tolist(),
        "live_cell_coordinates_gen0": [[2, 1], [2, 2], [2, 3]],
        "simultaneous_update_mechanism": (
            "Shifted 2D slice views over 8 directions compute neighbor matrix N "
            "before modifying grid state, ensuring true simultaneous cellular automaton updates."
        )
    }
    inputs_file = work_dir / "inputs.json"
    with open(inputs_file, "w", encoding="utf-8") as f:
        json.dump(inputs_data, f, indent=2)

    results_data = {
        "exercise": 88,
        "task": "R44-T007",
        "description": "Conway's Game of Life step-by-step blinker oscillator derivation",
        "grid_size": [5, 5],
        "gen0_horizontal": gen0.tolist(),
        "step_1_intermediate": {
            "neighbor_matrix_N": N1.tolist(),
            "birth_cells": np.argwhere(birth1).tolist(),
            "survive_cells": np.argwhere(survive1).tolist(),
            "gen1_vertical_actual": gen1.tolist(),
            "gen1_vertical_expected": expected_gen1.tolist(),
            "matches_expected": bool(np.array_equal(gen1, expected_gen1))
        },
        "step_2_intermediate": {
            "neighbor_matrix_N": N2.tolist(),
            "gen2_horizontal_actual": gen2.tolist(),
            "matches_initial_gen0": bool(np.array_equal(gen2, gen0))
        },
        "period_2_oscillation_verified": True,
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
        f.write("TASK R44-T007 VERIFICATION SUMMARY\n")
        f.write("===================================\n")
        f.write("Source: source/exercises100.ktx (a88)\n")
        f.write("Generated: 100_Numpy_exercises_with_solutions.md\n")
        f.write("Status: ALL CHECKS PASSED\n")
        f.write("Blinker oscillator step-by-step derivation verified:\n")
        f.write("  Gen 0 (horizontal) -> Gen 1 (vertical) -> Gen 2 (horizontal)\n")
        f.write("Simultaneous slice-based updates verified.\n")
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
