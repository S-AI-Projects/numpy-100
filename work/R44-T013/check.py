import os
import sys
import json
import platform
import numpy as np
import numpy.testing as npt

def extract_ktx_block(file_path, block_key):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    collecting = False
    block_lines = []
    target = f"< {block_key}"
    for line in lines:
        if line.strip().startswith('<'):
            if line.strip() == target:
                collecting = True
                continue
            elif collecting:
                break
        elif collecting:
            block_lines.append(line)
    return "".join(block_lines).strip()

def run_checks():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ktx_path = os.path.join(repo_root, 'source', 'exercises100.ktx')
    
    # 1. Extract published code
    a99_code = extract_ktx_block(ktx_path, 'a99')
    assert len(a99_code) > 0, "Failed to extract a99 block"
    
    ns = {'np': np}
    exec(a99_code, ns)
    assert 'M' in ns and 'M_alt' in ns, "Published code did not define M and M_alt"
    
    # Verify original example outputs
    X_orig = ns['X']
    npt.assert_array_equal(X_orig[ns['M']], np.array([[2.0, 0.0, 1.0, 1.0]]))
    npt.assert_array_equal(X_orig[ns['M_alt']], np.array([[2.0, 0.0, 1.0, 1.0]]))
    
    # 2. Test fixture for n = 3
    # Rows:
    # [1, 2, 0] -> valid (sum=3, non-negative integers)
    # [0, 0, 3] -> valid (sum=3, non-negative integers)
    # [1, 1, 0] -> invalid (sum=2 != 3)
    # [-1, 4, 0] -> invalid (negative count: draw count cannot be negative)
    # [1.5, 0.5, 1.0] -> invalid (non-integral counts)
    n = 3
    X_fixture = np.asarray([
        [1.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 1.0, 0.0],
        [-1.0, 4.0, 0.0],
        [1.5, 0.5, 1.0]
    ])
    
    # Method 1 (Source method with non-negative check)
    m1 = np.logical_and.reduce(np.mod(X_fixture, 1) == 0, axis=-1)
    m1 &= np.all(X_fixture >= 0, axis=-1)
    m1 &= (X_fixture.sum(axis=-1) == n)
    
    # Method 2 (Alternative using np.floor == X and np.all)
    m2 = np.all(np.floor(X_fixture) == X_fixture, axis=-1)
    m2 &= np.all(X_fixture >= 0, axis=-1)
    m2 &= (X_fixture.sum(axis=-1) == n)
    
    expected_mask = np.array([True, True, False, False, False])
    npt.assert_array_equal(m1, expected_mask, err_msg="Method 1 mask does not match expected")
    npt.assert_array_equal(m2, expected_mask, err_msg="Method 2 mask does not match expected")
    
    expected_selected = np.asarray([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    npt.assert_array_equal(X_fixture[m1], expected_selected)
    npt.assert_array_equal(X_fixture[m2], expected_selected)
    
    # 3. Deliberate failure check (e.g. accepting invalid negative row)
    deliberate_failure_caught = False
    try:
        # Deliberately assert that [-1, 4, 0] is accepted
        npt.assert_array_equal(m1, np.array([True, True, False, True, False]))
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 4. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T013",
        "exercise": 99,
        "n_degrees": n,
        "fixture": X_fixture.tolist(),
        "expected_mask": expected_mask.tolist(),
        "method1_mask": m1.tolist(),
        "method2_mask": m2.tolist(),
        "valid_rows_selected": expected_selected.tolist(),
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T013.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
