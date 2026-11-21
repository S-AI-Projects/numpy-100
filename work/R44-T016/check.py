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
    
    # 1. Extract and execute published code
    a24_code = extract_ktx_block(ktx_path, 'a24')
    assert len(a24_code) > 0, "Failed to extract a24 block"
    
    ns = {'np': np}
    exec(a24_code, ns)
    assert 'Z' in ns, "Published code did not define Z"
    
    Z = ns['Z']
    
    # 2. Assert shape is (5, 2)
    assert Z.shape == (5, 2), f"Expected shape (5, 2), got {Z.shape}"
    
    # 3. Assert every entry is 3.0
    expected = np.full((5, 2), 3.0)
    npt.assert_allclose(Z, expected, err_msg="Z entries do not all equal 3.0")
    
    # Verify intermediate dot product contraction
    A = np.ones((5, 3))
    B = np.ones((3, 2))
    row_0 = A[0, :]
    col_0 = B[:, 0]
    intermediate_mult = row_0 * col_0
    intermediate_sum = np.sum(intermediate_mult)
    npt.assert_array_equal(intermediate_mult, np.array([1.0, 1.0, 1.0]))
    assert intermediate_sum == 3.0
    
    # Check both np.matmul and @ operator produce identical output
    Z_matmul = np.matmul(A, B)
    Z_at = A @ B
    npt.assert_array_equal(Z_matmul, Z_at)
    
    # 4. Deliberate failure check (e.g. asserting entries are 5)
    deliberate_failure_caught = False
    try:
        npt.assert_allclose(Z, np.full((5, 2), 5.0))
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 5. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T016",
        "exercise": 24,
        "A_shape": [5, 3],
        "B_shape": [3, 2],
        "result_shape": list(Z.shape),
        "expected_value": 3.0,
        "all_entries_equal_3": bool(np.all(Z == 3.0)),
        "intermediate_products": intermediate_mult.tolist(),
        "intermediate_sum": float(intermediate_sum),
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T016.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
