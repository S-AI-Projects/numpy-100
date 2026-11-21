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
    a81_code = extract_ktx_block(ktx_path, 'a81')
    assert len(a81_code) > 0, "Failed to extract a81 block"
    
    ns = {'np': np}
    exec(a81_code, ns)
    assert 'R' in ns and 'R_view' in ns, "Published code did not define R and R_view"
    
    R = ns['R']
    R_view = ns['R_view']
    
    # 2. Check shape (11, 4)
    assert R.shape == (11, 4), f"Expected shape (11, 4), got {R.shape}"
    assert R_view.shape == (11, 4), f"Expected shape (11, 4), got {R_view.shape}"
    
    # 3. Check first row [1, 2, 3, 4] and last row [11, 12, 13, 14]
    expected_first = np.array([1, 2, 3, 4], dtype=np.uint32)
    expected_last = np.array([11, 12, 13, 14], dtype=np.uint32)
    
    npt.assert_array_equal(R[0], expected_first, err_msg="R first row mismatch")
    npt.assert_array_equal(R[-1], expected_last, err_msg="R last row mismatch")
    npt.assert_array_equal(R_view[0], expected_first, err_msg="R_view first row mismatch")
    npt.assert_array_equal(R_view[-1], expected_last, err_msg="R_view last row mismatch")
    
    # Check both methods produce identical sliding window view
    npt.assert_array_equal(R, R_view, err_msg="R and R_view mismatch")
    
    # 4. Deliberate failure check (e.g. asserting wrong last row)
    deliberate_failure_caught = False
    try:
        npt.assert_array_equal(R[-1], np.array([10, 11, 12, 13], dtype=np.uint32))
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 5. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T015",
        "exercise": 81,
        "input_range": [1, 15],
        "window_shape": 4,
        "output_shape": list(R.shape),
        "first_row": R[0].tolist(),
        "last_row": R[-1].tolist(),
        "as_strided_equals_sliding_window_view": True,
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T015.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
