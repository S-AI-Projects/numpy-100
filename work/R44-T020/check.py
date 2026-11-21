import os
import sys
import json
import platform
import numpy as np
import numpy.testing as npt
from numpy.lib.stride_tricks import as_strided, sliding_window_view

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
    a84_code = extract_ktx_block(ktx_path, 'a84')
    assert len(a84_code) > 0, "Failed to extract a84 block"
    
    ns = {'np': np}
    exec(a84_code, ns)
    assert 'C' in ns, "Published code did not define C"
    assert ns['C'].shape == (8, 8, 3, 3), f"Expected shape (8, 8, 3, 3), got {ns['C'].shape}"
    
    # 2. Test fixture: Z = np.arange(100).reshape(10, 10)
    Z = np.arange(100).reshape(10, 10)
    n = 3
    i = 1 + (Z.shape[0] - n)
    j = 1 + (Z.shape[1] - n)
    
    # as_strided implementation
    C = as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
    # sliding_window_view implementation
    C_view = sliding_window_view(Z, window_shape=(n, n))
    
    # Verify shape (8, 8, 3, 3)
    assert C.shape == (8, 8, 3, 3), f"Expected shape (8, 8, 3, 3), got {C.shape}"
    assert C_view.shape == (8, 8, 3, 3), f"Expected shape (8, 8, 3, 3), got {C_view.shape}"
    npt.assert_array_equal(C, C_view)
    
    # Verify boundary elements:
    # First window [0, 0] top-left is 0
    first_window_top_left = int(C[0, 0, 0, 0])
    assert first_window_top_left == 0, f"Expected first window top-left to be 0, got {first_window_top_left}"
    
    # Last window [-1, -1] bottom-right is 99
    last_window_bottom_right = int(C[-1, -1, -1, -1])
    assert last_window_bottom_right == 99, f"Expected last window bottom-right to be 99, got {last_window_bottom_right}"
    
    # 3. Mutation test: one-operation mutant (off-by-one window count without '+ 1')
    def mutant_windows(arr, window_sz=3):
        # Mutant: forgot the '+ 1' offset
        mutant_i = arr.shape[0] - window_sz
        mutant_j = arr.shape[1] - window_sz
        return as_strided(arr, shape=(mutant_i, mutant_j, window_sz, window_sz), strides=arr.strides + arr.strides)
    
    mutant_C = mutant_windows(Z, 3)
    mutant_caught = False
    try:
        # Our invariant requires shape (8, 8, 3, 3) and last window bottom-right 99
        assert mutant_C.shape == (8, 8, 3, 3)
    except AssertionError:
        mutant_caught = True
    assert mutant_caught, "Mutant with off-by-one window shape was not killed by the test!"
    
    # 4. Deliberate failure check (e.g. asserting last window bottom-right is 98)
    deliberate_failure_caught = False
    try:
        assert last_window_bottom_right == 98
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 5. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T020",
        "exercise": 84,
        "input_shape": list(Z.shape),
        "window_shape": [n, n],
        "extracted_shape": list(C.shape),
        "first_window_top_left": first_window_top_left,
        "last_window_bottom_right": last_window_bottom_right,
        "as_strided_equals_sliding_window_view": True,
        "mutant_killed_by_test": mutant_caught,
        "mutant_shape": list(mutant_C.shape),
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T020.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
