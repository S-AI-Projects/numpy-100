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
    a10_code = extract_ktx_block(ktx_path, 'a10')
    assert len(a10_code) > 0, "Failed to extract a10 block"
    
    ns = {'np': np}
    exec(a10_code, ns)
    assert 'nz' in ns, "Published code did not define nz"
    npt.assert_array_equal(ns['nz'][0], np.array([0, 1, 4]))
    
    # 2. Test fixture: [1, 2, 0, 0, 4, 0]
    input_arr = np.array([1, 2, 0, 0, 4, 0])
    nz = np.nonzero(input_arr)
    expected_indices = np.array([0, 1, 4])
    npt.assert_array_equal(nz[0], expected_indices, err_msg="Nonzero indices mismatch for [1,2,0,0,4,0]")
    
    # 3. Test all-zero vector: has no nonzero indices
    zero_arr = np.array([0, 0, 0, 0, 0, 0])
    nz_zero = np.nonzero(zero_arr)
    assert len(nz_zero[0]) == 0, f"Expected 0 nonzero indices for all-zero array, got {len(nz_zero[0])}"
    npt.assert_array_equal(nz_zero[0], np.array([], dtype=np.int64))
    
    # 4. Copy vs View / Memory isolation test
    # Verify that the returned index array is a newly allocated copy and does not share memory
    assert not np.shares_memory(input_arr, nz[0]), "np.nonzero output shares memory with input"
    
    # 5. Deliberate failure check (e.g. asserting nonzero indices include 2 or 3)
    deliberate_failure_caught = False
    try:
        npt.assert_array_equal(nz[0], np.array([0, 1, 2, 4]))
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 6. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T019",
        "exercise": 10,
        "input": input_arr.tolist(),
        "nonzero_indices": nz[0].tolist(),
        "zero_array": zero_arr.tolist(),
        "zero_array_nonzero_indices": nz_zero[0].tolist(),
        "shares_memory_with_input": bool(np.shares_memory(input_arr, nz[0])),
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T019.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
