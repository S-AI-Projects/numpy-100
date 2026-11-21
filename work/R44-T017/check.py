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
    a57_code = extract_ktx_block(ktx_path, 'a57')
    assert len(a57_code) > 0, "Failed to extract a57 block"
    
    ns = {'np': np}
    exec(a57_code, ns)
    assert 'Z' in ns, "Published code did not define Z"
    assert np.count_nonzero(ns['Z']) == 3, f"Expected 3 nonzero elements in original example, got {np.count_nonzero(ns['Z'])}"
    
    # 2. Place p=3 nonzero entries in 4x4 array without replacement using seed 42
    np.random.seed(42)
    n = 4
    p = 3
    Z_4x4 = np.zeros((n, n))
    selected_indices = np.random.choice(range(n * n), p, replace=False)
    np.put(Z_4x4, selected_indices, 1)
    
    nonzero_count = int(np.count_nonzero(Z_4x4))
    assert nonzero_count == 3, f"Expected exactly 3 nonzero entries, got {nonzero_count}"
    assert Z_4x4.shape == (4, 4), f"Expected shape (4, 4), got {Z_4x4.shape}"
    
    # 3. Reject p > 16 (sampling without replacement cannot exceed total population size 16)
    rejected_p_over_16 = False
    try:
        np.random.choice(range(16), 17, replace=False)
    except ValueError:
        rejected_p_over_16 = True
    assert rejected_p_over_16, "p > 16 without replacement did not raise ValueError as required"
    
    # 4. Deliberate failure check (e.g. asserting nonzero count is 5)
    deliberate_failure_caught = False
    try:
        assert nonzero_count == 5
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 5. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T017",
        "exercise": 57,
        "n": n,
        "p": p,
        "seed": 42,
        "selected_indices": selected_indices.tolist(),
        "resulting_array": Z_4x4.tolist(),
        "nonzero_count": nonzero_count,
        "rejected_p_over_16_with_value_error": rejected_p_over_16,
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T017.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
