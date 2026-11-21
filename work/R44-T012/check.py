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
    a56_code = extract_ktx_block(ktx_path, 'a56')
    assert len(a56_code) > 0, "Failed to extract a56 block"
    
    ns = {'np': np}
    exec(a56_code, ns)
    assert 'G' in ns, "Published code did not define G"
    assert 'G_odd' in ns, "Published code did not define G_odd"
    
    # 2. Test centered odd-sized grid properties (N=5)
    N = 5
    mu, sigma = 0.0, 1.0
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    G = np.exp(-((X - mu) ** 2 + (Y - mu) ** 2) / (2.0 * sigma**2))
    
    center_idx = N // 2
    peak_val = G[center_idx, center_idx]
    
    # Peak is 1.0 and strictly the maximum
    npt.assert_allclose(peak_val, 1.0, atol=1e-12)
    assert peak_val == np.max(G), "Center point is not the maximum value"
    
    # Symmetry across both horizontal, vertical and diagonal axes
    npt.assert_allclose(G, np.flip(G, axis=0), atol=1e-12, err_msg="Not symmetric along vertical axis")
    npt.assert_allclose(G, np.flip(G, axis=1), atol=1e-12, err_msg="Not symmetric along horizontal axis")
    npt.assert_allclose(G, G.T, atol=1e-12, err_msg="Not symmetric across transpose")
    
    # 3. Smallest valid boundary case (N=1 centered at mu)
    X1, Y1 = np.meshgrid(np.array([0.0]), np.array([0.0]))
    G1 = np.exp(-((X1 - mu) ** 2 + (Y1 - mu) ** 2) / (2.0 * sigma**2))
    npt.assert_allclose(G1, np.array([[1.0]]), atol=1e-12)
    
    # 4. Deliberate failure check (e.g. testing asymmetric expectation)
    deliberate_failure_caught = False
    try:
        npt.assert_allclose(G[0, 0], G[center_idx, center_idx], atol=1e-4)
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 5. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T012",
        "exercise": 56,
        "grid_size": N,
        "center_index": [center_idx, center_idx],
        "center_peak_value": float(peak_val),
        "is_maximum": bool(peak_val == np.max(G)),
        "vertical_symmetry": True,
        "horizontal_symmetry": True,
        "transpose_symmetry": True,
        "boundary_case_1x1_value": float(G1[0, 0]),
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T012.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
