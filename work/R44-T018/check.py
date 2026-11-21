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
    a69_code = extract_ktx_block(ktx_path, 'a69')
    assert len(a69_code) > 0, "Failed to extract a69 block"
    
    ns = {'np': np}
    exec(a69_code, ns)
    assert 'A' in ns and 'B' in ns, "Published code did not define A and B"
    
    # 2. Test concrete fixture: A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # Method 1: slow diag(dot(A, B))
    d1 = np.diag(np.dot(A, B))
    # Method 2: sum(A * B.T, axis=1)
    d2 = np.sum(A * B.T, axis=1)
    # Method 3: einsum
    d3 = np.einsum("ij,ji->i", A, B)
    
    expected_diag = np.array([19, 50])
    npt.assert_array_equal(d1, expected_diag, err_msg="diag(dot(A, B)) mismatch")
    npt.assert_array_equal(d2, expected_diag, err_msg="sum(A * B.T, axis=1) mismatch")
    npt.assert_array_equal(d3, expected_diag, err_msg="einsum mismatch")
    
    # 3. Common misconception: A * B without transposing B
    misconception_result = np.sum(A * B, axis=1)
    # misconception gives [1*5 + 2*6, 3*7 + 4*8] = [17, 53] which is incorrect
    npt.assert_array_equal(misconception_result, np.array([17, 53]))
    assert not np.array_equal(misconception_result, expected_diag), "Misconception should not equal correct diagonal"
    
    # 4. Empty-result / shape contrast: (0, 2) and (2, 0)
    A_empty = np.zeros((0, 2))
    B_empty = np.zeros((2, 0))
    d_empty = np.diag(np.dot(A_empty, B_empty))
    assert d_empty.shape == (0,), f"Expected shape (0,), got {d_empty.shape}"
    
    # 5. Deliberate failure check (e.g. asserting [19, 50] equals [20, 50])
    deliberate_failure_caught = False
    try:
        npt.assert_array_equal(d1, np.array([20, 50]))
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 6. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T018",
        "exercise": 69,
        "A": A.tolist(),
        "B": B.tolist(),
        "dot_product": np.dot(A, B).tolist(),
        "expected_diagonal": expected_diag.tolist(),
        "method1_diag_dot": d1.tolist(),
        "method2_sum_multiply_transpose": d2.tolist(),
        "method3_einsum": d3.tolist(),
        "misconception_sum_without_transpose": misconception_result.tolist(),
        "empty_result_shape": list(d_empty.shape),
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T018.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
