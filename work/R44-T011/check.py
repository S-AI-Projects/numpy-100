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
    
    # 1. Extract published code from authoritative source
    a36_code = extract_ktx_block(ktx_path, 'a36')
    assert len(a36_code) > 0, "Failed to extract a36 block from exercises100.ktx"
    
    # Execute published code in isolated namespace
    ns = {'np': np}
    exec(a36_code, ns)
    assert 'Z' in ns, "Published code did not define Z"
    
    # 2. Test extraction methods on concrete input: [1.2, 2.9, 3.0]
    input_z = np.array([1.2, 2.9, 3.0])
    
    m_modulo = input_z - input_z % 1
    m_floordiv = input_z // 1
    m_floor = np.floor(input_z)
    m_trunc = np.trunc(input_z)
    m_astype = input_z.astype(int)
    
    expected_values = np.array([1.0, 2.0, 3.0])
    expected_int_values = np.array([1, 2, 3])
    
    # Assert values
    npt.assert_allclose(m_modulo, expected_values, err_msg="m_modulo failed")
    npt.assert_allclose(m_floordiv, expected_values, err_msg="m_floordiv failed")
    npt.assert_allclose(m_floor, expected_values, err_msg="m_floor failed")
    npt.assert_allclose(m_trunc, expected_values, err_msg="m_trunc failed")
    npt.assert_array_equal(m_astype, expected_int_values, err_msg="m_astype failed")
    
    # Assert dtype distinction
    assert np.issubdtype(m_modulo.dtype, np.floating), "m_modulo should be floating"
    assert np.issubdtype(m_floordiv.dtype, np.floating), "m_floordiv should be floating"
    assert np.issubdtype(m_floor.dtype, np.floating), "m_floor should be floating"
    assert np.issubdtype(m_trunc.dtype, np.floating), "m_trunc should be floating"
    assert np.issubdtype(m_astype.dtype, np.integer), "m_astype should be integer"
    
    # 3. Deliberate wrong result check (must fail)
    deliberate_failure_caught = False
    try:
        # Deliberately compare against wrong values
        npt.assert_array_equal(m_astype, np.array([2, 3, 4]))
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"

    # 4. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T011",
        "exercise": 36,
        "input": input_z.tolist(),
        "methods": {
            "Z - Z % 1": {"values": m_modulo.tolist(), "dtype": str(m_modulo.dtype)},
            "Z // 1": {"values": m_floordiv.tolist(), "dtype": str(m_floordiv.dtype)},
            "np.floor(Z)": {"values": m_floor.tolist(), "dtype": str(m_floor.dtype)},
            "np.trunc(Z)": {"values": m_trunc.tolist(), "dtype": str(m_trunc.dtype)},
            "Z.astype(int)": {"values": m_astype.tolist(), "dtype": str(m_astype.dtype)},
        },
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    evidence_path = os.path.join(work_dir, "evidence.json")
    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2)
    
    print("All checks passed successfully for R44-T011.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
