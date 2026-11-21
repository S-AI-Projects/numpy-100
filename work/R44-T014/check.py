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
    a12_code = extract_ktx_block(ktx_path, 'a12')
    assert len(a12_code) > 0, "Failed to extract a12 block"
    
    ns = {'np': np}
    exec(a12_code, ns)
    assert 'Z' in ns, "Published code did not define Z"
    assert ns['Z'].shape == (3, 3, 3), "Z shape is not (3,3,3)"
    assert np.all((ns['Z'] >= 0.0) & (ns['Z'] < 1.0)), "Z values outside [0, 1)"
    
    # 2. Test reproducibility invariant with default_rng(42) on (3,3,3)
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    z1 = rng1.random((3, 3, 3))
    z2 = rng2.random((3, 3, 3))
    
    npt.assert_array_equal(z1, z2, err_msg="Two fresh default_rng(42) did not produce identical arrays")
    assert np.all((z1 >= 0.0) & (z1 < 1.0)), "Array contains values outside [0, 1)"
    
    # 3. Test on at least 5 bounded variations where domain permits
    variations = [
        (1,),
        (2, 2),
        (4, 3, 2),
        (1, 5, 1, 2),
        (5, 5),
        (0,)
    ]
    variation_results = []
    for var_shape in variations:
        var_arr = rng1.random(var_shape)
        assert var_arr.shape == var_shape, f"Shape mismatch for variation {var_shape}"
        if var_arr.size > 0:
            assert np.all((var_arr >= 0.0) & (var_arr < 1.0)), f"Values out of range [0, 1) for shape {var_shape}"
        variation_results.append({
            "shape": list(var_shape),
            "size": var_arr.size,
            "min": float(var_arr.min()) if var_arr.size > 0 else None,
            "max": float(var_arr.max()) if var_arr.size > 0 else None,
            "all_in_range": True
        })
        
    # 4. Deliberate failure check (e.g., comparing output of two different seeds)
    deliberate_failure_caught = False
    try:
        rng_diff = np.random.default_rng(999)
        z_diff = rng_diff.random((3, 3, 3))
        npt.assert_array_equal(z1, z_diff)
    except AssertionError:
        deliberate_failure_caught = True
    assert deliberate_failure_caught, "Deliberate failure assertion was not caught!"
    
    # 5. Save evidence
    work_dir = os.path.dirname(os.path.abspath(__file__))
    evidence = {
        "task_id": "R44-T014",
        "exercise": 12,
        "seed": 42,
        "fixture_shape": [3, 3, 3],
        "fixture_min": float(z1.min()),
        "fixture_max": float(z1.max()),
        "reproducibility_verified": True,
        "all_fixture_values_in_half_open_interval": True,
        "variations_tested": variation_results,
        "deliberate_wrong_result_failed_as_expected": deliberate_failure_caught,
        "tool_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform()
        }
    }
    with open(os.path.join(work_dir, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
        
    print("All checks passed successfully for R44-T014.")
    return 0

if __name__ == '__main__':
    sys.exit(run_checks())
