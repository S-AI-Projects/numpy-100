import re, subprocess, sys, numpy as np

ktx = open("source/exercises100.ktx", encoding="utf-8").read()
a1 = re.search(r"< a1\n(.*?)\n< q2", ktx, re.S).group(1).strip()

EXPECTED = ["numpy", "[0 1 2]", "[]", "(0,)"]

ns = {}
exec(a1, ns)
assert ns["np"] is np, "np is not the numpy module"
np.testing.assert_array_equal(np.arange(3), [0, 1, 2])

def run(code):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()

actual = run(a1)
assert actual == EXPECTED, f"expected {EXPECTED}, got {actual}"

wrong = run(a1.replace("np.arange(3)", "np.arange(4)"))
assert wrong != EXPECTED, "check did not catch the deliberate wrong result"

open("work/R44-T022/result.txt", "w").write(
    f"numpy {np.__version__}, python {sys.version.split()[0]}\n"
    f"expected: {EXPECTED}\nactual:   {actual}\nwrong:    {wrong}\n")
print("OK: published example matches, wrong version is caught")
