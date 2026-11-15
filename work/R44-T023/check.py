import re, subprocess, sys, io, contextlib, numpy as np

ktx = open("source/exercises100.ktx", encoding="utf-8").read()
a49 = re.search(r"< a49\n(.*?)\n< q50", ktx, re.S).group(1).strip()

HEAD = np.array2string(np.zeros((40, 40)), threshold=10**6)
TAIL = ["True", "False", "True", "True"]
EXPECTED = HEAD + "\n" + "\n".join(TAIL)

def run(code):
    r = subprocess.run([sys.executable, "-c", "import numpy as np\n" + code],
                       capture_output=True, text=True, check=True)
    return r.stdout.rstrip("\n")

actual = run(a49)
assert actual == EXPECTED, "published example output differs from expected"

wrong1 = run(a49.replace("method1 = str(Z)", "method1 = summarised"))
assert wrong1 != EXPECTED, "check missed a summarised (wrong) method 1"
wrong2 = run(a49.replace("np.set_printoptions(**old)", "pass"))
assert wrong2 != EXPECTED, "check missed a missing restore"

# Options must be restored even when the code fails halfway
before = np.get_printoptions()
ns = {"np": np}
try:
    with contextlib.redirect_stdout(io.StringIO()):
        exec(a49.replace("Z = np.arange(100)", "Z = np.arange(100); raise RuntimeError('boom')"), ns)
except RuntimeError:
    pass
assert np.get_printoptions() == before, "options not restored after a failure"

open("work/R44-T023/result.txt", "w").write(
    f"numpy {np.__version__}, python {sys.version.split()[0]}\n"
    f"tail expected: {TAIL}\nactual tail:   {actual.splitlines()[-4:]}\n"
    f"wrong1 tail:   {wrong1.splitlines()[-4:]}\nwrong2 tail:   {wrong2.splitlines()[-4:]}\n")
print("OK: published example matches, wrong versions caught, options restored on failure")
