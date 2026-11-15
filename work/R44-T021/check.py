import re, subprocess, sys, numpy as np

ktx = open("source/exercises100.ktx").read()
a60 = re.search(r"< a60\n(.*?)\n< ", ktx, re.S).group(1)

EXPECTED = ["[ True False  True]", "[False  True False]", "[1]", "True",
            "[1]", "[1 2]", "[False False  True]"]

def run(code):
    full = "import numpy as np\nnp.random.seed(0)\n" + code
    r = subprocess.run([sys.executable, "-c", full], capture_output=True, text=True, check=True)
    return r.stdout.strip().splitlines()

actual = run(a60)[1:]          # line 0 is the random example
assert actual == EXPECTED, f"expected {EXPECTED}, got {actual}"

wrong = run(a60.replace("~Z.any(axis=0)", "~Z.all(axis=0)"))[1:]
assert wrong != EXPECTED, "check did not catch the deliberate wrong result"

open("work/R44-T021/result.txt", "w").write(
    f"numpy {np.__version__}, python {sys.version.split()[0]}\n"
    f"expected: {EXPECTED}\nactual:   {actual}\nwrong:    {wrong}\n")
print("OK: published example matches, wrong version is caught")
