import re, subprocess, sys, numpy as np

N = 50
ktx = open("source/exercises100.ktx", encoding="utf-8").read()
code = re.search(r"< a%d\n(.*?)\n+< q%d\n" % (N, N + 1), ktx, re.S).group(1)

HEAD = [eval(e) for e in []]
TAIL = ['[-4 -1  3]', '[4 1 3]', '1', '4', '[5 2 2]', '4']
SKIP_FIRST = True
MUTANTS = [('(np.abs(Z - v)).argmin()', '(Z - v).argmin()'), ('.argmin()', '.argmax()')]


def run(src):
    r = subprocess.run([sys.executable, "-c", "import numpy as np\nnp.random.seed(0)\n" + src],
                       capture_output=True, text=True)
    lines = r.stdout.rstrip("\n").split("\n")
    return r.returncode, (lines[1:] if SKIP_FIRST else lines), r.stderr


EXPECTED = ("\n".join(HEAD).split("\n") if HEAD else []) + TAIL

rc, actual, err = run(code)
assert rc == 0, "published example failed:\n" + err
assert actual == EXPECTED, "published output differs.\nexpected: %r\nactual:   %r" % (EXPECTED, actual)

caught = []
for old, new in MUTANTS:
    assert old in code, "mutant target not found in the example: %r" % old
    mrc, mout, _ = run(code.replace(old, new))
    assert mrc != 0 or mout != EXPECTED, "mutant NOT caught: %r -> %r" % (old, new)
    caught.append((old, new, mrc))

with open("work/R44-T027/result.txt", "w") as f:
    f.write("numpy %s, python %s\n" % (np.__version__, sys.version.split()[0]))
    f.write("expected tail: %r\n" % (TAIL,))
    f.write("actual tail:   %r\n" % (actual[-len(TAIL):],))
    for old, new, mrc in caught:
        f.write("mutant %r -> %r caught (exit code %s)\n" % (old, new, mrc))
print("OK: published example matches, %d wrong version(s) caught" % len(caught))
