import re, subprocess, sys, numpy as np

N = 65
ktx = open("source/exercises100.ktx", encoding="utf-8").read()
code = re.search(r"< a%d\n(.*?)\n+< q%d\n" % (N, N + 1), ktx, re.S).group(1)

TAIL = ['[5, 4]', '[5, 4]']
MUTANTS = [('np.add.at(F_addat, I_fix, X_fix)', 'F_addat[I_fix] += X_fix')]

def run(src):
    r = subprocess.run([sys.executable, "-c", "import numpy as np\nnp.random.seed(0)\n" + src],
                       capture_output=True, text=True)
    lines = r.stdout.rstrip("\n").split("\n")
    return r.returncode, lines, r.stderr

rc, actual, err = run(code)
assert rc == 0, "published example failed:\n" + err
assert actual[-len(TAIL):] == TAIL, "published output differs.\nexpected tail: %r\nactual tail:   %r" % (TAIL, actual[-len(TAIL):])

caught = []
for old, new in MUTANTS:
    assert old in code, "mutant target not found in code: %r" % old
    mrc, mout, _ = run(code.replace(old, new, 1))
    assert mrc != 0 or mout[-len(TAIL):] != TAIL, "mutant NOT caught: %r -> %r" % (old, new)
    caught.append((old, new, mrc))

with open("work/R44-T033/result.txt", "w") as f:
    f.write("numpy %s, python %s\n" % (np.__version__, sys.version.split()[0]))
    f.write("expected tail: %r\n" % (TAIL,))
    f.write("actual tail:   %r\n" % (actual[-len(TAIL):],))
    for old, new, mrc in caught:
        f.write("mutant %r -> %r caught (exit code %s)\n" % (old, new, mrc))
print("OK: published example matches, %d wrong version(s) caught" % len(caught))
