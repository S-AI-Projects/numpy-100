import re, subprocess, sys, numpy as np

N = 76
ktx = open("source/exercises100.ktx", encoding="utf-8").read()
code = re.search(r"< a%d\n(.*?)\n+< q%d\n" % (N, N + 1), ktx, re.S).group(1)

HEAD = [eval(e) for e in ['str(np.lib.stride_tricks.sliding_window_view(np.arange(10), 3))', 'str(np.lib.stride_tricks.sliding_window_view(np.arange(10), 3))']]
TAIL = ['[[0, 1, 2], [1, 2, 3], [2, 3, 4]]', '(3, 3)', 'True', 'True', 'True']
SKIP_FIRST = False
MUTANTS = [('(a.size - window + 1, window)', '(a.size - window, window)'), ('R[i] == a[i:i + w]', 'R[i] == a[i + 1:i + w + 1]')]


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

with open("work/R44-T024/result.txt", "w") as f:
    f.write("numpy %s, python %s\n" % (np.__version__, sys.version.split()[0]))
    f.write("expected tail: %r\n" % (TAIL,))
    f.write("actual tail:   %r\n" % (actual[-len(TAIL):],))
    for old, new, mrc in caught:
        f.write("mutant %r -> %r caught (exit code %s)\n" % (old, new, mrc))
print("OK: published example matches, %d wrong version(s) caught" % len(caught))
