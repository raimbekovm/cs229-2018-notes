"""Pure-Python linear algebra for the chapter figure scripts.

numpy is not a dependency of this book, and the generators only ever handle
matrices of a few rows, so lists of lists are enough. Matrices are lists of
rows, vectors are flat lists. Chapters 14 and 15 each wrote their own copy of
these before they were collected here; anything chapter-specific (a model's
E-step, a log-density) stays in the chapter.

Reach it the same way as `_svg`:

    import pathlib, sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from _linalg import T, chol, eigh, inv, mm
"""

import math


def zeros(r, c):
    """An r-by-c zero matrix."""
    return [[0.0] * c for _ in range(r)]


def eye(n):
    """The n-by-n identity."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def T(A):
    """Transpose."""
    return [list(r) for r in zip(*A)]


def mm(A, B):
    """Matrix product."""
    Bt = T(B)
    return [[sum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def mv(A, v):
    """Matrix-vector product."""
    return [sum(a * b for a, b in zip(r, v)) for r in A]


def add(A, B, s=1.0):
    """A + s B."""
    return [[a + s * b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def outer(u, v):
    """U v^T."""
    return [[a * b for b in v] for a in u]


def chol(A):
    """Lower Cholesky factor of a symmetric positive-definite matrix."""
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError("not positive definite")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def inv(A):
    """Inverse of a symmetric positive-definite matrix, through its Cholesky factor."""
    n = len(A)
    L = chol(A)
    Li = zeros(n, n)
    for c in range(n):
        for i in range(n):
            s = (1.0 if i == c else 0.0) - sum(L[i][k] * Li[k][c] for k in range(i))
            Li[i][c] = s / L[i][i]
    return mm(T(Li), Li)


def rank(A, tol=1e-9):
    """Numerical rank by Gaussian elimination with partial pivoting."""
    M = [r[:] for r in A]
    n, m = len(M), len(M[0])
    scale = max(abs(x) for r in M for x in r) or 1.0
    r = 0
    for c in range(m):
        p = max(range(r, n), key=lambda i: abs(M[i][c]), default=None)
        if p is None or abs(M[p][c]) < tol * scale:
            continue
        M[r], M[p] = M[p], M[r]
        for i in range(r + 1, n):
            f = M[i][c] / M[r][c]
            M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
        if r == n:
            break
    return r


def mean(X):
    """Column means."""
    return [sum(c) / len(X) for c in zip(*X)]


def eigh(A, sweeps=60):
    """Eigenvalues and unit eigenvectors of a symmetric matrix, largest first (cyclic Jacobi)."""
    n = len(A)
    A = [r[:] for r in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-30:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                th = 0.5 * math.atan2(2 * A[p][q], A[q][q] - A[p][p])
                c, s = math.cos(th), math.sin(th)
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    pairs = sorted(((A[i][i], [V[k][i] for k in range(n)]) for i in range(n)), key=lambda t: -t[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def dot(u, v):
    """Inner product."""
    return sum(a * b for a, b in zip(u, v))
