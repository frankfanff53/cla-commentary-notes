import numpy as np
from numpy import random, linalg
import timeit


def orthog_cpts(v, Q):
    u = np.conj(Q).T @ v
    r = v - Q @ u
    return r, u


def solveQ(Q, b):
    # Qx = b => Q^*Qx = Q^*b => x = Q^*b since Q unitary.
    return np.conj(Q).T @ b


import timeit
from numpy import random, linalg
def time_solveQ():
    """
    Get some timings for solveQ.
    """
    for size in [100, 200, 400]:
        Q = random.randn(size, size)
        b = random.randn(size)
        print("--- Input matrix size n = {} ---".format(size))
        print("Timing for solveQ")
        print(timeit.Timer(lambda: solveQ(Q, b)).timeit(number=1))
        print("Timing for general purpose solve")
        print(timeit.Timer(lambda: linalg.solve(Q, b)).timeit(number=1))
        print("--- End for testing matrix with n = {} ---".format(size))


def orthog_proj(Q):
    return Q @ np.conj(Q).T


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.
    """
    _, n = V.shape
    # Get Q unitary from comoplete qr factorisation.
    Q, _ = np.linalg.qr(V, 'complete')

    # Then the subspace orthogonal to U should be spanned
    # by the remaining m - n col vectors by mutal orthogonality.
    return Q[:, n:]
