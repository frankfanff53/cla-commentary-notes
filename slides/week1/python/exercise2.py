import numpy as np
from numpy import random, linalg
import timeit


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors
    q_1,...q_n, compute v = r + u_1q_1 +  ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r.
    """
    r, u = v.copy(), np.array([])

    for i in range(len(Q[0])):
        qi = Q[:, i]
        # Find the scale factor of q_i in v,
        # and remove the orthogonal component.
        ui = np.conj(qi).dot(v)
        r -= ui * qi
        u = np.append(u, [ui])

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b,
    solve Qx=b for x.
    """
    return np.conj(Q).T.dot(b)


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
    """
    Given a vector v and an orthonormal set of vectors
    q_1,...q_n, compute the orthogonal projector P
    that projects vectors onto the subspace
    spanned by those vectors.
    """
    return Q.dot(np.conj(Q).T)


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
