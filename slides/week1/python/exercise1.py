import numpy as np
def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.
    ...
    :return b: m-dimensional numpy array
    """
    b = []
    for i in range(m):
        # Compute the value of b_i.
        for j in range(n):
            # Apply the basic formula in summation.
        # Add the computed b_i value to b.
    return np.array(b)


def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.
    ...
    :return b: m-dimensional numpy array
    """
    b = []
    m, n = A.shape
    for i in range(m):
        # Compute the value of b_i.
        bi = 0
        for j in range(n):
            # Apply the basic formula in summation.
            bi += A[i, j] * x[j]
        # Add the computed b_i value to b.
        b.append(bi)
    return np.array(b)


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product.
    ...
    :return b: an m-dimensional numpy array which is the product of A with x
    """
    b = np.zeros(len(A))
    for i, elem in enumerate(x):
        # Add the products up as a vector sum to the result.
        b += A[:, i] * elem
    return b


basic_matvec(A0, x0) # double-nested loop implementation

column_matvec(A0, x0) # single loop implementation

A0.dot(x0) # numpy implementation


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*v1^* + u2*v2^*.
    ...
    """
    # Construct the matrix from column vectors u1 and u2.
    # Using transpose is because matrix is constructed from rows by default.
    B = np.array([u1, u2]).T
    # Construct the conjugate matrix formed by v1 and v2.
    C = np.conj(np.array([v1, v2]))

    A = B.dot(C)

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where ...
    """
    # A^-1 = I - (uv^*)/(1 + v^*u)
    return np.eye(len(u)) - (np.outer(u, np.conj(v)) /
                             (1 + np.inner(np.conj(v), u)))


import timeit
import numpy.random as random
u0 = random.randn(400)
v0 = random.randn(400)


def timeable_basic_matinv():
    a_inv = rank1pert_inv(u0, v0)  # noqa


def timeable_numpy_matinv():
    a_inv = np.linalg.inv(np.eye(len(u0)) + np.outer(u0, np.conj(v0)))  # noqa


def time_matinvs():
    print("Timing for basic_matinv")
    print(timeit.Timer(timeable_basic_matinv).timeit(number=1))
    print("Timing for numpy matinv")
    print(timeit.Timer(timeable_numpy_matinv).timeit(number=1))


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, 
    where A = B + iC.
    """
    # Find B, C.
    # Get the part of B, C in Ahat without leading diagonal.
    B, C = np.tril(Ahat, -1), np.triu(Ahat, 1)

    B = B + B.T + np.diag(np.diag(Ahat))
    C = C - C.T

    # Find zr, zi using column space formulation.
    m, _ = Ahat.shape
    zr = np.zeros(m)
    zi = np.zeros(m)

    for j, (er, ei) in enumerate(zip(xr, xi)):
        zr += B[:, j] * er - C[:, j] * ei
        zi += B[:, j] * ei + C[:, j] * er

    return zr, zi
