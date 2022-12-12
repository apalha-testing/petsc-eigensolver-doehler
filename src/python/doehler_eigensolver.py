import numpy
import scipy

my_random_state = numpy.random.RandomState(seed=42)

# NOTE: Sometimes the matrix will be singular. If this is the case, rerun the code.
#       This happens because the matrix is randomly generated.

# Input parameters -------------------------------------------------------------
matrix_size = 100  # number of rows and columns
n_non_zeros_per_row = 4;  # number of nonzero elements per row
matrix_density = 0.2

n_eigs = 5  # number of eigenvalues to compute
# ------------------------------------------------------------------------------


def eigen_computation_scipy(A, M, n_eigs):
    # Generate random right hand side vector
    b = numpy.random.rand(matrix_size)


    # Solve the systems (just to be sure they are not singular)
    b = numpy.random.rand(matrix_size)
    x_A = scipy.sparse.linalg.spsolve(A, b)
    x_M = scipy.sparse.linalg.spsolve(M, b)


    # Find the first k_eigs of the system A x = l M x
    ws, vs = scipy.sparse.linalg.eigs(A, k=n_eigs, M=M)

    return ws, vs


def eigen_computation_scipy_dense(A, M, n_eigs):
    # Find the first n_eigs eigenvalues of the system A x = l M x
    ws, vs = scipy.linalg.eig(A.toarray(), b=M.toarray())

    return ws[:n_eigs], vs[:, :n_eigs]


def eigen_doehler(A, M, n_eigs, n_max_iter=100, tol=1e-5, normalize_S=True):
    A_size = A.shape[0]  # assume to be square

    # Initialize the algorithm with initial guesses
    numpy.random.seed(30)
    X = numpy.random.rand(A_size, n_eigs)  # the n_eigs eigenvectors we wish to find
    S = numpy.random.rand(A_size, n_eigs)  # the additional n_eigs eigenvectors we use
                                           # to correct the solution in each iteration

    scipy.io.savemat('doehler_start_X_S.mat', {'X': X, 'S': S})
    
    L_save = numpy.empty([n_max_iter + 1], dtype=object)
    Q_save = numpy.empty([n_max_iter + 1], dtype=object)

    # TODO Apply the projector to ensure X and S satisfy the constraint

    # Iterate to find corrected solutions to the eigenvalue
    iter_idx = 0  # initialize counter for number of interations performed
    error_max = 1.0  # initialize error max to determine if the loop is over or not
    while (iter_idx <= n_max_iter) and (error_max > tol):
        iter_idx += 1  # update counter for number of iterations performed

        # Compute the reduced matrices on the space spanned by [X, S]
        T = numpy.hstack((X, S))  # the projector to the reduced space
        A_p = T.conjugate().transpose() @ (A @ T)
        M_p = T.conjugate().transpose() @ (M @ T)

        # Make sure the resulting reduced matrices are still symmetric
        # Symmetry can be lost due to roundoff and accumulation errors
        A_p = 0.5 * (A_p + A_p.conjugate().transpose())
        M_p = 0.5 * (M_p + M_p.conjugate().transpose())

        # Compute the Ritz values (L) and Ritz vectors (Q) of the reduced eigenvalue problem
        L, Q = scipy.linalg.eig(A_p, b=M_p)

        # Normalize the eigenvectors using M_r as the mass matrix, or weigthed norm
        Q_norms = numpy.sqrt((Q * (M_p @ Q)).sum(0))  # compute the weighted norms of the Ritz eigenvectors
        Q = Q / Q_norms  # normalize the Ritz eigenvectors

        # Sort the eigenvalues and reorder the eigenvectors to express this
        L_order_idx = L.argsort()  # get the indices of the new ordering
        L = L[L_order_idx]  # reorder the eigenvalues
        Q = Q[:, L_order_idx]  # reorder the eigenvectors

        # Now we can reconstruct the eigenvectors, i.e., compute them in the full space
        W_r = T @ Q[:, n_eigs:]  # reconstruct the eigenvectors associated to the largest eigenvalues
        X = T @ Q[:, :n_eigs]  # update the eigenvectors solution (the lowest eigenvalues)

        # Compute the residual with the updated eigenvectors
        R = (A @ X) - M @ (X * L[:n_eigs])
        error_max = numpy.sqrt(numpy.abs(R * R.conjugate()).sum(0)).max()  # Compute the max "L2" error norm
        cond_A_p = numpy.linalg.cond(A_p)
        print(f"iter {iter_idx:4}: \t error_max = {error_max:.16e} \t cond(A_p)  = {cond_A_p:.16e}")

        # TODO Apply preconditioner

        # Compute the new augmented solution space (the correction space) and the new search space
        V = -((W_r.conjugate().transpose() @ (A @ R - M @ (R * L[:n_eigs]))) / (L[n_eigs:].reshape([-1, 1]) - L[:n_eigs].conjugate()));
        S = R + W_r @ V # new search directions
        
        # If nothing is done, the search directions result in vectors with very
        # small norms. This leads to very badly conditioned reduced matrices. A
        # solution to this problem is to normalize these vectors associated to
        # the new search directions. This essentially means normalizing the columns
        # of S.
        if normalize_S:
            norm_cols_S = numpy.sqrt(numpy.abs(R * R.conjugate()).sum(0))
            S = S / norm_cols_S

        # TODO Apply the projector to ensure X and S satisfy the
    
    return L[:n_eigs], X



# Pre-computations ------------------------------------------------------------
n_non_zeros = matrix_size * n_non_zeros_per_row  # number of nonzero elements in the matrix
# -----------------------------------------------------------------------------


# Generate random sparse matrices ---------------------------------------------
# Generate a random sparse matrices with (up to randomness) n_non_zeros_per_row
# elements per row
# rows_idx = numpy.arange(0, matrix_size).repeat(n_non_zeros_per_row)
# cols_idx = numpy.random.randint(0, high=(matrix_size-1), size=(n_non_zeros,))
# data = numpy.random.rand(n_non_zeros)
# A = scipy.sparse.csc_matrix((data, (rows_idx, cols_idx)), shape=(matrix_size, matrix_size))
# A.toarray()
A = scipy.sparse.rand(matrix_size, matrix_size, density=matrix_density, format='csc', random_state=my_random_state)
A = 0.5 * (A + A.transpose())  # to make the matrix symmetric
# M = scipy.sparse.rand(matrix_size, matrix_size, density=matrix_density, format='csc', random_state=my_random_state)
# M = 0.5 * (M + M.transpose())  # symmetrize the matrix, since this is a requirement
M = (2.0*scipy.sparse.eye(matrix_size)).tocsr()

scipy.io.savemat('doehler_example_matrices.mat', {'A': A, 'M': M})
# -----------------------------------------------------------------------------


# Compute eigenvalues with scipy (reference) ----------------------------------
ws_scipy, vs_scipy = eigen_computation_scipy(A, M, n_eigs)
R_scipy = (A @ vs_scipy) - M @ (vs_scipy * ws_scipy)
error_scipy =  numpy.sqrt(numpy.abs(R_scipy * R_scipy.conjugate()).sum(0)).max()
print(f"Error scipy: {error_scipy}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with scipy (reference) ----------------------------------
ws_scipy_dense, vs_scipy_dense = eigen_computation_scipy_dense(A, M, n_eigs)
R_scipy_dense = (A @ vs_scipy_dense) - M @ (vs_scipy_dense * ws_scipy_dense)
error_scipy_dense =  numpy.sqrt(numpy.abs(R_scipy_dense * R_scipy_dense.conjugate()).sum(0)).max()
print(f"Error scipy: {error_scipy_dense}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with doehler algorithm (normalising S) ------------------
ws_doehler, vs_doehler = eigen_doehler(A, M, n_eigs, n_max_iter=100, tol=1e-10)
R_doehler = (A @ vs_doehler) - M @ (vs_doehler * ws_doehler)
error_doehler =  numpy.sqrt(numpy.abs(R_doehler * R_doehler.conjugate()).sum(0)).max()
print(f"Error doehler: {error_doehler}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with doehler algorithm (without normalizing S) ----------
ws_doehler_no_norm_S, vs_doehler_no_norm_S = eigen_doehler(A, M, n_eigs, n_max_iter=100, tol=1e-10, normalize_S=False)
R_doehler_no_norm_S = (A @ vs_doehler_no_norm_S) - M @ (vs_doehler_no_norm_S * ws_doehler_no_norm_S)
error_doehler_no_norm_S =  numpy.sqrt(numpy.abs(R_doehler_no_norm_S * R_doehler_no_norm_S.conjugate()).sum(0)).max()
print(f"Error doehler: {error_doehler_no_norm_S}")
# -----------------------------------------------------------------------------
