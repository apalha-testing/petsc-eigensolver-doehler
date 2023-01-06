import numpy
import scipy

from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

import sys
if sys.path[-1] != '../../utils':
    sys.path.append('../../utils')
if 'petscIO' not in sys.modules:
    import petscIO

my_random_state = numpy.random.RandomState(seed=42)

# NOTE: Sometimes the matrix A will be singular. If this is the case, rerun the code.
#       This happens because the matrix is randomly generated.

# Input parameters -------------------------------------------------------------
matrix_size = 100  # number of rows and columns
n_non_zeros_per_row = 4;  # number of nonzero elements per row
matrix_density = 0.2

n_eigs = 5  # number of eigenvalues to compute
# ------------------------------------------------------------------------------

def create_BV_copy(X_bv, column_idx_start, column_idx_end, comm=MPI.COMM_WORLD):
    n_rows_info_X, n_columns_X = X_bv.getSizes()
    n_rows_Y = n_rows_info_X[1]
    n_columns_Y = column_idx_end - column_idx_start
    
    # This would be the easy way, but it does not work, the Y_bv in the end is garbage
    # Y_bv = SLEPc.BV().create(MPI.COMM_WORLD)
    # Y_bv.setSizes(n_rows_Y, n_columns_Y)
    # Y_bv.setFromOptions()
    
    # X_bv.setActiveColumns(column_idx_start, column_idx_end)
    # X_bv_Mat = X_bv.getMat()
    
    # Y_bv.createFromMat(X_bv_Mat)
    
    # X_bv.restoreMat(X_bv_Mat)
    # X_bv.setActiveColumns(0, n_columns_X)
    
    # Alternative going column by column by hand
    Y_bv = SLEPc.BV().create(MPI.COMM_WORLD)
    Y_bv.setSizes(n_rows_Y, n_columns_Y)
    Y_bv.setFromOptions()
    
    for column_idx in range(column_idx_start, column_idx_end):
        Y_bv_column_Vec = Y_bv.getColumn(column_idx - column_idx_start)  # subtract column_idx_start because the new vector will place the column with index column_idx_start at its column with index 0
        X_bv.copyVec(column_idx, Y_bv_column_Vec)
        Y_bv.restoreColumn(column_idx - column_idx_start, Y_bv_column_Vec)
        
    return Y_bv
    
    

def compute_residual_eigen_v(A_Mat, M_Mat, L_Vec, X_bv, comm=MPI.COMM_WORLD):
    # Computes:
    #   R = A_Mat @ X_bv - M_Mat @ (X_bv * L)
    #
    # Where:
    #   L: are n eigenvalues of the generalised eigenvalue problem A X = L M X
    #   X: are n vectors with the same dimensions as the eigenvectors of the generalised eigenvalue problem A X = L M X
    #   A_Mat: is the A matrix of the generalised eigenvalue problem
    #   M_Mat: is the M matrix of the generalised eigenvalue problem
    
    # Determine the number of eigenvalues in L 
    n_eigs = L_Vec[:].size
    
    # Compute X * L
    XL_bv = X_bv.copy()  # make a copy to hold X * L[:2*n_eigs]
    for low_eigen_v_idx in range(n_eigs):  # compute X * L[:2*n_eigs]
        XL_bv.scaleColumn(low_eigen_v_idx, L_Vec[low_eigen_v_idx])
    
    # Compute M @ (X * L)
    MXL_bv = XL_bv.copy()  # first make a copy to initialize memory
    XL_bv.matMult(M_Mat, Y=MXL_bv)  # make the multiplication M @ (X * L[:2*n_eigs])
    
    # Compute A @ X
    R_bv = X_bv.copy()  # first make a copy to initialize memory
    X_bv.matMult(A_Mat, Y=R_bv)  # make the multiplication A @ X
    
    # Compute (A @ X) - (M @ (X * L))
    # Since the following line does not work as SLEPc (sent pull request that got accepted)
    #    R_bv.mult(-1.0, 1.0, MXL_bv, NULL)
    # Need to use the version
    #    R_bv.mult(-1.0, 1.0, MXL_bv, Q)
    # with Q set to the identity matrix. This means we need to setup the identity matrix
    diag_values_Vec = PETSc.Vec().create(comm)
    diag_values_Vec.setName('D_values')
    diag_values_Vec.setSizes(n_eigs)
    diag_values_Vec.setFromOptions()
    diag_values_Vec.setUp()
    diag_values_Vec.assemblyBegin()  # assembling the vector makes it "useable".
    diag_values_Vec.assemblyEnd()
    diag_values_Vec.set(1.0)
    
    Identity_Mat = PETSc.Mat().createDense([n_eigs, n_eigs], comm=comm)  # this should be sequential
    Identity_Mat.setName('I')
    Identity_Mat.setFromOptions()
    Identity_Mat.setUp()
    Identity_Mat.assemblyBegin()  # assembling the matrix makes it "useable".
    Identity_Mat.assemblyEnd()
    Identity_Mat.setDiagonal(diag_values_Vec)
    
    R_bv.mult(-1.0, 1.0, MXL_bv, Identity_Mat)
    
    return R_bv
    
    
def eigen_computation_scipy(A, M, n_eigs, sigma=0.0):
    # Generate random right hand side vector
    b = numpy.random.rand(matrix_size)


    # Solve the systems (just to be sure they are not singular)
    b = numpy.random.rand(matrix_size)
    x_A = scipy.sparse.linalg.spsolve(A, b)
    x_M = scipy.sparse.linalg.spsolve(M, b)


    # Find the first k_eigs of the system A x = l M x
    # We use sigma (search eigenvalues close to sigma) so that we can compare to doehler
    ws, vs = scipy.sparse.linalg.eigs(A, k=n_eigs, M=M, sigma=sigma)

    return ws, vs


def eigen_computation_scipy_dense(A, M, n_eigs):
    # Find the first n_eigs eigenvalues of the system A x = l M x
    ws, vs = scipy.linalg.eig(A.toarray(), b=M.toarray())
    
    # We need to sort the eigenvalues, and then re-order the eigenvector correspondingly
    ws_order_idx = ws.argsort()
    ws = ws[ws_order_idx]
    vs = vs[:, ws_order_idx]
    return ws[:n_eigs], vs[:, :n_eigs]


def eigen_doehler(A, M, n_eigs, n_max_iter=100, tol=1e-5, verbose=False, normalize_S=True):
    A_size = A.shape[0]  # assume to be square

    # Initialize the algorithm with initial guesses
    numpy.random.seed(30)
    X = numpy.random.rand(A_size, n_eigs)  # the n_eigs eigenvectors we wish to find
    S = numpy.random.rand(A_size, n_eigs)  # the additional n_eigs eigenvectors we use
                                           # to correct the solution in each iteration
    scipy.io.savemat('doehler_start_X_S.mat', {'X': X, 'S': S})

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
        Q_norms = numpy.sqrt((Q.conjugate() * (M_p @ Q)).sum(0))  # compute the weighted norms of the Ritz eigenvectors
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
        if verbose:
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
    
    return L[:n_eigs], X, error_max, iter_idx


def eigen_doehler_petsc(A, M, n_eigs, n_max_iter=100, tol=1e-5, normalize_S=True, verbose=False, indefinite_dot=True, comm=MPI.COMM_WORLD):
    A_size = A.shape[0]  # assume to be square
    
    # Convert the matrices to PETSc sparse matrices
    A_Mat = petscIO.csc2mat(A, 'A', comm=comm)
    M_Mat = petscIO.csc2mat(M, 'M', comm=comm)

    # Initialize the algorithm with initial guesses
    
    # Initialize tranformation matrix T as a bv system (used to project to the reduced space)
    T_bv = SLEPc.BV().create(MPI.COMM_WORLD)
    T_bv.setSizes(A_size, 2*n_eigs)
    T_bv.setFromOptions()
    
    # Initialize eigenvector bv system to store the eigenvectors computed each iteration
    Q_bv = SLEPc.BV().create(MPI.COMM_WORLD)
    Q_bv.setSizes(2*n_eigs, 2*n_eigs)
    Q_bv.setFromOptions()
    

    # Set the X initial values
    # the n_eigs eigenvectors we wish to find
    T_bv.setActiveColumns(0, n_eigs)
    X_Mat = T_bv.getMat()
    numpy.random.seed(30)
    X = numpy.random.rand(A_size, n_eigs)
    X_Mat = petscIO.ndarray2mat(X, 'X', comm=comm, M_as_Mat=X_Mat)
    T_bv.restoreMat(X_Mat)
    
    # Set the S initial values
    # the additional n_eigs eigenvectors we use
    # to correct the solution in each iteration
    T_bv.setActiveColumns(n_eigs, 2*n_eigs)
    S_Mat = T_bv.getMat()
    S = numpy.random.rand(A_size, n_eigs)
    S_Mat = petscIO.ndarray2mat(S, 'S', comm=comm, M_as_Mat=S_Mat)
    T_bv.restoreMat(S_Mat)
    
    # Reset the active columns to all columns 
    T_bv.setActiveColumns(0, 2*n_eigs)

    # TODO Apply the projector to ensure X and S satisfy the constraint
    # TODO Consider forcing all small matrices and vectors to be sequential
    
    # Iterate to find corrected solutions to the eigenvalue
    iter_idx = 0  # initialize counter for number of interations performed
    error_max = 1.0  # initialize error max to determine if the loop is over or not
    while (iter_idx <= n_max_iter) and (error_max > tol):
        iter_idx += 1  # update counter for number of iterations performed

        # Compute the reduced matrices on the space spanned by T = [X, S]
        A_Mat_p = T_bv.matProject(A_Mat, T_bv)
        M_Mat_p = T_bv.matProject(M_Mat, T_bv)
        
        # Create temporary vector to store eigenvalues
        L_Vec = M_Mat_p.createVecRight()

        # Make sure the resulting reduced matrices are still symmetric
        # Symmetry can be lost due to roundoff and accumulation errors
        
        # Compute the Ritz values (L) and Ritz vectors (Q) of the reduced eigenvalue problem
        eigen_solver = SLEPc.EPS()
        eigen_solver.create()
        eigen_solver.setOperators(A_Mat_p, B=M_Mat_p)
        eigen_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eigen_solver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)  # sort eigenpairs by smallest to largest eigenvalue
        eigen_solver.setFromOptions()
        eigen_solver.solve()
        
        # Save all eigenvectors into the BV Q_bv
        # and normalise the eigenvectors using M_Mat_p as the inner product matrix
        if not indefinite_dot:
            Q_bv.setMatrix(M_Mat_p, False)  # use M_Mat_p to compute the inner products
        
        for eigen_v_idx in range(2*n_eigs):
            # Extract the eigenvector
            eigen_v = Q_bv.getColumn(eigen_v_idx)  # first get the column to update 
            L_Vec[eigen_v_idx] = eigen_solver.getEigenpair(eigen_v_idx, eigen_v)  # update the column with the eigenvector
            
            if indefinite_dot:
                # Normalise it using just the transpose, without conjugation
                M_mult_eigen_v = eigen_v.duplicate()  # create a temporary vector with the same shape for storing M_Mat_p * eigen_v
                M_Mat_p.mult(eigen_v, M_mult_eigen_v)  # compute M_Mat_p * eigen_v
                eigen_v_norm = numpy.sqrt(M_mult_eigen_v.tDot(eigen_v))  # compute the indefinite vector dot product 
                eigen_v.scale(1.0/eigen_v_norm)  # normalise it
                
                # Return the updated column vector to Q_bv
                Q_bv.restoreColumn(eigen_v_idx, eigen_v)  # restore the column vector to Q_bv
                
            else:
                # Return the updated column vector to Q_bv
                Q_bv.restoreColumn(eigen_v_idx, eigen_v)  # restore the column vector to Q_bv
                
                # Normalise it using BV (this uses the conjugate transpose)
                eigen_v_norm = Q_bv.normColumn(eigen_v_idx)  # compute the norm 
                Q_bv.scaleColumn(eigen_v_idx, eigen_v_norm)  # normalise it
                

        # Now we can reconstruct the eigenvectors, i.e., compute them in the full space
        T_bv_new = T_bv.copy()  # the updated reconstructed vectors
        
        # Reconstruct the eigenvectors (all at once)
        Q_Mat = Q_bv.getMat()  # get the matrix associated to the search space eigenvectors to use with mult below
        T_bv_new.mult(1.0, 0.0, T_bv, Q_Mat)  # make the multiplication T_bv_new = T_bv * Q_mat (reconstruct eigenvalues)
        Q_bv.restoreMat(Q_Mat)  # restore the matrix so that we can reuse Q_bv
        
        X_bv = create_BV_copy(T_bv_new, 0, n_eigs, comm=comm)
        W_r_bv = create_BV_copy(T_bv_new, n_eigs, 2*n_eigs, comm=comm)

        

        # Compute the residual with the updated eigenvectors
        R_bv = compute_residual_eigen_v(A_Mat, M_Mat, L_Vec[:n_eigs], X_bv, comm=comm)
        
        # Compute the max "L2" error norm
        error_max = 0.0  # initialise the error
        error_max_temp = 0.0
        for eigen_v_idx in range(n_eigs):  # check the error only for the solution space
            error_max_temp = R_bv.normColumn(eigen_v_idx)
            error_max = error_max_temp if error_max_temp > error_max else error_max
            
        cond_A_p = numpy.linalg.cond(A_Mat_p[:, :])
        if verbose:
            print(f"iter {iter_idx:4}: \t error_max = {error_max:.16e} \t cond(A_p)  = {cond_A_p:.16e}")

        # TODO Apply preconditioner

        # Compute the new augmented solution space (the correction space) and the new search space
        RR_bv = compute_residual_eigen_v(A_Mat, M_Mat, L_Vec[:n_eigs], R_bv, comm=comm)
        T_bv_new.setActiveColumns(n_eigs, 2*n_eigs)
        W_r_bv_Mat = W_r_bv.getMat()
        
        V_bv = SLEPc.BV().create(MPI.COMM_WORLD)
        V_bv.setSizes(n_eigs, n_eigs)
        V_bv.setFromOptions()
        
        V_bv = RR_bv.matMultHermitianTranspose(W_r_bv_Mat, Y=V_bv)
        
        W_r_bv.restoreMat(W_r_bv_Mat)
        
        # Divide by -(L[n_eigs:].reshape([-1, 1]) - L[:n_eigs].conjugate())
        for column_idx in range(n_eigs):
            V_bv_col_Vec = V_bv.getColumn(column_idx)
            
            for row_idx in range(n_eigs):
                V_value = V_bv_col_Vec.getValue(row_idx)
                V_value = -V_value / (L_Vec.getValue(row_idx + n_eigs) - numpy.conjugate(L_Vec.getValue(column_idx)))
                V_bv_col_Vec.setValue(row_idx, V_value, addv=PETSc.InsertMode.INSERT_VALUES)
            
            V_bv.restoreColumn(column_idx, V_bv_col_Vec)
        
        # Compute the new search space
        V_bv_Mat = V_bv.getMat()
        R_bv.mult(1.0, 1.0, W_r_bv, V_bv_Mat)  # note that R_bv is now S with respect to the algorithm
        V_bv.restoreMat(V_bv_Mat)
        
        # Restart T_bv
        
        # Update X part
        for column_idx in range(n_eigs):
            T_bv_column_Vec = T_bv.getColumn(column_idx)
            X_bv.copyVec(column_idx, T_bv_column_Vec)
            T_bv.restoreColumn(column_idx, T_bv_column_Vec)
            
        # Update S part
        for column_idx in range(n_eigs):
            T_bv_column_Vec = T_bv.getColumn(column_idx + n_eigs)  # we need to offset because now we are updating the S part
            R_bv.copyVec(column_idx, T_bv_column_Vec)
            
            # If nothing is done, the search directions result in vectors with very
            # small norms. This leads to very badly conditioned reduced matrices. A
            # solution to this problem is to normalize these vectors associated to
            # the new search directions. This essentially means normalizing the columns
            # of S.
            if normalize_S:
                T_bv_column_Vec.normalize()
                
            T_bv.restoreColumn(column_idx + n_eigs, T_bv_column_Vec)
            

        # TODO Apply the projector to ensure X and S satisfy the
        
        # TODO Update T (or update during the code so that at the restart of the
        #      loop, we have the new T ready)
    
    # Convert the eigenvector solution into a numpy.array
    X_bv_Mat = X_bv.getMat()
    X = X_bv_Mat[:,:]
    X_bv.restoreMat(X_bv_Mat)
    
    return L_Vec[:n_eigs], X, error_max, iter_idx



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


# Compute eigenvalues with doehler algorithm (normalising S) ------------------
ws_doehler, vs_doehler, _, iter_doehler = eigen_doehler(A, M, n_eigs, n_max_iter=300, tol=1e-14)
R_doehler = (A @ vs_doehler) - M @ (vs_doehler * ws_doehler)
error_doehler =  numpy.sqrt(numpy.abs(R_doehler * R_doehler.conjugate()).sum(0)).max()
print(f"Error doehler with normalization: {error_doehler} \t iter: {iter_doehler}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with doehler algorithm (without normalizing S) ----------
ws_doehler_no_norm_S, vs_doehler_no_norm_S, _, iter_doehler_no_norm_S = eigen_doehler(A, M, n_eigs, n_max_iter=300, tol=1e-10, normalize_S=False)
R_doehler_no_norm_S = (A @ vs_doehler_no_norm_S) - M @ (vs_doehler_no_norm_S * ws_doehler_no_norm_S)
error_doehler_no_norm_S =  numpy.sqrt(numpy.abs(R_doehler_no_norm_S * R_doehler_no_norm_S.conjugate()).sum(0)).max()
print(f"Error doehler no normalization: \t{error_doehler_no_norm_S} \t\t iter: {iter_doehler_no_norm_S}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with doehler algorithm (PETSc) --------------------------
ws_doehler_petsc, vs_doehler_petsc, _, iter_doehler_petsc = eigen_doehler_petsc(A, M, n_eigs, n_max_iter=300, tol=1e-14, normalize_S=True, verbose=False)
R_doehler_petsc = (A @ vs_doehler_petsc) - M @ (vs_doehler_petsc * ws_doehler_petsc)
error_doehler_petsc =  numpy.sqrt(numpy.abs(R_doehler_petsc * R_doehler_petsc.conjugate()).sum(0)).max()
print(f"Error doehler PETSc: \t\t\t\t{error_doehler_petsc} \t iter: {iter_doehler_petsc}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with scipy (reference) ----------------------------------
ws_scipy, vs_scipy = eigen_computation_scipy(A, M, n_eigs, sigma=ws_doehler[0])
R_scipy = (A @ vs_scipy) - M @ (vs_scipy * ws_scipy)
error_scipy =  numpy.sqrt(numpy.abs(R_scipy * R_scipy.conjugate()).sum(0)).max()
print(f"Error scipy sparse: \t\t\t\t{error_scipy}")
# -----------------------------------------------------------------------------


# Compute eigenvalues with scipy (reference) ----------------------------------
ws_scipy_dense, vs_scipy_dense = eigen_computation_scipy_dense(A, M, n_eigs)
R_scipy_dense = (A @ vs_scipy_dense) - M @ (vs_scipy_dense * ws_scipy_dense)
error_scipy_dense =  numpy.sqrt(numpy.abs(R_scipy_dense * R_scipy_dense.conjugate()).sum(0)).max()
print(f"Error scipy dense: \t\t\t\t{error_scipy_dense}")
# -----------------------------------------------------------------------------

