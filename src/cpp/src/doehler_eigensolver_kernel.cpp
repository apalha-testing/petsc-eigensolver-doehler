#include <iostream>

#include "petsc.h"
#include "slepc.h"

#include "doehler_eigensolver_kernel.hpp"


namespace doehler{
  void eigen_doehler_petsc(Mat &A_Mat, Mat &M_Mat, Mat &T_Mat_in, PetscInt n_eigs, PetscInt n_max_iter, 
      PetscReal tol, bool normalize_S, bool verbose, 
      bool indefinite_dot, bool py_inputs, MPI_Comm comm)
  {
    std::cout << "\n**************************************************************" << std::endl;
    std::cout << "* Doehler eingenvalue solver (PETSc) START" << std::endl;
    std::cout << "**************************************************************\n" << std::endl;
    
    // Initialize the algorithm with initial guesses
    
    // Compute initial parameters, like system size, etc 
    PetscInt A_n_rows, A_n_cols;  // number of rows and columns of matrix A
    MatGetSize(A_Mat, &A_n_rows, &A_n_cols);
    
    std::cout << "System size:" << std::endl;
    std::cout << "   n_rows: " << A_n_rows << std::endl;
    std::cout << "   n_cols: " << A_n_cols << std::endl;
    
    // Initialize tranformation matrix T as a bv system (used to project to the reduced space)
    BV T_bv;
    BVCreate(comm, &T_bv);
    
    BVSetSizes(T_bv, PETSC_DECIDE, A_n_rows, 2*n_eigs);
    BVSetFromOptions(T_bv);
    
    // Initialize eigenvector bv system to store the eigenvectors computed each iteration
    BV Q_bv;
    BVCreate(comm, &Q_bv);
    
    BVSetSizes(Q_bv, PETSC_DECIDE, 2*n_eigs, 2*n_eigs);
    BVSetFromOptions(Q_bv);
    
    // Set the initial (guess) values
    // (the n_eigs eigenvectors we wish to find and the 
    // n_eigs vector search directions)
    if(py_inputs)
    {
      // Use initial values from python for one to one comparison
      Mat T_Mat;
      MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, A_n_rows, 2*n_eigs, NULL, &T_Mat);
      MatSetUp(T_Mat); 
      BVGetMat(T_bv, &T_Mat);
      MatCopy(T_Mat_in, T_Mat, DIFFERENT_NONZERO_PATTERN);
      BVRestoreMat(T_bv, &T_Mat);
    }
    else
    {
      // Use randomly generated values to initialize T_bv
      BVSetRandom(T_bv);
    }
    
    // TODO Apply the projector to ensure X and S satisfy the constraint
    // TODO Consider forcing all small matrices and vectors to be sequential
    
    // Iterate to find corrected solutions to the eigenvalue
    PetscInt iter_idx = 0;  // initialize counter for number of interations performed
    PetscReal error_max = 1.0;  // initialize error max to determine if the loop is over or not
    
    // Declare variables required in while loop 
    
    // Reduced matrices obtained by projecting A_Mat and M_Mat
    // into the T_bv space (approximate eigenvectors \ocirc search space) 
    Mat A_Mat_p, M_Mat_p;
    MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, 2*n_eigs, 2*n_eigs, NULL, &A_Mat_p);
    MatSetUp(A_Mat_p); 
    MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, 2*n_eigs, 2*n_eigs, NULL, &M_Mat_p);
    MatSetUp(M_Mat_p); 
                           
    // Vector containing eigenvalues 
    Vec L_Vec;
    
    // Setup the (small) eigensolver 
    EPS eigen_solver;
    EPSCreate(comm, &eigen_solver);
    
    while((iter_idx <= 0) && (error_max > tol))
    {
      iter_idx++;  // update counter for number of iterations performed
      std::cout << "   iter: " << iter_idx << ";  error_max: " << error_max << std::endl;
      
      // Compute the reduced matrices on the space spanned by T = [X, S]
      BVMatProject(T_bv, A_Mat, T_bv, A_Mat_p);
      BVMatProject(T_bv, M_Mat, T_bv, M_Mat_p);
      
      std::cout << "\n\nThe A matrix:" << std::endl;
      MatView(A_Mat_p, PETSC_VIEWER_STDOUT_WORLD);

      std::cout << "\n\nThe M matrix:" << std::endl;
      MatView(M_Mat_p, PETSC_VIEWER_STDOUT_WORLD);
      
      // Create temporary vector to store eigenvalues
      MatCreateVecs(M_Mat_p, &L_Vec, NULL);
      
      std::cout << "\n\nThe L_Vec vector:" << std::endl;
      VecView(L_Vec, PETSC_VIEWER_STDOUT_WORLD);
      
      // Make sure the resulting reduced matrices are still symmetric
      // Symmetry can be lost due to roundoff and accumulation errors
      
      // Compute the Ritz values (L) and Ritz vectors (Q) of the reduced eigenvalue problem
      EPSSetOperators(eigen_solver, A_Mat_p, M_Mat_p);
      EPSSetProblemType(eigen_solver, EPS_GHEP);
      EPSSetWhichEigenpairs(eigen_solver, EPS_SMALLEST_REAL);
      EPSSetFromOptions(eigen_solver);
      EPSSolve(eigen_solver);
      
      // Save all eigenvectors into the BV Q_bv
      // and normalise the eigenvectors using M_Mat_p as the inner product matrix
      if(!indefinite_dot)  // need to check if bool is correct
        BVSetMatrix(Q_bv, M_Mat_p, PETSC_FALSE);
      
      for(PetscInt eigen_v_idx = 0; eigen_v_idx < 2*n_eigs; eigen_v_idx++)
      {
        Vec eigen_v;  // temporary eigenvector to extract from EPS and store in Q_bv
        PetscScalar L_value;  // temporary eigenvalue to extract from EPS and store in L_Vec
        
        // Extract the eigenvector and eigenvalue
        BVGetColumn(Q_bv, eigen_v_idx, &eigen_v);  // first get the column to update 
        EPSGetEigenpair(eigen_solver, eigen_v_idx, &L_value, NULL, eigen_v, NULL);  // update the column with the eigenvector
        VecSetValue(L_Vec, eigen_v_idx, L_value, INSERT_VALUES);  // update the eigenvalue
        std::cout << "\n\nEigenvector:"  << eigen_v_idx << std::endl;
        VecView(eigen_v, PETSC_VIEWER_STDOUT_WORLD);
        BVRestoreColumn(Q_bv, eigen_v_idx, &eigen_v);  // restore the column so that we can reuse Q_bv
      }
      std::cout << "\n\nEigenvalues:" << std::endl;
      VecView(L_Vec, PETSC_VIEWER_STDOUT_WORLD);
      
      //   for eigen_v_idx in range(2*n_eigs):
      //       # Extract the eigenvector
      //       eigen_v = Q_bv.getColumn(eigen_v_idx)  # first get the column to update 
      //       L_Vec[eigen_v_idx] = eigen_solver.getEigenpair(eigen_v_idx, eigen_v)  # update the column with the eigenvector
      // 
      //       if indefinite_dot:
      //           # Normalise it using just the transpose, without conjugation
      //           M_mult_eigen_v = eigen_v.duplicate()  # create a temporary vector with the same shape for storing M_Mat_p * eigen_v
      //           M_Mat_p.mult(eigen_v, M_mult_eigen_v)  # compute M_Mat_p * eigen_v
      //           eigen_v_norm = numpy.sqrt(M_mult_eigen_v.tDot(eigen_v))  # compute the indefinite vector dot product 
      //           eigen_v.scale(1.0/eigen_v_norm)  # normalise it
      // 
      //           # Return the updated column vector to Q_bv
      //           Q_bv.restoreColumn(eigen_v_idx, eigen_v)  # restore the column vector to Q_bv
      // 
      //       else:
      //           # Return the updated column vector to Q_bv
      //           Q_bv.restoreColumn(eigen_v_idx, eigen_v)  # restore the column vector to Q_bv
      // 
      //           # Normalise it using BV (this uses the conjugate transpose)
      //           eigen_v_norm = Q_bv.normColumn(eigen_v_idx)  # compute the norm 
      //           Q_bv.scaleColumn(eigen_v_idx, eigen_v_norm)  # normalise it
        
    }
    
    std::cout << "\n**************************************************************" << std::endl;
    std::cout << "* Doehler eingenvalue solver (PETSc) END" << std::endl;
    std::cout << "**************************************************************\n" << std::endl;
  }
}  // namespace doehler
