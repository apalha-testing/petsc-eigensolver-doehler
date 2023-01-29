#include <iostream>

#include "petsc.h"
#include "slepc.h"

#include "doehler_eigensolver_kernel.hpp"


namespace doehler{
  void eigen_doehler_petsc(Mat &A, Mat &M, PetscInt n_eigs, PetscInt n_max_iter, 
      PetscReal tol, bool normalize_S, bool verbose, 
      bool indefinite_dot, MPI_Comm comm)
  {
    std::cout << "\n**************************************************************" << std::endl;
    std::cout << "* Doehler eingenvalue solver (PETSc) START" << std::endl;
    std::cout << "**************************************************************\n" << std::endl;
    
    // Initialize the algorithm with initial guesses
    
    // Compute initial parameters, like system size, etc 
    PetscInt A_n_rows, A_n_cols;  // number of rows and columns of matrix A
    MatGetSize(A, &A_n_rows, &A_n_cols);
    
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
    
    // Set the X initial (guess) values
    // (the n_eigs eigenvectors we wish to find)
    
    
    BVSetActiveColumns(T_bv, 0, n_eigs);
    // X_Mat = T_bv.getMat()
    // numpy.random.seed(30)
    // X = numpy.random.rand(A_size, n_eigs)
    // X_Mat = petscIO.ndarray2mat(X, 'X', comm=comm, M_as_Mat=X_Mat)
    // T_bv.restoreMat(X_Mat)
    
    std::cout << "\n**************************************************************" << std::endl;
    std::cout << "* Doehler eingenvalue solver (PETSc) END" << std::endl;
    std::cout << "**************************************************************\n" << std::endl;
  }
}  // namespace doehler
