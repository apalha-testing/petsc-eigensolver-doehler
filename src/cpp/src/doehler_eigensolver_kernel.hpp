#ifndef DOEHLER_EIGENSOLVER_KERNEL_H
#define DOEHLER_EIGENSOLVER_KERNEL_H

#include "petsc.h"
#include "slepc.h"

namespace doehler{
  void eigen_doehler_petsc(Mat &A_Mat, Mat &M_Mat, Mat &T_Mat_in, PetscInt n_eigs, PetscInt n_max_iter=100, 
      PetscReal tol=1e-5, bool normalize_S=true, bool verbose=false, 
      bool indefinite_dot=true, bool py_inputs=false, MPI_Comm comm=PETSC_COMM_WORLD);
      
  void compute_residual_eigen_v(Mat &A_Mat, Mat &M_Mat, Vec &L_Vec, 
      BV &X_bv, PetscInt eigen_idx_start, PetscInt n_eigs, 
      BV &R_bv, MPI_Comm comm=PETSC_COMM_WORLD);
}

#endif  // DOEHLER_EIGENSOLVER_KERNEL_H
