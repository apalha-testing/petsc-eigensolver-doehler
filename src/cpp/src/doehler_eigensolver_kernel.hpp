#ifndef DOEHLER_EIGENSOLVER_KERNEL_H
#define DOEHLER_EIGENSOLVER_KERNEL_H

#include "petsc.h"

namespace doehler{
  void eigen_doehler_petsc(Mat &A_Mat, Mat &M_Mat, Mat &T_Mat_in, PetscInt n_eigs, PetscInt n_max_iter=100, 
      PetscReal tol=1e-5, bool normalize_S=true, bool verbose=false, 
      bool indefinite_dot=true, bool py_inputs=false, MPI_Comm comm=PETSC_COMM_WORLD);
}

#endif  // DOEHLER_EIGENSOLVER_KERNEL_H
