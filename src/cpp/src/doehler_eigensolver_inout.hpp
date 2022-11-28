#ifndef DOEHLER_EIGENSOLVER_INOUT_H
#define DOEHLER_EIGENSOLVER_INOUT_H

#include "petsc.h"
#include "slepc.h"

namespace doehler{
  PetscErrorCode read_matrix(Mat &M, char *name,  PetscViewer fd);
  void fill_tridiagonal_matrix(Mat &M, PetscInt m);
}

#endif  // DOEHLER_EIGENSOLVER_INOUT_H
