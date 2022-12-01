#ifndef DOEHLER_EIGENSOLVER_INOUT_H
#define DOEHLER_EIGENSOLVER_INOUT_H

#include <string>

#include "petsc.h"
#include "slepc.h"

namespace doehler{
  PetscErrorCode read_matrix(Mat &M, std::string &name, std::string &filename, std::string &read_format);
  PetscErrorCode write_matrix(Mat &M, std::string &filename, std::string &write_format);
  void fill_tridiagonal_matrix(Mat &M, std::string &M_name, PetscInt m);
}

#endif  // DOEHLER_EIGENSOLVER_INOUT_H
