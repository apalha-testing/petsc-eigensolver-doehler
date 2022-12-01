
#include <string>

#include "petsc.h"
#include "petscviewerhdf5.h"
#include "petscmat.h"
#include "slepc.h"

#include "doehler_eigensolver_inout.hpp"

PetscErrorCode doehler::read_matrix(Mat &M, std::string &name, std::string &filename, std::string &read_format) {
  // Data viewer (for reading matrices from file)
  PetscViewer     viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, read_format.data());
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_MAT);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, filename.data());


  // Create the matrix to store the saved data
  MatCreate(PETSC_COMM_WORLD, &M);
  PetscObjectSetName((PetscObject)M, name.data());

  // Read the matrix from file
  MatLoad(M, viewer);

  // Clean up
  PetscViewerDestroy(&viewer);

  return(0);
}

PetscErrorCode doehler::write_matrix(Mat &M, std::string &filename, std::string &write_format) {
  PetscViewer viewer;
  PetscErrorCode viewer_error;
  // viewer_error = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename.data(), FILE_MODE_WRITE, &viewer);
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, write_format.data());
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_MAT);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, filename.data());

  viewer_error = MatView(M, viewer);

  PetscViewerDestroy(&viewer);

  return(viewer_error);
}

void doehler::fill_tridiagonal_matrix(Mat &M, std::string &M_name, PetscInt m) {
    // M: the matrix to fill in with the triagonal filling corresponding to finite differences
    // m: the size of the matrix (assumed square)

    PetscInt row_idx;
    PetscInt col_idx[3];
    PetscInt lstart, lend;  // the ranges of matrix rows for each MPI process, for filling the matrix in parallel
    PetscScalar v[3];

    MatCreate(PETSC_COMM_WORLD, &M);
    PetscObjectSetName((PetscObject)M, M_name.data());
    MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, m, m);
    // MatSetOptionsPrefix(M, "matrix_");
    MatSetFromOptions(M);
    MatSetUp(M);
    MatGetOwnershipRange(M, &lstart, &lend);

    for (row_idx = lstart; row_idx < lend; row_idx++){
      if (row_idx == 0){
        v[0] = 3.0; v[1] = -1.0;
        col_idx[0] = 0; col_idx[1] = 1;
        MatSetValues(M, 1, &row_idx, 2, col_idx, v, INSERT_VALUES);
      }
      else{
        v[0] = 1.0; v[1] = 3.0; v[2] = -1.0;
        col_idx[0] = row_idx - 1; col_idx[1] = row_idx; col_idx[2] = row_idx + 1;
        if (row_idx == m - 1){
          MatSetValues(M, 1, &row_idx, 2, col_idx, v, INSERT_VALUES);
        }
        else{
          MatSetValues(M, 1, &row_idx, 3, col_idx, v, INSERT_VALUES);
        }
      }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
}
