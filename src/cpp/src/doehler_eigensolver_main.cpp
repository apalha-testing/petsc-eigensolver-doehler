#include <iostream>
#include <string>

#include "petsc.h"
#include "slepc.h"
#include "petscviewerhdf5.h"

#include "doehler_eigensolver_kernel.hpp"
#include "doehler_eigensolver_inout.hpp"

static char help[] = "Solve an eigenvalue problem Ax = lMx using Doehler's algorithm.";

int main(int argc, char *argv[]) {

  // Input data names
  std::string data_filename(PETSC_MAX_PATH_LEN, '\0');  // the name of the file from where to read the matrices, we set the maximum size to PETSc's max path length
  std::string A_name(128, '\0');  // the name of the system matrix (A matrix) to read from the data file
  std::string M_name(128, '\0');  // the name of the right hand side matrix (M matrix) to read from the data file
  std::string x0_name(128, '\0');  // the name of the initial guess vector (X0 vector) to read from the data file

  // Input data
  Mat A, M, x0;  // the matrices to store the system matrix, right hand side matrix, and initial guesses

  // Data viewer (for reading matrices in HDF5 file)
  PetscViewer     data_viewer;


  /*
    PETSc and SLEPc need to be initialized, if we initialize SLEPc, PETSc is
    also initialized, so we just initalize SLEPc with the helper message.
  */
  // PetscInitialize(&argc,&argv,PETSC_NULL,help);
  SlepcInitialize(&argc, &argv, PETSC_NULL, help);

  /*
    Determine files from which we read the linear system
    (matrix A, right-hand-side matrix M, and initial guess vector X0).
  */
  PetscOptionsGetString(NULL, NULL, "-f", data_filename.data(), data_filename.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-A_name", A_name.data(), A_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-M_name", M_name.data(), M_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-x0_name", x0_name.data(), x0_name.size(), NULL);

  std::cout << "Solving eigenvalue problem Ax = lMx from:" << data_filename << std::endl;
  std::cout << "   Data filename: " << data_filename << std::endl;
  std::cout << "   A matrix     : " << A_name << std::endl;
  std::cout << "   M matrix     : " << M_name << std::endl;
  std::cout << "   x0 vector    : " << x0_name << std::endl;

  // /*
  //   Read the matrices from the data file.
  // */
  // PetscViewerHDF5Open(PETSC_COMM_WORLD, data_filename.data(), FILE_MODE_READ, &data_viewer);
  // PetscViewerPushFormat(data_viewer, PETSC_VIEWER_HDF5_MAT);
  //
  // // load the matrices
  // doehler::read_matrix(A, A_name.data(), data_viewer);
  // doehler::read_matrix(M, M_name.data(), data_viewer);
  // doehler::read_matrix(x0, x0_name.data(), data_viewer);
  //
  // MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  PetscInt m = 10;
  std::cout << "Fill tridiagonal!" << std::endl;
  doehler::fill_tridiagonal_matrix(M, m);

  std::cout << "Hello, world!" << std::endl;
  doehler::hello_world();
  // MatDestroy(&A);
  // MatDestroy(&M);
  // MatDestroy(&x0);

  return PetscFinalize();
}
