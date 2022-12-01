#include <iostream>
#include <string>

#include "petsc.h"
#include "slepc.h"
#include "petscviewerhdf5.h"

#include "doehler_eigensolver_kernel.hpp"
#include "doehler_eigensolver_inout.hpp"

static char help[] = "Solve an eigenvalue problem Ax = lMx using Doehler's algorithm.";

// #define PETSCVIEWERSOCKET      "socket"
// #define PETSCVIEWERASCII       "ascii"
// #define PETSCVIEWERBINARY      "binary"
// #define PETSCVIEWERSTRING      "string"
// #define PETSCVIEWERDRAW        "draw"
// #define PETSCVIEWERVU          "vu"
// #define PETSCVIEWERMATHEMATICA "mathematica"
// #define PETSCVIEWERHDF5        "hdf5"
// #define PETSCVIEWERVTK         "vtk"
// #define PETSCVIEWERMATLAB      "matlab"
// #define PETSCVIEWERSAWS        "saws"
// #define PETSCVIEWERGLVIS       "glvis"
// #define PETSCVIEWERADIOS       "adios"
// #define PETSCVIEWEREXODUSII    "exodusii"
// #define PETSCVIEWERCGNS        "cgns"

int main(int argc, char *argv[]) {

  // Input data names
  std::string ifilename(PETSC_MAX_PATH_LEN, '\0');  // the name of the file from where to read the matrices, we set the maximum size to PETSc's max path length
  std::string ofilename(PETSC_MAX_PATH_LEN, '\0');  // the name of the file where to write the matrices, we set the maximum size to PETSc's max path length
  std::string A_name(128, '\0');  // the name of the system matrix (A matrix) to read from the data file
  std::string M_name(128, '\0');  // the name of the right hand side matrix (M matrix) to read from the data file
  std::string x0_name(128, '\0');  // the name of the initial guess vector (X0 vector) to read from the data file
  std::string data_format(128, '\0');  // the format to use when reading and writing matrices to the data file

  // Input data
  Mat A, M, x0;  // the matrices to store the system matrix, right hand side matrix, and initial guesses
  PetscInt m = 10;  // the size of the tridiagonal matrix

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
  PetscOptionsGetString(NULL, NULL, "-if", ifilename.data(), ifilename.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-of", ofilename.data(), ifilename.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-A_name", A_name.data(), A_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-M_name", M_name.data(), M_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-x0_name", x0_name.data(), x0_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-data_format", data_format.data(), data_format.size(), NULL);
  PetscOptionsGetInt(NULL, NULL, "-matrix_size", &m, NULL);

  std::cout << "Solving eigenvalue problem Ax = lMx" << std::endl;
  std::cout << "   Input filename : " << ifilename << std::endl;
  std::cout << "   Output filename: " << ofilename << std::endl;
  std::cout << "   Data format    : " << data_format << std::endl;
  std::cout << "   A matrix       : " << A_name << std::endl;
  std::cout << "   M matrix       : " << M_name << std::endl;
  std::cout << "   x0 vector      : " << x0_name << std::endl;

  // /*
  //   Read the matrices from the data file.
  // */
  // PetscViewerHDF5Open(PETSC_COMM_WORLD, data_filename.data(), FILE_MODE_WRITE, &data_viewer);
  // PetscViewerPushFormat(data_viewer, PETSC_VIEWER_HDF5_MAT);
  //
  // // load the matrices
  // doehler::read_matrix(A, A_name.data(), data_viewer);
  // doehler::read_matrix(M, M_name.data(), data_viewer);
  // doehler::read_matrix(x0, x0_name.data(), data_viewer);
  //
  // MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  std::cout << "Fill tridiagonal!" << std::endl;
  doehler::fill_tridiagonal_matrix(M, M_name, m);

  MatView(M, PETSC_VIEWER_STDOUT_WORLD);

  // Write the tridiagonal matrix
#if defined(PETSC_HAVE_HDF5)
  std::cout << "HDF5 is configured..." << std::endl;
#endif
  std::cout << "Writting matrix..." << std::endl;
  PetscErrorCode write_error;
  write_error = doehler::write_matrix(M, ofilename, data_format);
  std::cout << "Error output writing matrix: " << write_error << std::endl;

  // And then read it back
  std::cout << "Reading matrix..." << std::endl;
  doehler::read_matrix(A, M_name, ifilename, data_format);
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  std::cout << "Hello, world!" << std::endl;
  doehler::hello_world();
  // MatDestroy(&A);
  // MatDestroy(&M);
  // MatDestroy(&x0);

  return PetscFinalize();
}
