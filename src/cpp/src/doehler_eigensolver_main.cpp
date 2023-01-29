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
  std::string data_format(128, '\0');  // the format to use when reading and writing matrices to the data file
  PetscBool show_input_matrices =  PETSC_FALSE;  // flag to show or not the input matrices in the commandline
  
  // Input data
  Mat A, M;  // the matrices to store the system matrix, and right hand side matrix
  PetscInt m = 10;  // the size of the tridiagonal matrix

  /*
    PETSc and SLEPc need to be initialized, if we initialize SLEPc, PETSc is
    also initialized, so we just initalize SLEPc with the helper message.
  */
  // PetscInitialize(&argc,&argv,PETSC_NULL,help);
  SlepcInitialize(&argc, &argv, PETSC_NULL, help);

  /*
    Determine files from which we read the linear system
    (matrix A, right-hand-side matrix M, and initial guess vector C).
  */
  PetscOptionsGetString(NULL, NULL, "-if", ifilename.data(), ifilename.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-of", ofilename.data(), ifilename.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-A_name", A_name.data(), A_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-M_name", M_name.data(), M_name.size(), NULL);
  PetscOptionsGetString(NULL, NULL, "-data_format", data_format.data(), data_format.size(), NULL);
  PetscOptionsGetBool(NULL, NULL, "-show_input", &show_input_matrices, NULL);
  
  PetscOptionsGetInt(NULL, NULL, "-matrix_size", &m, NULL);

  std::cout << "Solving eigenvalue problem Ax = lMx" << std::endl;
  std::cout << "   Input filename : " << ifilename << std::endl;
  std::cout << "   Output filename: " << ofilename << std::endl;
  std::cout << "   Data format    : " << data_format << std::endl;
  std::cout << "   A matrix       : " << A_name << std::endl;
  std::cout << "   M matrix       : " << M_name << std::endl;
  std::cout << "   Show input     : " << show_input_matrices << std::endl;

  /*
    Read the matrices from the data file.
  */
  
  // load the matrices
  PetscViewer     viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, data_format.data());
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_MAT);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, ifilename.data());

  doehler::read_matrix(A, A_name, viewer);
  doehler::read_matrix(M, M_name, viewer);

  // Clean up viewer
  PetscViewerDestroy(&viewer);
  
  // Print input matrices to commandline
  if(show_input_matrices ==  PETSC_TRUE)
  {
    std::cout << "\n\nThe A matrix:" << std::endl;
    MatView(A, PETSC_VIEWER_STDOUT_WORLD);

    std::cout << "\n\nThe M matrix:" << std::endl;
    MatView(M, PETSC_VIEWER_STDOUT_WORLD);
  }
  
  /*
    Compute eigenvalues using Doehler algorithm
  */
  PetscInt n_eigs = 5;
  PetscInt n_max_iter = 500;
  PetscReal tol = 1e-10;
  bool normalize_S = true;
  bool verbose = true;
  bool indefinite_dot = true;

  doehler::eigen_doehler_petsc(A, M, n_eigs, n_max_iter, tol, normalize_S, verbose, indefinite_dot);

  // Write the tridiagonal matrix
#if defined(PETSC_HAVE_HDF5)
  std::cout << "HDF5 is configured..." << std::endl;
#endif
  std::cout << "Writting matrix..." << std::endl;
  PetscErrorCode write_error;
  write_error = doehler::write_matrix(M, ofilename, data_format);
  std::cout << "Error output writing matrix: " << write_error << std::endl;

  return PetscFinalize();
}
