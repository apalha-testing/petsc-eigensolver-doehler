import scipy

from petsc4py import PETSc
from mpi4py import MPI

import petscIO

file_format = "binary"

# Create Matrix to save
A = PETSc.Mat().create(MPI.COMM_WORLD)
A.setName("A")
A.setSizes([3, 3])
A.setFromOptions()
A.setUp()
lstart, lend = A.getOwnershipRange()
A.setValue(0, 0, 3) # Insert a single value into matrix.
A.assemblyBegin() # Assembling the matrix makes it "useable".
A.assemblyEnd()

print("Matrix A ready to be written to file: \n")
A.view()

print("Writting matrix A to file...")
petscIO.save_to_file(A, "A_matrix.dat", file_format, comm=MPI.COMM_WORLD)
print("DONE...\n")

print("Reading Matrix A2 from file: ")
A2 = petscIO.read_from_file("A_matrix.dat", file_format, comm=MPI.COMM_WORLD)
print("DONE...\n")

print("Matrix A2 read from file: ")
A2.view()


# Load matlab saved data
matlab_sparse_data = scipy.io.loadmat("D2N10P1_transformed.mat")
matlab_dense_data = scipy.io.loadmat("test_dense.mat")

u_petsc = petscIO.matlab2petsc(matlab_dense_data["u"], "u")
v_petsc = petscIO.matlab2petsc(matlab_dense_data["v"], "v")
A_petsc = petscIO.matlab2petsc(matlab_sparse_data["A"], "A")
