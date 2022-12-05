from petsc4py import PETSc
from mpi4py import MPI

A = PETSc.Mat().create(MPI.COMM_WORLD)
A.setName("A")
A.setSizes([3, 3])
A.setFromOptions()
A.setUp()
lstart, lend = A.getOwnershipRange()
A.setValue(0, 0, 3) # Insert a single value into matrix.
A.assemblyBegin() # Assembling the matrix makes it "useable".
A.assemblyEnd()


print("Matrix A ready to write to file: ")
A.view()

############ WRITE A and x ##############
viewer_A = PETSc.Viewer().createBinary("A.dat", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
#viewer_x = PETSc.Viewer().createBinary("x.dat", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
viewer_A(A)
#viewer_x(x)


############## READ A and x ################
viewer_A2 = PETSc.Viewer().createBinary("A.dat", mode=PETSc.Viewer.Mode.READ, comm=MPI.COMM_WORLD)
#viewer_x2 = PETSc.Viewer().createBinary("x.dat", mode=PETSc.Viewer.Mode.READ, comm=MPI.COMM_WORLD)
A2 = PETSc.Mat().load(viewer_A2)
#x2 = PETSc.Vec().load(viewer_x2)

print("Matrix A2 read from file: ")
A2.view()
