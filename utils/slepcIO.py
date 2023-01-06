import scipy
import numpy

from slepc4py import SLEPc
from mpi4py import MPI

import typing


def ndarray2bv(bv_as_ndarray: numpy.ndarray, name: str,
        comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD) -> SLEPc.BV:
    """
    Convert numpy.nparray (n, m) into a SLEPc BV (basis vectors).

    Parameters
    ----------
    M_as_ndarray: numpy.ndarray, shape (n, m)
        The numpy.ndarray array to convert into SLEPc BV.
    name: str, shape (1,)
        The name to assign to the SLEPc BV object. Mainly useful if later saved.
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.

    Returns
    -------
    bv_as_BV : SLEPc.BV, shape (n, m)
        SLEPc Basis Vectors, either complex or real, with the same content as bv_as_ndarray.
    """
    # Get info on csc sparse array
    n_rows, n_cols = M_as_ndarray.shape

    # Create array to save
    M_as_Mat = PETSc.Mat().createDense([n_rows, n_cols], comm=comm)
    M_as_Mat.setName(name)
    M_as_Mat.setFromOptions()
    M_as_Mat.setUp()

    # Set the values
    row_start, row_end = M_as_Mat.getOwnershipRange()
    row_idx = numpy.arange(row_start, row_end, dtype=PETSc.IntType)
    col_idx = numpy.arange(0, n_cols, dtype=PETSc.IntType)  # note that here we take all columns therefore the next line
    values = M_as_ndarray[row_idx, :]  # since we take all columns, we use :, otherwise we would only get the diagonal elements
    M_as_Mat.setValues(row_idx, col_idx, values)  # insert the values (the non-zeros) for the whole row

    # Once we are done we can assembly the matrix
    M_as_Mat.assemblyBegin()  # assembling the matrix makes it "useable".
    M_as_Mat.assemblyEnd()

    return M_as_Mat


def csc2mat(M_as_csc: scipy.sparse._csc.csc_matrix,
        name: str, comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD) -> PETSc.Mat:
    """
    Convert scipy.sparse._csc.csc_matrix into a PETSc sparse Mat.

    Parameters
    ----------
    M_as_csc: scipy.sparse._csc.csc_matrix, shape (n, m)
        The scipy.csc sparse array to convert into PETSc Mat.
    name: str, shape (1,)
        The name to assign to the PETSc Vec object. Mainly useful if later saved.
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.

    Returns
    -------
    M_as_Mat : PETSc.Mat, shape (n, m)
        PETSc matrix, either complex or real, with the same elements as M_as_csc.
    """

    # Get info on csc sparse array
    n_rows, n_cols = M_as_csc.shape

    # Convert to csr, to simplify reading and writing to PETSc since PETSc is csr
    M_as_csr = M_as_csc.tocsr()

    # Create array to save
    M_as_Mat = PETSc.Mat().create(comm)
    M_as_Mat.setName(name)
    M_as_Mat.setSizes([n_rows, n_cols])
    M_as_Mat.setFromOptions()
    M_as_Mat.setUp()

    # Set the values
    row_start, row_end = M_as_Mat.getOwnershipRange()
    for row_idx in numpy.arange(row_start, row_end, dtype=PETSc.IntType):
        col_idx_start = M_as_csr.indptr[row_idx]
        col_idx_end = M_as_csr.indptr[row_idx + 1]
        col_idx = M_as_csr.indices[col_idx_start:col_idx_end]  # the indices of columns of this wrow with non-zero values values
        values = M_as_csr.data[col_idx_start:col_idx_end]  # the data follows the same indexing as the columns
        n_values = values.size
        M_as_Mat.setValues(row_idx, col_idx, values)  # insert the values (the non-zeros) for the whole row

    # Once we are done we can assembly the matrix
    M_as_Mat.assemblyBegin()  # assembling the matrix makes it "useable".
    M_as_Mat.assemblyEnd()

    return M_as_Mat

# #viewer = PETSc.Viewer().createBinary("A.mat", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
#
# #viewer(A)
#
#
#
# ############ WRITE A and x ##############
# viewer_A = PETSc.Viewer().createBinary("A.dat", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
# #viewer_x = PETSc.Viewer().createBinary("x.dat", mode=PETSc.Viewer.Mode.WRITE, comm=MPI.COMM_WORLD)
# viewer_A(A)
# #viewer_x(x)
#
#
# ############## READ A and x ################
# viewer_A2 = PETSc.Viewer().createBinary("A.dat", mode=PETSc.Viewer.Mode.READ, comm=MPI.COMM_WORLD)
# #viewer_x2 = PETSc.Viewer().createBinary("x.dat", mode=PETSc.Viewer.Mode.READ, comm=MPI.COMM_WORLD)
# A2 = PETSc.Mat().load(viewer_A2)
# #x2 = PETSc.Vec().load(viewer_x2)
#
