import scipy
import numpy

from petsc4py import PETSc
from mpi4py import MPI

import typing

def save_to_file(M: typing.Union[
            typing.Dict[str, typing.Union[PETSc.Mat, PETSc.Vec]],
            typing.List[typing.Union[PETSc.Mat, PETSc.Vec]]
        ],
        filename: str, file_format: str,
        comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD):
    """
    Save all PETSc matrix or vector in the dict or list M to file in the specified format.

    Parameters
    ----------
    M : List or dict of PETSc.Mat or PETSc.Vec,
        PETSc matrix or vector, either complex or real.
    filename: str, shape (1,)
        The name  of the file where to save the data.
    file_format: str, shape (1,)
        The format to use to save the file, this can be:
        binary: PETSc own binary format
        hdf5: HDF5 format
        ascii: ASCII, human readable format
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.

    Returns
    -------
    Nothing

    """
    print(f'Writting to file {filename} ...')

    # Generate the view to save to file
    if file_format == "binary":
        viewer_petsc = PETSc.Viewer().createBinary(filename, mode=PETSc.Viewer.Mode.WRITE, comm=comm)
    elif file_format == "ascii":
        viewer_petsc = PETSc.Viewer().createASCII(filename, mode=PETSc.Viewer.Mode.WRITE, comm=comm)
    elif file_format == "hdf5":
        viewer_petsc = PETSc.Viewer().createHDF5(filename, mode=PETSc.Viewer.Mode.WRITE, comm=comm)
    else:
        raise ValueError(f'Format {file_format} is not valid for writing to file.')

    # Save each element of M to the viewer (file)

    # First check if the input is a list or a dict and use either a range or the
    # keys in the loop over the elements
    if isinstance(M, list):
        petsc_idx_list = range(0, len(M))  # get the indices for all elements of M (we save all to file)

    elif isinstance(M, dict):
        petsc_idx_list = M.keys()  # get the keys as the indices so that we do the for loop in the same way

    else:
        raise ValueError(f'Input M must be either a list or a dict but it is a {type(M)}.')

    for petsc_idx in petsc_idx_list:
        this_object_name = M[petsc_idx].getName()
        print(f'   Writting {this_object_name} ...')
        viewer_petsc(M[petsc_idx])


def read_from_file(filename: str, names: typing.List[str], isMat: typing.List[bool], file_format: str,
        comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD, as_list=True) \
                -> typing.Union[
                    typing.Dict[str, typing.Union[PETSc.Mat, PETSc.Vec]],
                    typing.List[typing.Union[PETSc.Mat, PETSc.Vec]]
                ]:
    """
    Read PETSc matrix or vector from file in the specified format

    Parameters
    ----------
    filename: str, shape (1,)
        The name  of the file from where to read the data.
    names: list[str], shape (n,)
        The names of the n PETSc objects to load from file.
        NOTE: this list of names must be in the order they were saved to the file,
              i.e., the matrix that was saved first will be read first. It is not
              possible to know the original name of the saved matrix. If you give
              a wrong naming you will read the files and assign the naming you
              provided.
    file_format: str, shape (1,)
        The format to use to read the file, this can be:
        binary: PETSc own binary format
        hdf5: HDF5 format
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.

    Returns
    -------
    read_objects : The list or dict of PETSc.Mat or PETSc.Vec, (n,)
        PETSc matrix or vector, either complex or real, read from file.
    """

    print(f'Reading from file {filename} ...')

    # Generate the viewer
    if file_format == "binary":
        viewer_petsc = PETSc.Viewer().createBinary(filename, mode=PETSc.Viewer.Mode.READ, comm=comm)
    elif file_format == "hdf5":
        viewer_petsc = PETSc.Viewer().createHDF5(filename, mode=PETSc.Viewer.Mode.READ, comm=comm)
    else:
        raise ValueError(f'Format {file_format} is not valid for reading from file.')

    # Read all requested objects, taking into account their type (Mat or Vec)
    if as_list:
        read_objects = []  # output a list in the order present in names
    else:
        read_objects = {}  # output a dictionary using names as keys

    for petsc_idx, petsc_name in enumerate(names):
        print(f'   Reading {petsc_name} ...')

        if isMat[petsc_idx]:
            petsc_obj = PETSc.Mat().create(comm)  # create a PETSc.Mat since it is a matrix

        else:
            petsc_obj = PETSc.Vec().create(comm)  # create a PETSc.Vec since it is a vector

        # Give it the name, otherwise it will be a generic name
        petsc_obj.setName(petsc_name)

        # Read the object and add it to the output variable to return all read objects
        if as_list:
            read_objects.append(petsc_obj.load(viewer_petsc))  # store in dictionary

        else:
            read_objects[petsc_name] = petsc_obj.load(viewer_petsc)  # store in list

    return read_objects


def matlab2petsc(M: typing.Union[scipy.sparse._csc.csc_matrix, numpy.ndarray],
        name: str, comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD) -> typing.Union[PETSc.Mat, PETSc.Vec]:
    """
    Convert matlab (scipy) array (dense or sparse) or vector into a dense or sparse PETSc Mat
    or Vec, respectively.

    Parameters
    ----------
    M: scipy.sparse._csc.csc_matrix or numpy.ndarray, shape (n,m)
        The matrix (dense or sparse) or vector to convert from matlab (scipy) format
        into PETSc Mat or Vec.
    name: str, shape (1,)
        The name to assign to the PETSc object (Mat or Vec). Mainly useful if later saved.
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.

    Returns
    -------
    M : PETSc.Mat or PETSc.Vec, shape (n,m) or (k,)
        PETSc (sparse or dense) matrix or vector, either complex or real.
    """

    # Check if input M is a sparse matrix or a numpy.ndarray
    if isinstance(M, scipy.sparse._csc.csc_matrix):
        print(f"{name} is a sparse csc matrix")
        M_petsc = csc2mat(M, name, comm)

    elif isinstance(M, numpy.ndarray):
        print(f"{name} is a dense matrix")
        if M.squeeze().ndim == 1:
            print(f"   a vector, actually")
            M_petsc = nparray2vec(M, name, comm)

        elif M.squeeze().ndim == 2:
            print(f"   a matrix, actually")
            M_petsc = ndarray2mat(M, name, comm)

        else:
            print(f"   the number of dimensions of {name} is not supported, only 1 or 2 is allowed")
            M_petsc = None

    else:
        raise ValueError(f"M is of unsupported type {type(M)}!")

    return M_petsc

def nparray2vec(v_as_ndarray: numpy.ndarray, name: str,
        comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD) -> PETSc.Vec:
    """
    Convert numpy.nparray (n, 1) into a PETSc Vec.

    Parameters
    ----------
    v_as_ndarray: numpy.ndarray, shape (n, 1)
        The numpy.ndarray vector to convert into PETSc Vec.
    name: str, shape (1,)
        The name to assign to the PETSc Vec object. Mainly useful if later saved.
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.

    Returns
    -------
    v_as_Vec : PETSc.Vec, shape (n, 1)
        PETSc vector, either complex or real, with the same elements as v_as_ndarray.
    """

    # Get info on matlab vector
    v_size = v_as_ndarray.size

    # Create vector to save
    v_as_Vec = PETSc.Vec().create(comm)
    v_as_Vec.setName(name)
    v_as_Vec.setSizes(v_size)
    v_as_Vec.setFromOptions()
    v_as_Vec.setUp()

    # Set the values
    lstart, lend = v_as_Vec.getOwnershipRange()
    my_idx = numpy.arange(lstart, lend, dtype=PETSc.IntType)  # get my local range of indices to fill up
    v_as_Vec.setValues(my_idx, v_as_ndarray[my_idx])  # insert the values of the input ndarray vector
    v_as_Vec.assemblyBegin()  # assembling the vector makes it "useable".
    v_as_Vec.assemblyEnd()

    return v_as_Vec


def ndarray2mat(M_as_ndarray: numpy.ndarray, name: str,
        comm: typing.Optional[MPI.Intracomm] = MPI.COMM_WORLD,
        M_as_Mat: typing.Optional[typing.Union[None, PETSc.Mat]] = None) -> PETSc.Mat:
    """
    Convert numpy.nparray (n, m) into a PETSc Mat dense.

    Parameters
    ----------
    M_as_ndarray: numpy.ndarray, shape (n, m)
        The numpy.ndarray array to convert into PETSc Mat dense.
    name: str, shape (1,)
        The name to assign to the PETSc Vec object. Mainly useful if later saved.
    comm: MPI.Intracomm, shape (1,)
        The MPI communicator to use for parallel runs.
    M_as_Mat: None or PETSc.Mat, shape (n, m)
        PETSc matrix, either complex or real, with the same elements as M_as_ndarray.
        If the user provides a matrix, then this matrix is reused and returned.
        If None is provided, then a new matrix is created.

    Returns
    -------
    M_as_Mat : PETSc.Mat, shape (n, m)
        PETSc matrix, either complex or real, with the same elements as M_as_ndarray.
    """
    # Get info on csc sparse array
    n_rows, n_cols = M_as_ndarray.shape

    # Create array to save
    if M_as_Mat is None:
        # If M_as_Mat was not provided create a new one, otherwise reuse it
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
