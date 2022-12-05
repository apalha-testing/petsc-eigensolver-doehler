import scipy
import random

from petsc4py import PETSc
from mpi4py import MPI

import petscIO

file_format = "binary"
mat_filename = "D2N10P1.mat"
petsc_filename = "D2N10P1_petsc.dat"
data_keys = ['A', 'M', 'C']

# Read data from Mat file ------------------------------------------------------
matlab_data = scipy.io.loadmat(mat_filename)

# Reading the data returns a dictionary with the keys of the data we saved plus
# some additional internal keys with internal information. To get only the data
# we saved, we ignore the internal keys
keys_to_ignore = ['__header__', '__version__', '__globals__']
data_keys_in_matlab = list(matlab_data.keys() - set(keys_to_ignore))
if not set(data_keys).issubset(set(data_keys_in_matlab)):
    raise ValueError(f"Not all object to read are present in matlab data file!")
# ------------------------------------------------------------------------------


# Convert matlab data to PETSc format ------------------------------------------
petsc_data = {key: petscIO.matlab2petsc(matlab_data.get(key, None), key) for key in data_keys}
# ------------------------------------------------------------------------------


# Save PETSc data to file ------------------------------------------------------
petscIO.save_to_file(petsc_data, petsc_filename, file_format)
# ------------------------------------------------------------------------------


# Read PETSc data from file ------------------------------------------------------
petsc_data_from_file = petscIO.read_from_file(petsc_filename, data_keys, [True, True, True], file_format, as_list = True)
# ------------------------------------------------------------------------------
