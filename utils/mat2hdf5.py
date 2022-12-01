import h5py
from scipy.io import loadmat
import numpy

# def convert_hdf5(filename, out=None):
#
#     ref5 = h5py.File(filename,'r')
#     if out is None:
#         out = filename.split('.')[0]+'_copy.hdf5'
#     new5 = h5py.File(out,'w')
#
#
#     for grp in list(ref5.keys()):
#         new5.create_group(grp)
#         new5[grp].attrs.update(ref5[grp].attrs)
#         new5[grp].create_dataset('data',data=ref5[grp]['data'])
#         new5[grp].create_dataset('ir',data=ref5[grp]['ir'])
#         new5[grp].create_dataset('jc',data=ref5[grp]['jc'])
#
#     ref5.close()
#     new5.close()

def mat2hdf5(mat_filename, keys, hdf5_filename=None):

    ref = loadmat(mat_filename)

    if hdf5_filename is None:
        hdf5_filename = mat_filename.split('.')[0]+'_copy.hdf5'

    new5 = h5py.File(hdf5_filename, 'w')


    for grp in keys:

        original_mat = ref[grp]

        new5.create_group(grp)
        new5[grp].attrs.create('MATLAB_class', b'complex')
        new5[grp].attrs.create('MATLAB_sparse',original_mat.shape[0])

        new_data = original_mat.data + 0.0j
        new5[grp].create_dataset('data',data=new_data) #original_mat.data)
        new5[grp].create_dataset('ir',data=original_mat.indices)
        new5[grp].create_dataset('jc',data=original_mat.indptr)

    # return new5
    new5.close()

if __name__ == "__main__":
    mat_filename = 'D2N10P1_transformed.mat'
    keys = ['A','M','C']
    mat2hdf5(mat_filename, keys)
