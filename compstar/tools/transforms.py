from collections import OrderedDict

import numpy as np
import h5py
from mpi4py import MPI

import dedalus.public as d3


class SHTransformer():
    """ 
    Class which takes spherical harmonic transforms of a Dedalus scalar or vector field 
    which is defined on an S2 sphere in grid space (i.e. phi, theta). 
    """

    def __init__(self, nphi, ntheta, dtype=np.float64, dealias=1, radius=1):
        """
        Initialize the transformer object.
    
        Parameters
        ----------
        nphi : int
            Number of phi grid points used in the Dedalus output task.
        ntheta : int
            Number of theta grid points used in the Dedalus output task.
        dtype : numpy dtype
            Data type of the Dedalus output task.
        dealias : int
            Dealiasing factor used in the Dedalus output task.
        radius : float
            Radius of the sphere used in the Dedalus output task.
        """
        resolution = (nphi, ntheta)
        self.dtype = dtype
        if self.dtype not in (np.float64, np.complex128):
            raise ValueError("Invalid dtype")

        # Create coordinate system, distributor, and sphere basis.
        c = d3.SphericalCoordinates('phi', 'theta', 'r')
        dealias_tuple = (dealias, dealias)
        self.dealias = dealias
        dist = d3.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
        basis = d3.SphereBasis(c.S2coordsys, resolution, radius=radius, dtype=dtype, dealias=dealias_tuple)
        self.phi, self.theta = basis.global_grids(basis.dealias)

        # Create fields which will be used to take transforms
        self.scalar_field = dist.Field(bases=basis)
        self.vector_field = dist.VectorField(bases=basis, coordsys=c)
        self.conj_vector_field = dist.VectorField(bases=basis, coordsys=c)
        self.power_scalar_op = d3.Average(self.scalar_field*np.conj(self.scalar_field))
        self.power_vector_op = d3.Average(self.conj_vector_field@self.vector_field)

        # Create a dictionary with keys 'ell,m' (e.g., '1,1') which contain index slices of that ell and m in the dedalus fields.
        self.ell_values = []
        self.m_values = []

        self.slices = dict()
        for i in range(self.scalar_field['c'].shape[0]):
            for j in range(self.scalar_field['c'].shape[1]):
                groups = basis.elements_to_groups((False, False), (np.array((i,)),np.array((j,))))
                m = groups[0][0]
                ell = groups[1][0]
                key = '{},{}'.format(ell, m)
                this_slice = (slice(i, i+1, 1), slice(j, j+1, 1))
                if key not in self.slices.keys():
                    self.slices[key] = [this_slice]
                else:
                    self.slices[key].append(this_slice)

                if ell not in self.ell_values:
                    self.ell_values.append(ell)
                if m not in self.m_values:
                    self.m_values.append(m)
        self.ell_values = np.sort(self.ell_values)[:,None]
        self.m_values = np.sort(self.m_values)[None,:]
        self.out_field = None
        self.vector = False

    def transform_scalar_field(self, grid_data, normalization=1/2):
        """ 
        Takes spherical harmonic transform of a scalar field.
        
        Parameters
        ----------
        grid_data : numpy array
            Array of shape (nphi, ntheta, 1) which contains the scalar field data.
        normalization : float
            Normalization factor to apply to the transform to satisfy Parseval's theorem. Default is 1/2.
        """
        self.vector = False
        self.scalar_field.change_scales(self.dealias)
        self.scalar_field['g'] = grid_data
        power_grid = self.power_scalar_op.evaluate()['g'].ravel()[0]

        out_field = np.zeros((self.ell_values.size, self.m_values.size), dtype=np.complex128)
        #loop over ell and m values and store the transform coefficients
        for i, ell in enumerate(self.ell_values.ravel()):
            for j, m in enumerate(self.m_values.ravel()):
                sl_key = '{},{}'.format(ell,m)
                if sl_key in self.slices.keys():
                    if self.dtype is np.complex128:
                        out_field[i,j] = self.scalar_field['c'][self.slices[sl_key][0]].ravel()
                        out_field[i,j] *= np.sqrt(2)*normalization
                    else:
                        cosmphi, sinmphi = self.slices[sl_key]
                        out_field[i,j] = self.scalar_field['c'][cosmphi].ravel()[0] \
                                        + 1j*self.scalar_field['c'][sinmphi].ravel()[0]
                        out_field[i,j] *= normalization
                        if m == 0:
                            out_field[i,j] *= np.sqrt(2) #normalize so that sum(coeff*conj(coeffs)) == 4*s2_avg(scalar_field**2)
        #check power
        power_transform = np.sum(out_field * np.conj(out_field)).real
        if (np.allclose(power_transform, 0) and np.allclose(power_grid, 0)) or np.allclose(power_transform/power_grid, 1): 
            self.out_field = out_field
            return out_field
        else:
            self.out_field = None
            raise ValueError("Scalar Transform is not conserving power; ratio: {}, vals: {}, {}".format(power_transform/power_grid, power_transform, power_grid))

    def transform_vector_field(self, grid_data, normalization=1/2):
        """ 
        Takes spherical harmonic transform of a scalar field.
        
        Parameters
        ----------
        grid_data : numpy array
            Array of shape (coords, nphi, ntheta, 1) which contains the scalar field data.
        normalization : float
            Normalization factor to apply to the transform to satisfy Parseval's theorem. Default is 1/2.
        """
        self.vector = True
        self.vector_field.change_scales(self.dealias)
        self.conj_vector_field.change_scales(self.dealias)
        self.vector_field['g'] = grid_data
        self.conj_vector_field['g'] = np.conj(self.vector_field['g'])
        power_grid = self.power_vector_op.evaluate()['g'].ravel()[0]

        out_field = np.zeros((grid_data.shape[0], self.ell_values.size, self.m_values.size), dtype=np.complex128)
        #loop over ell and m values and store the transform coefficients
        for i, ell in enumerate(self.ell_values.ravel()):
            for j, m in enumerate(self.m_values.ravel()):
                sl_key = '{},{}'.format(ell,m)
                if sl_key in self.slices.keys():
                    for v  in range(grid_data.shape[0]):
                        if self.dtype is np.complex128:
                            index = (slice(v, v+1, 1), *self.slices[sl_key][0])
                            out_field[v,i,j] = self.vector_field['c'][index].ravel()
                            out_field[v,i,j] *= np.sqrt(2)*normalization
                        else:
                            cosmphi, sinmphi = self.slices[sl_key]
                            cosmphi = (slice(v, v+1, 1), *cosmphi)
                            sinmphi = (slice(v, v+1, 1), *sinmphi)
                            out_field[v,i,j] = self.vector_field['c'][cosmphi].ravel()[0] \
                                             + 1j*self.vector_field['c'][sinmphi].ravel()[0]
                            out_field[v,i,j] *= normalization
                            if m == 0:
                                out_field[v,i,j] *= np.sqrt(2) #normalize so that sum(coeff*conj(coeffs)) == 4*s2_avg(vector_field**2)
        #check power
        power_transform = np.sum(out_field * np.conj(out_field)).real
        if (np.allclose(power_transform, 0) and np.allclose(power_grid, 0)) or np.allclose(power_transform/power_grid, 1): 
            self.out_field = out_field
            return out_field
        else:
            self.out_field = None
            raise ValueError("Vector Transform is not conserving power; ratio: {}, vals: {}, {}".format(power_transform/power_grid, power_transform, power_grid))

    def get_ell_m_value(self, ell, m):
        """ Returns the transform coefficient for a given ell and m value. """
        if self.out_field is None:
            raise ValueError("Must transform field before finding ell,m value")

        index = (self.ell_values == ell)*(self.m_values == m)
        if self.vector:
            return self.out_field[:,index]
        else:
            return self.out_field[index]

class DedalusShellSHTransformer():
    """ 
    A wrapper which takes all tasks in a dedalus output file and performs a spherical harmonic transform on them.
    Assumes all output tasks are defined on S2.

    Relies on plotpal for file reading.
    """
    def __init__(self, nphi, ntheta, root_dir, data_dir, dtype=np.float64, dealias=1, radius=1, **kwargs):
        """
        Instantiates a DedalusShellSHTransformer object, including a file reader.

        Parameters
        ----------
        nphi, ntheta : int
            Number of phi, theta grid points that the grid data is expanded on.
        root_dir : str
            Root directory of the dedalus simulation handlers.
        data_dir : str
            Directory of the dedalus output files (handler name).
        dtype : numpy dtype
            Data type of the grid data. Default is np.float64.
        dealias : int
            Dealiasing factor of the dedalus simulation. Default is 1.
        radius : float
            Radius of the sphere. Default is 1.
        **kwargs : dict
            Keyword arguments to pass to the file reader.     
        """
        from plotpal.file_reader import SingleTypeReader as SR
        self.root_dir = root_dir
        self.out_dir = 'SH_transform_{}'.format(data_dir)
        self.reader = SR(root_dir, data_dir, self.out_dir, distribution='even-file', **kwargs) #important to use even-file so that one input file -> one output file.

        # Parameters
        self.transformer = SHTransformer(nphi, ntheta, dtype=dtype, dealias=dealias, radius=radius)
        if not self.reader.idle:
            with h5py.File(self.reader.files[0], 'r') as f:
                self.fields = list(f['tasks'].keys())
        else:
            self.fields = None

    def write_transforms(self):
        """
        Reads in dedalus output from all output files, takes SH transforms, and writes transformmed fields to a new file.
        """
        if not self.reader.idle:
            while self.reader.writes_remain():
                dsets, ni = self.reader.get_dsets(self.fields)
                file_name = self.reader.current_file_name
                file_num = int(file_name.split('_s')[-1].split('.h5')[0])
                if ni == 0:
                    file_mode = 'w'
                else:
                    file_mode = 'a'
                output_file_name = '{}/{}/{}_s{}.h5'.format(self.root_dir, self.out_dir, self.out_dir, file_num)

                
                with h5py.File(output_file_name, file_mode) as of:
                    #Set up output file; output ells, ms, time, etc.
                    sim_times = self.reader.current_file_handle['scales/sim_time'][()]
                    write_numbers = self.reader.current_file_handle['scales/write_number'][()]
                    if ni == 0:
                        of['ells'] = self.transformer.ell_values[None,:,:,None]
                        of['ms']   = self.transformer.m_values[None,:,:,None]
                        of['time'] = sim_times[()]
                        of['write_number'] = write_numbers[()]
                        for attr in ['writes', 'set_number', 'handler_name']:
                            of.attrs[attr] = self.reader.current_file_handle.attrs[attr]

                    #Transform each field and save to output file.
                    for f in self.fields:
                        task_data = dsets[f][ni,:]
                        if len(task_data.squeeze().shape) == len(self.transformer.vector_field['g'].squeeze().shape):
                            out_field = self.transformer.transform_vector_field(task_data)
                        else:
                            out_field = self.transformer.transform_scalar_field(task_data)
                        if ni == 0:
                            of.create_dataset(name='tasks/'+f, shape=(len(sim_times),)+out_field.shape, dtype=np.complex128)
                        of['tasks/'+f][ni,:] =  np.copy(out_field)



