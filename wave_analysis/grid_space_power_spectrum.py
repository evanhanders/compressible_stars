"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_mollweide_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: surface_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --plot_only
"""
import gc
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from scipy import sparse
from mpi4py import MPI

from plotpal.file_reader import SingleFiletypePlotter as SFP
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

args = docopt(__doc__)

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

star_file = '../mesa_stars/MESA_Models_Dedalus_Full_Sphere_6_ballShell/ballShell_nccs_B63_S63.h5'
with h5py.File(star_file, 'r') as f:
    tau = f['tau'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L'][()]
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

# Create Plotter object, tell it which fields to plot
out_dir = 'grid_power_spectra'.format(data_dir)
full_out_dir = '{}/{}'.format(root_dir, out_dir)
if not args['--plot_only']:
    # Parameters
    dtype = np.float64
    Lmax = int(root_dir.split('Re')[-1].split('_')[1].split('x')[0])

    # Bases
    dealias = 1
    c = coords.S2Coordinates('φ', 'θ')
    d = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF)
    b = basis.SWSH(c, (2*(Lmax+2), Lmax+1), radius=float(args['--radius']), dtype=dtype)
    φ, θ = b.local_grids((dealias, dealias))
    φg, θg = b.global_grids((dealias, dealias))

    weight_φ = np.gradient(φg.flatten()).reshape(φg.shape)#(np.ones_like(φg)*np.pi/((b.Lmax+2)*dealias))
    weight_θ = b.global_colatitude_weights(dealias)
    weight = weight_θ * weight_φ
    volume = np.sum(weight)

    field = field.Field(dist=d, bases=(b,), dtype=dtype)

    plotter = SFP(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
    fields = ['s1_surf',]#, 'u_theta_surf',]

    times = []
    print('getting times...')
    while plotter.files_remain([], fields):
        print('reading file {}...'.format(plotter.current_filenum+1))
        file_name = plotter.files[plotter.current_filenum]
        with h5py.File('{}'.format(file_name), 'r') as f:
            times.append(f['scales/sim_time'][()])
        plotter.current_filenum += 1

    times = np.concatenate(times)
    data_cube = np.zeros((times.shape[0], *tuple(field['g'].shape)), dtype=dtype)

    print('filling datacube...')
    writes = 0
    while plotter.files_remain([], fields):
        print('reading file {}...'.format(plotter.current_filenum+1))
        file_name = plotter.files[plotter.current_filenum]
        with h5py.File('{}'.format(file_name), 'r') as f:
            this_file_writes = len(f['scales/sim_time'][()].flatten())
            data_cube[writes:writes+this_file_writes,:] = f['tasks/s1_surf'][()].squeeze()
            writes += this_file_writes
        plotter.current_filenum += 1

    print('taking transform')
    dt = np.mean(np.gradient(times))
    freqs = np.fft.fftfreq(times.shape[0], d=dt)
    window = np.hanning(times.shape[0]).reshape((times.shape[0], 1))
    transform = np.zeros(tuple(data_cube.shape), dtype=np.complex128)
    full_power = np.zeros(tuple(data_cube.shape), dtype=np.float64)
    for i in range(data_cube.shape[2]):
        print('taking transform {}/{}'.format(i+1, data_cube.shape[2]))
        transform[:,:,i] = np.fft.fft(window*data_cube[:,:,i], axis=0)
        gc.collect()
    del window
    del data_cube
    gc.collect()
    full_power[:] = (transform*np.conj(transform)).real / (freqs.shape[0]/2)**2

    #Sum over theta and phi, use weights & volume to get vol_avg, then scale by true vol (4pi)
    power = 4*np.pi*np.sum(np.sum(weight*full_power, axis=2), axis=1)/volume
    del full_power
    gc.collect()

    with h5py.File('{}/grid_power_spectra.h5'.format(full_out_dir), 'w') as f:
        f['power'] = power
        f['freqs'] = freqs 
        f['freqs_inv_day'] = freqs/tau
else:  
    with h5py.File('{}/grid_power_spectra.h5'.format(full_out_dir), 'r') as f:
        power = f['power'][()]
        freqs = f['freqs'][()]

freqs /= tau
good = freqs >= 0
min_freq = 1e-1
max_freq = freqs.max()
        
ymin = power[(freqs > 5e-2)*(freqs < max_freq)][-1].min()/2
ymax = power[(freqs > 5e-2)*(freqs <= max_freq)].max()*2

plt.plot(freqs[good], power[good], c = 'k')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'Power (simulation s1 units squared)')
plt.xlabel(r'Frequency (day$^{-1}$)')
plt.axvline(np.sqrt(N2plateau)/(2*np.pi), c='k')
plt.xlim(min_freq, max_freq)
plt.ylim(ymin, ymax)
plt.legend(loc='best')
plt.savefig('{}/summed_power.png'.format(full_out_dir), dpi=600)
