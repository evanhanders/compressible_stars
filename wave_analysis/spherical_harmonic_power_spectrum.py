"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_mollweide_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_surface_shell_slices]
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

from power_spectrum_functions import clean_cfft, normalize_cfft_power

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
out_dir = 'SH_power_spectra'.format(data_dir)
full_out_dir = '{}/{}'.format(root_dir, out_dir)
if not args['--plot_only']:
    plotter = SFP(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
    fields = ['s1_surf',]#, 'u_theta_surf',]

    times = []
    print('getting times...')
    while plotter.files_remain([], fields):
        print('reading file {}...'.format(plotter.current_filenum+1))
        file_name = plotter.files[plotter.current_filenum]
        with h5py.File('{}'.format(file_name), 'r') as f:
            if plotter.current_filenum == 0:
                ells = f['ells'][()]
                ms = f['ms'][()]
            times.append(f['time'][()])
        plotter.current_filenum += 1

    times = np.concatenate(times)
    data_cube = np.zeros((times.shape[0], ells.shape[1], ms.shape[2]), dtype=np.complex128)

    print('filling datacube...')
    writes = 0
    while plotter.files_remain([], fields):
        print('reading file {}...'.format(plotter.current_filenum+1))
        file_name = plotter.files[plotter.current_filenum]
        with h5py.File('{}'.format(file_name), 'r') as f:
            this_file_writes = len(f['time'][()])
            data_cube[writes:writes+this_file_writes,:] = f['s1_surf'][:,:,:]
            writes += this_file_writes
        plotter.current_filenum += 1

    print('taking transform')
    transform = np.zeros(data_cube.shape, dtype=np.complex128)
    for ell in range(data_cube.shape[1]):
        print('taking transforms {}/{}'.format(ell+1, data_cube.shape[1]))
        for m in range(data_cube.shape[2]):
            if m > ell: continue
            freqs, transform[:,ell,m] = clean_cfft(times, data_cube[:,ell,m])
    del data_cube
    gc.collect()
    freqs, full_power = normalize_cfft_power(freqs, transform)
    del transform
    gc.collect()

    power_per_ell = np.sum(full_power, axis=2) #sum over m's
    del full_power
    gc.collect()

    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'w') as f:
        f['power_per_ell'] = power_per_ell
        f['ells']  = ells
        f['freqs'] = freqs 
        f['freqs_inv_day'] = freqs/tau
else:  
    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as f:
        power_per_ell = f['power_per_ell'][()]
        ells = f['ells'][()]
        freqs = f['freqs'][()]

freqs /= tau
good = freqs >= 0
min_freq = 1e-1
max_freq = freqs.max()
sum_power = np.sum(power_per_ell, axis=1)
sum_power2 = np.zeros(power_per_ell.shape[0])
#sum_power2 = np.sum(power_per_ell, axis=1)
for i, ell in enumerate(ells.flatten()):
    if ell <= 10:
        sum_power2 += power_per_ell[:,i].real
        
ymin = sum_power[(freqs > 5e-2)*(freqs < max_freq)][-1].min()/2
ymax = sum_power[(freqs > 5e-2)*(freqs <= max_freq)].max()*2
print(ymin, ymax)

plt.plot(freqs[good], sum_power2[good], c = 'green', ls='--', label=r'$\ell <= 10$')
plt.plot(freqs[good], sum_power[good], c = 'k', label=r'all $\ell$ values')
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'Power (simulation s1 units squared)')
plt.xlabel(r'Frequency (day$^{-1}$)')
plt.axvline(np.sqrt(N2plateau)/(2*np.pi), c='k')
plt.xlim(min_freq, max_freq)
plt.ylim(ymin, ymax)
plt.legend(loc='best')
plt.savefig('{}/summed_power.png'.format(full_out_dir), dpi=600)

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
for i, ell in enumerate(ells.flatten()):
    if ell == 0:
        continue
    plt.plot(freqs[good], power_per_ell[good,i].real/ell, c='k')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.text(0.05, 0.95, r'$\ell = {{{}}}$'.format(ell), transform=ax1.transAxes)
    ax1.set_ylabel(r'Power$/\ell$ (simulation s1 units squared)')
    ax1.set_xlabel(r'Frequency (day$^{-1}$)')
    ax1.set_xlim(min_freq, max_freq)
    ax1.set_ylim(ymin, ymax)
    fig.savefig('{}/power_ell{}.png'.format(full_out_dir, ell), dpi=600)
    ax1.cla()
    if ell > 10:
        break