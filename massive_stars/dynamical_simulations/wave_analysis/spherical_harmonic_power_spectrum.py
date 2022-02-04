"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    spherical_harmonic_power_spectrum.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_wave_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --plot_only
    --field=<f>                         If specified, only transform this field
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

from plotpal.file_reader import SingleTypeReader as SR
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

star_file = '../mesa_stars/nccs_40msol/ballShell_nccs_B96_S96_Re1e3_de1.5.h5'
with h5py.File(star_file, 'r') as f:
    tau = f['tau'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L'][()]
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2plateau_sim = f['N2plateau'][()] * f['tau'][()]**2
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

# Create Plotter object, tell it which fields to plot
if 'SH_transform' in data_dir:
    out_dir = data_dir.replace('SH_transform', 'SH_power')
else:
    out_dir = 'SH_power_spectra'
full_out_dir = '{}/{}'.format(root_dir, out_dir)
reader = SR(root_dir, data_dir, out_dir, start_file=start_file, n_files=n_files, distribution='single')
if args['--field'] is None:
    with h5py.File(reader.files[0], 'r') as f:
        fields = list(f['tasks'].keys())
else:
    fields = [args['--field'],]


if not args['--plot_only']:
    times = []
    print('getting times...')
    ells = ms = None
    while reader.writes_remain():
        dsets, ni = reader.get_dsets([])
        times.append(reader.current_file_handle['time'][ni])
        if ells is None and ms is None:
            ells = reader.current_file_handle['ells'][()]
            ms = reader.current_file_handle['ms'][()]

    times = np.array(times)

    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'w') as wf:
        wf['ells']  = ells
        wf['times']  = times
    with h5py.File('{}/transforms.h5'.format(full_out_dir), 'w') as wf:
        wf['ells']  = ells

    #TODO: only load in one ell and m at a time, that'll save memory.
    for i, f in enumerate(fields):
        print('reading field {}'.format(f))

        print('filling datacube...')
        writes = 0
        first = True
        while reader.writes_remain():
            print('reading file {}...'.format(reader.current_file_number+1))
            dsets, ni = reader.get_dsets([f,])
            if first:
                shape = list(dsets[f][()].shape)[:-1]
                shape[0] = times.shape[0]
                shape[-1] = ms.shape[2]
                shape[-2] = ells.shape[1]
                data_cube = np.zeros(shape, dtype=np.complex128)
                first = False
            data_cube[writes,:] = dsets[f][ni,:].squeeze()
            writes += 1
        print(data_cube.shape)

        print('taking transform')
        transform = np.zeros(data_cube.shape, dtype=np.complex128)
        for ell in range(data_cube.shape[1]):
            print('taking transforms on {}: {}/{}'.format(f, ell+1, data_cube.shape[1]))
            for m in range(data_cube.shape[2]):
                if m > ell: continue
                if len(data_cube.shape) == 4:
                    for v in range(data_cube.shape[1]):
                        freqs, transform[:,v,ell,m] = clean_cfft(times, data_cube[:,v,ell,m])
                else:
                    freqs, transform[:,ell,m] = clean_cfft(times, data_cube[:,ell,m])
#        transform -= np.mean(transform, axis=0) #remove the temporal average; don't care about it.
        gc.collect()
        freqs0 = freqs
        if len(data_cube.shape) == 4:
            full_power = []
            for v in range(data_cube.shape[1]):
                freqs, this_power = normalize_cfft_power(freqs0, transform[:,v,:])
                full_power.append(this_power)
            full_power = np.array(full_power)
        else:
            freqs, full_power = normalize_cfft_power(freqs0, transform)
        power_per_ell = np.sum(full_power, axis=-1) #sum over m's
        with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r+') as wf:
            wf['{}_power_per_ell'.format(f)] = power_per_ell
            if len(data_cube.shape) == 4:
                wf['{}_ell1_data'.format(f)] = data_cube[:, :, 1, :]
            else:
                wf['{}_ell1_data'.format(f)] = data_cube[:, 1, :]
            if i == 0:
                wf['freqs'] = freqs 
                wf['freqs_inv_day'] = freqs/tau
        with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r+') as wf:
            wf['{}_cft'.format(f)] = transform
            if i == 0:
                wf['freqs'] = freqs0
                wf['freqs_inv_day'] = freqs0/tau


powers_per_ell = OrderedDict()
with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as out_f:
    for f in fields:
        powers_per_ell[f] = out_f['{}_power_per_ell'.format(f)][()]
    ells = out_f['ells'][()]
    freqs = out_f['freqs'][()]


good = freqs >= 0
min_freq = 3e-1
max_freq = freqs.max()
for k, powspec in powers_per_ell.items():
    if len(powspec.shape) != 3:
        powspec = np.expand_dims(powspec, axis=0)
    full_powspec = np.copy(powspec)
    for v in range(full_powspec.shape[0]):
        powspec = full_powspec[v,:]
        good_axis = np.arange(len(powspec.shape))[np.array(powspec.shape) == len(ells.flatten())][0]
                

        print(k, good_axis, powspec.shape, len(ells.flatten()))
        sum_power = np.sum(powspec, axis=good_axis).squeeze()
        sum_power_ell2 = np.sum((powspec/ells[:,:,0]**2).reshape(powspec.shape)[:,ells[0,:,0] > 0], axis=good_axis).squeeze()
            
        ymin = sum_power[(freqs > min_freq)*(freqs < max_freq)][-1].min()/2
        ymax = sum_power[(freqs > min_freq)*(freqs <= max_freq)].max()*2

        plt.figure()
        if len(sum_power.shape) > 1:
            for i in range(sum_power.shape[1]):
                plt.plot(freqs[good], sum_power[good, i], c = 'k', label=r'axis {}, sum over $\ell$ values'.format(i))
                plt.plot(freqs[good], sum_power_ell2[good, i], c = 'orange', label=r'axis {}, sum over $\ell$ values with $\ell^{-2}$'.format(i))
        else:
            plt.plot(freqs[good], sum_power[good], c = 'k', label=r'sum over $\ell$ values')
            plt.plot(freqs[good], sum_power_ell2[good], c = 'orange', label=r'sum over $\ell$ values with $\ell^{-2}$')
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'Power ({})'.format(k))
        plt.xlabel(r'Frequency (sim units)')
        plt.axvline(np.sqrt(N2plateau_sim)/(2*np.pi), c='k')
        plt.xlim(min_freq, max_freq)
        plt.ylim(ymin, ymax)
        plt.legend(loc='best')
        k_out = k.replace('(', '_').replace(')', '_').replace('=', '')

        plt.savefig('{}/{}_v{}_summed_power.png'.format(full_out_dir, k_out, v), dpi=600)

        plt.clf()

        for ell in range(1, 11):
            plt.loglog(freqs[good], powspec[:,ells.flatten()==ell], c='k')
            plt.xlim(min_freq, max_freq)
            plt.ylim(ymin, ymax)
            plt.title('ell={}'.format(ell))
            plt.xlabel('frequency (sim units)')
            plt.ylabel('Power')
            plt.savefig('{}/{}_v{}_ell_{:03d}.png'.format(full_out_dir, k_out, v, ell), dpi=200, bbox_inches='tight')
            plt.clf()
