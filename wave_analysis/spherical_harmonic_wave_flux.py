"""
This script computes the wave flux in a d3 spherical simulation

Usage:
    spherical_harmonci_wave_flux.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_wave_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --no_ft                             Do the base fourier transforms
"""
import re
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
from scipy import sparse
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

from power_spectrum_functions import clean_cfft, normalize_cfft_power

args = docopt(__doc__)
res = re.compile('(.*)\(r=(.*)\)')

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
    rB = f['rB'][()]
    rS = f['rS'][()]
    ρB = np.exp(f['ln_ρB'][()])
    ρS = np.exp(f['ln_ρS'][()])
    r = np.concatenate((rB.flatten(), rS.flatten()))
    ρ = np.concatenate((ρB.flatten(), ρS.flatten()))
    rho_func = interp1d(r,ρ)
    tau = f['tau'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L'][()]
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

# Create Plotter object, tell it which fields to plot
out_dir = 'SH_wave_flux_spectra'.format(data_dir)
full_out_dir = '{}/{}'.format(root_dir, out_dir)
reader = SR(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
with h5py.File(reader.files[0], 'r') as f:
    fields = list(f['tasks'].keys())
radii = []
for f in fields:
    if res.match(f):
        radius = float(f.split('r=')[-1].split(')')[0])
        if radius not in radii:
            radii.append(radius)
if not args['--no_ft']:
    times = []
    print('getting times...')
    first = True
    while reader.writes_remain():
        dsets, ni = reader.get_dsets([])
        times.append(reader.current_file_handle['time'][ni])
        if first:
            ells = reader.current_file_handle['ells'][()]
            ms = reader.current_file_handle['ms'][()]
            first = False

    times = np.array(times)

    with h5py.File('{}/transforms.h5'.format(full_out_dir), 'w') as wf:
        wf['ells']  = ells

    #TODO: only load in one ell and m at a time, that'll save memory.
    for i, f in enumerate(fields):
        print('reading field {}'.format(f))
        data_cube = np.zeros((times.shape[0], ells.shape[1], ms.shape[2]), dtype=np.complex128)

        print('filling datacube...')
        writes = 0
        while reader.writes_remain():
            print('reading file {}...'.format(reader.current_file_number+1))
            dsets, ni = reader.get_dsets([])
            rf = reader.current_file_handle
            data_cube[writes,:] = rf['tasks'][f][ni,:].squeeze()
            writes += 1

        print('taking transform')
        transform = np.zeros(data_cube.shape, dtype=np.complex128)
        for ell in range(data_cube.shape[1]):
            print('taking transforms {}/{}'.format(ell+1, data_cube.shape[1]))
            for m in range(data_cube.shape[2]):
                if m > ell: continue
                freqs, transform[:,ell,m] = clean_cfft(times, data_cube[:,ell,m])
        transform -= np.mean(transform, axis=0) #remove the temporal average; don't care about it.
        del data_cube
        gc.collect()

        with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r+') as wf:
            wf['{}_cft'.format(f)] = transform
            if i == 0:
                wf['freqs'] = freqs 
                wf['freqs_inv_day'] = freqs/tau

#Get spectrum = rho*(real(ur*conj(p)))
with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r+') as wf:
    if 'wave_flux(r={})'.format(radii[0]) not in wf.keys():
        raw_freqs = wf['freqs'][()]
        raw_freqs_invDay = wf['freqs_inv_day'][()]
        for i, radius in enumerate(radii):
            ur = wf['u(r={})_cft'.format(radius)][()]
            p  = wf['pomega(r={})_cft'.format(radius)][()]
            spectrum = radius**2*rho_func(radius)*(ur*np.conj(p)).real
            # Collapse negative frequencies
            for f in raw_freqs:
                if f < 0:
                    spectrum[raw_freqs == -f] += spectrum[raw_freqs == f]
            # Sum over m's.
            spectrum = spectrum[raw_freqs >= 0,:]
            spectrum = np.sum(spectrum, axis=2)
            wf['wave_flux(r={})'.format(radius)] = spectrum
            if i == 0:
                wf['real_freqs'] = raw_freqs[raw_freqs >= 0]
                wf['real_freqs_inv_day'] = raw_freqs_invDay[raw_freqs_invDay >= 0]
    with h5py.File('{}/wave_flux.h5'.format(full_out_dir), 'w') as of:
        print('saving wave flux at r={}'.format(radii[1]))
        of['wave_flux'] = wf['wave_flux(r={})'.format(radii[1])][()]
        of['real_freqs'] = wf['real_freqs'][()]
        of['real_freqs_inv_day'] = wf['real_freqs_inv_day'][()]
        of['ells'] = wf['ells'][()]
        

fig = plt.figure()
freqs_for_dfdell = [1, 5, 8]
with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as rf:
    freqs = rf['real_freqs'][()]
    ells = rf['ells'][()].flatten()
    for f in freqs_for_dfdell:
        print('plotting f = {}'.format(f))
        f_ind = np.argmin(np.abs(freqs - f))
        for radius in radii:
            if radius < 1: continue
            wave_flux = rf['wave_flux(r={})'.format(radius)][f_ind, :]
            plt.loglog(ells, ells*wave_flux, label='r={}'.format(radius))
            if radius == radii[1]:
                shift = (ells*wave_flux)[ells == 2]
        plt.loglog(ells, shift*(ells/2)**4, c='k', label=r'$\ell^4$')
        plt.legend(loc='best')
        plt.title('f = {} 1/day'.format(f))
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\frac{\partial^2 F}{\partial\ln\ell}$')
        plt.ylim(1e-25, 1e-9)
        plt.xlim(1, ells.max())
        fig.savefig('{}/ell_spectrum_freq{}.png'.format(full_out_dir, f), dpi=300, bbox_inches='tight')
        plt.clf()
    
 
for ell in range(11):
    if ell == 0: continue
    print('plotting ell = {}'.format(ell))
    with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as rf:
        freqs = rf['real_freqs'][()]
        for radius in radii:
            if radius < 1: continue
            wave_flux = rf['wave_flux(r={})'.format(radius)][:,ell]
            plt.loglog(freqs, freqs*wave_flux, label='r={}'.format(radius))
            if radius == radii[1]:
                shift = (freqs*wave_flux)[freqs > 1][0]
    plt.loglog(freqs, shift*10*(freqs/freqs[freqs > 1][0])**(-13/2), c='k', label=r'$f^{-13/2}$')
    plt.legend(loc='best')
    plt.title('ell={}'.format(ell))
    plt.xlabel('freqs (sim units)')
    plt.ylabel(r'$\frac{\partial^2 F}{\partial \ln f}$')
    plt.ylim(1e-25, 1e-9)
    fig.savefig('{}/freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
    plt.clf()
    
    

