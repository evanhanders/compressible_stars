"""
Dedalus script for Boussinesq convection in a sphere (spherical, rotating, rayleigh-benard convection).

This script is functional with and dependent on commit 7eda513 of the d3_eval_backup branch of dedalus (https://github.com/DedalusProject/dedalus).
This script is also functional with and dependent on commit 7eac019f of the d3 branch of dedalus_sphere (https://github.com/DedalusProject/dedalus_sphere/tree/d3).
While the inputs are Ra, Ek, and Pr, the control parameters are the modified Rayleigh number (Ram), the Prandtl number (Pr) and the Ekman number (Ek).
Ram = Ra * Ek / Pr, where Ra is the traditional Rayleigh number.

Usage:
    ballShell_split_interped_check.py <root_dir> --res_fracB=<f> --res_fracS_<f> --mesh=<m> --mesa_file=<f> [options]
    ballShell_split_interped_check.py <root_dir> --L_frac=<f> --N_fracB=<f> --N_fracS=<f> --mesh=<m> --mesa_file=<f> [options]

Options:
    --L=<L>                    initial resolution
    --NB=<N>                    initial resolution
    --NS=<N>                    initial resolution
    --node_size=<n>            Size of node; use to reduce memory constraints for large files
"""
import h5py
from fractions import Fraction
import time
from collections import OrderedDict

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'
import dedalus_sphere

from output.averaging    import VolumeAverager, EquatorSlicer, PhiAverager
from output.writing      import ScalarWriter,  MeridionalSliceWriter, EquatorialSliceWriter, SphericalShellWriter

import logging
logger = logging.getLogger(__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING) 

args   = docopt(__doc__)
root_dir = args['<root_dir>']
if args['--res_fracB'] is not None:
    L_fracB = N_fracB = float(Fraction(args['--res_fracB']))
    L_fracS = N_fracS = float(Fraction(args['--res_fracS']))
else:
    L_fracB = L_fracS = float(Fraction(args['--L_frac']))
    N_fracB = float(Fraction(args['--N_fracB']))
    N_fracS = float(Fraction(args['--N_fracS']))

res_strings = root_dir.split('Re')[-1].split('_')[1:3]
res_strB, res_strS = res_strings
resolutions = [[], []]
if args['--L'] is None or args['--NS'] is None or args['--NB'] is None:
    for r in res_strB.split('x'):
        if r[-1] == '/':
            resolutions[0].append(int(r[:-1]))
        else:
            resolutions[0].append(int(r))
    for r in res_strS.split('x'):
        if r[-1] == '/':
            resolutions[1].append(int(r[:-1]))
        else:
            resolutions[1].append(int(r))
        
else:
    resolutions[0] = [int(args['--L']), int(args['--NB'])]
    resolutions[1] = [int(args['--L']), int(args['--NS'])]
LmaxB, NmaxB = resolutions[0]
LmaxS, NmaxS = resolutions[1]


new_LmaxB = int((LmaxB+2)*L_fracB) - 2
new_LmaxS = int((LmaxS+2)*L_fracS) - 2
new_NmaxB = int((NmaxB+1)*N_fracB) - 1
new_NmaxS = int((NmaxS+1)*N_fracS) - 1
dealias = 1
dtype   = np.float64
with h5py.File(args['--mesa_file'], 'r') as f:
    r_inner = f['r_inner'][()]
    r_outer = f['r_outer'][()]
mesh    = args['--mesh'].split(',')
mesh = [int(m) for m in mesh]

# Bases
c       = coords.SphericalCoordinates('φ', 'θ', 'r')
d       = distributor.Distributor((c,), mesh=mesh)
b2B       = basis.BallBasis(c, (2*(new_LmaxB+2), new_LmaxB+1, new_NmaxB+1), radius=r_inner, dtype=dtype, dealias=(dealias, dealias, dealias))
b2S       = basis.SphericalShellBasis(c, (2*(new_LmaxS+2), new_LmaxS+1, new_NmaxS+1), radii=(r_inner, r_outer), dtype=dtype, dealias=(dealias, dealias, dealias))
φB2, θB2, rB2    = b2B.local_grids((dealias, dealias, dealias))
φBg2, θBg2, rBg2 = b2B.global_grids((dealias, dealias, dealias))
φS2,  θS2,  rS2  = b2S.local_grids((dealias, dealias, dealias))
φSg2, θSg2, rSg2 = b2S.global_grids((dealias, dealias, dealias))

uB2 = field.Field(dist=d, bases=(b2B,), tensorsig=(c,), dtype=dtype)
sB2 = field.Field(dist=d, bases=(b2B,), dtype=dtype)
uS2 = field.Field(dist=d, bases=(b2S,), tensorsig=(c,), dtype=dtype)
sS2 = field.Field(dist=d, bases=(b2S,), dtype=dtype)


check_str = 'checkpoint_LB{:.2f}_NB{:.2f}_LS{:.2f}_NS{:.2f}'.format(L_fracB, N_fracB, L_fracS, N_fracS)
out_dir = '{:s}/{:s}'.format(root_dir, check_str)

node_size = args['--node_size']
if node_size is not None: 
    node_size = int(node_size)
else:
    node_size = 1

import sys
for i in range(node_size):
    if d.comm_cart.rank % node_size == i:
        print('reading on node rank {}'.format(i))
        sys.stdout.flush()
        with h5py.File('{:s}/{:s}_s1.h5'.format(out_dir, check_str), 'r') as f:
            rBg = f['rBg'][()]
            φBg = f['φBg'][()]
            θBg = f['θBg'][()]
            rSg = f['rSg'][()]
            φSg = f['φSg'][()]
            θSg = f['θSg'][()]

            rBgood = np.zeros_like(rBg, dtype=bool)
            φBgood = np.zeros_like(φBg, dtype=bool)
            θBgood = np.zeros_like(θBg, dtype=bool)
            rSgood = np.zeros_like(rSg, dtype=bool)
            φSgood = np.zeros_like(φSg, dtype=bool)
            θSgood = np.zeros_like(θSg, dtype=bool)
            for rv in rB2.flatten():
                if rv in rBg:
                    rBgood[0, 0, rBg.flatten() == rv] = True
            for φv in φB2.flatten():
                if φv in φBg:
                    φBgood[φBg.flatten() == φv, 0, 0] = True
            for θv in θB2.flatten():
                if θv in θBg:
                    θBgood[0, θBg.flatten() == θv, 0] = True
            for rv in rS2.flatten():
                if rv in rSg:
                    rSgood[0, 0, rSg.flatten() == rv] = True
            for φv in φS2.flatten():
                if φv in φSg:
                    φSgood[φSg.flatten() == φv, 0, 0] = True
            for θv in θS2.flatten():
                if θv in θSg:
                    θSgood[0, θSg.flatten() == θv, 0] = True

            global_goodB = rBgood*φBgood*θBgood
            global_goodS = rSgood*φSgood*θSgood
            for i in range(3):
                uB2['g'][i,:] = f['tasks']['uB'][()][i][global_goodB].reshape(uB2['g'][i,:].shape)
                uS2['g'][i,:] = f['tasks']['uS'][()][i][global_goodS].reshape(uS2['g'][i,:].shape)
            sB2['g'] = f['tasks']['s1B'][()][global_goodB].reshape(sB2['g'].shape)
            sS2['g'] = f['tasks']['s1S'][()][global_goodS].reshape(sS2['g'].shape)
            del global_goodB
            del global_goodS
    else:
        for i in range(3):
            uB2['g'][i,:] = uB2['g'][i,:]
            uS2['g'][i,:] = uS2['g'][i,:]
        sB2['g'] = sB2['g']
        sS2['g'] = sS2['g']
    d.comm_cart.barrier()

split_out_dir = '{:s}/{:s}_s1/'.format(out_dir, check_str)
import os
if d.comm_cart.rank == 0:
    if not os.path.exists('{:s}/'.format(split_out_dir)):
        os.makedirs('{:s}/'.format(split_out_dir))
d.comm_cart.Barrier()

with h5py.File('{:s}/{:s}_s1_p{}.h5'.format(split_out_dir, check_str, int(d.comm_cart.rank)), 'w') as f:
    task_group = f.create_group('tasks')
    f['tasks']['uB'] = np.expand_dims(uB2['c'], axis=0)
    f['tasks']['s1B'] = np.expand_dims(sB2['c'], axis=0)
    f['tasks']['uS'] = np.expand_dims(uS2['c'], axis=0)
    f['tasks']['s1S'] = np.expand_dims(sS2['c'], axis=0)
