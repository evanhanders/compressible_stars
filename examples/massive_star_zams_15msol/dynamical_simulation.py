"""
d3 script for anelastic convection in a stitched BallBasis and ShellBasis domain.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Note that options specified in a cnofig file override command line arguments.

Usage:
    ivp_ball_2shells_AN.py [options]
    ivp_ball_2shells_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 4]
    --nrB=<res>          Number of radial grid points in ball (Nmax+1)   [default: 24]
    --nrS1=<res>          Number of radial grid points in first shell (Nmax+1)   [default: 8]
    --nrS2=<res>          Number of radial grid points in second shell (Nmax+1)   [default: 8]
    --sponge             If flagged, add a damping layer in the shell that damps out waves.
    --tau_factor=<f>     Multiplication factor on sponge term [default: 1]

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --mesh=<n,m>         The processor mesh over which to distribute the cores

    --RK222              Use RK222 (default is SBDF2)
    --SBDF4              Use SBDF4 (default is SBDF2)
    --safety=<s>         Timestep CFL safety factor [default: 0.2]
    --CFL_max_r=<r>      zero out velocities above this radius value for CFL

    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --A0=<A>             Amplitude of random noise initial conditions [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory
    --cutoff=<c>         NCC cutoff magnitude [default: 1e-10]

    --rotation_time=<t>  Rotation timescale, in days (if ncc_file is not None) or sim units (for polytrope)

"""
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)


from d3_stars.simulations.anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem
from d3_stars.simulations.parser import parse_std_config
from d3_stars.simulations.outputs import initialize_outputs

# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

if __name__ == '__main__':
    # Read options
    config, raw_config, star_dir, star_file = parse_std_config('controls.cfg')
    out_dir = './'
    ntheta = config['ntheta']
    nphi = 2*ntheta
    L_dealias = config['l_dealias']
    N_dealias = config['n_dealias']
    ncc_cutoff = config['ncc_cutoff']
    dtype = np.float64

    # Parameters
    resolutions = []
    for nr in config['nr']:
        resolutions.append((nphi, ntheta, nr))
    Re  = config['reynolds_target'] 
    Pr  = config['prandtl']
    Pe  = Pr*Re
    ncc_file  = star_file

    if ncc_file is not None:
        with h5py.File(ncc_file, 'r') as f:
            r_stitch = f['r_stitch'][()]
            r_outer = f['r_outer'][()]
            tau_nd = f['tau_nd'][()]
            m_nd = f['m_nd'][()]
            L_nd = f['L_nd'][()]
            T_nd = f['T_nd'][()]
            rho_nd = f['rho_nd'][()]
            s_nd = f['s_nd'][()]

            tau_day = tau_nd/(60*60*24)
            N2_mesa = f['N2_mesa'][()]
            S1_mesa = f['S1_mesa'][()]
            r_mesa = f['r_mesa'][()]
            Re_shift = f['Re_shift'][()]
    else:
        r_stitch = (1.1, 1.4)
        r_outer = 1.5
        tau_day = 1
        Re_shift = 1
    logger.info('r_stitch: {} / r_outer: {:.2f}'.format(r_stitch, r_outer))

    Re *= Re_shift
    Pe *= Re_shift



    wall_hours = config['wall_hours']
    buoy_end_time = config['buoy_end_time']
    sponge = config['sponge']
    tau_factor = config['tau_factor']

    # rotation
    do_rotation = False
    if 'rotation_time' in config.keys():
        do_rotation = True
        rotation_time = config['rotation_time']
        dimensional_Omega = 2*np.pi / rotation_time  #radians / day [in MESA units]

    # Initial conditions
    if 'restart' in config.keys():
        restart = config['restart']
    else:
        restart = None
    A0 = config['a0']

    # Timestepper
    ts = None
    if 'timestepper' in config.keys():
        if config['timestepper'] == 'SBDF4':
            logger.info('using timestepper SBDF4')
            ts = d3.SBDF4
            timestepper_history = [0, 1, 2, 3]
        elif config['timestepper'] == 'RK222':
            logger.info('using timestepper RK222')
            ts = d3.RK222
            timestepper_history = [0, ]
    if ts is None:
        logger.info('using default timestepper SBDF2')
        ts = d3.SBDF2
        timestepper_history = [0, 1,]
    hermitian_cadence = 100
    safety = config['safety']
    if 'CFL_max_r' in config.keys():
        CFL_max_r = config['CFL_max_r']
    else:
        CFL_max_r = np.inf

    # Processor mesh
    ncpu = MPI.COMM_WORLD.size
    mesh = None
    if 'mesh' in config.keys():
        mesh = config['mesh']
    else:
        log2 = np.log2(ncpu)
        if log2 == int(log2):
            mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

    # Read in domain bound values
    L_shell = r_outer - r_stitch[0]
    sponge_function = lambda r: zero_to_one(r, r_stitch[0] + 2*L_shell/3, 0.1*L_shell)

    stitch_radii = r_stitch
    radius = r_outer
    coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=(L_dealias, L_dealias, N_dealias), dtype=dtype, mesh=mesh)

    vec_fields = ['u',]
    scalar_fields = ['p', 's1', 'inv_T', 'H', 'rho', 'T']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_ln_rho', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_chi_rad']
    scalar_nccs = ['ln_rho', 'ln_T', 'chi_rad', 'sponge', 'nu_diff']
    variables = make_fields(bases, coords, dist, 
                            vec_fields=vec_fields, scalar_fields=scalar_fields, 
                            vec_taus=vec_taus, scalar_taus=scalar_taus, 
                            vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                            sponge=sponge, do_rotation=do_rotation, sponge_function=sponge_function)


    variables, timescales = fill_structure(bases, dist, variables, ncc_file, r_outer, Pe,
                                            vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                            sponge=sponge, do_rotation=do_rotation)

    for i, bn in enumerate(bases.keys()):
        variables['sponge_{}'.format(bn)]['g'] *= tau_factor
    max_dt, t_buoy, t_rot = timescales
    logger.info('timescales -- max_dt {}, t_buoy {}, t_rot {}'.format(max_dt, t_buoy, t_rot))

    # Put nccs and fields into locals()
    locals().update(variables)


    # Problem
    prob_variables = get_anelastic_variables(bases, bases_keys, variables)
    problem = d3.IVP(prob_variables, namespace=locals())

    problem = set_anelastic_problem(problem, bases, bases_keys, stitch_radii=stitch_radii)

    logger.info("Problem built")
    # Solver
    solver = problem.build_solver(ts, ncc_cutoff=config['ncc_cutoff'])
    solver.stop_sim_time = buoy_end_time*t_buoy
    solver.stop_wall_time = wall_hours * 60 * 60
    logger.info("solver built")

    # Initial conditions / Checkpoint
    write_mode = 'overwrite'
    timestep = None
    if restart is not None:
        write, timestep = solver.load_state(restart)
        write_mode = 'append'
    else:
        # Initial conditions
        for bk in bases_keys:
            variables['s1_{}'.format(bk)].fill_random(layout='g', seed=42, distribution='normal', scale=A0)
            variables['s1_{}'.format(bk)].low_pass_filter(scales=0.25)
            variables['s1_{}'.format(bk)]['g'] *= np.sin(variables['theta1_{}'.format(bk)])
            variables['s1_{}'.format(bk)]['g'] *= np.cos(np.pi*variables['r1_{}'.format(bk)]/r_outer)

    analysis_tasks, even_analysis_tasks = initialize_outputs(solver, coords, variables, bases, timescales, out_dir=out_dir)

    ## Loop output Setup
    logger_handler = solver.evaluator.add_dictionary_handler(iter=1)
    for bn, basis in bases.items():
        vol_avg = variables['vol_avg_{}'.format(bn)]
        u = variables['u_{}'.format(bn)]
        nu_diff = variables['nu_diff_{}'.format(bn)]
        rho = variables['rho_{}'.format(bn)]
        u_squared = d3.dot(u, u)
        logger_handler.add_task(vol_avg((u_squared)**(1/2)/nu_diff), name='Re_avg_{}'.format(bn), layout='g')
        logger_handler.add_task(d3.integ(rho*(u_squared)/2), name='KE_{}'.format(bn), layout='g')

    #CFL setup
    heaviside_cfl = dist.Field(name='heaviside_cfl', bases=bases['B'])
    heaviside_cfl['g'] = 1
    if np.sum(r1_B > CFL_max_r) > 0:
        heaviside_cfl['g'][:,:, r1_B.flatten() > CFL_max_r] = 0
    heaviside_cfl = d3.Grid(heaviside_cfl).evaluate()

    initial_max_dt = max_dt
    while initial_max_dt < max_dt:
        max_dt /= 2
    if timestep is None:
        timestep = initial_max_dt
    my_cfl = d3.CFL(solver, timestep, safety=safety, cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
    my_cfl.add_velocity(heaviside_cfl*u_B)

    # Main loop
    start_time = time.time()
    start_iter = solver.iteration
    max_dt_check = True
    current_max_dt = my_cfl.max_dt
    slice_process = False
    just_wrote    = False
    slice_time = np.inf
    Re0 = 0
    outer_shell_dt = np.min(even_analysis_tasks['output_dts'])*2
    surface_shell_slices = even_analysis_tasks['shells']
    try:
        while solver.proceed:
            if max_dt_check and timestep < outer_shell_dt:
                #throttle max_dt timestep CFL early in simulation once timestep is below the output cadence.
                my_cfl.max_dt = max_dt
                max_dt_check = False
                just_wrote = True
                slice_time = solver.sim_time + outer_shell_dt

            timestep = my_cfl.compute_timestep()

            if just_wrote:
                just_wrote = False
                num_steps = np.ceil(outer_shell_dt / timestep)
                timestep = current_max_dt = my_cfl.stored_dt = outer_shell_dt/num_steps
            elif max_dt_check:
                timestep = np.min((timestep, current_max_dt))
            else:
                my_cfl.stored_dt = timestep = current_max_dt

            t_future = solver.sim_time + timestep
            if t_future >= slice_time*(1-1e-8):
               slice_process = True

            if solver.iteration % hermitian_cadence in timestepper_history:
                for f in solver.state:
                    f.require_grid_space()

            solver.step(timestep)

            if solver.iteration % 10 == 0 or solver.iteration <= 10:
                Re_avg = logger_handler.fields['Re_avg_B']
                KE_shell = logger_handler.fields['KE_S1']
                if dist.comm_cart.rank == 0:
                    KE0 = KE_shell['g'].min()
                    Re0 = Re_avg['g'].min()
                else:
                    KE0 = None
                    Re0 = None
                Re0 = dist.comm_cart.bcast(Re0, root=0)
                KE0 = dist.comm_cart.bcast(KE0, root=0)
                this_str = "iteration = {:08d}, t = {:f}, timestep = {:f}, Re = {:.4e}".format(solver.iteration, solver.sim_time, timestep, Re0)
                this_str += ", KE = {:.4e}".format(KE0)
                logger.info(this_str)


            if slice_process:
                slice_process = False
                wall_time = time.time() - solver.start_time
                solver.evaluator.evaluate_handlers([surface_shell_slices],wall_time=wall_time, sim_time=solver.sim_time, iteration=solver.iteration,world_time = time.time(),timestep=timestep)
                slice_time = solver.sim_time + outer_shell_dt
                just_wrote = True

            if np.isnan(Re0):
                logger.info('exiting with NaN')
                break

    except:
        logger.info('something went wrong in main loop.')
        raise
    finally:
        solver.log_stats()

        logger.info('making final checkpoint')
        fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
        fcheckpoint.add_tasks(solver.state, layout='g')
        solver.step(timestep)

        logger.info('Stitching open virtual files...')
        for handler in analysis_tasks:
            if not handler.check_file_limits():
                file = handler.get_file()
                file.close()
                if dist.comm_cart.rank == 0:
                    handler.process_virtual_file()
