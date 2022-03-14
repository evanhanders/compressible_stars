"""
d3 script for eigenvalue problem of anelastic convection / waves in a massive star.

Usage:
    evp_ball_2shells_AN.py [options]
    evp_ball_2shells_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           Angular resolution (Lmax   [default: 1]
    --nrB=<Nmax>          The ball radial degrees of freedom (Nmax+1)   [default: 64]
    --nrS1=<Nmax>          The shell radial degrees of freedom (Nmax+1)   [default: 64]
    --nrS2=<Nmax>          The shell radial degrees of freedom (Nmax+1)   [default: 16]
    --nrB_hi=<Nmax>       The hires-ball radial degrees of freedom (Nmax+1)
    --nrS1_hi=<Nmax>       The hires-shell radial degrees of freedom (Nmax+1)
    --nrS2_hi=<Nmax>       The hires-shell radial degrees of freedom (Nmax+1)

    --label=<label>      A label to add to the end of the output directory

    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --ncc_file_hi=<f>   path to a .h5 file of ICCs, curated from a MESA model (for hires solve)
"""
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

from anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem

def solve_dense(solver, ell):
    """
    Do a dense eigenvalue solve at a specified ell.
    Sort the eigenvalues and eigenvectors according to damping rate.
    """
    for subproblem in solver.subproblems:
        this_ell = subproblem.group[1]
        if this_ell != ell:
            continue
        #TODO: Output to file.
        logger.info("solving ell = {}".format(ell))
        solver.solve_dense(subproblem)

        values = solver.eigenvalues 
        vectors = solver.eigenvectors

        #filter out nans
        cond1 = np.isfinite(values)
        values = values[cond1]
        vectors = vectors[:, cond1]

        #Only take positive frequencies
        cond2 = values.real > 0
        values = values[cond2]
        vectors = vectors[:, cond2]

        #Sort by real frequency magnitude
        order = np.argsort(1/values.real)
        values = values[order]
        vectors = vectors[:, order]

        #Update solver
        solver.eigenvalues = values
        solver.eigenvectors = vectors
        return solver

def combine_eigvecs(field, ell_m_bool, bases, namespace, scales=None, shift=True):
    fields = []
    ef_base = []
    vector = False
    for i, bn in enumerate(bases.keys()):
        fields.append(namespace['{}_{}'.format(field, bn)])
        this_field = fields[-1]


        if len(this_field.tensorsig) == 1:
            vector = True

        if scales is not None:
            this_field.change_scales(scales[i])
        else:
            this_field.change_scales((1,1,1))
        this_field['c']
        this_field.towards_grid_space()

        if vector:
            ef_base.append(this_field.data[:,ell_m_bool,:].squeeze())
        else:
            ef_base.append(this_field.data[ell_m_bool,:].squeeze())

    ef = np.concatenate(ef_base, axis=-1)

    if shift:
        if vector:
            #Getting argmax then indexing ensures we're not off by a negative sign flip.
            ix = np.argmax(np.abs(ef[2,:]))
            divisor = ef[2,:][ix]
        else:
            ix = np.argmax(np.abs(ef))
            divisor = ef[ix]

        ef /= divisor
        for i in range(len(bases.keys())):
            ef_base[i] /= divisor

    if vector:
        #go from plus/minus to theta/phi
        ef_u = np.zeros_like(ef)
        ef_u[0,:] = (1j/np.sqrt(2))*(ef[1,:] - ef[0,:])
        ef_u[1,:] = ( 1/np.sqrt(2))*(ef[1,:] + ef[0,:])
        ef_u[2,:] = ef[2,:]
        ef[:] = ef_u[:]

    return ef, ef_base



def check_eigen(solver1, solver2, bases1, bases2, namespace1, namespace2, ell, r_cz=1, cutoff=1e-1):
    """
    Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
    Only keep the solutions that match to within the specified cutoff between the two cases.
    """
    good_values1 = []
    good_values2 = []

    scales = []
    rho2 = []
    r1 = []
    r2 = []
    for i, bn in enumerate(bases1.keys()):
        scales.append((1, 1, namespace2['resolutions_hi'][i][-1]/namespace1['resolutions'][i][-1]))
        rho2.append(namespace2['rho_{}'.format(bn)]['g'][0,0,:])
        r1.append(namespace1['r_{}'.format(bn)].flatten())
        r2.append(namespace2['r_{}'.format(bn)].flatten())

    rho2 = np.concatenate(rho2)
    r1 = np.concatenate(r1)
    r2 = np.concatenate(r2)

    for sbsys in solver1.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem1 = sbsys
            break
    for sbsys in solver2.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem2 = sbsys
            break

    print('finding good eigenvalues with cutoff {}'.format(cutoff))
    shape = list(namespace1['s1_B']['c'].shape[:2])
    good = np.zeros(shape, bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid_space = (False,False)
            elements = (np.array((i,)),np.array((j,)))
            m, this_ell = bases1['B'].sphere_basis.elements_to_groups(grid_space, elements)
            if this_ell == ell and m == 1:
                good[i,j] = True

    rough_dr2 = np.gradient(r2, edge_order=2)

    for i, v1 in enumerate(solver1.eigenvalues):
        for j, v2 in enumerate(solver2.eigenvalues):
            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
            goodness = np.abs(v1 - v2)/np.abs(v1).min()
            if goodness < cutoff:# or (j == 0 and (i == 2 or i == 3)):# and (np.abs(v1.imag - v2.imag)/np.abs(v1.imag)).min() < 1e-1:
                
                solver1.set_state(i, subsystem1)
                solver2.set_state(j, subsystem2)

                ef_u1, ef_u1_pieces = combine_eigvecs('u', good, bases1, namespace1, scales=scales)
                ef_u2, ef_u2_pieces = combine_eigvecs('u', good, bases2, namespace2)

                #If mode KE is inside of the convection zone then it's a bad mode.
                mode_KE = rho2*np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
                cz_KE = np.sum((mode_KE*4*np.pi*r2**2*rough_dr2)[r2 <= r_cz])
                tot_KE = np.sum((mode_KE*4*np.pi*r2**2*rough_dr2))
                cz_KE_frac = cz_KE/tot_KE
                vector_diff = np.max(np.abs(ef_u1 - ef_u2))
                print(v1/(2*np.pi), v2/(2*np.pi), vector_diff)
#                if vector_diff < np.sqrt(cutoff) and cz_KE_frac < 0.5:
                if vector_diff < np.sqrt(cutoff):
                    print('good evalue w/ vdiff', vector_diff, 'czfrac', cz_KE_frac.real)
                    if cz_KE_frac.real > 0.5:
                        print('evalue is in the CZ, skipping')
                    else:
                        plt.figure()
                        plt.plot(r2, ef_u1[2,:])
                        plt.plot(r2, ef_u2[2,:])
                        plt.show()
                        good_values1.append(i)
                        good_values2.append(j)


    solver1.eigenvalues = solver1.eigenvalues[good_values1]
    solver2.eigenvalues = solver2.eigenvalues[good_values2]
    solver1.eigenvectors = solver1.eigenvectors[:, good_values1]
    solver2.eigenvectors = solver2.eigenvectors[:, good_values2]
    return solver1, solver2

def calculate_duals(vel_ef_lists, bases, namespace):
    """
    Calculate the dual basis of the velocity eigenvectors.
    """
    work_fields = []
    vel_ef_lists = list(vel_ef_lists)
    int_field = None
    for i, bn in enumerate(bases.keys()):
        vel_ef_lists[i] = np.array(vel_ef_lists[i])
        work_fields.append(namespace['dist'].Field(bases=bases[bn]))
        if int_field is None:
            int_field = d3.integ(namespace['rho_{}'.format(bn)]*work_fields[-1])
        else:
            int_field += d3.integ(namespace['rho_{}'.format(bn)]*work_fields[-1])

    print(vel_ef_lists)

    def IP(velocity_list1, velocity_list2):
        """ Integrate the bra-ket of two eigenfunctions of velocity. """
        for i, bn in enumerate(bases.keys()):
            velocity1 = velocity_list1[i]
            velocity2 = velocity_list2[i]
            work_fields[i]['g'] = np.sum(velocity1*np.conj(velocity2), axis=0)
        return int_field.evaluate()['g'].min()


    n_modes = vel_ef_lists[0].shape[0]
    IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
    for i in range(n_modes):
        if i % 1 == 0: logger.info("duals {}/{}".format(i, n_modes))
        for j in range(n_modes):
            velocity_list1 = []
            velocity_list2 = []
            for k, bn in enumerate(bases.keys()):
                velocity_list1.append(vel_ef_lists[k][i])
                velocity_list2.append(vel_ef_lists[k][j])
            IP_matrix[i,j] = IP(velocity_list1, velocity_list2)
    
#    print('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
    IP_inv = np.linalg.inv(IP_matrix)

    vel_duals = []
    for i, bn in enumerate(bases.keys()):
        vel_duals.append(np.zeros_like(vel_ef_lists[i]))
        for j in range(3):
            vel_duals[-1][:,j,:] = np.einsum('ij,ik->kj', vel_ef_lists[i][:,j,:], np.conj(IP_inv))

    return np.concatenate(vel_duals, axis=-1)



# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

if __name__ == '__main__':
    # Read in parameters and create output directory
    args   = docopt(__doc__)
    if args['<config>'] is not None: 
        config_file = Path(args['<config>'])
        config = ConfigParser()
        config.read(str(config_file))
        for n, v in config.items('parameters'):
            for k in args.keys():
                if k.split('--')[-1].lower() == n:
                    if v == 'true': v = True
                    args[k] = v

    # Parameters
    Lmax = int(args['--L'])
    ntheta = Lmax + 1
    nphi = 2*ntheta
    nrB = int(args['--nrB'])
    nrS1 = int(args['--nrS1'])
    nrS2 = int(args['--nrS2'])
    resolutionB = (nphi, ntheta, nrB)
    resolutionS1 = (nphi, ntheta, nrS1)
    resolutionS2 = (nphi, ntheta, nrS2)
    Re  = float(args['--Re'])
    Pr  = 1
    Pe  = Pr*Re
    ncc_file  = args['--ncc_file']

    do_hires = (args['--nrB_hi'] is not None and args['--nrS1_hi'] is not None and args['--nrS2_hi'] is not None)

    if do_hires:
        nrB_hi = int(args['--nrB_hi'])
        nrS1_hi = int(args['--nrS1_hi'])
        nrS2_hi = int(args['--nrS2_hi'])
        ncc_file_hi = args['--ncc_file_hi']
        resolutionB_hi = (nphi, ntheta, nrB_hi)
        resolutionS1_hi = (nphi, ntheta, nrS1_hi)
        resolutionS2_hi = (nphi, ntheta, nrS2_hi)
    else:
        nrB_hi = nrS1_hi = nrS2_hi = ncc_file_hi = None

    out_dir = './' + sys.argv[0].split('.py')[0]
    if args['--ncc_file'] is None:
        out_dir += '_polytrope'
    out_dir += '_Re{}_Lmax{}_nr{}+{}+{}'.format(args['--Re'], Lmax, nrB, nrS1, nrS2)
    if do_hires:
        out_dir += '_nrhi{}+{}+{}'.format(nrB_hi, nrS1_hi, nrS2_hi)
    if args['--label'] is not None:
        out_dir += '_{:s}'.format(args['--label'])
    logger.info('saving data to {:s}'.format(out_dir))
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(out_dir)):
            os.makedirs('{:s}/'.format(out_dir))

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
            r_mesa = f['r_mesa'][()]
    else:
        r_stitch = (1.1, 1.4)
        r_outer = 1.5
        tau_day = 1

    sponge = False
    do_rotation = False
    dealias = (1,1,1) 

    logger.info('r_stitch: {} / r_outer: {:.2f}'.format(r_stitch, r_outer))

    #Create bases
    resolutions = (resolutionB, resolutionS1, resolutionS2)
    stitch_radii = r_stitch
    radius = r_outer
    coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)

    vec_fields = ['u',]
    scalar_fields = ['p', 's1', 'inv_T', 'H', 'rho', 'T']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_ln_rho', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_chi_rad']
    scalar_nccs = ['ln_rho', 'ln_T', 'chi_rad', 'sponge']
    variables = make_fields(bases, coords, dist, 
                            vec_fields=vec_fields, scalar_fields=scalar_fields, 
                            vec_taus=vec_taus, scalar_taus=scalar_taus, 
                            vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                            sponge=sponge, do_rotation=do_rotation)


    variables, timescales = fill_structure(bases, dist, variables, ncc_file, r_outer, Pe, 
                                            vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                            sponge=sponge, do_rotation=do_rotation)
    variables.update(locals())
    omega = dist.Field(name='omega')
    variables['dt'] = lambda A: -1j * omega * A

    prob_variables = get_anelastic_variables(bases, bases_keys, variables)

    problem = d3.EVP(prob_variables, eigenvalue=omega, namespace=variables)

    problem = set_anelastic_problem(problem, bases, bases_keys, stitch_radii=stitch_radii)
    logger.info('problem built')

    solver = problem.build_solver()
    logger.info('solver built')

    if do_hires:
        resolutions_hi = (resolutionB_hi, resolutionS1_hi, resolutionS2_hi)
        coords_hi, dist_hi, bases_hi, bases_keys_hi = make_bases(resolutions_hi, stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)
        variables_hi = make_fields(bases_hi, coords_hi, dist_hi, 
                                vec_fields=vec_fields, scalar_fields=scalar_fields, 
                                vec_taus=vec_taus, scalar_taus=scalar_taus, 
                                vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                sponge=sponge, do_rotation=do_rotation)


        variables_hi, timescales_hi = fill_structure(bases_hi, dist_hi, variables_hi, ncc_file_hi, r_outer, Pe, 
                                                vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                                sponge=sponge, do_rotation=do_rotation)


        variables_hi.update(locals())
        omega_hi = dist_hi.Field(name='omega_hi')
        variables_hi['dt'] = lambda A: -1j * omega_hi * A
        prob_variables_hi = get_anelastic_variables(bases_hi, bases_keys_hi, variables_hi)
        problem_hi = d3.EVP(prob_variables_hi, eigenvalue=omega_hi, namespace=variables_hi)
        problem_hi = set_anelastic_problem(problem_hi, bases_hi, bases_keys_hi, stitch_radii=stitch_radii)
        logger.info('hires problem built')
        solver_hi = problem_hi.build_solver()
        logger.info('hires solver built')
#    for ncc in vec_nccs:
#        for bn in bases_keys:
#            r1 = variables['r_{}'.format(bn)]
#            ncc_1 = variables['{}_{}'.format(ncc, bn)]
#            r2 = variables_hi['r_{}'.format(bn)]
#            ncc_2 = variables_hi['{}_{}'.format(ncc, bn)]
#
#            if ncc in ['grad_T', 'grad_ln_rho', 'grad_ln_T']:
#                plt.plot(r1.ravel(), -ncc_1['g'][2,:].ravel(), c='k')
#                plt.plot(r2.ravel(), -ncc_2['g'][2,:].ravel(), c='r', ls='--')
#            else:
#                plt.plot(r1.ravel(), ncc_1['g'][2,:].ravel(), c='k')
#                plt.plot(r2.ravel(), ncc_2['g'][2,:].ravel(), c='r', ls='--')
#            if ncc == 'grad_T':
#                print(ncc_1['g'][2,0,0,:], ncc_2['g'][2, 0,0,:])
#        plt.yscale('log')
#        plt.ylabel(ncc)
#        plt.show()
#    for ncc in scalar_nccs:
#        for bn in bases_keys:
#            r1 = variables['r_{}'.format(bn)]
#            ncc_1 = variables['{}_{}'.format(ncc, bn)]
#            r2 = variables_hi['r_{}'.format(bn)]
#            ncc_2 = variables_hi['{}_{}'.format(ncc, bn)]
#
#            if ncc in ['ln_rho', 'ln_T']:
#                plt.plot(r1.ravel(), -ncc_1['g'].ravel(), c='k')
#                plt.plot(r2.ravel(), -ncc_2['g'].ravel(), c='r', ls='--')
#            else:
#                plt.plot(r1.ravel(), ncc_1['g'].ravel(), c='k')
#                plt.plot(r2.ravel(), ncc_2['g'].ravel(), c='r', ls='--')
#        plt.yscale('log')
#        plt.ylabel(ncc)
#        plt.show()
#
#    sys.exit()

    #ell = 1 solve
    for i in range(Lmax):
        ell = i + 1
        logger.info('solving lores eigenvalue with nr = ({}, {}, {})'.format(nrB, nrS1, nrS2))
        solve_dense(solver, ell)

        if do_hires:
            logger.info('solving hires eigenvalue with nr = ({}, {}, {})'.format(nrB_hi, nrS1_hi, nrS2_hi))
            solve_dense(solver_hi, ell)
            print(variables['u_B']['g'].shape, variables_hi['u_B']['g'].shape, 'meh')
            solver, solver_hi = check_eigen(solver, solver_hi, bases, bases_hi, variables, variables_hi, ell)
    
        #Calculate 'optical depths' of each mode.
        depths = []
        if ncc_file is not None:
            chi_rads = []
            for i, bn in enumerate(bases_keys):
                chi_rad = np.ones_like(r_mesa) / Pe
                local_r_inner, local_r_outer = 0, 0
                if i == 0:
                    local_r_inner = 0
                else:
                    local_r_inner = stitch_radii[i-1]
                if len(bases_keys) > 1 and i < len(bases_keys) - 1:
                    local_r_outer = stitch_radii[i]
                else:
                    local_r_outer = radius
                r_mesa_nd = r_mesa/L_nd
                good_r = (r_mesa_nd > local_r_inner)*(r_mesa_nd <= local_r_outer)
                chi_rad[good_r] = interp1d(variables['r_{}'.format(bn)].flatten(), variables['grad_chi_rad_{}'.format(bn)]['g'][2,0,0,:], 
                                           bounds_error=False, fill_value='extrapolate')(r_mesa_nd[good_r])
                chi_rad *= (L_nd**2 / tau_nd)

            # from Shiode et al 2013 eqns 4-8 
            for om in solver.eigenvalues.real:
                Lambda = np.sqrt(ell*(ell+1))
                kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_nd))
                v_group = (om/tau_nd) / kr_cm
                gamma_rad = chi_rad * kr_cm**2
                depth_integrand = gamma_rad/v_group

                #No optical depth in CZs, or outside of simulation domain...
                depth_integrand[N2_mesa < 0] = 0
                depth_integrand[r_mesa/L_nd > r_outer] = 0

                #Numpy integrate
                opt_depth = np.trapz(depth_integrand, x=r_mesa)
                depths.append(opt_depth)

        shape = list(variables['s1_B']['c'].shape[:2])
        good = np.zeros(shape, bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid_space = (False,False)
                elements = (np.array((i,)),np.array((j,)))
                m, this_ell = bases['B'].sphere_basis.elements_to_groups(grid_space, elements)
                if this_ell == ell and m == 1:
                    good[i,j] = True

        integ_energy_op = None
        rho_fields = []
        for i, bn in enumerate(bases_keys):
            p, u = [variables['{}_{}'.format(f, bn)] for f in ['p', 'u']]
            variables['pomega_hat_{}'.format(bn)] = p - 0.5*d3.dot(u,u)
            variables['KE_{}'.format(bn)] = dist.Field(bases=bases[bn], name='KE_{}'.format(bn))
            rho_fields.append(variables['rho_{}'.format(bn)]['g'][0,0,:])

            if integ_energy_op is None:
                integ_energy_op = d3.integ(variables['KE_{}'.format(bn)])
            else:
                integ_energy_op += d3.integ(variables['KE_{}'.format(bn)])
            if i == len(bases_keys)-1:
                s1_surf = variables['s1_{}'.format(bn)](r=radius)
        rho_full = np.concatenate(rho_fields, axis=-1)

        integ_energies = np.zeros_like(solver.eigenvalues, dtype=np.float64) 
        s1_amplitudes = np.zeros_like(solver.eigenvalues, dtype=np.float64)  
        velocity_eigenfunctions = []
        velocity_eigenfunctions_pieces = []
        entropy_eigenfunctions = []
        wave_flux_eigenfunctions = []

        for sbsys in solver.subsystems:
            ss_m, ss_ell, r_couple = sbsys.group
            if ss_ell == ell and ss_m == 1:
                subsystem = sbsys
                break

        for i, e in enumerate(solver.eigenvalues):
            solver.set_state(i, subsystem)

            #Get eigenvectors
            for i, bn in enumerate(bases_keys):
                variables['pomega_hat_field_{}'.format(bn)] = variables['pomega_hat_{}'.format(bn)].evaluate()

            ef_u, ef_u_pieces = combine_eigvecs('u', good, bases, variables, shift=False)
            ef_s1, ef_s1_pieces = combine_eigvecs('s1', good, bases, variables, shift=False)
            ef_pom, ef_pom_pieces = combine_eigvecs('pomega_hat_field', good, bases, variables, shift=False)

            #normalize & store eigenvectors
            shift = np.max(np.abs(ef_u[2,:]))
            for data in [ef_u, ef_s1, ef_pom]:
                data[:] /= shift
            for piece_tuple in [ef_u_pieces, ef_s1_pieces, ef_pom_pieces]:
                for data in piece_tuple:
                    data[:] /= shift

            velocity_eigenfunctions.append(ef_u)
            velocity_eigenfunctions_pieces.append(ef_u_pieces)
            entropy_eigenfunctions.append(ef_s1)

            #Wave flux
            wave_flux = rho_full*ef_u[2,:]*np.conj(ef_pom).squeeze()
            wave_flux_eigenfunctions.append(wave_flux)

#            #Kinetic energy
            for i, bn in enumerate(bases_keys):
                rho = variables['rho_{}'.format(bn)]['g'][0,0,:]
                print(ef_u_pieces[i].shape)
                u_squared = np.sum(ef_u_pieces[i]*np.conj(ef_u_pieces[i]), axis=0)
                variables['KE_{}'.format(bn)]['g'] = (rho*u_squared.real/2)[None,None,:]
            integ_energy = integ_energy_op.evaluate()
            integ_energies[i] = integ_energy['g'].min().real / 2 #factor of 2 accounts for spherical harmonic integration (we're treating the field like an ell = 0 one)

            #Surface entropy perturbations
            for i, bn in enumerate(bases_keys):
                s1 = variables['s1_{}'.format(bn)]
                s1['g'] = 0
                s1['g'] = ef_s1_pieces[i]
            s1_surf_vals = s1_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
            s1_amplitudes[i] = np.abs(s1_surf_vals.max())

        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
            f['good_evalues'] = solver.eigenvalues
            f['good_omegas']  = solver.eigenvalues.real
            f['good_evalues_inv_day'] = solver.eigenvalues/tau_day
            f['good_omegas_inv_day']  = solver.eigenvalues.real/tau_day
            f['s1_amplitudes']  = s1_amplitudes
            f['integ_energies'] = integ_energies
            f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
            f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
            f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
            for i, bn in enumerate(bases_keys):
                f['r_{}'.format(bn)] = variables['r1_{}'.format(bn)]
                f['rho_{}'.format(bn)] = variables['rho_{}'.format(bn)]['g']
            f['rho_full'] = rho_full
            f['depths'] = np.array(depths)

            #Pass through nondimensionalization

            if ncc_file is not None:
                f['tau_nd'] = tau_nd 
                f['m_nd']   = m_nd   
                f['L_nd']   = L_nd   
                f['T_nd']   = T_nd   
                f['rho_nd'] = rho_nd 
                f['s_nd']   = s_nd   


        #TODO: Fix dual calculation
        velocity_duals = calculate_duals(velocity_eigenfunctions_pieces, bases, variables)
        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
            f['good_evalues'] = solver.eigenvalues
            f['good_omegas']  = solver.eigenvalues.real
            f['good_evalues_inv_day'] = solver.eigenvalues/tau_day
            f['good_omegas_inv_day']  = solver.eigenvalues.real/tau_day
            f['s1_amplitudes']  = s1_amplitudes
            f['integ_energies'] = integ_energies
            f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
            f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
            f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
            f['velocity_duals'] = velocity_duals
            for i, bn in enumerate(bases_keys):
                f['r_{}'.format(bn)] = variables['r1_{}'.format(bn)]
                f['rho_{}'.format(bn)] = variables['rho_{}'.format(bn)]['g']
            f['rho_full'] = rho_full
            f['depths'] = np.array(depths)

            #Pass through nondimensionalization

            if ncc_file is not None:
                f['tau_nd'] = tau_nd 
                f['m_nd']   = m_nd   
                f['L_nd']   = L_nd   
                f['T_nd']   = T_nd   
                f['rho_nd'] = rho_nd 
                f['s_nd']   = s_nd   

        gc.collect()