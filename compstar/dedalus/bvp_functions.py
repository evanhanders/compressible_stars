
from collections import OrderedDict

import h5py
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from mpi4py import MPI
from .compressible_functions import make_bases

from scipy.interpolate import interp1d
from ..tools.general import one_to_zero, zero_to_one
import logging
logger = logging.getLogger(__name__)

interp_kwargs = {'fill_value' : 'extrapolate', 'bounds_error' : False}

def HSE_solve(coords, dist, bases, grad_ln_rho_func, N2_func, Fconv_func, r_stitch=[], r_outer=1, low_nr=16, \
              R=1, gamma=5/3, nondim_radius=1, ncc_cutoff=1e-9, tolerance=1e-9, HSE_tolerance = 1e-4, smooth_edge=True):
    """
    Solves for hydrostatic equilibrium in a calorically perfect ideal gas.
    The solution for density, entropy, and gravity is found given a specified function of N^2 and grad ln rho.
    The heating term associated with a convective luminosity is also found given a specified function of the convective flux, Fconv.

    Arguments
    ---------
    coords : Dedalus CoordinateSystem object
        The coordinate system in which the solution is found.
    dist : Dedalus Distributor object
        The distributor object associated with the bases; should NOT be in parallel.
    bases : dict
        A dictionary of Dedalus bases, with keys 'B', 'S1', 'S2', etc. for the Ball basis, first Shell basis, second Shell basis, etc.
    grad_ln_rho_func : function
        A function of radius that returns the gradient of the log of density. Input r should be nondimensionalized.
    N2_func : function
        A function of radius that returns the nondimensionalized Brunt-Vaisala frequency squared. Input r should be nondimensionalized.
    Fconv_func : function
        A function of radius that returns the nondimensionalized convective flux. Input r should be nondimensionalized.
    r_stitch : list
        A list of radii at which to stitch together the solutions from different bases. 
        The first element should be the radius of the outer boundary of the BallBasis.
        If there is only one basis, r_stitch should be an empty list.
    r_outer : float
        The radius of the outer boundary of the simulation domain.
    low_nr : int
        The number of radial points in the low resolution domain; used to set up background fields for solve. #TODO: make this by-basis.
    R : float
        The nondimensional value of the gas constant divided by the mean molecular weight.
    gamma : float
        The adiabatic index of the gas.
    nondim_radius : float
        The radius where thermodynamics are nondimensionalized.
    ncc_cutoff : float
        The NCC floor for the solver. See Dedalus.core.solvers.SolverBase
    tolerance : float
        The tolerance for perturbation norm of the newton iteration.
    HSE_tolerance : float
        The tolerance for hydrostatic equilibrium of the BVP solve.
    smooth_edge : bool
        Whether to smooth the heating function from its value to zero at the boundary of the domain.
    
    Returns
    -------
    atmosphere : dict
        A dictionary of interpolated functions which return atmospheric quantities as a function of nondimensional radius.
    """
    # Parameters
    namespace = dict()
    namespace['R'] = dist.Field(name='R')
    namespace['R']['g'] = R
    namespace['Cp'] = Cp = dist.Field(name='Cp')
    namespace['Cp']['g'] = R*gamma/(gamma-1)
    namespace['gamma'] = dist.Field(name='gamma')
    namespace['gamma']['g'] = gamma
    namespace['log'] = np.log

    #Loop over bases, set up fields and operators.
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        # Make problem variables and taus.
        namespace['g_phi_{}'.format(k)] = Q = dist.Field(name='g_phi', bases=basis)
        namespace['Q_{}'.format(k)] = Q = dist.Field(name='Q', bases=basis)
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['g_{}'.format(k)] = g = dist.VectorField(basis.coordsystem, name='g', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_g_phi_{}'.format(k)] = tau_g_phi = dist.Field(name='tau_g_phi', bases=S2_basis)

        # Set up some fundamental grid data
        low_scales = low_nr/basis.radial_basis.radial_size 
        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,low_scales))
        namespace['r_de_{}'.format(k)] = r_de
        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r
        namespace['r_squared_{}'.format(k)] = r_squared = dist.Field(bases=basis.radial_basis)
        r_squared['g'] = r**2       

        # Make lift operators for BCs
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis.derivative_basis(2), -1)

        # Make a field of ones for converting NCCs to full fields.
        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1

        #Make a field that smooths at the edge of the ball basis.
        namespace['edge_smoothing_{}'.format(k)] = edge_smooth = dist.Field(bases=basis, name='edge_smooth')
        if smooth_edge:
            edge_smooth['g'] = one_to_zero(r, 0.95*bases['B'].radius, width=0.03*bases['B'].radius)
        else:
            edge_smooth['g'] = 1

        # Get a high-resolution N^2 in the ball; low-resolution elsewhere where it transitions more gradually.
        namespace['N2_{}'.format(k)] = N2 = dist.Field(bases=basis, name='N2')
        if k == 'B':
            N2['g'] = N2_func(r)
        else:
            N2.change_scales(low_scales)
            N2['g'] = N2_func(r_low)

        #Set grad ln rho.
        grad_ln_rho.change_scales(low_scales)
        grad_ln_rho['g'][2] = grad_ln_rho_func(r_low)

        # Set the convective flux.
        namespace['Fconv_{}'.format(k)] = Fconv   = dist.VectorField(coords, name='Fconv', bases=basis)
        Fconv.change_scales(low_scales)
        Fconv['g'][2] = Fconv_func(r_low)

        # Create important operations from the fields.
        namespace['ln_pomega_LHS_{}'.format(k)] = ln_pomega_LHS = gamma*(s/Cp + ((gamma-1)/gamma)*ln_rho*ones)
        namespace['ln_pomega_{}'.format(k)] = ln_pomega = ln_pomega_LHS + np.log(R)
        namespace['pomega_{}'.format(k)] = pomega = np.exp(ln_pomega)
        namespace['P_{}'.format(k)] = P = pomega*np.exp(ln_rho)
        namespace['HSE_{}'.format(k)] = HSE = gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) - g*ones
        namespace['N2_op_{}'.format(k)] = N2_op = -g@d3.grad(s)/Cp
        namespace['rho_{}'.format(k)] = rho = np.exp(ln_rho*ones)
        namespace['T_{}'.format(k)] = T = pomega/R
        namespace['ln_T_{}'.format(k)] = ln_T = ln_pomega - np.log(R)
        namespace['grad_pomega_{}'.format(k)] = d3.grad(pomega)
        namespace['grad_ln_pomega_{}'.format(k)] = d3.grad(ln_pomega)
        namespace['grad_s_{}'.format(k)] = grad_s = d3.grad(s)
        namespace['r_vec_g_{}'.format(k)] = r_vec@g
        namespace['g_op_{}'.format(k)] = gamma * pomega * (grad_s/Cp + grad_ln_rho)
        namespace['s0_{}'.format(k)] = Cp * ((1/gamma)*(ln_pomega + ln_rho) - ln_rho) #s with an offset so s0 = cp * (1/gamma * lnP - ln_rho)
        namespace['L_heat_{}'.format(k)] = 4*np.pi*r_squared*Fconv

    namespace['pi'] = np.pi
    locals().update(namespace)

    #Solve for ln_rho.
    variables, taus = [], []
    for k, basis in bases.items():
        variables += [namespace['ln_rho_{}'.format(k)],]
        taus      += [namespace['tau_rho_{}'.format(k)],]

    problem = d3.NLBVP(variables + taus, namespace=locals())
    for k, basis in bases.items():
        #Equation is just definitional.
        problem.add_equation("grad(ln_rho_{0}) - grad_ln_rho_{0} + r_vec_{0}*lift_{0}(tau_rho_{0}) = 0".format(k))
    
    #Set boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B' and k != 'S0':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
        elif k == 'S0':
            problem.add_equation("ln_rho_S0(r=nondim_radius) = 0")
        elif k == 'B':
            problem.add_equation("ln_rho_B(r=nondim_radius) = 0")
        iter += 1

    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
    logger.info('ln_rho found')

    #Solve for everything else given ln_rho.
    variables, taus = [], []
    for k, basis in bases.items():
        variables += [namespace['s_{}'.format(k)], namespace['g_{}'.format(k)], namespace['Q_{}'.format(k)], namespace['g_phi_{}'.format(k)]]
        taus += [namespace['tau_s_{}'.format(k)], namespace['tau_g_phi_{}'.format(k)]]

    problem = d3.NLBVP(variables + taus, namespace=locals())
    for k, basis in bases.items():
        #Set a decent initial guess for s.
        namespace['s_{}'.format(k)].change_scales(basis.dealias)
        namespace['s_{}'.format(k)]['g'] = -(R*namespace['ln_rho_{}'.format(k)]).evaluate()['g']

        #Set the equations: hydrostatic equilibrium, gravity, Q.
        problem.add_equation("grad(ln_rho_{0})@(grad(s_{0})/Cp) + lift_{0}(tau_s_{0}) = -N2_{0}/(gamma*pomega_{0}) - grad(s_{0})@grad(s_{0}) / Cp**2".format(k))
        problem.add_equation("g_{0} = g_op_{0} ".format(k))
        problem.add_equation("Q_{0} = edge_smoothing_{0}*div(Fconv_{0})".format(k))
        problem.add_equation("grad(g_phi_{0}) + g_{0} + r_vec_{0}*lift_{0}(tau_g_phi_{0}) = 0".format(k))
    
    #Set the boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B' and k != 'S0':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("g_phi_{0}(r={2}) - g_phi_{1}(r={2}) = 0".format(k, k_old, r_s))
        else:
            problem.add_equation("ln_pomega_LHS_{}(r=nondim_radius) = 0".format(k))
        iter += 1
        if iter == len(bases.items()):
            problem.add_equation("g_phi_{0}(r=r_outer) = 0".format(k))

    #Solve with tolerances on pert_norm and hydrostatic equilibrium.
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance or HSE_err > HSE_tolerance:
        HSE_err = 0
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        for k, basis in bases.items():
            this_HSE = np.max(np.abs(namespace['HSE_{}'.format(k)].evaluate()['g']))
            logger.info('HSE in {}:{:.3e}'.format(k, this_HSE))
            if this_HSE > HSE_err:
                HSE_err = this_HSE

    # Stitch together the fields for creation of interpolators that span the full simulation domain.
    #Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'grad_s', 'g', 'pomega', 'rho', 'ln_rho', 'g_phi', 'r_vec', 'HSE', 'N2_op', 'Q', 's0', 'L_heat']
    for f in fields:
        stitch_fields[f] = []
    
    for k, basis in bases.items():
        for f in fields:
            stitch_fields[f] += [np.copy(namespace['{}_{}'.format(f, k)].evaluate()['g'])]

    if len(stitch_fields['r_vec']) == 1:
        for f in fields:
            stitch_fields[f] = stitch_fields[f][0]
    else:
        for f in fields:
            stitch_fields[f] = np.concatenate(stitch_fields[f], axis=-1)

    grad_pom = stitch_fields['grad_pomega'][2,:].ravel()
    grad_ln_pom = stitch_fields['grad_ln_pomega'][2,:].ravel()
    grad_ln_rho = stitch_fields['grad_ln_rho'][2,:].ravel()
    grad_s = stitch_fields['grad_s'][2,:].ravel()
    g = stitch_fields['g'][2,:].ravel()
    HSE = stitch_fields['HSE'][2,:].ravel()
    r = stitch_fields['r_vec'][2,:].ravel()

    pom = stitch_fields['pomega'].ravel()
    rho = stitch_fields['rho'].ravel()
    ln_rho = stitch_fields['ln_rho'].ravel()
    g_phi = stitch_fields['g_phi'].ravel()
    N2 = stitch_fields['N2_op'].ravel()
    Q = stitch_fields['Q'].ravel()
    s0 = stitch_fields['s0'].ravel()
    L_heat = stitch_fields['L_heat'][2,:].ravel()


    #Plot the results.
    fig = plt.figure()
    ax1 = fig.add_subplot(4,2,1)
    ax2 = fig.add_subplot(4,2,2)
    ax3 = fig.add_subplot(4,2,3)
    ax4 = fig.add_subplot(4,2,4)
    ax5 = fig.add_subplot(4,2,5)
    ax6 = fig.add_subplot(4,2,6)
    ax7 = fig.add_subplot(4,2,7)
    ax8 = fig.add_subplot(4,2,8)
    ax1.plot(r, grad_pom, label='grad pomega')
    ax1.legend()
    ax2.plot(r, grad_ln_rho, label='grad ln rho')
    ax2.legend()
    ax3.plot(r, pom/R, label='pomega/R')
    ax3.plot(r, rho, label='rho')
    ax3.legend()
    ax4.plot(r, HSE, label='HSE')
    ax4.legend()
    ax5.plot(r, g, label='g')
    ax5.legend()
    ax6.plot(r, L_heat, label='L_heat')
    ax6.legend()
    ax7.plot(r, N2, label=r'$N^2$')
    ax7.plot(r, -N2, label=r'$-N^2$')
    ax7.plot(r, (N2_func(r)), label=r'$N^2$ goal', ls='--')
    ax7.set_yscale('log')
    if 'B' in bases.keys():
        yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
        ax7.set_yticks(yticks)
        ax7.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
    ax7.legend()
    ax8.plot(r, grad_s, label='grad s')
    ax8.set_yscale('log')
    ax8.legend()
    fig.savefig('stratification.png', bbox_inches='tight', dpi=300)

    #Create interpolators for the atmosphere.
    atmosphere = dict()
    atmosphere['grad_pomega'] = interp1d(r, grad_pom, **interp_kwargs)
    atmosphere['grad_ln_pomega'] = interp1d(r, grad_ln_pom, **interp_kwargs)
    atmosphere['grad_ln_rho'] = interp1d(r, grad_ln_rho, **interp_kwargs)
    atmosphere['grad_s'] = interp1d(r, grad_s, **interp_kwargs)
    atmosphere['g'] = interp1d(r, g, **interp_kwargs)
    atmosphere['pomega'] = interp1d(r, pom, **interp_kwargs)
    atmosphere['rho'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['g_phi'] = interp1d(r, g_phi, **interp_kwargs)
    atmosphere['N2'] = interp1d(r, N2, **interp_kwargs)
    atmosphere['Q'] = interp1d(r, Q, **interp_kwargs)
    atmosphere['s0'] = interp1d(r, s0, **interp_kwargs)
    atmosphere['L_heat'] = interp1d(r, L_heat, **interp_kwargs)
    return atmosphere

def get_fastICs(problem, ncc_file, namespace, NuvRe, Re, equations='FC_HD', boundary='upper'):
    """
    Get the initial conditions for a convection problem which has a thermal boundary layer.
    Assumes that the background stratification is adiabatic.

    
    Parameters
    ----------
    problem : CompressibleProblem object
        The dedalus problem object.
    ncc_file : str
        The name of the HDF5 file containing the stellar stratification.
    namespace : dict
        The namespace of the dedalus IVP.
    NuvRe : function
        A function which returns the Nusselt number as a function of the heating-timescale Reynolds number.
    Re : float
        The heating-timescale Reynolds number.    
    equations : str
        The dedalus equation formulation. Currently just 'FC_HD' is supported.
    boundary : str
        The boundary where the FastIC boundary layer is created. Currently just 'upper' is supported.
    """
    if equations != 'FC_HD':
        raise NotImplementedError('Only FC_HD is currently supported.')
    if boundary != 'upper':
        raise NotImplementedError('Only upper boundary is currently supported.')

    resolutions = [list(n) for n in problem.resolutions]
    for i in range(len(resolutions)):
        resolutions[i][0] = 1
        resolutions[i][1] = 1
        resolutions[i] = tuple(resolutions[i])
    
    #make a local copy of the dedalus coords and bases.
    coords, dist, bases, bases_keys = make_bases(resolutions, problem.stitch_radii, problem.radius, r_inner=problem.r_inner, comm=MPI.COMM_SELF)
    
    #get L_heat, kappa_rad, r, ln_rho, pom0, and s0 from the NCC file for each basis.
    fields = ['L_heat', 'kappa_rad', 'r', 'ln_rho0', 'pom0', 's0', 'g']
    stratification = OrderedDict()
    local_ns = OrderedDict()
    with h5py.File(ncc_file, 'r') as f:
        for b, basis in bases.items():
            for field in fields:
                name = '{}_{}'.format(field, b)
                global_field = namespace['{}_{}'.format(field, b)]
                if isinstance(global_field, d3.Field):
                    local_ns[name] = dist.Field(bases=basis.radial_basis, name=name, tensorsig=tuple([coords]*len(global_field.tensorsig)), dtype=global_field.dtype)
                    local_ns[name].change_scales(basis.dealias)
                    local_ns[name]['g'] = f['dedalus']['{}_{}'.format(field, b)][()]
                else:
                    local_ns[name] = f['dedalus']['{}_{}'.format(field, b)][()]
            local_ns['ones_{}'.format(b)] = dist.Field(bases=basis, name='ones')
            local_ns['ones_{}'.format(b)]['g'] = 1
        R_gas = f['scalars']['R'][()]
        gamma_gas = f['scalars']['gamma1'][()]
        Cp_gas = f['scalars']['Cp'][()]
    
    #Solve for temperature gradient amplitude at top of atmosphere
    radius = problem.radius
    Nu = NuvRe(Re)
    kappa = (local_ns['ones_{}'.format(bases_keys[-1])]*local_ns['kappa_rad_{}'.format(bases_keys[-1])])(r=radius).evaluate()['g'].min()
    dTdr = (-(local_ns['ones_{}'.format(bases_keys[-1])]*local_ns['L_heat_{}'.format(bases_keys[-1])]/kappa)(r=radius).evaluate()['g'][2] / (4*np.pi*radius**2)).min()
    print(kappa, dTdr)


    # Newton iteration to solve for the temperature gradient's shape
    L_conv_int = np.sum([np.trapz(local_ns['L_heat_{}'.format(b)]['g'], x=local_ns['r_{}'.format(b)]) for b in bases_keys])
    lorentzian = lambda r, delta: (1/np.pi)*(delta)/((r-radius)**2 + (delta)**2)
    norm_lorentzian = lambda r, delta: lorentzian(r, delta)/lorentzian(radius, delta)
    #dTdr_func = lambda r, delta: dTdr*(zero_to_one(r, radius-delta, width=delta/2) + 0.5*norm_lorentzian(r, delta))
    dTdr_func = lambda r, delta: dTdr*norm_lorentzian(r, delta)
    del_bl = (3*L_conv_int/(-dTdr*4*np.pi*Nu*kappa))**(1/3) #guess for the boundary layer thickness
    err = 1e-5
    tol = 1e-10
    compterm = L_conv_int/(4*np.pi*Nu)
    while np.abs(err) > tol:
        logger.debug('FastIC err {}, del_bl {}'.format(err, del_bl))
        #calculate magnitude of int(Lconv dr) /(4 pi Nu) + int(kappa*dTdr*r^2*dr) == err
        err = compterm + np.sum([np.trapz(local_ns['kappa_rad_{}'.format(b)]['g']*dTdr_func(local_ns['r_{}'.format(b)], del_bl)*local_ns['r_{}'.format(b)]**2, x=local_ns['r_{}'.format(b)]) for b in bases_keys])
        err /= compterm #fractional err
        del_bl *= 1 + err/100
    logger.debug('FastIC err {}, del_bl {}'.format(err, del_bl))
    logger.info('Fast IC with Nu = {} using del_bl = {}'.format(Nu, del_bl))
    
    #calculate the temperature gradient as dedalus fields
    for i, b in enumerate(bases_keys):
        local_ns['grad_T1_{}'.format(b)] = dist.VectorField(coords, bases=basis, name='grad_T_{}'.format(b))
        local_ns['grad_T1_{}'.format(b)].change_scales(basis.dealias)
        local_ns['grad_T1_{}'.format(b)]['g'][2] = dTdr_func(local_ns['r_{}'.format(b)], del_bl)

    

    # Parameters
    namespace = dict()
    namespace['R'] = R = dist.Field(name='R')
    namespace['R']['g'] = R_gas
    namespace['gamma'] = gamma = dist.Field(name='gamma')
    namespace['gamma']['g'] = gamma_gas
    namespace['Cp'] = Cp = dist.Field(name='Cp')
    namespace['Cp']['g'] = Cp_gas
    log = np.log
    
    #Set up a bvp that forces grad_s and grad_ln_rho to produce this grad T, then solves for s1 and ln_rho1.
    variables = []
    taus = []
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        # Make problem variables and taus
        namespace['pom0_{}'.format(k)] = pom0 = local_ns['pom0_{}'.format(k)]
        namespace['s0_{}'.format(k)] = local_ns['s0_{}'.format(k)]
        namespace['ln_rho0_{}'.format(k)] = local_ns['ln_rho0_{}'.format(k)]
        namespace['rho0_{}'.format(k)] = rho0 = np.exp(local_ns['ln_rho0_{}'.format(k)]).evaluate()
        namespace['g_{}'.format(k)] = g = local_ns['g_{}'.format(k)]
        namespace['grad_pom1_{}'.format(k)] = (namespace['R']*local_ns['grad_T1_{}'.format(k)]).evaluate()

        namespace['er_{}'.format(k)] = er = dist.VectorField(coords, name='er', bases=basis.radial_basis)
        er['g'][2] = 1
        namespace['pom1_{}'.format(k)] = pom1 = dist.Field(name='pom1', bases=basis)
        namespace['s1_{}'.format(k)] = s1 = dist.Field(name='s', bases=basis)
        namespace['ln_rho1_{}'.format(k)] = ln_rho1 = dist.Field(name='ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_s2_{}'.format(k)] = tau_s2 = dist.Field(name='tau_s2', bases=S2_basis)
        

        namespace['pom1_d_pom0_{}'.format(k)] = pom1_d_pom0 = gamma*s1/Cp + (gamma-1)*ln_rho1
        namespace['pom2_d_pom0_{}'.format(k)] = pom2_d_pom0 = np.exp(pom1_d_pom0) - (1 + pom1_d_pom0)
        namespace['pomfluc_{}'.format(k)] = pomfluc = pom0*(pom1_d_pom0 + pom2_d_pom0)
        namespace['HSE_base_{}'.format(k)] = HSE_base = gamma*(d3.grad(s1)/Cp + d3.grad(ln_rho1))

        # Make lift operators for BCs
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A, k=-1: d3.Lift(A, basis, k)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A, k=-1: d3.Lift(A, basis.derivative_basis(2), k)
    
        variables += [namespace['{}_{}'.format(var, k)] for var in ['s1', 'ln_rho1']]
        taus += [tau_s]
        if k == 'B' or k == 'S0':
            namespace['tau_M'.format(k)] = tau_M = dist.Field(name='tau_M', bases=S2_basis)
            if k == 'B':
                taus += [tau_M,]
            else:
                taus += [tau_M, tau_s2]
    
    locals().update(namespace)
    problem = d3.NLBVP(variables + taus, namespace=locals())
    for k, basis in bases.items():
        #Equation is just definitional.
        if k == 'B' or k == 'S0':
            problem.add_equation("grad(pom0_{0}*pom1_d_pom0_{0}) + er_{0}*lift_{0}(tau_M) = grad_pom1_{0}".format(k))
        else:
            problem.add_equation("grad(pom0_{0}*pom1_d_pom0_{0}) = grad_pom1_{0}".format(k))
        if k == 'S0':
            problem.add_equation("div(pom0_{0}*HSE_base_{0} + g_{0}*pom1_d_pom0_{0}) + lift_{0}(tau_s_{0}) + lift_{0}(tau_s2_{0}, k=-2) = -div(pomfluc_{0}*HSE_base_{0} + g_{0}*pom2_d_pom0_{0})".format(k))
        else:
            problem.add_equation("div(pom0_{0}*HSE_base_{0} + g_{0}*pom1_d_pom0_{0}) + lift_{0}(tau_s_{0}) = -div(pomfluc_{0}*HSE_base_{0} + g_{0}*pom2_d_pom0_{0})".format(k))
    
    #Set boundary conditions.
    iter = 0
    mass_integ_L = 0
    mass_integ_R = 0
    for k, basis in bases.items():
        mass_integ_L += d3.integ(namespace['rho0_{}'.format(k)]*namespace['ln_rho1_{}'.format(k)])
        mass_integ_R += -d3.integ(namespace['rho0_{}'.format(k)]*(np.exp(namespace['ln_rho1_{}'.format(k)]) - 1 - namespace['ln_rho1_{}'.format(k)]))
        if k == 'S0':
            problem.add_equation("radial(grad(pom1_d_pom0_{0})(r={1})) = 0".format(k, basis.radii[0]))
        if iter < len(bases)-1:
            k_next = list(bases.keys())[iter+1]
            r_s = problem.stitch_radii[iter]
            problem.add_equation("s1_{0}(r={2}) - s1_{1}(r={2}) = 0".format(k, k_next, r_s))
        else:
            #integ of mass == 0
            problem.add_equation("s1_{0}(r={1}) = 0".format(k, radius))
            problem.add_equation((mass_integ_L, mass_integ_R))
        iter += 1

    solver = problem.build_solver()
    pert_norm = np.inf
    while pert_norm > 1e-8:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
    logger.info('FastIC found')
    logger.info('mass conservation: {}'.format((mass_integ_L - mass_integ_R).evaluate()['g']))

#    r = local_ns['r_S0']
#    plt.plot(r.ravel(), s1['g'].ravel())

#    plt.figure()
#    plt.plot(r.ravel(), local_ns['grad_T1_S0']['g'][2].ravel())

#    plt.figure()
#    plt.plot(r.ravel(), namespace['ln_rho1_S0']['g'].ravel())
#    plt.show()
#    import sys
#    sys.exit()
    return namespace

