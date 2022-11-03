import os, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import h5py
from mpi4py import MPI
import matplotlib.pyplot as plt
import dedalus.public as d3
import mesa_reader as mr

from astropy import units as u
from astropy import constants
from scipy.interpolate import interp1d

import d3_stars
from .anelastic_functions import make_bases
from .parser import name_star
import d3_stars.defaults.config as config

import logging
logger = logging.getLogger(__name__)

interp_kwargs = {'fill_value' : 'extrapolate', 'bounds_error' : False}
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




def HSE_solve(coords, dist, bases, grad_ln_rho_func, N2_func, Fconv_func, r_stitch=[], r_outer=1, dtype=np.float64, \
              R=1, gamma=5/3, comm=MPI.COMM_SELF, nondim_radius=1, g_nondim=1, s_motions=1, smooth_edge=True):

    Cp = R*gamma/(gamma-1)
    Cv = Cp/gamma

    # Parameters
    scales = bases['B'].dealias[-1]
    namespace = dict()
    namespace['R'] = R
    namespace['Cp'] = Cp
    namespace['gamma'] = gamma
    namespace['log'] = np.log
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        namespace['g_phi_{}'.format(k)] = Q = dist.Field(name='g_phi', bases=basis)
        namespace['Q_{}'.format(k)] = Q = dist.Field(name='Q', bases=basis)
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['g_{}'.format(k)] = g = dist.VectorField(basis.coordsystem, name='g', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_g_phi_{}'.format(k)] = tau_g_phi = dist.Field(name='tau_g_phi', bases=S2_basis)

        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        low_scales = 16/basis.radial_basis.radial_size
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,low_scales))
        namespace['r_de_{}'.format(k)] = r_de
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis.derivative_basis(2), -1)

        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1

        namespace['edge_smoothing_{}'.format(k)] = edge_smooth = dist.Field(bases=basis, name='edge_smooth')
        if smooth_edge:
            edge_smooth['g'] = one_to_zero(r, 0.95*bases['B'].radius, width=0.03*bases['B'].radius)
        else:
            edge_smooth['g'] = 1
        namespace['N2_{}'.format(k)] = N2 = dist.Field(bases=basis, name='N2')

        if k == 'B':
            N2['g'] = (r/basis.radius)**2 * (N2_func(basis.radius)) * zero_to_one(r, basis.radius*0.9, width=basis.radius*0.03)
        else:
            N2.change_scales(low_scales)
            N2['g'] = N2_func(r_low)

        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r
        namespace['r_squared_{}'.format(k)] = r_squared = dist.Field(bases=basis.radial_basis)
        r_squared['g'] = r**2

        grad_ln_rho.change_scales(low_scales)
        grad_ln_rho['g'][2] = grad_ln_rho_func(r_low)


        namespace['Fconv_{}'.format(k)] = Fconv   = dist.VectorField(coords, name='Fconv', bases=basis)
        Fconv['g'][2] = Fconv_func(r)

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



    namespace['pi'] = pi = np.pi
    locals().update(namespace)
    ncc_cutoff=1e-6
    tolerance=1e-6
    HSE_tolerance = 1e-3

    #Solve for ln_rho.
    variables = []
    for k, basis in bases.items():
        variables += [namespace['ln_rho_{}'.format(k)],]
    for k, basis in bases.items():
        variables += [namespace['tau_rho_{}'.format(k)],]

    problem = d3.NLBVP(variables, namespace=locals())
    for k, basis in bases.items():
        problem.add_equation("grad(ln_rho_{0}) - grad_ln_rho_{0} + r_vec_{0}*lift_{0}(tau_rho_{0}) = 0".format(k))
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
        iter += 1
    problem.add_equation("ln_rho_B(r=nondim_radius) = 0")
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')

    logger.info('ln_rho found')

    #solve for everything else.
    variables = []
    for k, basis in bases.items():
        variables += [namespace['s_{}'.format(k)], namespace['g_{}'.format(k)], namespace['Q_{}'.format(k)], namespace['g_phi_{}'.format(k)]]
    for k, basis in bases.items():
        variables += [namespace['tau_s_{}'.format(k)], namespace['tau_g_phi_{}'.format(k)]]


    problem = d3.NLBVP(variables, namespace=locals())

    for k, basis in bases.items():
        #initial condition
        namespace['s_{}'.format(k)].change_scales(basis.dealias)
        namespace['s_{}'.format(k)]['g'] = -(R*namespace['ln_rho_{}'.format(k)]).evaluate()['g']
        problem.add_equation("grad(ln_rho_{0})@(grad(s_{0})/Cp) + lift_{0}(tau_s_{0}) = -N2_{0}/(gamma*pomega_{0}) - grad(s_{0})@grad(s_{0}) / Cp**2".format(k))
        problem.add_equation("g_{0} = g_op_{0} ".format(k))
        problem.add_equation("Q_{0} = edge_smoothing_{0}*div(Fconv_{0})".format(k))
        problem.add_equation("grad(g_phi_{0}) + g_{0} + r_vec_{0}*lift_{0}(tau_g_phi_{0}) = 0".format(k))
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("g_phi_{0}(r={2}) - g_phi_{1}(r={2}) = 0".format(k, k_old, r_s))
        iter += 1
        if iter == len(bases.items()):
            problem.add_equation("g_phi_{0}(r=r_outer) = 0".format(k))
    problem.add_equation("ln_pomega_LHS_B(r=nondim_radius) = 0")


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
#        plt.plot(namespace['r_de_B'].ravel(),  namespace['g_op_B'].evaluate()['g'][2].ravel())
#        plt.plot(namespace['r_de_S1'].ravel(), namespace['g_op_S1'].evaluate()['g'][2].ravel())
#        plt.show()

    #Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'grad_s', 'g', 'pomega', 'rho', 'ln_rho', 'g_phi', 'r_vec', 'HSE', 'N2_op', 'Q', 's0']
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
    ax6.plot(r, g_phi, label='g_phi')
    ax6.legend()
    ax7.plot(r, N2, label=r'$N^2$')
    ax7.plot(r, -N2, label=r'$-N^2$')
    ax7.plot(r, (N2_func(r)), label=r'$N^2$ goal', ls='--')
    ax7.set_yscale('log')
    yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
    ax7.set_yticks(yticks)
    ax7.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
    ax7.legend()
    ax8.plot(r, grad_s, label='grad s')
    ax8.axhline(s_motions)
    ax8.set_yscale('log')
    ax8.legend()
    fig.savefig('stratification.png', bbox_inches='tight', dpi=300)
#    plt.show()

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
    return atmosphere


### Function definitions
def plot_ncc_figure(rvals, mesa_func, dedalus_vals, Ns, ylabel="", fig_name="", out_dir='.', zero_line=False, log=False, r_int=None, ylim=None, axhline=None, ncc_cutoff=1e-6):
    """ Plots up a figure to compare a dedalus field to the MESA field it's based on. """
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    if axhline is not None:
        ax1.axhline(axhline, c='k')

    for r, y in zip(rvals, dedalus_vals):
        mesa_y = mesa_func(r)
        ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
        ax1.plot(r, y, label='dedalus', c='red')

        diff = np.abs(1 - mesa_y/y)
        ax2.plot(r, diff)

    ax1.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    if log:
        ax1.set_yscale('log')
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2.axhline(1e-1, c='k', lw=0.5)
    ax2.axhline(1e-2, c='k', lw=0.5)
    ax2.axhline(1e-3, c='k', lw=0.5)
    ax2.set_ylabel('abs(1 - mesa/dedalus)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')

    ax2.set_ylim(1e-4, 1)
    fig.suptitle('coeff bandwidth = {}; cutoff = {:e}'.format(Ns, ncc_cutoff))
    if r_int is not None:
        for ax in [ax1, ax2]:
            for rval in r_int:
                ax.axvline(rval, c='k')
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

def make_NCC(basis, coords, dist, interp_func, Nmax=32, vector=False, grid_only=False, ncc_cutoff=1e-6):
    if grid_only:
        scales = basis.dealias
    else:
        scales = basis.dealias
        scales_small = (1, 1, Nmax/basis.radial_basis.radial_size)
    rvals = basis.global_grid_radius(scales[2])
    if vector:
        this_field = dist.VectorField(coords, bases=basis)
        this_field.change_scales(scales)
        this_field['g'][2] = interp_func(rvals)
    else:
        this_field = dist.Field(bases=basis)
        this_field.change_scales(scales)
        this_field['g'] = interp_func(rvals)
    if not grid_only:
        this_field.change_scales(scales_small)
        this_field['g']
        this_field['c'][np.abs(this_field['c']) < ncc_cutoff] = 0
        this_field.change_scales(basis.dealias)
    return this_field

def build_nccs(plot_nccs=False):
    # Read in parameters and create output directory
    out_dir, out_file = name_star()
    ncc_dict = config.nccs

    package_path = Path(d3_stars.__file__).resolve().parent
    stock_path = package_path.joinpath('stock_models')
    mesa_file_path = None
    if os.path.exists(config.star['path']):
        mesa_file_path = config.star['path']
    else:
        stock_file_path = stock_path.joinpath(config.star['path'])
        if os.path.exists(stock_file_path):
            mesa_file_path = str(stock_file_path)
        else:
            raise ValueError("Cannot find MESA profile file in {} or {}".format(config.star['path'], stock_file_path))

    #TODO: figure out how to make MESA the file path w.r.t. stock model path w/o supplying full path here 
    logger.info("Reading MESA file {}".format(mesa_file_path))
    p = mr.MesaData(mesa_file_path)
    mass           = (p.mass[::-1] * u.M_sun).cgs
    r              = (p.radius[::-1] * u.R_sun).cgs
    rho            = 10**p.logRho[::-1] * u.g / u.cm**3
    P              = p.pressure[::-1] * u.g / u.cm / u.s**2
    T              = p.temperature[::-1] * u.K
    R_gas          = P / (rho * T)
    nablaT         = p.gradT[::-1] #dlnT/dlnP
    nablaT_ad      = p.grada[::-1]
    chiRho         = p.chiRho[::-1]
    chiT           = p.chiT[::-1]
    cp             = p.cp[::-1]  * u.erg / u.K / u.g
    opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
    Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
    conv_L_div_L   = p.lum_conv_div_L[::-1]
    csound         = p.csound[::-1] * u.cm / u.s
    N2 = N2_mesa   = p.brunt_N2[::-1] / u.s**2
    N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
    N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
    eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
    mu             = p.mu[::-1] * u.g / u.mol 
    lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r

    R_star = (p.photosphere_r * u.R_sun).cgs
    
    #Put all MESA fields into cgs and calculate secondary MESA fields
    R_gas           = constants.R.cgs / mu[0]
    g               = constants.G.cgs*mass/r**2
    dlogPdr         = -rho*g/P
    gamma1          = dlogPdr/(-g/csound**2)
    dlogrhodr       = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
    dlogTdr         = dlogPdr*(nablaT)
    grad_s_over_cp  = N2/g #entropy gradient, for NCC, includes composition terms
    grad_s          = cp * grad_s_over_cp
    L_conv          = conv_L_div_L*Luminosity
    dTdr            = (T)*dlogTdr

    # Calculate k_rad and radiative diffusivity using luminosities and smooth things.
    k_rad = rad_cond = -(Luminosity - L_conv)/(4*np.pi*r**2*dTdr)
    rad_diff        = k_rad / (rho * cp)
    #rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs # this is less smooth

    ### CORE CONVECTION LOGIC - Find boundary of core convection zone  & setup simulation domain
    ### Split up the domain
    # Find edge of core cz
    cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1]) #rudimentary but works
    core_index  = np.argmin(np.abs(mass - mass[cz_bool][-1]))
    core_cz_radius = r[core_index]

    # Specify fraction of total star to simulate
    r_bounds = list(config.star['r_bounds'])
    r_bools = []
    for i, rb in enumerate(r_bounds):
        if type(rb) == str:
            if 'R' in rb:
                r_bounds[i] = float(rb.replace('R', ''))*R_star
            elif 'L' in rb:
                if rb == 'L':
                    r_bounds[i] = core_cz_radius
                    use_heat_nd = True
                else:
                    r_bounds[i] = float(rb.replace('L', ''))*core_cz_radius
                    use_heat_nd = False
            else:
                try:
                    r_bounds[i] = float(r_bounds[i]) * u.cm
                except:
                    raise ValueError("index {} ('{}') of r_bounds is poorly specified".format(i, rb))
            r_bounds[i] = core_cz_radius*np.around(r_bounds[i]/core_cz_radius, decimals=2)
    for i, rb in enumerate(r_bounds):
        if i < len(r_bounds) - 1:
            r_bools.append((r > r_bounds[i])*(r <= r_bounds[i+1]))
    logger.info('fraction of FULL star simulated: {:.2f}, up to r={:.3e}'.format(r_bounds[-1]/R_star, r_bounds[-1]))
    sim_bool      = (r > r_bounds[0])*(r <= r_bounds[-1])

    #Get N2 info
    N2max_sim = N2[sim_bool].max()
    shell_points = np.sum(sim_bool*(r > core_cz_radius))
    N2plateau = np.median(N2[r > core_cz_radius][int(shell_points*0.25):int(shell_points*0.75)])
    f_brunt = np.sqrt(N2max_sim)/(2*np.pi)
 
    #Nondimensionalization
    L_CZ    = core_cz_radius
    m_core  = rho[0] * L_CZ**3
    T_core  = T[0]
    H0      = (rho*eps_nuc)[0]
    tau_heat  = ((H0*L_CZ/m_core)**(-1/3)).cgs #heating timescale
    L_nd    = L_CZ
    m_nd    = rho[r==L_nd][0] * L_nd**3 #mass at core cz boundary
    T_nd    = T[r==L_nd][0] #temp at core cz boundary
    if use_heat_nd:
        tau_nd = tau_heat.cgs
    else:
        tau_nd  = (1/f_brunt).cgs #timescale of max N^2
    rho_nd  = m_nd/L_nd**3
    u_nd    = L_nd/tau_nd
    s_nd    = L_nd**2 / tau_nd**2 / T_nd
    H_nd    = (m_nd / L_nd) * tau_nd**-3
    s_motions    = L_nd**2 / tau_heat**2 / T[0]
    lum_nd  = L_nd**2 * m_nd / (tau_nd**2) / tau_nd
    nondim_R_gas = (R_gas / s_nd).cgs.value
    nondim_gamma1 = (gamma1[0]).value
    nondim_cp = nondim_R_gas * nondim_gamma1 / (nondim_gamma1 - 1)
    nondim_G = (constants.G * (rho_nd * tau_nd**2)).value
    u_heat_nd = (L_nd/tau_heat) / u_nd
    Ma2_r0 = ((u_nd*(tau_nd/tau_heat))**2 / ((gamma1[0]-1)*cp[0]*T[0])).cgs
    logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
    logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(nondim_cp, nondim_R_gas, nondim_gamma1))
    logger.info('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))
    logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(Ma2_r0), tau_heat))

#    g_phi           = u_nd**2 + np.cumsum(g*np.gradient(r)) #gvec = -grad phi; set g_phi = 1 at r = 0
    g_over_cp       = g / cp
    g_phi           = np.cumsum(g*np.gradient(r))  #gvec = -grad phi; 
    g_phi -= g_phi[-1] - u_nd**2 #set g_phi = -1 at r = R_star
    grad_ln_g_phi   = g / g_phi
    s_over_cp       = np.cumsum(grad_s_over_cp*np.gradient(r))
    pomega_tilde    = np.cumsum(s_over_cp * g * np.gradient(r)) #TODO: should this be based on the actual grad s used in the simulation?
# integrate by parts:    pomega_tilde    = s_over_cp * g_phi - np.cumsum(grad_s_over_cp * g_phi * np.gradient(r)) #TODO: should this be based on the actual grad s used in the simulation?

    #construct simulation diffusivity profiles
    rad_diff_nd = rad_diff * (tau_nd / L_nd**2)
    rad_diff_cutoff = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((L_CZ**2/tau_heat) / (L_nd**2/tau_nd))
    sim_rad_diff = np.copy(rad_diff_nd) + rad_diff_cutoff
    sim_nu_diff = config.numerics['prandtl']*rad_diff_cutoff*np.ones_like(sim_rad_diff)
    Re_shift = ((L_nd**2/tau_nd) / (L_CZ**2/tau_heat))

    logger.info('u_heat_nd: {:.3e}'.format(u_heat_nd))
    logger.info('rad_diff cutoff: {:.3e}'.format(rad_diff_cutoff))
    
    #MESA radial values at simulation joints & across full star in simulation units
    r_bound_nd = [(rb/L_nd).value for rb in r_bounds]
    r_nd = (r/L_nd).cgs
    
    ### entropy gradient
    ### More core convection zone logic here
    if core_cz_radius == r_bounds[-1]:
        grad_s_smooth = np.zeros_like(grad_s)
        N2_func = interp1d(r_nd, np.zeros_like(grad_s_smooth), **interp_kwargs)
    else:
        #Build a nice function for our basis in the ball
        grad_s_width = 0.05
        grad_s_transition_point = r_bound_nd[1] - grad_s_width
        logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
        logger.info('using default grad s width = {}'.format(grad_s_width))
        grad_s_center =  grad_s_transition_point - 0.5*grad_s_width
        grad_s_width *= (L_CZ/L_nd).value
        grad_s_center *= (L_CZ/L_nd).value
       
    #    grad_s_outer_ball = grad_s[r/L_nd <= r_bound_nd[1]][-1]
        grad_s_smooth = np.copy(grad_s)
    #    grad_s_smooth[r/L_nd <= r_bound_nd[1]] = (grad_s_outer_ball * (r/L_nd / r_bound_nd[1])**2)[r/L_nd <= r_bound_nd[1]]
        flat_value  = np.interp(grad_s_transition_point, r/L_nd, grad_s)
        grad_s_smooth += (r/L_nd)**2 *  flat_value
        grad_s_smooth *= zero_to_one(r/L_nd, grad_s_transition_point, width=grad_s_width)
    #    plt.plot(r/L_nd, grad_s_smooth)
    #    plt.yscale('log')
    #    plt.show()
        #construct N2 function #TODO: blend logic here & in BVP.
        smooth_N2 = np.copy(N2_mesa)
        stitch_value = np.interp(bases['B'].radius, r/L_nd, N2_mesa)
        smooth_N2[r/L_nd < bases['B'].radius] = (r[r/L_nd < bases['B'].radius]/L_nd / bases['B'].radius)**2 * stitch_value
        smooth_N2 *= zero_to_one(r/L_nd, grad_s_transition_point, width=grad_s_width)
        N2_func = interp1d(r_nd, tau_nd**2 * smooth_N2, **interp_kwargs)

    
   
    ### Make dedalus domain and bases
    resolutions = [(1, 1, nr) for nr in config.star['nr']]
    stitch_radii = r_bound_nd[1:-1]
    dtype=np.float64
    mesh=None
    dealias = config.numerics['N_dealias']
    c, d, bases, bases_keys = make_bases(resolutions, stitch_radii, r_bound_nd[-1], dealias=(1,1,dealias), dtype=dtype, mesh=mesh)
    dedalus_r = OrderedDict()
    for bn in bases.keys():
        phi, theta, r_vals = bases[bn].global_grids((1, 1, dealias))
        dedalus_r[bn] = r_vals


    
    if config.star['smooth_h']:
        #smooth CZ-RZ transition
        L_conv_sim = np.copy(L_conv)
        L_conv_sim *= one_to_zero(r, 0.9*core_cz_radius, width=0.05*core_cz_radius)
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)
        L_conv_sim /= (r/L_nd)**2 * (4*np.pi)
        F_conv_func = interp1d(r/L_nd, L_conv_sim/lum_nd, **interp_kwargs)
    elif config.star['heat_only']:
        #smooth CZ-RZ transition
        L_conv_sim = np.copy(Luminosity)
#        L_conv_sim *= one_to_zero(r, 0.9*core_cz_radius, width=0.05*core_cz_radius)
#        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)
        L_conv_sim /= (r/L_nd)**2 * (4*np.pi)
        F_conv_func = interp1d(r/L_nd, L_conv_sim/lum_nd, **interp_kwargs)
    else:
        raise NotImplementedError("must use smooth_h")


    # Get some timestepping & wave frequency info
    f_nyq = 2*tau_nd*np.sqrt(N2max_sim)/(2*np.pi)
    nyq_dt   = (1/f_nyq) 
    kepler_tau     = 30*60*u.s
    max_dt_kepler  = kepler_tau/tau_nd
    max_dt = max_dt_kepler
    logger.info('needed nyq_dt is {} s / {} % of a nondimensional time (Kepler 30 min is {} %) '.format(nyq_dt*tau_nd, nyq_dt*100, max_dt_kepler*100))
 
    #Create interpolations of the various fields that may be used in the problem
    interpolations = OrderedDict()
    interpolations['ln_rho0'] = interp1d(r_nd, np.log(rho/rho_nd), **interp_kwargs)
    interpolations['ln_T0'] = interp1d(r_nd, np.log(T/T_nd), **interp_kwargs)
    interpolations['grad_ln_rho0'] = interp1d(r_nd, dlogrhodr*L_nd, **interp_kwargs)
    interpolations['grad_ln_T0'] = interp1d(r_nd, dlogTdr*L_nd, **interp_kwargs)
    interpolations['T0'] = interp1d(r_nd, T/T_nd, **interp_kwargs)
    interpolations['nu_diff'] = interp1d(r_nd, sim_nu_diff, **interp_kwargs)
    interpolations['chi_rad'] = interp1d(r_nd, sim_rad_diff, **interp_kwargs)
    interpolations['grad_chi_rad'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd), **interp_kwargs)
    interpolations['g'] = interp1d(r_nd, -g * (tau_nd**2/L_nd), **interp_kwargs)
    interpolations['g_phi'] = interp1d(r_nd, g_phi * (tau_nd**2 / L_nd**2), **interp_kwargs)
    interpolations['pomega_tilde'] = interp1d(r_nd, pomega_tilde * (tau_nd**2 / L_nd**2), **interp_kwargs)

    ln_rho_func = interpolations['ln_rho0']
    grad_ln_rho_func = interpolations['grad_ln_rho0']
    atmo = HSE_solve(c, d, bases,  grad_ln_rho_func, N2_func, F_conv_func,
              r_outer=r_bound_nd[-1], r_stitch=stitch_radii, dtype=np.float64, \
              R=nondim_R_gas, gamma=nondim_gamma1, comm=MPI.COMM_SELF, \
              nondim_radius=1, g_nondim=interpolations['g'](1), s_motions=s_motions/s_nd, smooth_edge=not(config.star['heat_only']))

    interpolations['ln_rho0'] = atmo['ln_rho']
    interpolations['Q'] = atmo['Q']
    interpolations['g'] = atmo['g']
    interpolations['g_phi'] = atmo['g_phi']
    interpolations['grad_s0'] = atmo['grad_s']
    interpolations['s0'] = atmo['s0']
    interpolations['pom0'] = atmo['pomega']
    interpolations['grad_ln_pom0'] = atmo['grad_ln_pomega']


    interpolations['kappa_rad'] = interp1d(r_nd, np.exp(interpolations['ln_rho0'](r_nd))*nondim_cp*sim_rad_diff, **interp_kwargs)
    interpolations['grad_kappa_rad'] = interp1d(r_nd, np.gradient(interpolations['kappa_rad'](r_nd), r_nd), **interp_kwargs)

    for ncc in ncc_dict.keys():
        for i, bn in enumerate(bases.keys()):
            ncc_dict[ncc]['Nmax_{}'.format(bn)] = ncc_dict[ncc]['nr_max'][i]
            ncc_dict[ncc]['field_{}'.format(bn)] = None
        if ncc in interpolations.keys():
            ncc_dict[ncc]['interp_func'] = interpolations[ncc]
        else:
            ncc_dict[ncc]['interp_func'] = None

    for bn, basis in bases.items():
        rvals = dedalus_r[bn]
        for ncc in ncc_dict.keys():
            interp_func = ncc_dict[ncc]['interp_func']
            if interp_func is not None and not ncc_dict[ncc]['from_grad']:
                Nmax = ncc_dict[ncc]['Nmax_{}'.format(bn)]
                vector = ncc_dict[ncc]['vector']
                grid_only = ncc_dict[ncc]['grid_only']
                ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, c, d, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                if ncc_dict[ncc]['get_grad']:
                    name = ncc_dict[ncc]['grad_name']
                    logger.info('getting {}'.format(name))
                    grad_field = d3.grad(ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                    grad_field.change_scales((1,1,(Nmax+1)/resolutions[bases_keys == bn][2]))
                    grad_field.change_scales(basis.dealias)
                    ncc_dict[name]['field_{}'.format(bn)] = grad_field
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                if ncc_dict[ncc]['get_inverse']:
                    name = 'inv_{}'.format(ncc)
                    inv_func = lambda r: 1/interp_func(r)
                    ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, c, d, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax


        if 'neg_g' in ncc_dict.keys():
            if 'g' not in ncc_dict.keys():
                ncc_dict['g'] = OrderedDict()
            name = 'g'
            ncc_dict['g']['field_{}'.format(bn)] = (-ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
            ncc_dict['g']['vector'] = True
            ncc_dict['g']['interp_func'] = interpolations['g']
            ncc_dict['g']['Nmax_{}'.format(bn)] = ncc_dict['neg_g']['Nmax_{}'.format(bn)]
            ncc_dict['g']['from_grad'] = True 
        
    
#        #Evaluate for grad chi rad
#        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['g'] = d3.grad(ncc_dict['chi_rad']['field_{}'.format(bn)]).evaluate()['g']
    
#    #Further post-process work to make grad_s nice in the ball
#    nr_post = ncc_dict['grad_s0']['nr_post']
#
#    for i, bn in enumerate(bases.keys()):
#        ncc_dict['grad_s0']['field_{}'.format(bn)]['g'][2] *= zero_to_one(dedalus_r[bn], grad_s_center, width=grad_s_width)
#        ncc_dict['grad_s0']['field_{}'.format(bn)]['c'][:,:,:,nr_post[i]:] = 0
#        ncc_dict['grad_s0']['field_{}'.format(bn)]['c'][np.abs(ncc_dict['grad_s0']['field_{}'.format(bn)]['c']) < config.numerics['ncc_cutoff']] = 0
#    
#    #Post-processing for grad chi rad - doesn't work great...
#    nr_post = ncc_dict['grad_chi_rad']['nr_post']
#    for i, bn in enumerate(bases.keys()):
#        if bn == 'B': continue
#        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['c'][:,:,:,nr_post[i]:] = 0
#        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['c'][np.abs(ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['c']) < config.numerics['ncc_cutoff']] = 0

    
    interpolations['ln_rho0'] = interp1d(r_nd, np.log(rho/rho_nd), **interp_kwargs)
    interpolations['ln_T0'] = interp1d(r_nd, np.log(T/T_nd), **interp_kwargs)
    
    if plot_nccs:
        for ncc in ncc_dict.keys():
            if ncc_dict[ncc]['interp_func'] is None:
                continue
            axhline = None
            log = False
            ylim = None
            rvals = []
            dedalus_yvals = []
            nvals = []
            for bn, basis in bases.items():
                rvals.append(dedalus_r[bn].ravel())
                nvals.append(ncc_dict[ncc]['Nmax_{}'.format(bn)])
                if ncc_dict[ncc]['vector']:
                    dedalus_yvals.append(np.copy(ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:]))
                else:
                    dedalus_yvals.append(np.copy(ncc_dict[ncc]['field_{}'.format(bn)]['g'][0,0,:]))
    
            if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad']:
                log = True
            if ncc == 'grad_s0': 
                axhline = (s_motions / s_nd)
            elif ncc in ['chi_rad', 'grad_chi_rad']:
                axhline = rad_diff_cutoff
    
            interp_func = ncc_dict[ncc]['interp_func']
            if ncc == 'H':
                interp_func = interp1d(r_vals, ( one_to_zero(r_vals, 1.5*r_bound_nd[1], width=0.05*r_bound_nd[1])*sim_H_eff ) * (1/H_nd), **interp_kwargs )
            elif ncc == 'grad_s0':
                interp_func = interp1d(r_nd, (L_nd/s_nd) * grad_s, **interp_kwargs)
            elif ncc in ['ln_T0', 'ln_rho0', 'grad_s0']:
                interp_func = interpolations[ncc]
    
            if ncc in ['grad_T', 'grad_kappa_rad']:
                interp_func = lambda r: -ncc_dict[ncc]['interp_func'](r)
                ylabel='-{}'.format(ncc)
                for i in range(len(dedalus_yvals)):
                    dedalus_yvals[i] *= -1
            else:
                ylabel = ncc
    
            plot_ncc_figure(rvals, interp_func, dedalus_yvals, nvals, \
                        ylabel=ylabel, fig_name=ncc, out_dir=out_dir, log=log, ylim=ylim, \
                        r_int=stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

    plt.figure()
    N2s = []
    HSEs = []
    EOSs = []
    grad_s0s = []
    grad_ln_rho0s = []
    grad_ln_pom0s = []
    rs = []
    for bn in bases_keys:
        rs.append(dedalus_r[bn].ravel())
        grad_ln_rho0 = ncc_dict['grad_ln_rho0']['field_{}'.format(bn)]
        grad_ln_pom0 = ncc_dict['grad_ln_pom0']['field_{}'.format(bn)]
        pom0 = ncc_dict['pom0']['field_{}'.format(bn)]
        ln_rho0 = ncc_dict['ln_rho0']['field_{}'.format(bn)]
        gvec = ncc_dict['g']['field_{}'.format(bn)]
        grad_s0 = ncc_dict['grad_s0']['field_{}'.format(bn)]
        s0 = ncc_dict['s0']['field_{}'.format(bn)]
        pom0 = ncc_dict['pom0']['field_{}'.format(bn)]
        HSE = (nondim_gamma1*pom0*(grad_ln_rho0 + grad_s0 / nondim_cp) - gvec).evaluate()
        EOS = s0/nondim_cp - ( (1/nondim_gamma1) * (np.log(pom0) - np.log(nondim_R_gas)) - ((nondim_gamma1-1)/nondim_gamma1) * ln_rho0 )
        N2_val = -gvec['g'][2,:] * grad_s0['g'][2,:] / nondim_cp 
        N2s.append(N2_val)
        HSEs.append(HSE['g'][2,:])
        EOSs.append(EOS.evaluate()['g'])
        grad_ln_rho0s.append(grad_ln_rho0['g'][2,:])
        grad_ln_pom0s.append(grad_ln_pom0['g'][2,:])
    r_dedalus = np.concatenate(rs, axis=-1)
    N2_dedalus = np.concatenate(N2s, axis=-1).ravel()
    HSE_dedalus = np.concatenate(HSEs, axis=-1).ravel()
    EOS_dedalus = np.concatenate(EOSs, axis=-1).ravel()
    grad_ln_rho0_dedalus = np.concatenate(grad_ln_rho0s, axis=-1).ravel()
    grad_ln_pom0_dedalus = np.concatenate(grad_ln_pom0s, axis=-1).ravel()
    plt.plot(r_nd, tau_nd**2*N2_mesa, label='mesa')
    plt.plot(r_nd, atmo['N2'](r_nd), label='atmosphere')
    plt.plot(r_dedalus, N2_dedalus, ls='--', label='dedalus')
    plt.legend()
    plt.ylabel(r'$N^2$')
    plt.xlabel('r')
    plt.yscale('log')
    plt.savefig('star/N2_goodness.png')
#    plt.show()

    plt.figure()
    plt.axhline(s_motions/nondim_cp / s_nd, c='k')
    plt.plot(r_dedalus, np.abs(HSE_dedalus))
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel("HSE")
    plt.savefig('star/HSE_goodness.png')

    plt.figure()
    plt.axhline(s_motions/nondim_cp / s_nd, c='k')
    plt.plot(r_dedalus, np.abs(EOS_dedalus))
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel("EOS")
    plt.savefig('star/EOS_goodness.png')


    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    plt.plot(r_dedalus, grad_ln_rho0_dedalus)
    plt.xlabel('r')
    plt.ylabel("grad_ln_rho0")
    ax2 = fig.add_subplot(2,1,2)
    plt.plot(r_dedalus, grad_ln_pom0_dedalus)
    plt.xlabel('r')
    plt.ylabel("grad_ln_pom0")
    plt.savefig('star/ln_thermo_goodness.png')
#    plt.show()


       
    integral = 0
    for bn in bases.keys():
        integral += d3.integ(ncc_dict['Q']['field_{}'.format(bn)])
    C = integral.evaluate()['g']
    vol = (4/3) * np.pi * (r_bound_nd[-1])**3
#    C = d3.integ(ncc_dict['Q']['field_B']).evaluate()['g']
#    vol = (4/3)*np.pi * bases['B'].radius**3
    adj = C / vol
    logger.info('adjusting dLdt for energy conservation; subtracting {} from H'.format(adj))
    for bn in bases.keys():
        ncc_dict['Q']['field_{}'.format(bn)]['g'] -= adj 

#    dLdt = d3.integ(4*np.pi*ncc_dict['H']['field_B']).evaluate()['g']
    
    with h5py.File('{:s}'.format(out_file), 'w') as f:
        # Save output fields.
        # slicing preserves dimensionality
        for bn, basis in bases.items():
            f['r_{}'.format(bn)] = dedalus_r[bn]
            for ncc in ncc_dict.keys():
                this_field = ncc_dict[ncc]['field_{}'.format(bn)]
                if ncc_dict[ncc]['vector']:
                    f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                    f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = ncc_dict[ncc]['Nmax_{}'.format(bn)]/resolutions[bases_keys == bn][2]
                else:
                    f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
                    f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = ncc_dict[ncc]['Nmax_{}'.format(bn)]/resolutions[bases_keys == bn][2]
    
        f['Cp'] = nondim_cp
        f['R_gas'] = nondim_R_gas
        f['gamma1'] = nondim_gamma1

        #Save properties of the star, with units.
        f['L_nd']   = L_nd
        f['L_nd'].attrs['units'] = str(L_nd.unit)
        f['rho_nd']  = rho_nd
        f['rho_nd'].attrs['units']  = str(rho_nd.unit)
        f['T_nd']  = T_nd
        f['T_nd'].attrs['units']  = str(T_nd.unit)
        f['tau_heat'] = tau_heat
        f['tau_heat'].attrs['units'] = str(tau_heat.unit)
        f['tau_nd'] = tau_nd 
        f['tau_nd'].attrs['units'] = str(tau_nd.unit)
        f['m_nd'] = m_nd 
        f['m_nd'].attrs['units'] = str(m_nd.unit)
        f['s_nd'] = s_nd
        f['s_nd'].attrs['units'] = str(s_nd.unit)
        f['P_r0']  = P[0]
        f['P_r0'].attrs['units']  = str(P[0].unit)
        f['H_nd']  = H_nd
        f['H_nd'].attrs['units']  = str(H_nd.unit)
        f['H0']  = H0
        f['H0'].attrs['units']  = str(H0.unit)
        f['N2max_sim'] = N2max_sim
        f['N2max_sim'].attrs['units'] = str(N2max_sim.unit)
        f['N2plateau'] = N2plateau
        f['N2plateau'].attrs['units'] = str(N2plateau.unit)
        f['cp_surf'] = cp[sim_bool][-1]
        f['cp_surf'].attrs['units'] = str(cp[sim_bool][-1].unit)
        f['r_mesa'] = r
        f['r_mesa'].attrs['units'] = str(r.unit)
        f['N2_mesa'] = N2
        f['N2_mesa'].attrs['units'] = str(N2.unit)
        f['S1_mesa'] = lamb_freq(1)
        f['S1_mesa'].attrs['units'] = str(lamb_freq(1).unit)
        f['g_mesa'] = g 
        f['g_mesa'].attrs['units'] = str(g.unit)
        f['cp_mesa'] = cp
        f['cp_mesa'].attrs['units'] = str(cp.unit)

        #TODO: put sim lum back
        f['lum_r_vals'] = lum_r_vals = np.linspace(r_bound_nd[0], r_bound_nd[-1], 1000)
        f['sim_lum'] = (4*np.pi*lum_r_vals**2)*F_conv_func(lum_r_vals)
        f['r_stitch']   = stitch_radii
        f['Re_shift'] = Re_shift
        f['r_outer']   = r_bound_nd[-1] 
        f['max_dt'] = max_dt
        f['Ma2_r0'] = Ma2_r0
        for k in ['r_stitch', 'r_outer', 'max_dt', 'Ma2_r0', 'Re_shift', 'lum_r_vals', 'sim_lum',\
                    'Cp', 'R_gas', 'gamma1']:
            f[k].attrs['units'] = 'dimensionless'
    logger.info('finished saving NCCs to {}'.format(out_file))
    logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(out_dir))

