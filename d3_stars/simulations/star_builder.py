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




def HSE_solve(coords, dist, bases, grad_s_func, g_func, r_stitch=[], r_outer=1, dtype=np.float64, \
              R=1, gamma=5/3, Cp=2.5, G=1, comm=MPI.COMM_SELF, nondim_radius=1, g_nondim=1, T_func=None, ln_rho_func=None):

    Cv = Cp/gamma
    N2_func = lambda r: grad_s_func(r)*(-g_func(r)) / Cp

    # Parameters
    scales = bases['B'].dealias[-1]
    namespace = dict()
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis

        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()
        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,0.5))
        namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)

        N2_func = lambda r: (-g_func(r)*grad_s_func(r) / Cp)
        N2_outer = N2_func(r_outer)
        N2_extra = lambda r: (r/r_outer)**2 * N2_outer
        namespace['N2_{}'.format(k)] = N2 = dist.Field(bases=basis, name='N2')
        N2['g'] = N2_extra(r) * zero_to_one(r, 1.02, width=0.05)
        N2.change_scales(basis.dealias)
        N2 = d3.Grid(N2).evaluate()
#        plt.plot(r_de.ravel(),N2['g'].ravel())
#        plt.plot(r_de.ravel(), N2_func(r_de.ravel()))
##        plt.plot(r_de.ravel(),N2_extra(r_de.ravel()) -  N2['g'].ravel())
#        plt.yscale('log')
#        plt.show()

        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r

        namespace['g_{}'.format(k)] = g   = dist.VectorField(coords, name='g', bases=basis)
        namespace['g_ncc_{}'.format(k)] = g_ncc   = dist.VectorField(coords, name='g', bases=basis.radial_basis)
        g.change_scales(basis.dealias)
        g['g'][2] = g_func(r_de)
        g_ncc['g'][2] = g_func(r)

        namespace['T_{}'.format(k)] = T = dist.Field(name='T', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['tau_T_{}'.format(k)] = tau_T = dist.Field(name='tau_T', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_T', bases=S2_basis)

        namespace['ln_T_{}'.format(k)] = ln_T = np.log(T)
#        namespace['T_{}'.format(k)] = T = np.exp(ln_T)
        namespace['rho_{}'.format(k)] = rho = np.exp(ln_rho)
        namespace['dS_dr_{}'.format(k)] = dS_dr = Cp * ((1/gamma) * d3.grad(ln_T) - ((gamma-1)/gamma)*d3.grad(ln_rho))
        namespace['N2_op_{}'.format(k)] = -g@dS_dr/Cp

        if T_func is None:
            T['g']   = -(r**2 - radius**2) + 1
        else:
            T['g'] = T_func(r)
#        if T_func is None:
#            ln_T['g']   = np.log(-(r**2 - radius**2) + 1)
#        else:
#            ln_T['g'] = np.log(T_func(r))
        if ln_rho_func is None:
            ln_rho['g'] = np.log(-(r**2 - radius**2) + 1)
        else:
            ln_rho['g'] = ln_rho_func(r)

    namespace['pi'] = pi = np.pi
    locals().update(namespace)

    variables = []
    for k, basis in bases.items():
        variables += [namespace['T_{}'.format(k)], namespace['ln_rho_{}'.format(k)]]
#        variables += [namespace['ln_T_{}'.format(k)], namespace['ln_rho_{}'.format(k)]]
    for k, basis in bases.items():
        variables += [namespace['tau_T_{}'.format(k)], namespace['tau_rho_{}'.format(k)]]


    problem = d3.NLBVP(variables, namespace=locals())

    for k, basis in bases.items():
        problem.add_equation("- R*(grad(T_{0})) + r_vec_{0}*lift(tau_T_{0}) = + R*T_{0}*grad(ln_rho_{0}) - g_{0}".format(k))
        problem.add_equation("r_vec_{0}@grad(ln_rho_{0}) - lift(tau_rho_{0}) = r_vec_{0}@grad(ln_rho_{0}) + g_{0}@dS_dr_{0}/Cp + N2_{0}".format(k))
#        problem.add_equation("- R*(grad(ln_T_{0}) + grad(ln_rho_{0})) + r_vec_{0}*lift(tau_T_{0}) = - g_{0}/T_{0}".format(k))
#        problem.add_equation("((gamma-1)/gamma)*g_ncc_{0}@grad(ln_rho_{0}) - lift(tau_rho_{0}) = (1/gamma)*g_{0}@grad(ln_T_{0}) + N2_{0}".format(k))
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("T_{0}(r={2}) - T_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
        iter += 1
    problem.add_equation("T_B(r=nondim_radius) = 1")
    problem.add_equation("ln_rho_B(r=nondim_radius) = 0")


    ncc_cutoff=1e-10
    tolerance=1e-10
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
#        plt.plot(r_de.ravel(), np.abs(namespace['N2_op_B'].evaluate()['g'].ravel()))
#        plt.yscale('log')
#        plt.show()

    rs = []
    gs = []
    N2s = []
    HSEs = []
    ln_Ts = []
    Ts = []
    ln_rhos = []
    dS_drs = []

    for k, basis in bases.items():
        ln_T = namespace['ln_T_{}'.format(k)].evaluate()
        T = namespace['T_{}'.format(k)].evaluate()
        ln_rho = namespace['ln_rho_{}'.format(k)]
        dS_dr = namespace['dS_dr_{}'.format(k)].evaluate()
        g = namespace['g_{}'.format(k)]
        N2_op = namespace['N2_op_{}'.format(k)]
        N2 = N2_op.evaluate()

        grad_ln_T = d3.grad(ln_T).evaluate()
        grad_ln_rho = d3.grad(ln_rho).evaluate()
        HSE = (-R*(d3.grad(ln_T) + d3.grad(ln_rho)) + g/T).evaluate()

        phi, theta, r = dist.local_grids(basis, scales=(1,1,scales))
        dS_dr.change_scales((1,1,scales))
        ln_T.change_scales((1,1,scales))
        T.change_scales((1,1,scales))
        ln_rho.change_scales((1,1,scales))
        N2.change_scales((1,1,scales))
        HSE.change_scales((1,1,scales))
        g.change_scales((1,1,scales))

        rs.append(r)
        gs.append(g['g'])
        N2s.append(N2['g'])
        ln_Ts.append(ln_T['g'])
        Ts.append(T['g'])
        ln_rhos.append(ln_rho['g'])
        dS_drs.append(dS_dr['g'])
        HSEs.append(HSE['g'])

    r = np.concatenate(rs, axis=-1)
    g = np.concatenate(gs, axis=-1)
    N2 = np.concatenate(N2s, axis=-1)
    HSE = np.concatenate(HSEs, axis=-1)
    ln_T = np.concatenate(ln_Ts, axis=-1)
    T = np.concatenate(Ts, axis=-1)
    ln_rho = np.concatenate(ln_rhos, axis=-1)
    dS_dr = np.concatenate(dS_drs, axis=-1)

    fig = plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2)
    ax3 = fig.add_subplot(4,1,3)
    ax4 = fig.add_subplot(4,1,4)
    ax1.plot(r.ravel(), ln_T.ravel(), label='ln_T')
    ax1.plot(r.ravel(), ln_rho.ravel(), label='ln_rho')
    ax1.legend()
    ax2.plot(r.ravel(), HSE[2,:].ravel(), label='HSE')
    ax2.legend()
    ax3.plot(r.ravel(), g[2,:].ravel(), label=r'$g$')
    ax3.legend()
    ax4.plot(r.ravel(), N2.ravel(), label=r'$N^2$')
    ax4.plot(r.ravel(), -N2.ravel(), label=r'$-N^2$')
    ax4.plot(r.ravel(), (N2_func(r)).ravel(), label=r'$N^2$ goal', ls='--')
    ax4.set_yscale('log')
    yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
    ax4.legend()
    fig.savefig('stratification.png', bbox_inches='tight', dpi=300)
#    plt.show()


    atmosphere = dict()
    atmosphere['r'] = r
    atmosphere['ln_T'] = ln_T
    atmosphere['T'] = T
    atmosphere['N2'] = N2
    atmosphere['ln_rho'] = ln_rho
    atmosphere['dS_dr'] = dS_dr
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
    print(ncc_dict.keys())

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
    N2             = p.brunt_N2[::-1] / u.s**2
    N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
    N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
    eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
    lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r

    R_star = (p.photosphere_r * u.R_sun).cgs
    
    #Put all MESA fields into cgs and calculate secondary MESA fields
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
                r_bounds[i] = float(rb.replace('L', ''))*core_cz_radius
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
    tau_nd  = (1/f_brunt).cgs #timescale of max N^2
    rho_nd  = m_nd/L_nd**3
    u_nd    = L_nd/tau_nd
    s_nd    = L_nd**2 / tau_nd**2 / T_nd
    H_nd    = (m_nd / L_nd) * tau_nd**-3
    s_motions    = L_nd**2 / tau_heat**2 / T[0]
    nondim_cp = (cp[r==L_nd][0]/s_nd).value
    nondim_gamma1 = (gamma1[r==L_nd][0]).value
    nondim_R_gas = nondim_cp * (nondim_gamma1 - 1) / nondim_gamma1
    nondim_G = (constants.G * (rho_nd * tau_nd**2)).value
    g               = constants.G.cgs*mass/r**2
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
    #Build a nice function for our basis in the ball
    grad_s_transition_point = 1.05
    logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
    grad_s_width = 0.025
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








    
    # Calculate internal heating function
    # Goal: H_eff= np.gradient(L_conv,r, edge_order=1)/(4*np.pi*r**2) # Heating, for ncc, H = rho*eps - portion carried by radiation
    # (1/4pir^2) dL_conv/dr = rho * eps + (1/r^2)d/dr (r^2 k_rad dT/dr) -> chain rule
#    eo=2
#    H_eff = (1/(4*np.pi*r**2))*np.gradient(Luminosity, r, edge_order=eo) + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
#    H_eff_secondary = rho*eps_nuc + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
#    H_eff[:2] = H_eff_secondary[:2]
#    
#    sim_H_eff = np.copy(H_eff)
#    L_conv_sim = np.zeros_like(L_conv)
#    L_eps = np.zeros_like(Luminosity)
#    for i in range(L_conv_sim.shape[0]):
#        L_conv_sim[i] = np.trapz((4*np.pi*r**2*sim_H_eff)[:1+i], r[:1+i])
#        L_eps[i] = np.trapz((4*np.pi*r**2*rho*eps_nuc)[:i+1], r[:i+1])
#    L_excess = L_conv_sim[-5] - Luminosity[-5]
    
    if config.star['smooth_h']:
        #smooth CZ-RZ transition
        L_conv_sim = np.copy(L_conv)
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.15*core_cz_radius)
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)

        r_vals = []
        sim_H_eff = []
        sim_lum = []
        for i, bn in enumerate(bases.keys()):
            field = d.Field(bases=bases[bn])
            field.change_scales((1,1,0.5))
            phi, theta, rv = bases[bn].global_grids((1, 1, 0.5))
            field['g'] = interp1d(r/L_nd, L_conv_sim, **interp_kwargs)(rv)
            field.change_scales(bases[bn].dealias)
            sim_lum.append(field['g'][0,0,:]/(H_nd*L_nd**3))
            r_vals.append(dedalus_r[bn].ravel())
            sim_H_eff.append((1/L_nd)**3*(1/(4*np.pi*dedalus_r[bn].ravel()**2)) * d3.grad(field).evaluate()['g'][2,0,0,:])
        sim_H_eff = np.concatenate(sim_H_eff)
        r_vals = np.concatenate(r_vals)
        sim_lum = np.concatenate(sim_lum)

#        #TODO: do a dedalus numerical derivative here.    
#        transition_region = (r > 0.5*core_cz_radius)
#        sim_H_eff[transition_region] = ((1/(4*np.pi*r**2))*np.gradient(L_conv_sim, r, edge_order=eo))[transition_region]
    else:
        raise NotImplementedError("must use smooth_h")
#        sim_H_eff = H_eff


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
#    interpolations['grad_T'] = interp1d(r_nd, (L_nd/T_nd)*dTdr, **interp_kwargs)
    if config.star['smooth_h']:
        interpolations['H'] = interp1d(r_vals, ( sim_H_eff / H_nd), **interp_kwargs)
    else:
        interpolations['H'] = interp1d(r_nd, ( sim_H_eff / H_nd), **interp_kwargs)
#    interpolations['H'] = interp1d(r_nd, ( sim_H_eff/(rho) ) * (rho_nd/H_nd))
#    interpolations['grad_s0'] = interp1d(r_nd, L_nd * grad_s_smooth / s_nd, **interp_kwargs)
    interpolations['nu_diff'] = interp1d(r_nd, sim_nu_diff, **interp_kwargs)
    interpolations['chi_rad'] = interp1d(r_nd, sim_rad_diff, **interp_kwargs)
    interpolations['grad_chi_rad'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd), **interp_kwargs)
    interpolations['g'] = interp1d(r_nd, -g * (tau_nd**2/L_nd), **interp_kwargs)
    interpolations['g_phi'] = interp1d(r_nd, g_phi * (tau_nd**2 / L_nd**2), **interp_kwargs)
    interpolations['pomega_tilde'] = interp1d(r_nd, pomega_tilde * (tau_nd**2 / L_nd**2), **interp_kwargs)


    grad_s_func = interp1d(r_nd, grad_s_smooth * L_nd/s_nd, **interp_kwargs)
    g_func = interpolations['g']
    atmo = HSE_solve(c, d, bases, grad_s_func, g_func,
              r_outer=r_bound_nd[-1], r_stitch=stitch_radii, dtype=np.float64, \
              R=nondim_R_gas, gamma=nondim_gamma1, Cp=nondim_cp, comm=MPI.COMM_SELF, \
              nondim_radius=1, g_nondim=interpolations['g'](1), G=nondim_G,
              T_func = interpolations['T0'], ln_rho_func = interpolations['ln_rho0'])


    interpolations['grad_s0'] = interp1d(atmo['r'].ravel(), atmo['dS_dr'][2,:].ravel(), **interp_kwargs)
    interpolations['T0'] = interp1d(atmo['r'].ravel(), atmo['T'].ravel(), **interp_kwargs)
    interpolations['ln_rho0'] = interp1d(atmo['r'].ravel(), atmo['ln_rho'].ravel(), **interp_kwargs)

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
                    grad_field = d3.grad(ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                    grad_field.change_scales((1,1,(Nmax+1)/resolutions[bases_keys == bn][2]))
                    grad_field.change_scales(basis.dealias)
                    ncc_dict[name]['field_{}'.format(bn)] = grad_field
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                    print(name, np.sum(np.abs(ncc_dict[name]['field_{}'.format(bn)]['c']) > 1e-10))

        if 'neg_g' in ncc_dict.keys():
            if 'g' not in ncc_dict.keys():
                ncc_dict['g'] = OrderedDict()
            name = 'g'
            ncc_dict['g']['field_{}'.format(bn)] = (-ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
            ncc_dict['g']['vector'] = True
            ncc_dict['g']['interp_func'] = interpolations['g']
            ncc_dict['g']['Nmax_{}'.format(bn)] = ncc_dict['neg_g']['Nmax_{}'.format(bn)]
            ncc_dict['g']['from_grad'] = True 
        
    
        #Evaluate for grad chi rad
        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['g'] = d3.grad(ncc_dict['chi_rad']['field_{}'.format(bn)]).evaluate()['g']
    
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
    
            if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0']:
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
                print(rvals, dedalus_yvals)
            elif ncc in ['ln_T', 'ln_rho']:
                interp_func = interpolations[ncc]
    
            if ncc == 'grad_T':
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
    grad_s0s = []
    rs = []
    for bn in bases_keys:
        rs.append(dedalus_r[bn].ravel())
        T0 = ncc_dict['T0']['field_{}'.format(bn)]
        grad_T0 = ncc_dict['grad_T0']['field_{}'.format(bn)]
        grad_ln_rho0 = ncc_dict['grad_ln_rho0']['field_{}'.format(bn)]
        gvec = ncc_dict['g']['field_{}'.format(bn)]
        grad_s0 = nondim_cp * ((1/nondim_gamma1) * (grad_T0['g']/T0['g'] - (nondim_gamma1-1) * grad_ln_rho0['g']))
        N2_val = -gvec['g'][2,:] * grad_s0[2,:] / nondim_cp 
        N2s.append(N2_val)
#        N2s.append((-1*d3.dot(grad_s0, ncc_dict['g']['field_{}'.format(bn)])/nondim_cp).evaluate()['g'])
#        grad_s0s.append(grad_s0.evaluate()['g'][2,:])
    r_dedalus = np.concatenate(rs, axis=-1)
#    grad_s0_dedalus = np.concatenate(grad_s0s, axis=-1).ravel()
    N2_dedalus = np.concatenate(N2s, axis=-1).ravel()
#    plt.plot(r_nd, tau_nd**2*g*grad_s_over_cp, label='mesa')
#    plt.plot(r_dedalus, grad_s0_dedalus, ls='--', label='dedalus')
    plt.plot(atmo['r'].ravel(), atmo['N2'].ravel(), label='atmosphere')
    plt.plot(r_dedalus, N2_dedalus, ls='--', label='dedalus')
    plt.ylabel(r'$N^2$')
    plt.xlabel('r')
    plt.yscale('log')
    plt.savefig('star/N2_goodness.png')

        

    C = d3.integ(ncc_dict['H']['field_B']).evaluate()['g']
    vol = (4/3)*np.pi * bases['B'].radius**3
    adj = C / vol
    logger.info('adjusting dLdt for energy conservation; subtracting {} from H_B'.format(adj))
    ncc_dict['H']['field_B']['g'] -= adj 

    dLdt = d3.integ(4*np.pi*ncc_dict['H']['field_B']).evaluate()['g']
    print(dLdt)
    
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

        f['lum_r_vals'] = r_vals
        f['sim_lum'] = sim_lum
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

