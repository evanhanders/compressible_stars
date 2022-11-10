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
from .compressible_functions import make_bases
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

def HSE_solve(coords, dist, bases, grad_ln_rho_func, N2_func, Fconv_func=None, Q_func=None, r_stitch=[], r_outer=1, dtype=np.float64, \
              R=1, gamma=5/3, comm=MPI.COMM_SELF, nondim_radius=1, g_nondim=1, s_motions=1):

    Cp = R*gamma/(gamma-1)
    Cv = Cp/gamma

    # Parameters
    scales = bases[list(bases.keys())[0]].dealias[-1]
    namespace = dict()
    namespace['R'] = R
    namespace['Cp'] = Cp
    namespace['gamma'] = gamma
    namespace['log'] = np.log
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        namespace['g_phi_{}'.format(k)] = g_phi = dist.Field(name='g_phi', bases=basis)
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['g_{}'.format(k)] = g = dist.VectorField(basis.coordsystem, name='g', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_g_phi_{}'.format(k)] = tau_g_phi = dist.Field(name='tau_g_phi', bases=S2_basis)
        namespace['tau_Fconv_{}'.format(k)] = tau_Fconv = dist.Field(name='tau_Fconv', bases=S2_basis)

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


        Q_var = Fconv_var = False
        namespace['Q_{}'.format(k)] = Q = dist.Field(name='Q', bases=basis)
        namespace['Fconv_{}'.format(k)] = Fconv   = dist.VectorField(coords, name='Fconv', bases=basis)
        if Fconv_func is not None:
            Fconv['g'][2] = Fconv_func(r)
            Q_var = True
        elif Q_func is not None:
            Q['g'] = Q_func(r)
            Fconv_var = True

        namespace['ln_T_LHS_{}'.format(k)] = ln_T_LHS = gamma*(s/Cp + ((gamma-1)/gamma)*ln_rho*ones)
        namespace['ln_pomega_{}'.format(k)] = ln_pomega = ln_T_LHS + np.log(R)
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
        namespace['s0_{}'.format(k)] = s0 = Cp * ((1/gamma)*(ln_pomega + ln_rho) - ln_rho) #s with an offset so s0 = cp * (1/gamma * lnP - ln_rho)


    namespace['pi'] = pi = np.pi
    locals().update(namespace)
    ncc_cutoff=1e-12
    tolerance=1e-7
    HSE_tolerance = 1e-1

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
        if k not in ['B', 'S0']:
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
        iter += 1
    if 'B' in bases.keys():
        problem.add_equation("ln_rho_B(r=nondim_radius) = 0")
    else:
        for k, basis in bases.items():
            if basis.radii[0] <= nondim_radius and basis.radii[1] >= nondim_radius:
                problem.add_equation("ln_rho_{}(r=nondim_radius) = 0".format(k))


    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')

    logger.info('ln_rho found')
    print('ln_rho', ln_rho['g'])

    #solve for everything else.
    variables = []
    for k, basis in bases.items():
        if Q_var:
            variables += [namespace['s_{}'.format(k)], namespace['g_{}'.format(k)], namespace['Q_{}'.format(k)], namespace['g_phi_{}'.format(k)]]
        elif Fconv_var:
            variables += [namespace['s_{}'.format(k)], namespace['g_{}'.format(k)], namespace['Fconv_{}'.format(k)], namespace['g_phi_{}'.format(k)]]
    for k, basis in bases.items():
        variables += [namespace['tau_s_{}'.format(k)], namespace['tau_g_phi_{}'.format(k)]]
        if k == 'S0' and Fconv_var:
            logger.info('using fconv var')
            variables += [namespace['tau_Fconv_{}'.format(k)]]


    problem = d3.NLBVP(variables, namespace=locals())

    for k, basis in bases.items():
        #initial condition
        namespace['s_{}'.format(k)].change_scales(basis.dealias)
        namespace['s_{}'.format(k)]['g'] = -(R*namespace['ln_rho_{}'.format(k)]).evaluate()['g']
        problem.add_equation("grad(ln_rho_{0})@(grad(s_{0})/Cp) + lift_{0}(tau_s_{0}) = -N2_{0}/(gamma*pomega_{0}) - grad(s_{0})@grad(s_{0}) / Cp**2".format(k))
        problem.add_equation("g_{0} = g_op_{0} ".format(k))
        if Q_var:
            problem.add_equation("Q_{0} = div(Fconv_{0})".format(k))
        elif Fconv_var:
            if k == 'S0':
                logger.info('lifting in fconv eqn')
                problem.add_equation("div(Fconv_{0}) + lift_{0}(tau_Fconv_{0}) = Q_{0}".format(k))
            else:
                problem.add_equation("div(Fconv_{0}) = Q_{0}".format(k))
        problem.add_equation("grad(g_phi_{0}) + g_{0} + r_vec_{0}*lift_{0}(tau_g_phi_{0}) = 0".format(k))
    iter = 0
    for k, basis in bases.items():
        if k not in ['B', 'S0']:
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            logger.info('adding {}'.format("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s)))
            problem.add_equation("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s))
            logger.info('adding {}'.format("g_phi_{0}(r={2}) - g_phi_{1}(r={2}) = 0".format(k, k_old, r_s)))
            problem.add_equation("g_phi_{0}(r={2}) - g_phi_{1}(r={2}) = 0".format(k, k_old, r_s))
        if k == 'S0':
            logger.info("adding {}".format("radial(Fconv_{0}(r={1})) = 0".format(k, basis.radii[0])))
            problem.add_equation("radial(Fconv_{0}(r={1})) = 0".format(k, basis.radii[0]))
        iter += 1
        if iter == len(bases.items()):
            logger.info('adding {}'.format("g_phi_{0}(r=r_outer) = 0".format(k)))
            problem.add_equation("g_phi_{0}(r=r_outer) = 0".format(k))
    if 'B' in bases.keys():
        problem.add_equation("ln_T_LHS_B(r=nondim_radius) = 0")
    else:
        for k, basis in bases.items():
            if basis.radii[0] <= nondim_radius and basis.radii[1] >= nondim_radius:
                logger.info('adding {}'.format("ln_T_LHS_{}(r=nondim_radius) = 0".format(k)))
                problem.add_equation("ln_T_LHS_{}(r=nondim_radius) = 0".format(k))


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

    #Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'grad_s', 'g', 'pomega', 'rho', 'ln_rho', 'g_phi', 'r_vec', 'HSE', 'N2_op', 'Q', 's0', 'Fconv']
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
    Fconv = stitch_fields['Fconv'][2,:].ravel()
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
#    yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
#    ax7.set_yticks(yticks)
#    ax7.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
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
    if np.max(N2_func(r)) == 0:
        atmosphere['grad_T0_superad'] = lambda r: 0*r
    else:
        atmosphere['grad_T0_superad'] = lambda r: atmosphere['grad_pomega'](r)/R - atmosphere['g'](r)/Cp
    atmosphere['pomega'] = interp1d(r, pom, **interp_kwargs)
    atmosphere['rho'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['g_phi'] = interp1d(r, g_phi, **interp_kwargs)
    atmosphere['N2'] = interp1d(r, N2, **interp_kwargs)
    atmosphere['Q'] = interp1d(r, Q, **interp_kwargs)
    atmosphere['Fconv'] = interp1d(r, Fconv, **interp_kwargs)
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

    first = True
    for r, y in zip(rvals, dedalus_vals):
        mesa_y = mesa_func(r)
        if first:
            ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
            ax1.plot(r, y, label='dedalus', c='red')
            first = False
        else:
            ax1.plot(r, mesa_y, c='k', lw=3)
            ax1.plot(r, y, c='red')

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


class DedalusMesaReader:

    def __init__(self):
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
        self.p = p = mr.MesaData(mesa_file_path)
        self.mass           = (p.mass[::-1] * u.M_sun).cgs
        self.r              = (p.radius[::-1] * u.R_sun).cgs
        self.rho            = 10**p.logRho[::-1] * u.g / u.cm**3
        self.P              = p.pressure[::-1] * u.g / u.cm / u.s**2
        self.T              = p.temperature[::-1] * u.K
        self.nablaT         = p.gradT[::-1] #dlnT/dlnP
        self.nablaT_ad      = p.grada[::-1]
        self.chiRho         = p.chiRho[::-1]
        self.chiT           = p.chiT[::-1]
        self.cp             = p.cp[::-1]  * u.erg / u.K / u.g
        self.opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
        self.Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
        self.conv_L_div_L   = p.lum_conv_div_L[::-1]
        self.csound         = p.csound[::-1] * u.cm / u.s
        self.N2 = self.N2_mesa   = p.brunt_N2[::-1] / u.s**2
        self.N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
        self.N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
        self.eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
        self.mu             = p.mu[::-1] * u.g / u.mol 
        self.lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * self.csound/self.r


        self.R_star = (p.photosphere_r * u.R_sun).cgs
        
        #Put all MESA fields into cgs and calculate secondary MESA fields
        self.R_gas           = constants.R.cgs / self.mu[0]
        self.g               = constants.G.cgs * self.mass / self.r**2
        self.dlogPdr         = -self.rho*self.g/self.P
        self.gamma1          = self.dlogPdr/(-self.g/self.csound**2)
        self.dlogrhodr       = self.dlogPdr*(self.chiT/self.chiRho)*(self.nablaT_ad - self.nablaT) - self.g/self.csound**2
        self.dlogTdr         = self.dlogPdr*(self.nablaT)
        self.grad_s_over_cp  = self.N2/self.g #entropy gradient, for NCC, includes composition terms
        self.grad_s          = self.cp * self.grad_s_over_cp
        self.L_conv          = self.conv_L_div_L*self.Luminosity
        self.dTdr            = self.T*self.dlogTdr


        # Calculate k_rad and radiative diffusivity using luminosities and smooth things.
        self.k_rad = rad_cond = -(self.Luminosity - self.L_conv)/(4*np.pi*self.r**2*self.dTdr)
        self.rad_diff        = self.k_rad / (self.rho * self.cp)
        #rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs # this is less smooth

        self.g_over_cp       = self.g / self.cp
        self.g_phi           = np.cumsum(self.g*np.gradient(self.r))  #gvec = -grad phi; 
        self.s_over_cp       = np.cumsum(self.grad_s_over_cp*np.gradient(self.r))

    def customize_star(self):
        pass

    def plot_star(self):
        pass

    def save_star(self):
        pass

class MassiveStarBuilder(DedalusMesaReader):

    def __init__(self, plot_nccs=False):
        """ Create nondimensionalization and Dedalus domain / bases. """
        super().__init__()
        self.plot_nccs = plot_nccs
        self.out_dir, self.out_file = name_star()
        self.ncc_dict = config.nccs.copy()

        # Find edge of core cz
        self.mesa_core_bool = (self.L_conv.value > 1)*(self.mass < 0.9*self.mass[-1]) #rudimentary but works
        self.mesa_core_bound_ind  = np.argmin(np.abs(self.mass - self.mass[self.mesa_core_bool][-1]))
        self.mesa_core_radius = self.r[self.mesa_core_bound_ind]

        # User specifics to only do core CZ or the fraction of total star to simulate
        self.cz_only = config.star['cz_only']
        if self.cz_only:
            self.mesa_basis_bounds = [0, self.mesa_core_radius]
        else:
            self.mesa_basis_bounds = list(config.star['r_bounds'])
            for i, rb in enumerate(self.mesa_basis_bounds):
                if type(rb) == str:
                    if 'R' in rb:
                        self.mesa_basis_bounds[i] = float(rb.replace('R', ''))*self.R_star
                    elif 'L' in rb:
                        if rb == 'L':
                            self.mesa_basis_bounds[i] = self.mesa_core_radius
                            self.cz_only = True
                        else:
                            self.mesa_basis_bounds[i] = float(rb.replace('L', ''))*self.mesa_core_radius
                    else:
                        try:
                            self.mesa_basis_bounds[i] = float(self.mesa_basis_bounds[i]) * u.cm
                        except:
                            raise ValueError("index {} ('{}') of self.mesa_basis_bounds is poorly specified".format(i, rb))
                    self.mesa_basis_bounds[i] = self.mesa_core_radius*np.around(self.mesa_basis_bounds[i]/self.mesa_core_radius, decimals=2)
        logger.info('fraction of FULL star simulated: {:.2f}, up to r={:.3e}'.format(self.mesa_basis_bounds[-1]/self.R_star, self.mesa_basis_bounds[-1]))
        self.mesa_sim_bool      = (self.r > self.mesa_basis_bounds[0])*(self.r <= self.mesa_basis_bounds[-1])

        #Get N2 info
        self.mesa_N2_max_sim = self.N2[self.mesa_sim_bool].max()
        num_shell_points = np.sum(self.mesa_sim_bool*(self.r > self.mesa_core_radius))
        self.mesa_N2_plateau = np.median(self.N2[self.r > self.mesa_core_radius][int(num_shell_points*0.25):int(num_shell_points*0.75)])
     
        #Characteristic scales:
        self.L_CZ    = self.mesa_core_radius
        m_core  = self.rho[0] * self.L_CZ**3
        T_core  = self.T[0]
        self.H0      = (self.rho*self.eps_nuc)[0]
        self.tau_heat  = ((self.H0*self.L_CZ/m_core)**(-1/3)).cgs #heating timescale
        self.tau_cp = np.sqrt(self.L_CZ**2 / (self.cp[0] * T_core))
        max_f_brunt = np.sqrt(self.mesa_N2_max_sim)/(2*np.pi)

        #Fundamental Nondimensionalization -- length (L_nd), mass (m_nd), temp (T_nd), time (tau_nd)
        self.L_nd    = self.L_CZ
        self.m_nd    = self.rho[self.r==self.L_nd][0] * self.L_nd**3 #mass at core cz boundary
        self.T_nd    = self.T[self.r==self.L_nd][0] #temp at core cz boundary
        if self.cz_only:
            self.tau_nd = self.tau_cp
        else:
            self.tau_nd  = (1/max_f_brunt).cgs #timescale of max N^2
        logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(self.L_nd, self.T_nd, self.m_nd, self.tau_nd))

        #Extra useful nondimensionalized quantities
        self.rho_nd  = self.m_nd/self.L_nd**3
        self.u_nd    = self.L_nd/self.tau_nd
        self.s_nd    = self.L_nd**2 / self.tau_nd**2 / self.T_nd
        self.H_nd    = (self.m_nd / self.L_nd) * self.tau_nd**-3
        self.lum_nd  = self.L_nd**2 * self.m_nd / (self.tau_nd**2) / self.tau_nd
        self.s_motions    = self.L_nd**2 / self.tau_heat**2 / self.T[0]
        self.R_gas_nd = (self.R_gas / self.s_nd).cgs.value
        self.gamma1_nd = (self.gamma1[0]).value
        self.cp_nd = self.R_gas_nd * self.gamma1_nd / (self.gamma1_nd - 1)
        self.Ma2_r0 = ((self.u_nd*(self.tau_nd/self.tau_heat))**2 / ((self.gamma1[0]-1)*self.cp[0]*self.T[0])).cgs
        logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(self.cp_nd, self.R_gas_nd, self.gamma1_nd))
        logger.info('m_nd/M_\odot: {:.3f}'.format((self.m_nd/constants.M_sun).cgs))
        logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(self.Ma2_r0), self.tau_heat))

       
        #MESA radial values at simulation joints & across full star in simulation units
        self.mesa_r_nd = (self.r/self.L_nd).cgs
        self.nd_basis_bounds = [(rb/self.L_nd).value for rb in self.mesa_basis_bounds]
        self.r_inner = self.nd_basis_bounds[0]
        self.r_outer = self.nd_basis_bounds[-1]
      
        ### Make dedalus domain and bases
        self.resolutions = [(1, 1, nr) for nr in config.star['nr']]
        self.stitch_radii = self.nd_basis_bounds[1:-1]
        self.dtype=np.float64
        mesh=None
        dealias = config.numerics['N_dealias']
        self.coords, self.dist, self.bases, self.bases_keys = make_bases(self.resolutions, self.stitch_radii, self.r_outer, dealias=(1,1,dealias), dtype=self.dtype, mesh=mesh)
        self.dedalus_r = OrderedDict()
        for bn in self.bases.keys():
            phi, theta, r_vals = self.bases[bn].global_grids((1, 1, dealias))
            self.dedalus_r[bn] = r_vals


    def customize_star(self):

        #Adjust gravitational potential so that it doesn't cross zero somewhere wonky
        self.g_phi           -= self.g_phi[-1] - self.u_nd**2 #set g_phi = -1 at r = R_star

        #Construct simulation diffusivity profiles -- MANY CHOICES COULD BE MADE HERE!!
        self.mesa_rad_diff_nd = self.rad_diff * (self.tau_nd / self.L_nd**2)
        self.rad_diff_cutoff_nd = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((self.L_CZ**2/self.tau_heat) / (self.L_nd**2/self.tau_nd))
        self.simulation_rad_diff_nd = np.copy(self.mesa_rad_diff_nd) + self.rad_diff_cutoff_nd
        self.simulation_visc_diff_nd = config.numerics['prandtl']*self.rad_diff_cutoff_nd*np.ones_like(self.simulation_rad_diff_nd)
        logger.info('rad_diff cutoff: {:.3e}'.format(self.rad_diff_cutoff_nd))
        
        ### entropy gradient
        if self.cz_only:
            #Entropy gradient is zero everywhere
            grad_s_smooth = np.zeros_like(self.grad_s)
            self.N2_func = interp1d(self.mesa_r_nd, np.zeros_like(grad_s_smooth), **interp_kwargs)
        else:
            #Build a smooth function in the ball, match the star in the shells.
            grad_s_width = 0.05
            grad_s_transition_point = self.nd_basis_bounds[1] - grad_s_width
            logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
            logger.info('using default grad s width = {}'.format(grad_s_width))
            grad_s_center =  grad_s_transition_point - 0.5*grad_s_width
            grad_s_width *= (self.L_CZ/self.L_nd).value
            grad_s_center *= (self.L_CZ/self.L_nd).value
           
            grad_s_smooth = np.copy(self.grad_s)
            flat_value  = np.interp(grad_s_transition_point, self.mesa_r_nd, self.grad_s)
            grad_s_smooth += (self.mesa_r_nd)**2 *  flat_value
            grad_s_smooth *= zero_to_one(self.mesa_r_nd, grad_s_transition_point, width=grad_s_width)

            #construct N2 function #TODO: blend logic here & in BVP?
            smooth_N2 = np.copy(self.N2_mesa)
            stitch_value = np.interp(self.bases['B'].radius, r/self.L_nd, self.N2_mesa)
            smooth_N2[r/self.L_nd < self.bases['B'].radius] = (r[r/self.L_nd < self.bases['B'].radius]/self.L_nd / self.bases['B'].radius)**2 * stitch_value
            smooth_N2 *= zero_to_one(r/self.L_nd, grad_s_transition_point, width=grad_s_width)
            self.N2_func = interp1d(self.mesa_r_nd, self.tau_nd**2 * smooth_N2, **interp_kwargs)

        ### Internal heating / cooling function
        interp_r = np.linspace(0, 1, 1000)
        if config.star['smooth_h']:
            #smooth CZ-RZ transition
            argmax = np.argmax(self.L_conv/self.lum_nd)
            max_r = self.r[argmax]/self.L_nd
            max_L = (self.L_conv[argmax]/self.lum_nd).cgs

            #Heating layer
            Q_base = lambda r : one_to_zero(r, max_r*0.8, width=max_r*0.2)
            Q = Q_base(self.mesa_r_nd)
            cumsum = np.cumsum(Q*np.gradient(self.mesa_r_nd) * 4*np.pi*((self.mesa_r_nd)**2))
            first_adjust = np.copy(max_L / cumsum[-1])
            Q_func_heat = lambda r: first_adjust * Q_base(r)
            heat_lum = np.trapz(4*np.pi*interp_r**2 * interp1d(self.mesa_r_nd, Q_func_heat(self.mesa_r_nd), **interp_kwargs)(interp_r), x=interp_r)

            #Cooling layer
            Qcool_base = lambda r: -zero_to_one(r, 0.85, width=0.07)
            cool_lum = np.trapz(4*np.pi*interp_r**2 * interp1d(self.mesa_r_nd, Qcool_base(self.mesa_r_nd), **interp_kwargs)(interp_r), x=interp_r)
            adjustment = np.abs(heat_lum/cool_lum)
            Q_func_cool = lambda r: adjustment * Qcool_base(r)
            self.Q_func = lambda r: Q_func_heat(r) + Q_func_cool(r)

        elif config.star['heat_only']:
            #smooth CZ-RZ transition
            argmax = np.argmax(self.L_conv/self.lum_nd)
            max_r = self.r[argmax]/self.L_nd
            max_L = (self.L_conv[argmax]/self.lum_nd).cgs

            #Heating layer
            Q_base = lambda r : one_to_zero(r, max_r*0.8, width=max_r*0.2)
            Q = Q_base(self.mesa_r_nd)
            cumsum = np.cumsum(Q*np.gradient(self.mesa_r_nd) * 4*np.pi*((self.mesa_r_nd)**2))
            first_adjust = np.copy(max_L / cumsum[-1])
            self.Q_func = lambda r: first_adjust * Q_base(r)
        else:
            raise NotImplementedError("must use smooth_h or heat_only")

        # Create interpolations of the various fields that may be used in the problem
        self.mesa_interpolations = OrderedDict()
        self.mesa_interpolations['ln_rho0'] = interp1d(self.mesa_r_nd, np.log(self.rho/self.rho_nd), **interp_kwargs)
        self.mesa_interpolations['rho0'] = lambda r: np.exp(self.mesa_interpolations['ln_rho0'](r))
        self.mesa_interpolations['ln_T0'] = interp1d(self.mesa_r_nd, np.log(self.T/self.T_nd), **interp_kwargs)
        self.mesa_interpolations['Q'] = interp1d(self.mesa_r_nd, (1/(4*np.pi*self.mesa_r_nd**2))*np.gradient(self.L_conv/self.lum_nd, self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['grad_ln_rho0'] = interp1d(self.mesa_r_nd, self.dlogrhodr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['grad_ln_T0'] = interp1d(self.mesa_r_nd, self.dlogTdr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['grad_ln_pom0'] = interp1d(self.mesa_r_nd, self.dlogTdr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['T0'] = interp1d(self.mesa_r_nd, self.T/self.T_nd, **interp_kwargs)
        self.mesa_interpolations['pom0'] = interp1d(self.mesa_r_nd, self.R_gas_nd * self.T/self.T_nd, **interp_kwargs)
        self.mesa_interpolations['nu_diff'] = interp1d(self.mesa_r_nd, self.simulation_visc_diff_nd, **interp_kwargs)
        self.mesa_interpolations['chi_rad'] = interp1d(self.mesa_r_nd, self.simulation_rad_diff_nd, **interp_kwargs)
        self.mesa_interpolations['grad_chi_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.mesa_rad_diff_nd, self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['g'] = interp1d(self.mesa_r_nd, -self.g * (self.tau_nd**2/self.L_nd), **interp_kwargs)
        self.mesa_interpolations['g_phi'] = interp1d(self.mesa_r_nd, self.g_phi * (self.tau_nd**2 / self.L_nd**2), **interp_kwargs)
        self.mesa_interpolations['grad_s0'] = interp1d(self.mesa_r_nd, self.grad_s_over_cp*self.cp * (self.L_nd/self.s_nd), **interp_kwargs)
        self.mesa_interpolations['s0'] = interp1d(self.mesa_r_nd, self.s_over_cp*self.cp  / self.s_nd, **interp_kwargs)
        self.mesa_interpolations['kappa_rad'] = interp1d(self.mesa_r_nd, self.mesa_interpolations['rho0'](self.mesa_r_nd)*self.cp_nd*self.simulation_rad_diff_nd, **interp_kwargs)
        self.mesa_interpolations['grad_kappa_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.mesa_interpolations['kappa_rad'](self.mesa_r_nd), self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['grad_T0_superad'] = interp1d(self.mesa_r_nd, self.mesa_interpolations['grad_ln_T0'](self.mesa_r_nd)*self.mesa_interpolations['T0'](self.mesa_r_nd)  - \
                                                                               self.mesa_interpolations['g'](self.mesa_r_nd)/(self.cp / self.s_nd).cgs, **interp_kwargs)
        self.interpolations = self.mesa_interpolations.copy()

        #Solve hydrostatic equilibrium BVP for consistency with evolved equations.
        ln_rho_func = self.interpolations['ln_rho0']
        grad_ln_rho_func = self.interpolations['grad_ln_rho0']
        self.atmo = HSE_solve(self.coords, self.dist, self.bases,  grad_ln_rho_func, self.N2_func, Q_func=self.Q_func,
                  r_outer=self.r_outer, r_stitch=self.stitch_radii, dtype=self.dtype, \
                  R=self.R_gas_nd, gamma=self.gamma1_nd, comm=MPI.COMM_SELF, \
                  nondim_radius=1, g_nondim=self.interpolations['g'](1), s_motions=self.s_motions/self.s_nd)

        #Update self.interpolations of important quantities from HSE BVP
        self.F_conv_func = self.atmo['Fconv']
        self.interpolations['ln_rho0'] = self.atmo['ln_rho']
        self.interpolations['rho0'] = lambda r: np.exp(self.interpolations['ln_rho0'](r))
        self.interpolations['Q'] = self.Q_func
        self.interpolations['grad_s0'] = self.atmo['grad_s']
        self.interpolations['pom0'] = self.atmo['pomega']
        self.interpolations['grad_ln_pom0'] = self.atmo['grad_ln_pomega']
        self.interpolations['s0'] = self.atmo['s0']
        self.interpolations['g'] = self.atmo['g']
        self.interpolations['g_phi'] = self.atmo['g_phi']
        self.interpolations['grad_T0_superad'] = self.atmo['grad_T0_superad']
        self.interpolations['kappa_rad'] = interp1d(self.mesa_r_nd, self.interpolations['rho0'](self.mesa_r_nd)*self.cp_nd*self.simulation_rad_diff_nd, **interp_kwargs)
        self.interpolations['grad_kappa_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.interpolations['kappa_rad'](self.mesa_r_nd), self.mesa_r_nd), **interp_kwargs)

        #Prep NCCs for construction.
        for ncc in self.ncc_dict.keys():
            for i, bn in enumerate(self.bases.keys()):
                self.ncc_dict[ncc]['Nmax_{}'.format(bn)] = self.ncc_dict[ncc]['nr_max'][i]
                self.ncc_dict[ncc]['field_{}'.format(bn)] = None
            if ncc in self.interpolations.keys():
                self.ncc_dict[ncc]['interp_func'] = self.interpolations[ncc]
            else:
                self.ncc_dict[ncc]['interp_func'] = None

        #Construct NCCs
        for bn, basis in self.bases.items():
            rvals = self.dedalus_r[bn]
            for ncc in self.ncc_dict.keys():
                interp_func = self.ncc_dict[ncc]['interp_func']
                if interp_func is not None and not self.ncc_dict[ncc]['from_grad']:
                    Nmax = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]
                    vector = self.ncc_dict[ncc]['vector']
                    grid_only = self.ncc_dict[ncc]['grid_only']
                    self.ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    if self.ncc_dict[ncc]['get_grad']:
                        name = self.ncc_dict[ncc]['grad_name']
                        logger.info('getting {}'.format(name))
                        grad_field = d3.grad(self.ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                        grad_field.change_scales((1,1,(Nmax+1)/self.resolutions[self.bases_keys == bn][2]))
                        grad_field.change_scales(basis.dealias)
                        self.ncc_dict[name]['field_{}'.format(bn)] = grad_field
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                    if self.ncc_dict[ncc]['get_inverse']:
                        name = 'inv_{}'.format(ncc)
                        inv_func = lambda r: 1/interp_func(r)
                        self.ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax


            if 'neg_g' in self.ncc_dict.keys():
                if 'g' not in self.ncc_dict.keys():
                    self.ncc_dict['g'] = OrderedDict()
                name = 'g'
                self.ncc_dict['g']['field_{}'.format(bn)] = (-self.ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
                self.ncc_dict['g']['vector'] = True
                self.ncc_dict['g']['interp_func'] = self.interpolations['g']
                self.ncc_dict['g']['Nmax_{}'.format(bn)] = self.ncc_dict['neg_g']['Nmax_{}'.format(bn)]
                self.ncc_dict['g']['from_grad'] = True 
 
        #Adjust heating function so luminosity integrates to zero when appropriate.  
        if not config.star['heat_only']:
            integral = 0
            for bn in self.bases.keys():
                integral += d3.integ(self.ncc_dict['Q']['field_{}'.format(bn)])
            C = integral.evaluate()['g']
            vol = (4/3) * np.pi * (self.r_outer)**3
            adj = C / vol
            logger.info('adjusting dLdt for energy conservation; subtracting {} from H'.format(adj))
            for bn in self.bases.keys():
                self.ncc_dict['Q']['field_{}'.format(bn)]['g'] -= adj 

    def plot_star(self):
       
        #Make plots of the NCCs
        if self.plot_nccs:
            for ncc in self.ncc_dict.keys():
                if self.ncc_dict[ncc]['interp_func'] is None:
                    continue
                axhline = None
                log = False
                ylim = None
                rvals = []
                dedalus_yvals = []
                nvals = []
                for bn, basis in self.bases.items():
                    rvals.append(self.dedalus_r[bn].ravel())
                    nvals.append(self.ncc_dict[ncc]['Nmax_{}'.format(bn)])
                    if self.ncc_dict[ncc]['vector']:
                        dedalus_yvals.append(np.copy(self.ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:]))
                    else:
                        dedalus_yvals.append(np.copy(self.ncc_dict[ncc]['field_{}'.format(bn)]['g'][0,0,:]))
        
                interp_func = self.mesa_interpolations[ncc]
                if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad']:
                    log = True
                if ncc == 'grad_s0': 
                    axhline = (self.s_motions / self.s_nd)
                elif ncc in ['chi_rad', 'grad_chi_rad']:
                    if ncc == 'chi_rad':
                        interp_func = interp1d(self.mesa_r_nd, (self.L_nd**2/self.tau_nd).value*self.mesa_rad_diff_nd, **interp_kwargs)
                        for ind in range(len(dedalus_yvals)):
                            dedalus_yvals[ind] *= (self.L_nd**2/self.tau_nd).value
                    axhline = self.rad_diff_cutoff_nd*(self.L_nd**2/self.tau_nd).value
        
                if ncc == 'H':
                    interp_func = interp1d(r_vals, ( one_to_zero(r_vals, 1.5*self.nd_basis_bounds[1], width=0.05*self.nd_basis_bounds[1])*sim_H_eff ) * (1/self.H_nd), **interp_kwargs )
                elif ncc == 'grad_s0':
                    interp_func = interp1d(self.mesa_r_nd, (self.L_nd/self.s_nd) * self.grad_s, **interp_kwargs)
                elif ncc in ['ln_T0', 'ln_rho0', 'grad_s0']:
                    interp_func = self.interpolations[ncc]
        
                if ncc in ['grad_T', 'grad_kappa_rad']:
                    interp_func = lambda r: -self.ncc_dict[ncc]['interp_func'](r)
                    ylabel='-{}'.format(ncc)
                    for i in range(len(dedalus_yvals)):
                        dedalus_yvals[i] *= -1
                elif ncc == 'chi_rad':
                    ylabel = 'radiative diffusivity (cm^2/s)'
                else:
                    ylabel = ncc

        
                plot_ncc_figure(rvals, interp_func, dedalus_yvals, nvals, \
                            ylabel=ylabel, fig_name=ncc, out_dir=self.out_dir, log=log, ylim=ylim, \
                            r_int=self.stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

        plt.figure()
        N2s = []
        HSEs = []
        EOSs = []
        grad_s0s = []
        grad_ln_rho0s = []
        grad_ln_pom0s = []
        rs = []
        for bn in self.bases_keys:
            rs.append(self.dedalus_r[bn].ravel())
            grad_ln_rho0 = self.ncc_dict['grad_ln_rho0']['field_{}'.format(bn)]
            grad_ln_pom0 = self.ncc_dict['grad_ln_pom0']['field_{}'.format(bn)]
            pom0 = self.ncc_dict['pom0']['field_{}'.format(bn)]
            ln_rho0 = self.ncc_dict['ln_rho0']['field_{}'.format(bn)]
            gvec = self.ncc_dict['g']['field_{}'.format(bn)]
            grad_s0 = self.ncc_dict['grad_s0']['field_{}'.format(bn)]
            s0 = self.ncc_dict['s0']['field_{}'.format(bn)]
            pom0 = self.ncc_dict['pom0']['field_{}'.format(bn)]
            HSE = (self.gamma1_nd*pom0*(grad_ln_rho0 + grad_s0 / self.cp_nd) - gvec).evaluate()
            EOS = s0/self.cp_nd - ( (1/self.gamma1_nd) * (np.log(pom0) - np.log(self.R_gas_nd)) - ((self.gamma1_nd-1)/self.gamma1_nd) * ln_rho0 )
            N2_val = -gvec['g'][2,:] * grad_s0['g'][2,:] / self.cp_nd 
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
        plt.plot(self.mesa_r_nd, self.tau_nd**2*self.N2_mesa, label='mesa')
        plt.plot(self.mesa_r_nd, self.atmo['N2'](self.mesa_r_nd), label='atmosphere')
        plt.plot(r_dedalus, N2_dedalus, ls='--', label='dedalus')
        plt.legend()
        plt.ylabel(r'$N^2$')
        plt.xlabel('r')
        plt.yscale('log')
        plt.savefig('star/N2_goodness.png')
    #    plt.show()

        plt.figure()
        plt.axhline(self.s_motions/self.cp_nd / self.s_nd, c='k')
        plt.plot(r_dedalus, np.abs(HSE_dedalus))
        plt.yscale('log')
        plt.xlabel('r')
        plt.ylabel("HSE")
        plt.savefig('star/HSE_goodness.png')

        plt.figure()
        plt.axhline(self.s_motions/self.cp_nd / self.s_nd, c='k')
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

    def save_star(self):

        # Get some timestepping & wave frequency info
        f_nyq = 2*self.tau_nd*np.sqrt(self.mesa_N2_max_sim)/(2*np.pi)
        nyq_dt   = (1/f_nyq) 
        kepler_tau     = 30*60*u.s
        max_dt_kepler  = kepler_tau/self.tau_nd
        max_dt = max_dt_kepler
        logger.info('needed nyq_dt is {} s / {} % of a nondimensional time (Kepler 30 min is {} %) '.format(nyq_dt*self.tau_nd, nyq_dt*100, max_dt_kepler*100))
           
    #    dLdt = d3.integ(4*np.pi*self.ncc_dict['H']['field_B']).evaluate()['g']
        
        with h5py.File('{:s}'.format(self.out_file), 'w') as f:
            # Save output fields.
            # slicing preserves dimensionality
            for bn, basis in self.bases.items():
                f['r_{}'.format(bn)] = self.dedalus_r[bn]
                for ncc in self.ncc_dict.keys():
                    this_field = self.ncc_dict[ncc]['field_{}'.format(bn)]
                    if self.ncc_dict[ncc]['vector']:
                        f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                        f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
                    else:
                        f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
                        f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
        
            f['Cp'] = self.cp_nd
            f['R_gas'] = self.R_gas_nd
            f['gamma1'] = self.gamma1_nd

            #Save properties of the star, with units.
            f['L_nd']   = self.L_nd
            f['L_nd'].attrs['units'] = str(self.L_nd.unit)
            f['rho_nd']  = self.rho_nd
            f['rho_nd'].attrs['units']  = str(self.rho_nd.unit)
            f['T_nd']  = self.T_nd
            f['T_nd'].attrs['units']  = str(self.T_nd.unit)
            f['tau_heat'] = self.tau_heat
            f['tau_heat'].attrs['units'] = str(self.tau_heat.unit)
            f['tau_nd'] = self.tau_nd 
            f['tau_nd'].attrs['units'] = str(self.tau_nd.unit)
            f['m_nd'] = self.m_nd 
            f['m_nd'].attrs['units'] = str(self.m_nd.unit)
            f['s_nd'] = self.s_nd
            f['s_nd'].attrs['units'] = str(self.s_nd.unit)
            f['P_r0']  = self.P[0]
            f['P_r0'].attrs['units']  = str(self.P[0].unit)
            f['H_nd']  = self.H_nd
            f['H_nd'].attrs['units']  = str(self.H_nd.unit)
            f['H0']  = self.H0
            f['H0'].attrs['units']  = str(self.H0.unit)
            f['mesa_N2_max_sim'] = self.mesa_N2_max_sim
            f['mesa_N2_max_sim'].attrs['units'] = str(self.mesa_N2_max_sim.unit)
            f['mesa_N2_plateau'] = self.mesa_N2_plateau
            f['mesa_N2_plateau'].attrs['units'] = str(self.mesa_N2_plateau.unit)
            f['cp_surf'] = self.cp[self.mesa_sim_bool][-1]
            f['cp_surf'].attrs['units'] = str(self.cp[self.mesa_sim_bool][-1].unit)
            f['r_mesa'] = self.r
            f['r_mesa'].attrs['units'] = str(self.r.unit)
            f['N2_mesa'] = self.N2
            f['N2_mesa'].attrs['units'] = str(self.N2.unit)
            f['S1_mesa'] = self.lamb_freq(1)
            f['S1_mesa'].attrs['units'] = str(self.lamb_freq(1).unit)
            f['g_mesa'] = self.g 
            f['g_mesa'].attrs['units'] = str(self.g.unit)
            f['cp_mesa'] = self.cp
            f['cp_mesa'].attrs['units'] = str(self.cp.unit)

            #TODO: put sim lum back
            f['lum_r_vals'] = lum_r_vals = np.linspace(self.nd_basis_bounds[0], self.r_outer, 1000)
            f['sim_lum'] = (4*np.pi*lum_r_vals**2)*self.F_conv_func(lum_r_vals)
            f['r_stitch']   = self.stitch_radii
            f['r_outer']   = self.r_outer 
            f['max_dt'] = max_dt
            f['Ma2_r0'] = self.Ma2_r0
            for k in ['r_stitch', 'r_outer', 'max_dt', 'Ma2_r0', 'lum_r_vals', 'sim_lum',\
                        'Cp', 'R_gas', 'gamma1']:
                f[k].attrs['units'] = 'dimensionless'
        logger.info('finished saving NCCs to {}'.format(self.out_file))
        logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(self.out_dir))

class MdwarfBuilder(DedalusMesaReader):

    def __init__(self, plot_nccs=False):
        """ Create nondimensionalization and Dedalus domain / bases. """
        super().__init__()
        self.plot_nccs = plot_nccs
        self.out_dir, self.out_file = name_star()
        self.ncc_dict = config.nccs.copy()
        self.cz_only = config.star['cz_only']

        # Find edge of dedalus domain
        core_rho = self.rho[0]
        ln_rho = np.log(self.rho/core_rho) #runs from 0 (at core) to negative numbers
        self.mesa_domain_bool = ln_rho > - config.star['n_rho']
        self.mesa_domain_bound_ind  = np.argmin(np.abs(self.mass - self.mass[self.mesa_domain_bool][-1]))
        self.mesa_domain_radius = self.r[self.mesa_domain_bound_ind]
        self.mesa_basis_bounds = [0, self.mesa_domain_radius]

        logger.info('fraction of FULL star simulated: {:.2f}, up to r={:.3e}'.format(self.mesa_basis_bounds[-1]/self.R_star, self.mesa_basis_bounds[-1]))
        self.mesa_sim_bool      = (self.r > self.mesa_basis_bounds[0])*(self.r <= self.mesa_basis_bounds[-1])

        #Characteristic scales:
        self.L_CZ    = self.mesa_domain_radius
        m_core  = self.rho[0] * self.L_CZ**3
        T_core  = self.T[0]
        self.H0      = (self.rho*self.eps_nuc)[0]
        self.tau_heat  = ((self.H0*self.L_CZ/m_core)**(-1/3)).cgs #heating timescale
        self.tau_cp = np.sqrt(self.L_CZ**2 / (self.cp[0] * T_core))

        #Fundamental Nondimensionalization -- length (L_nd), mass (m_nd), temp (T_nd), time (tau_nd)
        self.L_nd    = self.L_CZ
        self.m_nd    = self.rho[self.r==self.L_nd][0] * self.L_nd**3 #mass at core cz boundary
        self.T_nd    = self.T[self.r==self.L_nd][0] #temp at core cz boundary
        self.tau_nd  = self.tau_cp.cgs
#        self.tau_nd  = self.tau_heat.cgs
        logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(self.L_nd, self.T_nd, self.m_nd, self.tau_nd))

        #Extra useful nondimensionalized quantities
        self.rho_nd  = self.m_nd/self.L_nd**3
        self.u_nd    = self.L_nd/self.tau_nd
        self.s_nd    = self.L_nd**2 / self.tau_nd**2 / self.T_nd
        self.H_nd    = (self.m_nd / self.L_nd) * self.tau_nd**-3
        self.lum_nd  = self.L_nd**2 * self.m_nd / (self.tau_nd**2) / self.tau_nd
        self.s_motions    = self.L_nd**2 / self.tau_heat**2 / self.T[0]
        self.R_gas_nd = (self.R_gas / self.s_nd).cgs.value
        self.gamma1_nd = (self.gamma1[0]).value
        self.cp_nd = self.R_gas_nd * self.gamma1_nd / (self.gamma1_nd - 1)
        self.Ma2_r0 = ((self.u_nd*(self.tau_nd/self.tau_heat))**2 / ((self.gamma1[0]-1)*self.cp[0]*self.T[0])).cgs
        logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(self.cp_nd, self.R_gas_nd, self.gamma1_nd))
        logger.info('m_nd/M_\odot: {:.3f}'.format((self.m_nd/constants.M_sun).cgs))
        logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(self.Ma2_r0), self.tau_heat))

       
        #MESA radial values at simulation joints & across full star in simulation units
        self.mesa_r_nd = (self.r/self.L_nd).cgs
        self.nd_basis_bounds = [(rb/self.L_nd).value for rb in self.mesa_basis_bounds]
        self.r_inner = self.nd_basis_bounds[0]
        self.r_outer = self.nd_basis_bounds[-1]
      
        ### Make dedalus domain and bases
        self.resolutions = [(1, 1, nr) for nr in config.star['nr']]
        self.stitch_radii = self.nd_basis_bounds[1:-1]
        self.dtype=np.float64
        mesh=None
        dealias = config.numerics['N_dealias']
        self.coords, self.dist, self.bases, self.bases_keys = make_bases(self.resolutions, self.stitch_radii, self.r_outer, dealias=(1,1,dealias), dtype=self.dtype, mesh=mesh)
        self.dedalus_r = OrderedDict()
        for bn in self.bases.keys():
            phi, theta, r_vals = self.bases[bn].global_grids((1, 1, dealias))
            self.dedalus_r[bn] = r_vals

    def customize_star(self):
        #Adjust gravitational potential so that it doesn't cross zero somewhere wonky
        self.g_phi           -= self.g_phi[-1] - self.u_nd**2 #set g_phi = -1 at r = R_star

        #Construct simulation diffusivity profiles -- MANY CHOICES COULD BE MADE HERE!!
        self.mesa_rad_diff_nd = self.rad_diff * (self.tau_nd / self.L_nd**2)
        self.rad_diff_cutoff_nd = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((self.L_CZ**2/self.tau_heat) / (self.L_nd**2/self.tau_nd))
        self.simulation_rad_diff_nd = np.copy(self.mesa_rad_diff_nd) + self.rad_diff_cutoff_nd
        self.simulation_visc_diff_nd = config.numerics['prandtl']*self.rad_diff_cutoff_nd*np.ones_like(self.simulation_rad_diff_nd)
        logger.info('rad_diff cutoff: {:.3e}'.format(self.rad_diff_cutoff_nd))
        
        ### entropy gradient
        if self.cz_only:
            #Entropy gradient is zero everywhere
            grad_s_smooth = np.zeros_like(self.grad_s)
            self.N2_func = interp1d(self.mesa_r_nd, np.zeros_like(grad_s_smooth), **interp_kwargs)
        else:
            #Build a smooth function in the ball, match the star in the shells.
            grad_s_width = 0.05
            grad_s_transition_point = self.nd_basis_bounds[1] - grad_s_width
            logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
            logger.info('using default grad s width = {}'.format(grad_s_width))
            grad_s_center =  grad_s_transition_point - 0.5*grad_s_width
            grad_s_width *= (self.L_CZ/self.L_nd).value
            grad_s_center *= (self.L_CZ/self.L_nd).value
           
            grad_s_smooth = np.copy(self.grad_s)
            flat_value  = np.interp(grad_s_transition_point, self.mesa_r_nd, self.grad_s)
            grad_s_smooth += (self.mesa_r_nd)**2 *  flat_value
            grad_s_smooth *= zero_to_one(self.mesa_r_nd, grad_s_transition_point, width=grad_s_width)

            #construct N2 function #TODO: blend logic here & in BVP?
            smooth_N2 = np.copy(self.N2_mesa)
            stitch_value = np.interp(self.bases['B'].radius, r/self.L_nd, self.N2_mesa)
            smooth_N2[r/self.L_nd < self.bases['B'].radius] = (r[r/self.L_nd < self.bases['B'].radius]/self.L_nd / self.bases['B'].radius)**2 * stitch_value
            smooth_N2 *= zero_to_one(r/self.L_nd, grad_s_transition_point, width=grad_s_width)
            self.N2_func = interp1d(self.mesa_r_nd, self.tau_nd**2 * smooth_N2, **interp_kwargs)

        ### Internal heating / cooling function
        interp_r = np.linspace(0, 1, 1000)
        if config.star['smooth_h']:
            #smooth CZ-RZ transition
            argmax = np.argmax(self.L_conv/self.lum_nd)
            max_r = self.r[argmax]/self.L_nd
            max_L = (self.L_conv[argmax]/self.lum_nd).cgs

            #Heating layer
            Q_base = lambda r : one_to_zero(r, 0.4, width=0.3)
            Q = Q_base(self.mesa_r_nd)
            cumsum = np.cumsum(Q*np.gradient(self.mesa_r_nd) * 4*np.pi*((self.mesa_r_nd)**2))
            first_adjust = np.copy(max_L / cumsum[self.mesa_domain_bound_ind])
            Q_func_heat = lambda r: first_adjust * Q_base(r)
            heat_lum = np.trapz(4*np.pi*interp_r**2 * interp1d(self.mesa_r_nd, Q_func_heat(self.mesa_r_nd), **interp_kwargs)(interp_r), x=interp_r)

            #Cooling layer
            Qcool_base = lambda r: -zero_to_one(r, 0.85, width=0.07)
            cool_lum = np.trapz(4*np.pi*interp_r**2 * interp1d(self.mesa_r_nd, Qcool_base(self.mesa_r_nd), **interp_kwargs)(interp_r), x=interp_r)
            adjustment = np.abs(heat_lum/cool_lum)
            Q_func_cool = lambda r: adjustment * Qcool_base(r)
            self.Q_func = lambda r: Q_func_heat(r) + Q_func_cool(r)

        elif config.star['heat_only']:
            #smooth CZ-RZ transition
            argmax = np.argmax(self.L_conv/self.lum_nd)
            max_L = (self.L_conv[argmax]/self.lum_nd).cgs

            #Heating layer
            Q_base = lambda r : one_to_zero(r, 0.4, width=0.3)
            Q = Q_base(self.mesa_r_nd)
            cumsum = np.cumsum(Q*np.gradient(self.mesa_r_nd) * 4*np.pi*((self.mesa_r_nd)**2))
            first_adjust = np.copy(max_L / cumsum[self.mesa_domain_bound_ind])
            self.Q_func = lambda r: first_adjust * Q_base(r)

#            Q_mesa = np.gradient(self.L_conv/self.lum_nd, self.mesa_r_nd) / (4*np.pi*self.mesa_r_nd**2)
#            plt.plot(self.mesa_r_nd, Q_mesa)
#            plt.plot(self.mesa_r_nd, self.Q_func(self.mesa_r_nd))
#            plt.yscale('log')
#            plt.xlim(0, 1)
#            plt.figure()
#            plt.plot(self.mesa_r_nd, np.cumsum(4*np.pi*self.mesa_r_nd**2*np.gradient(self.mesa_r_nd)*self.Q_func(self.r/self.L_nd)))
#            plt.plot(self.mesa_r_nd, self.L_conv/self.lum_nd)
#            plt.xlim(0, 1)
#            plt.show()
        else:
            raise NotImplementedError("must use smooth_h or heat_only")

        # Create interpolations of the various fields that may be used in the problem
        self.mesa_interpolations = OrderedDict()
        self.mesa_interpolations['ln_rho0'] = interp1d(self.mesa_r_nd, np.log(self.rho/self.rho_nd), **interp_kwargs)
        self.mesa_interpolations['rho0'] = lambda r: np.exp(self.mesa_interpolations['ln_rho0'](r))
        self.mesa_interpolations['ln_T0'] = interp1d(self.mesa_r_nd, np.log(self.T/self.T_nd), **interp_kwargs)
        self.mesa_interpolations['Q'] = interp1d(self.mesa_r_nd, (1/(4*np.pi*self.mesa_r_nd**2))*np.gradient(self.L_conv/self.lum_nd, self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['grad_ln_rho0'] = interp1d(self.mesa_r_nd, self.dlogrhodr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['grad_ln_T0'] = interp1d(self.mesa_r_nd, self.dlogTdr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['grad_ln_pom0'] = interp1d(self.mesa_r_nd, self.dlogTdr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['T0'] = interp1d(self.mesa_r_nd, self.T/self.T_nd, **interp_kwargs)
        self.mesa_interpolations['pom0'] = interp1d(self.mesa_r_nd, self.R_gas_nd * self.T/self.T_nd, **interp_kwargs)
        self.mesa_interpolations['nu_diff'] = interp1d(self.mesa_r_nd, self.simulation_visc_diff_nd, **interp_kwargs)
        self.mesa_interpolations['chi_rad'] = interp1d(self.mesa_r_nd, self.simulation_rad_diff_nd, **interp_kwargs)
        self.mesa_interpolations['grad_chi_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.mesa_rad_diff_nd, self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['g'] = interp1d(self.mesa_r_nd, -self.g * (self.tau_nd**2/self.L_nd), **interp_kwargs)
        self.mesa_interpolations['g_phi'] = interp1d(self.mesa_r_nd, self.g_phi * (self.tau_nd**2 / self.L_nd**2), **interp_kwargs)
        self.mesa_interpolations['grad_s0'] = interp1d(self.mesa_r_nd, self.grad_s_over_cp*self.cp * (self.L_nd/self.s_nd), **interp_kwargs)
        self.mesa_interpolations['s0'] = interp1d(self.mesa_r_nd, self.s_over_cp*self.cp  / self.s_nd, **interp_kwargs)
        self.mesa_interpolations['kappa_rad'] = interp1d(self.mesa_r_nd, self.mesa_interpolations['rho0'](self.mesa_r_nd)*self.cp_nd*self.simulation_rad_diff_nd, **interp_kwargs)
        self.mesa_interpolations['grad_kappa_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.mesa_interpolations['kappa_rad'](self.mesa_r_nd), self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['grad_T0_superad'] = interp1d(self.mesa_r_nd, self.mesa_interpolations['grad_ln_T0'](self.mesa_r_nd)*self.mesa_interpolations['T0'](self.mesa_r_nd)  - \
                                                                               self.mesa_interpolations['g'](self.mesa_r_nd)/(self.cp / self.s_nd).cgs, **interp_kwargs)
        self.interpolations = self.mesa_interpolations.copy()

        #Solve hydrostatic equilibrium BVP for consistency with evolved equations.
        ln_rho_func = self.interpolations['ln_rho0']
        grad_ln_rho_func = self.interpolations['grad_ln_rho0']
        self.atmo = HSE_solve(self.coords, self.dist, self.bases,  grad_ln_rho_func, self.N2_func, Q_func=self.Q_func,
                  r_outer=self.r_outer, r_stitch=self.stitch_radii, dtype=self.dtype, \
                  R=self.R_gas_nd, gamma=self.gamma1_nd, comm=MPI.COMM_SELF, \
                  nondim_radius=1, g_nondim=self.interpolations['g'](1), s_motions=self.s_motions/self.s_nd)

        #Update self.interpolations of important quantities from HSE BVP
        self.F_conv_func = self.atmo['Fconv']
        self.interpolations['ln_rho0'] = self.atmo['ln_rho']
        self.interpolations['rho0'] = lambda r: np.exp(self.interpolations['ln_rho0'](r))
        self.interpolations['Q'] = self.Q_func
        self.interpolations['grad_s0'] = self.atmo['grad_s']
        self.interpolations['pom0'] = self.atmo['pomega']
        self.interpolations['grad_ln_pom0'] = self.atmo['grad_ln_pomega']
        self.interpolations['s0'] = self.atmo['s0']
        self.interpolations['g'] = self.atmo['g']
        self.interpolations['g_phi'] = self.atmo['g_phi']
        self.interpolations['grad_T0_superad'] = self.atmo['grad_T0_superad']
        self.interpolations['kappa_rad'] = interp1d(self.mesa_r_nd, self.interpolations['rho0'](self.mesa_r_nd)*self.cp_nd*self.simulation_rad_diff_nd, **interp_kwargs)
        self.interpolations['grad_kappa_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.interpolations['kappa_rad'](self.mesa_r_nd), self.mesa_r_nd), **interp_kwargs)

        #Prep NCCs for construction.
        for ncc in self.ncc_dict.keys():
            for i, bn in enumerate(self.bases.keys()):
                self.ncc_dict[ncc]['Nmax_{}'.format(bn)] = self.ncc_dict[ncc]['nr_max'][i]
                self.ncc_dict[ncc]['field_{}'.format(bn)] = None
            if ncc in self.interpolations.keys():
                self.ncc_dict[ncc]['interp_func'] = self.interpolations[ncc]
            else:
                self.ncc_dict[ncc]['interp_func'] = None

        #Construct NCCs
        for bn, basis in self.bases.items():
            rvals = self.dedalus_r[bn]
            for ncc in self.ncc_dict.keys():
                interp_func = self.ncc_dict[ncc]['interp_func']
                if interp_func is not None and not self.ncc_dict[ncc]['from_grad']:
                    Nmax = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]
                    vector = self.ncc_dict[ncc]['vector']
                    grid_only = self.ncc_dict[ncc]['grid_only']
                    self.ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    if self.ncc_dict[ncc]['get_grad']:
                        name = self.ncc_dict[ncc]['grad_name']
                        logger.info('getting {}'.format(name))
                        grad_field = d3.grad(self.ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                        grad_field.change_scales((1,1,(Nmax+1)/self.resolutions[self.bases_keys == bn][2]))
                        grad_field.change_scales(basis.dealias)
                        self.ncc_dict[name]['field_{}'.format(bn)] = grad_field
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                    if self.ncc_dict[ncc]['get_inverse']:
                        name = 'inv_{}'.format(ncc)
                        inv_func = lambda r: 1/interp_func(r)
                        self.ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax


            if 'neg_g' in self.ncc_dict.keys():
                if 'g' not in self.ncc_dict.keys():
                    self.ncc_dict['g'] = OrderedDict()
                name = 'g'
                self.ncc_dict['g']['field_{}'.format(bn)] = (-self.ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
                self.ncc_dict['g']['vector'] = True
                self.ncc_dict['g']['interp_func'] = self.interpolations['g']
                self.ncc_dict['g']['Nmax_{}'.format(bn)] = self.ncc_dict['neg_g']['Nmax_{}'.format(bn)]
                self.ncc_dict['g']['from_grad'] = True 
 
        #Adjust heating function so luminosity integrates to zero when appropriate.  
        if not config.star['heat_only']:
            integral = 0
            for bn in self.bases.keys():
                integral += d3.integ(self.ncc_dict['Q']['field_{}'.format(bn)])
            C = integral.evaluate()['g']
            vol = (4/3) * np.pi * (self.r_outer)**3
            adj = C / vol
            logger.info('adjusting dLdt for energy conservation; subtracting {} from H'.format(adj))
            for bn in self.bases.keys():
                self.ncc_dict['Q']['field_{}'.format(bn)]['g'] -= adj 


    def plot_star(self):
        #Make plots of the NCCs
        if self.plot_nccs:
            for ncc in self.ncc_dict.keys():
                if self.ncc_dict[ncc]['interp_func'] is None:
                    continue
                axhline = None
                log = False
                ylim = None
                rvals = []
                dedalus_yvals = []
                nvals = []
                for bn, basis in self.bases.items():
                    rvals.append(self.dedalus_r[bn].ravel())
                    nvals.append(self.ncc_dict[ncc]['Nmax_{}'.format(bn)])
                    if self.ncc_dict[ncc]['vector']:
                        dedalus_yvals.append(np.copy(self.ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:]))
                    else:
                        dedalus_yvals.append(np.copy(self.ncc_dict[ncc]['field_{}'.format(bn)]['g'][0,0,:]))
        
                interp_func = self.mesa_interpolations[ncc]
                if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad']:
                    log = True
                if ncc == 'grad_s0': 
                    axhline = (self.s_motions / self.s_nd)
                elif ncc in ['chi_rad', 'grad_chi_rad']:
                    if ncc == 'chi_rad':
                        interp_func = interp1d(self.mesa_r_nd, (self.L_nd**2/self.tau_nd).value*self.mesa_rad_diff_nd, **interp_kwargs)
                        for ind in range(len(dedalus_yvals)):
                            dedalus_yvals[ind] *= (self.L_nd**2/self.tau_nd).value
                    axhline = self.rad_diff_cutoff_nd*(self.L_nd**2/self.tau_nd).value
        
                if ncc == 'H':
                    interp_func = interp1d(r_vals, ( one_to_zero(r_vals, 1.5*self.nd_basis_bounds[1], width=0.05*self.nd_basis_bounds[1])*sim_H_eff ) * (1/self.H_nd), **interp_kwargs )
                elif ncc == 'grad_s0':
                    interp_func = interp1d(self.mesa_r_nd, (self.L_nd/self.s_nd) * self.grad_s, **interp_kwargs)
                elif ncc in ['ln_T0', 'ln_rho0', 'grad_s0']:
                    interp_func = self.interpolations[ncc]
        
                if ncc in ['grad_T', 'grad_kappa_rad']:
                    interp_func = lambda r: -self.ncc_dict[ncc]['interp_func'](r)
                    ylabel='-{}'.format(ncc)
                    for i in range(len(dedalus_yvals)):
                        dedalus_yvals[i] *= -1
                elif ncc == 'chi_rad':
                    ylabel = 'radiative diffusivity (cm^2/s)'
                else:
                    ylabel = ncc

        
                plot_ncc_figure(rvals, interp_func, dedalus_yvals, nvals, \
                            ylabel=ylabel, fig_name=ncc, out_dir=self.out_dir, log=log, ylim=ylim, \
                            r_int=self.stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

    def save_star(self):

        # Get some timestepping info
        max_dt = 0.05*self.tau_heat/self.tau_nd
        # Save output fields.
        with h5py.File('{:s}'.format(self.out_file), 'w') as f:
            # slicing preserves dimensionality
            for bn, basis in self.bases.items():
                f['r_{}'.format(bn)] = self.dedalus_r[bn]
                for ncc in self.ncc_dict.keys():
                    this_field = self.ncc_dict[ncc]['field_{}'.format(bn)]
                    if self.ncc_dict[ncc]['vector']:
                        f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                        f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
                    else:
                        f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
                        f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
        
            f['Cp'] = self.cp_nd
            f['R_gas'] = self.R_gas_nd
            f['gamma1'] = self.gamma1_nd

            #Save properties of the star, with units.
            f['L_nd']   = self.L_nd
            f['L_nd'].attrs['units'] = str(self.L_nd.unit)
            f['rho_nd']  = self.rho_nd
            f['rho_nd'].attrs['units']  = str(self.rho_nd.unit)
            f['T_nd']  = self.T_nd
            f['T_nd'].attrs['units']  = str(self.T_nd.unit)
            f['tau_heat'] = self.tau_heat
            f['tau_heat'].attrs['units'] = str(self.tau_heat.unit)
            f['tau_nd'] = self.tau_nd 
            f['tau_nd'].attrs['units'] = str(self.tau_nd.unit)
            f['m_nd'] = self.m_nd 
            f['m_nd'].attrs['units'] = str(self.m_nd.unit)
            f['s_nd'] = self.s_nd
            f['s_nd'].attrs['units'] = str(self.s_nd.unit)
            f['P_r0']  = self.P[0]
            f['P_r0'].attrs['units']  = str(self.P[0].unit)
            f['H_nd']  = self.H_nd
            f['H_nd'].attrs['units']  = str(self.H_nd.unit)
            f['H0']  = self.H0
            f['H0'].attrs['units']  = str(self.H0.unit)
            f['cp_surf'] = self.cp[self.mesa_sim_bool][-1]
            f['cp_surf'].attrs['units'] = str(self.cp[self.mesa_sim_bool][-1].unit)
            f['r_mesa'] = self.r
            f['r_mesa'].attrs['units'] = str(self.r.unit)
            f['g_mesa'] = self.g 
            f['g_mesa'].attrs['units'] = str(self.g.unit)
            f['cp_mesa'] = self.cp
            f['cp_mesa'].attrs['units'] = str(self.cp.unit)

            #TODO: put sim lum back
            f['lum_r_vals'] = lum_r_vals = np.linspace(self.nd_basis_bounds[0], self.r_outer, 1000)
            f['sim_lum'] = (4*np.pi*lum_r_vals**2)*self.F_conv_func(lum_r_vals)
            f['r_stitch']   = self.stitch_radii
            f['r_outer']   = self.r_outer 
            f['max_dt'] = max_dt
            f['Ma2_r0'] = self.Ma2_r0
            for k in ['r_stitch', 'r_outer', 'max_dt', 'Ma2_r0', 'lum_r_vals', 'sim_lum',\
                        'Cp', 'R_gas', 'gamma1']:
                f[k].attrs['units'] = 'dimensionless'
        logger.info('finished saving NCCs to {}'.format(self.out_file))
        logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(self.out_dir))


class EnvelopeStarBuilder(DedalusMesaReader):

    def __init__(self, plot_nccs=False):
        """ Create nondimensionalization and Dedalus domain / bases. """
        super().__init__()
        self.plot_nccs = plot_nccs
        self.out_dir, self.out_file = name_star()
        self.ncc_dict = config.nccs.copy()
        self.cz_only = config.star['cz_only']

        # Find edges of dedalus domain
        self.mesa_conv_bool = self.conv_L_div_L > 1e-6
        ln_rho = np.log(self.rho/self.rho[0])
        ln_rho_cz_base = ln_rho[self.mesa_conv_bool][0]
        self.mesa_domain_bool = (ln_rho <= ln_rho_cz_base)*(ln_rho > ln_rho_cz_base - config.star['n_rho'])
        self.mesa_domain_inner_bound_ind  = np.argmin(np.abs(self.mass - self.mass[self.mesa_domain_bool][0]))
        self.mesa_domain_outer_bound_ind  = np.argmin(np.abs(self.mass - self.mass[self.mesa_domain_bool][-1]))
        self.mesa_domain_r_inner = self.r[self.mesa_domain_inner_bound_ind]
        self.mesa_domain_r_outer = self.r[self.mesa_domain_outer_bound_ind]
        self.mesa_basis_bounds = [self.mesa_domain_r_inner, self.mesa_domain_r_outer]
        self.mesa_domain_radius = self.mesa_domain_r_outer - self.mesa_domain_r_inner

        logger.info('fraction of FULL star simulated: {:.2f}, from r = {:.3e} up to r={:.3e}'.format(self.mesa_domain_radius/self.R_star, self.mesa_basis_bounds[0], self.mesa_basis_bounds[-1]))
        self.mesa_sim_bool      = (self.r > self.mesa_basis_bounds[0])*(self.r <= self.mesa_basis_bounds[-1])

        #Characteristic scales:
        self.L_CZ    = self.mesa_domain_radius
        m_bot        = self.rho[self.mesa_domain_inner_bound_ind] * self.L_CZ**3
        T_bot        = self.T[self.mesa_domain_inner_bound_ind]
        heating      = np.gradient(self.L_conv, self.r) / (4*np.pi*self.r**2)
        self.H0      = heating[self.mesa_sim_bool].max().cgs
        self.tau_heat  = ((self.H0*self.L_CZ/m_bot)**(-1/3)).cgs #heating timescale
        self.tau_cp = np.sqrt(self.L_CZ**2 / (self.cp[self.mesa_domain_inner_bound_ind] * T_bot))

        #Fundamental Nondimensionalization -- length (L_nd), mass (m_nd), temp (T_nd), time (tau_nd)
        self.L_nd    = self.L_CZ
        self.m_nd    = self.rho[self.mesa_domain_outer_bound_ind] * self.L_nd**3 #mass at core cz boundary
        self.T_nd    = self.T[self.mesa_domain_outer_bound_ind] #temp at core cz boundary
        self.tau_nd  = self.tau_cp.cgs
#        self.tau_nd  = self.tau_heat.cgs
        logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(self.L_nd, self.T_nd, self.m_nd, self.tau_nd))

        #Extra useful nondimensionalized quantities
        self.rho_nd  = self.m_nd/self.L_nd**3
        self.u_nd    = self.L_nd/self.tau_nd
        self.s_nd    = self.L_nd**2 / self.tau_nd**2 / self.T_nd
        self.H_nd    = (self.m_nd / self.L_nd) * self.tau_nd**-3
        self.lum_nd  = self.L_nd**2 * self.m_nd / (self.tau_nd**2) / self.tau_nd
        self.s_motions    = self.L_nd**2 / self.tau_heat**2 / self.T[0]
        self.R_gas_nd = (self.R_gas / self.s_nd).cgs.value
        self.gamma1_nd = (self.gamma1[0]).value
        self.cp_nd = self.R_gas_nd * self.gamma1_nd / (self.gamma1_nd - 1)
        self.Ma2_r0 = ((self.u_nd*(self.tau_nd/self.tau_heat))**2 / ((self.gamma1[0]-1)*self.cp[0]*self.T[0])).cgs
        logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(self.cp_nd, self.R_gas_nd, self.gamma1_nd))
        logger.info('m_nd/M_\odot: {:.3f}'.format((self.m_nd/constants.M_sun).cgs))
        logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(self.Ma2_r0), self.tau_heat))


      
        #MESA radial values at simulation joints & across full star in simulation units
        self.mesa_r_nd = (self.r/self.L_nd).cgs
        self.nd_basis_bounds = [(rb/self.L_nd).value for rb in self.mesa_basis_bounds]
        self.r_inner = self.nd_basis_bounds[0]
        self.r_outer = self.nd_basis_bounds[-1]
      
        ### Make dedalus domain and bases
        self.resolutions = [(1, 1, nr) for nr in config.star['nr']]
        self.stitch_radii = self.nd_basis_bounds[1:-1]
        self.dtype=np.float64
        mesh=None
        dealias = config.numerics['N_dealias']
        self.coords, self.dist, self.bases, self.bases_keys = make_bases(self.resolutions, self.stitch_radii, self.r_outer, r_inner=self.r_inner, dealias=(1,1,dealias), dtype=self.dtype, mesh=mesh)
        self.dedalus_r = OrderedDict()
        for bn in self.bases.keys():
            phi, theta, r_vals = self.bases[bn].global_grids((1, 1, dealias))
            self.dedalus_r[bn] = r_vals

    def customize_star(self):
        #Adjust gravitational potential so that it doesn't cross zero somewhere wonky
        self.g_phi           -= self.g_phi[-1] - self.u_nd**2 #set g_phi = -1 at r = R_star

        #Construct simulation diffusivity profiles -- MANY CHOICES COULD BE MADE HERE!!
        self.mesa_rad_diff_nd = self.rad_diff * (self.tau_nd / self.L_nd**2)
        self.rad_diff_cutoff_nd = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((self.L_CZ**2/self.tau_heat) / (self.L_nd**2/self.tau_nd))
        self.simulation_rad_diff_nd = np.copy(self.mesa_rad_diff_nd) + self.rad_diff_cutoff_nd
        self.simulation_visc_diff_nd = config.numerics['prandtl']*self.rad_diff_cutoff_nd*np.ones_like(self.simulation_rad_diff_nd)
        logger.info('rad_diff cutoff: {:.3e}'.format(self.rad_diff_cutoff_nd))
        
        ### entropy gradient
        if self.cz_only:
            #Entropy gradient is zero everywhere
            grad_s_smooth = np.zeros_like(self.grad_s)
            self.N2_func = interp1d(self.mesa_r_nd, np.zeros_like(grad_s_smooth), **interp_kwargs)
        else:
            #Build a smooth function in the ball, match the star in the shells.
            grad_s_width = 0.05
            grad_s_transition_point = self.nd_basis_bounds[1] - grad_s_width
            logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
            logger.info('using default grad s width = {}'.format(grad_s_width))
            grad_s_center =  grad_s_transition_point - 0.5*grad_s_width
            grad_s_width *= (self.L_CZ/self.L_nd).value
            grad_s_center *= (self.L_CZ/self.L_nd).value
           
            grad_s_smooth = np.copy(self.grad_s)
            flat_value  = np.interp(grad_s_transition_point, self.mesa_r_nd, self.grad_s)
            grad_s_smooth += (self.mesa_r_nd)**2 *  flat_value
            grad_s_smooth *= zero_to_one(self.mesa_r_nd, grad_s_transition_point, width=grad_s_width)

            #construct N2 function #TODO: blend logic here & in BVP?
            smooth_N2 = np.copy(self.N2_mesa)
            stitch_value = np.interp(self.bases['B'].radius, r/self.L_nd, self.N2_mesa)
            smooth_N2[r/self.L_nd < self.bases['B'].radius] = (r[r/self.L_nd < self.bases['B'].radius]/self.L_nd / self.bases['B'].radius)**2 * stitch_value
            smooth_N2 *= zero_to_one(r/self.L_nd, grad_s_transition_point, width=grad_s_width)
            self.N2_func = interp1d(self.mesa_r_nd, self.tau_nd**2 * smooth_N2, **interp_kwargs)

        ### Internal heating / cooling function
        interp_r = np.linspace(self.r_inner, self.r_outer, 1000)
        if config.star['smooth_h']:
            #smooth CZ-RZ transition
            max_L = (self.L_conv/self.lum_nd).cgs.max()

            #Heating layer
            Q_base = lambda r : zero_to_one(r, self.r_inner, width=0.00001) * (0.25 + r - self.r_inner)**(-2)
            Q = Q_base(self.mesa_r_nd)
            cumsum = np.cumsum(Q*np.gradient(self.mesa_r_nd) * 4*np.pi*((self.mesa_r_nd)**2))
            first_adjust = np.copy(max_L / cumsum[self.mesa_domain_outer_bound_ind])
            Q_func_heat = lambda r: first_adjust * Q_base(r)
            heat_lum = np.trapz(4*np.pi*interp_r**2 * interp1d(self.mesa_r_nd, Q_func_heat(self.mesa_r_nd), **interp_kwargs)(interp_r), x=interp_r)

            #Cooling layer
            Qcool_base = lambda r: -zero_to_one(r, self.r_inner + 0.85, width=0.07)
            cool_lum = np.trapz(4*np.pi*interp_r**2 * interp1d(self.mesa_r_nd, Qcool_base(self.mesa_r_nd), **interp_kwargs)(interp_r), x=interp_r)
            adjustment = np.abs(heat_lum/cool_lum)
            Q_func_cool = lambda r: adjustment * Qcool_base(r)
            self.Q_func = lambda r: Q_func_heat(r) + Q_func_cool(r)

#            plt.figure()
#            Q_mesa = np.gradient(self.L_conv/self.lum_nd, self.mesa_r_nd) / (4*np.pi*self.mesa_r_nd**2)
#            plt.plot(self.mesa_r_nd, self.Q_func(self.mesa_r_nd))
#            plt.plot(self.mesa_r_nd, Q_mesa)
#            plt.ylabel('Q')
##            plt.yscale('log')
#            plt.xlim(0, 4)
#            plt.ylim(0, 2e-8)
#            plt.figure()
#            plt.plot(self.mesa_r_nd, np.cumsum(4*np.pi*self.mesa_r_nd**2*np.gradient(self.mesa_r_nd)*self.Q_func(self.r/self.L_nd)))
#            plt.plot(self.mesa_r_nd, self.L_conv/self.lum_nd)
#            plt.ylabel('Lconv')
#            plt.xlim(0, 4)
#            plt.show()
#

        elif config.star['heat_only']:
            #smooth CZ-RZ transition
            max_L = (self.L_conv/self.lum_nd).cgs.max()

            #Heating layer
            Q_base = lambda r : zero_to_one(r, self.r_inner, width=0.00001) * (0.25 + r - self.r_inner)**(-2)
            Q = Q_base(self.mesa_r_nd)
            cumsum = np.cumsum(Q*np.gradient(self.mesa_r_nd) * 4*np.pi*((self.mesa_r_nd)**2))
            first_adjust = np.copy(max_L / cumsum[self.mesa_domain_outer_bound_ind])
            self.Q_func = lambda r: first_adjust * Q_base(r)

#            plt.figure()
#            Q_mesa = np.gradient(self.L_conv/self.lum_nd, self.mesa_r_nd) / (4*np.pi*self.mesa_r_nd**2)
#            plt.plot(self.mesa_r_nd, Q_mesa)
#            plt.plot(self.mesa_r_nd, self.Q_func(self.mesa_r_nd))
#            plt.yscale('log')
#            plt.xlim(3, 4.5)
#            plt.figure()
#            plt.plot(self.mesa_r_nd, np.cumsum(4*np.pi*self.mesa_r_nd**2*np.gradient(self.mesa_r_nd)*self.Q_func(self.r/self.L_nd)))
#            plt.plot(self.mesa_r_nd, self.L_conv/self.lum_nd)
#            plt.xlim(3, 4.5)
#            plt.show()
        else:
            raise NotImplementedError("must use smooth_h or heat_only")

        # Create interpolations of the various fields that may be used in the problem
        self.mesa_interpolations = OrderedDict()
        self.mesa_interpolations['ln_rho0'] = interp1d(self.mesa_r_nd, np.log(self.rho/self.rho_nd), **interp_kwargs)
        self.mesa_interpolations['rho0'] = lambda r: np.exp(self.mesa_interpolations['ln_rho0'](r))
        self.mesa_interpolations['ln_T0'] = interp1d(self.mesa_r_nd, np.log(self.T/self.T_nd), **interp_kwargs)
        self.mesa_interpolations['Q'] = interp1d(self.mesa_r_nd, (1/(4*np.pi*self.mesa_r_nd**2))*np.gradient(self.L_conv/self.lum_nd, self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['grad_ln_rho0'] = interp1d(self.mesa_r_nd, self.dlogrhodr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['grad_ln_T0'] = interp1d(self.mesa_r_nd, self.dlogTdr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['grad_ln_pom0'] = interp1d(self.mesa_r_nd, self.dlogTdr*self.L_nd, **interp_kwargs)
        self.mesa_interpolations['T0'] = interp1d(self.mesa_r_nd, self.T/self.T_nd, **interp_kwargs)
        self.mesa_interpolations['pom0'] = interp1d(self.mesa_r_nd, self.R_gas_nd * self.T/self.T_nd, **interp_kwargs)
        self.mesa_interpolations['nu_diff'] = interp1d(self.mesa_r_nd, self.simulation_visc_diff_nd, **interp_kwargs)
        self.mesa_interpolations['chi_rad'] = interp1d(self.mesa_r_nd, self.simulation_rad_diff_nd, **interp_kwargs)
        self.mesa_interpolations['grad_chi_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.mesa_rad_diff_nd, self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['g'] = interp1d(self.mesa_r_nd, -self.g * (self.tau_nd**2/self.L_nd), **interp_kwargs)
        self.mesa_interpolations['g_phi'] = interp1d(self.mesa_r_nd, self.g_phi * (self.tau_nd**2 / self.L_nd**2), **interp_kwargs)
        self.mesa_interpolations['grad_s0'] = interp1d(self.mesa_r_nd, self.grad_s_over_cp*self.cp * (self.L_nd/self.s_nd), **interp_kwargs)
        self.mesa_interpolations['s0'] = interp1d(self.mesa_r_nd, self.s_over_cp*self.cp  / self.s_nd, **interp_kwargs)
        self.mesa_interpolations['kappa_rad'] = interp1d(self.mesa_r_nd, self.mesa_interpolations['rho0'](self.mesa_r_nd)*self.cp_nd*self.simulation_rad_diff_nd, **interp_kwargs)
        self.mesa_interpolations['grad_kappa_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.mesa_interpolations['kappa_rad'](self.mesa_r_nd), self.mesa_r_nd), **interp_kwargs)
        self.mesa_interpolations['grad_T0_superad'] = interp1d(self.mesa_r_nd, self.mesa_interpolations['grad_ln_T0'](self.mesa_r_nd)*self.mesa_interpolations['T0'](self.mesa_r_nd)  - \
                                                                               self.mesa_interpolations['g'](self.mesa_r_nd)/(self.cp / self.s_nd).cgs, **interp_kwargs)
        self.interpolations = self.mesa_interpolations.copy()

        #Solve hydrostatic equilibrium BVP for consistency with evolved equations.
        ln_rho_func = self.interpolations['ln_rho0']
        grad_ln_rho_func = self.interpolations['grad_ln_rho0']
        self.atmo = HSE_solve(self.coords, self.dist, self.bases,  grad_ln_rho_func, self.N2_func, Q_func=self.Q_func,
                  r_outer=self.r_outer, r_stitch=self.stitch_radii, dtype=self.dtype, \
                  R=self.R_gas_nd, gamma=self.gamma1_nd, comm=MPI.COMM_SELF, \
                  nondim_radius=self.r_outer, g_nondim=self.interpolations['g'](1), s_motions=self.s_motions/self.s_nd)

        #Update self.interpolations of important quantities from HSE BVP
        self.F_conv_func = self.atmo['Fconv']
        self.interpolations['ln_rho0'] = self.atmo['ln_rho']
        self.interpolations['rho0'] = lambda r: np.exp(self.interpolations['ln_rho0'](r))
        self.interpolations['Q'] = self.Q_func
        self.interpolations['grad_s0'] = self.atmo['grad_s']
        self.interpolations['pom0'] = self.atmo['pomega']
        self.interpolations['grad_ln_pom0'] = self.atmo['grad_ln_pomega']
        self.interpolations['s0'] = self.atmo['s0']
        self.interpolations['g'] = self.atmo['g']
        self.interpolations['g_phi'] = self.atmo['g_phi']
        self.interpolations['grad_T0_superad'] = self.atmo['grad_T0_superad']
        self.interpolations['kappa_rad'] = interp1d(self.mesa_r_nd, self.interpolations['rho0'](self.mesa_r_nd)*self.cp_nd*self.simulation_rad_diff_nd, **interp_kwargs)
        self.interpolations['grad_kappa_rad'] = interp1d(self.mesa_r_nd, np.gradient(self.interpolations['kappa_rad'](self.mesa_r_nd), self.mesa_r_nd), **interp_kwargs)

        #Prep NCCs for construction.
        for ncc in self.ncc_dict.keys():
            for i, bn in enumerate(self.bases.keys()):
                self.ncc_dict[ncc]['Nmax_{}'.format(bn)] = self.ncc_dict[ncc]['nr_max'][i]
                self.ncc_dict[ncc]['field_{}'.format(bn)] = None
            if ncc in self.interpolations.keys():
                self.ncc_dict[ncc]['interp_func'] = self.interpolations[ncc]
            else:
                self.ncc_dict[ncc]['interp_func'] = None

        #Construct NCCs
        for bn, basis in self.bases.items():
            rvals = self.dedalus_r[bn]
            for ncc in self.ncc_dict.keys():
                interp_func = self.ncc_dict[ncc]['interp_func']
                if interp_func is not None and not self.ncc_dict[ncc]['from_grad']:
                    Nmax = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]
                    vector = self.ncc_dict[ncc]['vector']
                    grid_only = self.ncc_dict[ncc]['grid_only']
                    self.ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    if self.ncc_dict[ncc]['get_grad']:
                        name = self.ncc_dict[ncc]['grad_name']
                        logger.info('getting {}'.format(name))
                        grad_field = d3.grad(self.ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                        grad_field.change_scales((1,1,(Nmax+1)/self.resolutions[self.bases_keys == bn][2]))
                        grad_field.change_scales(basis.dealias)
                        self.ncc_dict[name]['field_{}'.format(bn)] = grad_field
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                    if self.ncc_dict[ncc]['get_inverse']:
                        name = 'inv_{}'.format(ncc)
                        inv_func = lambda r: 1/interp_func(r)
                        self.ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax


            if 'neg_g' in self.ncc_dict.keys():
                if 'g' not in self.ncc_dict.keys():
                    self.ncc_dict['g'] = OrderedDict()
                name = 'g'
                self.ncc_dict['g']['field_{}'.format(bn)] = (-self.ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
                self.ncc_dict['g']['vector'] = True
                self.ncc_dict['g']['interp_func'] = self.interpolations['g']
                self.ncc_dict['g']['Nmax_{}'.format(bn)] = self.ncc_dict['neg_g']['Nmax_{}'.format(bn)]
                self.ncc_dict['g']['from_grad'] = True 
 
        #Adjust heating function so luminosity integrates to zero when appropriate.  
        if not config.star['heat_only']:
            integral = 0
            for bn in self.bases.keys():
                integral += d3.integ(self.ncc_dict['Q']['field_{}'.format(bn)])
            C = integral.evaluate()['g']
            vol = (4/3) * np.pi * (self.r_outer)**3
            adj = C / vol
            logger.info('adjusting dLdt for energy conservation; subtracting {} from H'.format(adj))
            for bn in self.bases.keys():
                self.ncc_dict['Q']['field_{}'.format(bn)]['g'] -= adj 


    def plot_star(self):
        #Make plots of the NCCs
        if self.plot_nccs:
            for ncc in self.ncc_dict.keys():
                if self.ncc_dict[ncc]['interp_func'] is None:
                    continue
                axhline = None
                log = False
                ylim = None
                rvals = []
                dedalus_yvals = []
                nvals = []
                for bn, basis in self.bases.items():
                    rvals.append(self.dedalus_r[bn].ravel())
                    nvals.append(self.ncc_dict[ncc]['Nmax_{}'.format(bn)])
                    if self.ncc_dict[ncc]['vector']:
                        dedalus_yvals.append(np.copy(self.ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:]))
                    else:
                        dedalus_yvals.append(np.copy(self.ncc_dict[ncc]['field_{}'.format(bn)]['g'][0,0,:]))
        
                interp_func = self.mesa_interpolations[ncc]
                if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad']:
                    log = True
                if ncc == 'grad_s0': 
                    axhline = (self.s_motions / self.s_nd)
                elif ncc in ['chi_rad', 'grad_chi_rad']:
                    if ncc == 'chi_rad':
                        interp_func = interp1d(self.mesa_r_nd, (self.L_nd**2/self.tau_nd).value*self.mesa_rad_diff_nd, **interp_kwargs)
                        for ind in range(len(dedalus_yvals)):
                            dedalus_yvals[ind] *= (self.L_nd**2/self.tau_nd).value
                    axhline = self.rad_diff_cutoff_nd*(self.L_nd**2/self.tau_nd).value
        
                if ncc == 'H':
                    interp_func = interp1d(r_vals, ( one_to_zero(r_vals, 1.5*self.nd_basis_bounds[1], width=0.05*self.nd_basis_bounds[1])*sim_H_eff ) * (1/self.H_nd), **interp_kwargs )
                elif ncc == 'grad_s0':
                    interp_func = interp1d(self.mesa_r_nd, (self.L_nd/self.s_nd) * self.grad_s, **interp_kwargs)
                elif ncc in ['ln_T0', 'ln_rho0', 'grad_s0']:
                    interp_func = self.interpolations[ncc]
        
                if ncc in ['grad_T', 'grad_kappa_rad']:
                    interp_func = lambda r: -self.ncc_dict[ncc]['interp_func'](r)
                    ylabel='-{}'.format(ncc)
                    for i in range(len(dedalus_yvals)):
                        dedalus_yvals[i] *= -1
                elif ncc == 'chi_rad':
                    ylabel = 'radiative diffusivity (cm^2/s)'
                else:
                    ylabel = ncc

        
                plot_ncc_figure(rvals, interp_func, dedalus_yvals, nvals, \
                            ylabel=ylabel, fig_name=ncc, out_dir=self.out_dir, log=log, ylim=ylim, \
                            r_int=self.stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

    def save_star(self):

        # Get some timestepping info
        max_dt = 0.05*self.tau_heat/self.tau_nd
        # Save output fields.
        with h5py.File('{:s}'.format(self.out_file), 'w') as f:
            # slicing preserves dimensionality
            for bn, basis in self.bases.items():
                f['r_{}'.format(bn)] = self.dedalus_r[bn]
                for ncc in self.ncc_dict.keys():
                    this_field = self.ncc_dict[ncc]['field_{}'.format(bn)]
                    if self.ncc_dict[ncc]['vector']:
                        f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                        f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
                    else:
                        f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
                        f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
        
            f['Cp'] = self.cp_nd
            f['R_gas'] = self.R_gas_nd
            f['gamma1'] = self.gamma1_nd

            #Save properties of the star, with units.
            f['L_nd']   = self.L_nd
            f['L_nd'].attrs['units'] = str(self.L_nd.unit)
            f['rho_nd']  = self.rho_nd
            f['rho_nd'].attrs['units']  = str(self.rho_nd.unit)
            f['T_nd']  = self.T_nd
            f['T_nd'].attrs['units']  = str(self.T_nd.unit)
            f['tau_heat'] = self.tau_heat
            f['tau_heat'].attrs['units'] = str(self.tau_heat.unit)
            f['tau_nd'] = self.tau_nd 
            f['tau_nd'].attrs['units'] = str(self.tau_nd.unit)
            f['m_nd'] = self.m_nd 
            f['m_nd'].attrs['units'] = str(self.m_nd.unit)
            f['s_nd'] = self.s_nd
            f['s_nd'].attrs['units'] = str(self.s_nd.unit)
            f['P_r0']  = self.P[0]
            f['P_r0'].attrs['units']  = str(self.P[0].unit)
            f['H_nd']  = self.H_nd
            f['H_nd'].attrs['units']  = str(self.H_nd.unit)
            f['H0']  = self.H0
            f['H0'].attrs['units']  = str(self.H0.unit)
            f['cp_surf'] = self.cp[self.mesa_sim_bool][-1]
            f['cp_surf'].attrs['units'] = str(self.cp[self.mesa_sim_bool][-1].unit)
            f['r_mesa'] = self.r
            f['r_mesa'].attrs['units'] = str(self.r.unit)
            f['g_mesa'] = self.g 
            f['g_mesa'].attrs['units'] = str(self.g.unit)
            f['cp_mesa'] = self.cp
            f['cp_mesa'].attrs['units'] = str(self.cp.unit)

            #TODO: put sim lum back
            f['lum_r_vals'] = lum_r_vals = np.linspace(self.nd_basis_bounds[0], self.r_outer, 1000)
            f['sim_lum'] = (4*np.pi*lum_r_vals**2)*self.F_conv_func(lum_r_vals)
            f['r_inner']   = self.r_inner
            f['r_stitch']   = self.stitch_radii
            f['r_outer']   = self.r_outer 
            f['max_dt'] = max_dt
            f['Ma2_r0'] = self.Ma2_r0
            for k in ['r_stitch', 'r_outer', 'r_inner', 'max_dt', 'Ma2_r0', 'lum_r_vals', 'sim_lum',\
                        'Cp', 'R_gas', 'gamma1']:
                f[k].attrs['units'] = 'dimensionless'
        logger.info('finished saving NCCs to {}'.format(self.out_file))
        logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(self.out_dir))








def build_nccs(plot_nccs=False):

    if config.star['type'].lower() == 'massive':
        star_builder = MassiveStarBuilder(plot_nccs=plot_nccs)
    elif config.star['type'].lower() == 'dwarf':
        star_builder = MdwarfBuilder(plot_nccs=plot_nccs)
    elif config.star['type'].lower() == 'envelope':
        star_builder = EnvelopeStarBuilder(plot_nccs=plot_nccs)
    else:
        raise ValueError('unknown star_type')
    star_builder.customize_star()
    star_builder.plot_star()
    star_builder.save_star()



