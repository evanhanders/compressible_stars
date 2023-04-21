import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3

from astropy import units as u
from astropy import constants
from scipy.interpolate import interp1d

import compstar 
from .compressible_functions import make_bases
from .parser import name_star
from .bvp_functions import HSE_solve
from compstar.tools.mesa import DimensionalMesaReader, find_core_cz_radius
from compstar.tools.general import one_to_zero, zero_to_one
import compstar.defaults.config as config

import logging
logger = logging.getLogger(__name__)

interp_kwargs = {'fill_value' : 'extrapolate', 'bounds_error' : False}

### Function definitions
def plot_ncc_figure(rvals, mesa_func, dedalus_vals, ylabel="", fig_name="", out_dir='.', 
                    zero_line=False, log=False, r_int=None, ylim=None, axhline=None, Ns=None, ncc_cutoff=1e-6):
    """ 
    Plots a figure which compares a dedalus field and the MESA profile that the Dedalus field is based on. 

    Parameters
    ----------
    rvals : list of arrays
        The radial values of the dedalus field
    mesa_func : function
        A function which takes a radius and returns the corresponding MESA value
    dedalus_vals : list of arrays
        The dedalus field values
    
    ylabel : str
        The label for the y-axis
    fig_name : str
        The name of the figure (will be saved as a png)
    out_dir : str
        The directory to save the figure in
    zero_line : bool
        Whether to plot a horizontal line at y=0
    log : bool
        Whether to plot the y-axis on a log scale
    r_int : list of floats
        The radii to plot vertical lines at
    ylim : list of floats
        The limits of the y-axis
    axhline : float
        A value to plot a horizontal line at
    Ns : list of int
        The number of coefficients used in the dedalus field expansion
    ncc_cutoff : float
        The NCC cutoff used.
    """
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
    """
    Given a function which returns some MESA profile as a function of nondimensional radius,
    this function returns a dedalus field which is the Dedalus expansion of that function.

    Arguments
    ---------
    basis : dedalus basis
        The dedalus basis to use for the field
    coords : dedalus coordinates
        The dedalus coordinates to use for the field
    dist : dedalus distributor
        The dedalus distributor to use for the field
    interp_func : function
        A function which takes a radius and returns the corresponding (nondimensional) MESA value
    Nmax : int
        The maximum number of coefficients to use in the dedalus expansion
    vector : bool
        Whether the field is a vector field; if False, the field is a scalar field
    grid_only : bool
        If True, this field will never be transformed into coefficient space (i.e. it will always be in grid space)
    ncc_cutoff : float
        The NCC cutoff used.
    """
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


class ConvectionSimStarBuilder:
    """
    An abstract class for building the background stratification needed
    for a convection simulation. The stratification is based on a MESA
    stellar model.
    """

    def __init__(self, profile_file=None, plot=True):
        """
        Initializes the ConvectionSimStarBuilder. Reads in the MESA profile.

        Arguments
        ---------
        profile_file : str
            The path to the MESA profile file to use; if None, uses the path from the config file.
        plot : bool
            Whether to plot comparisons between the MESA profile and the dedalus profile.
        """
        # Read in parameters and create output directory
        self.out_dir, self.out_file = name_star()
        self.ncc_dict = config.nccs
        self.resolutions = [(1, 1, nr) for nr in config.star['nr']]
        self.dealias = config.numerics['N_dealias']
        self.dtype = np.float64
        self.r_bound_nd = None

        # Find the path to the MESA profile file
        package_path = Path(compstar.__file__).resolve().parent
        stock_path = package_path.joinpath('stock_models')
        self.mesa_file_path = None
        if profile_file is not None and os.path.exists(profile_file):
            self.mesa_file_path = profile_file
        elif os.path.exists(config.star['path']):
            self.mesa_file_path = config.star['path']
        else:
            stock_file_path = stock_path.joinpath(config.star['path'])
            if os.path.exists(stock_file_path):
                self.mesa_file_path = str(stock_file_path)
            else:
                raise ValueError("Cannot find MESA profile file in {} or {}".format(config.star['path'], stock_file_path))
        logger.info('Generating stratification associated with MESA profile file: {}'.format(self.mesa_file_path))
        self.reader = DimensionalMesaReader(self.mesa_file_path)

        self._define_cz_bounds()
        self._nondimensionalize()
        self._make_bases()
        self._construct_diffusivities()
        self._interpolate_mesa_fields()
        self._get_Fconv()
        self._get_stability()
        self._hydrostatic_nlbvp()
        self._construct_nccs()
        self._save_star()
        if plot:
            self.plot_nccs()
  
    def _make_bases(self):
        """ Construct the dedalus bases """
        stitch_radii = self.r_bound_nd[1:-1]
        self.coords, self.dist, self.bases, self.bases_keys = make_bases(self.resolutions, stitch_radii, self.r_bound_nd[-1], dealias=(1,1,self.dealias), dtype=self.dtype, mesh=None)
        self.dedalus_r = OrderedDict()
        for bn in self.bases.keys():
            phi, theta, r_vals = self.bases[bn].global_grids((1, 1, self.dealias))
            self.dedalus_r[bn] = r_vals

    def _interpolate_mesa_fields(self):
        """ creates smooth, nondimensional interpolation functions of important MESA fields. """
        nd = SimpleNamespace(**self.nd)
        structure = SimpleNamespace(**self.reader.structure)
        r_nd = structure.r/nd.L_nd
        self.interpolations = OrderedDict()
        if config.numerics['equations'] == 'FC_HD':
            self.interpolations['grad_s0'] = interp1d(r_nd, (structure.cp*structure.grad_s_over_cp*nd.L_nd/nd.s_nd), **interp_kwargs)
            #self.interpolations['s0'] = interp1d(r_nd, np.cumsum(self.interpolations['grad_s0'](r_nd)*np.gradient(r_nd)), **interp_kwargs)
            self.interpolations['ln_rho0'] = interp1d(r_nd, np.log(structure.rho/nd.rho_nd), **interp_kwargs)
            self.interpolations['ln_T0'] = interp1d(r_nd, np.log(structure.T/nd.T_nd), **interp_kwargs)
            self.interpolations['pom0'] = interp1d(r_nd, nd.R*structure.T/nd.T_nd, **interp_kwargs)
            self.interpolations['grad_ln_rho0'] = interp1d(r_nd, structure.dlogrhodr*nd.L_nd, **interp_kwargs)
            self.interpolations['grad_ln_T0'] = interp1d(r_nd, structure.dlogTdr*nd.L_nd, **interp_kwargs)
            self.interpolations['grad_ln_pom0'] = self.interpolations['grad_ln_T0']
            self.interpolations['T0'] = interp1d(r_nd, structure.T/nd.T_nd, **interp_kwargs)
            self.interpolations['nu_diff'] = interp1d(r_nd, structure.nu_diff*(nd.tau_nd/nd.L_nd**2), **interp_kwargs)
            self.interpolations['chi_rad'] = interp1d(r_nd, structure.rad_diff*(nd.tau_nd/nd.L_nd**2), **interp_kwargs)
            self.interpolations['kappa_rad'] = lambda r: self.interpolations['chi_rad'](r)*np.exp(self.interpolations['ln_rho0'](r))*nd.Cp
            self.interpolations['grad_kappa_rad'] = interp1d(r_nd, np.gradient(self.interpolations['kappa_rad'](r_nd), r_nd), **interp_kwargs)
            self.interpolations['g'] = interp1d(r_nd, -structure.g * (nd.tau_nd**2/nd.L_nd), **interp_kwargs)
            self.interpolations['g_phi'] = interp1d(r_nd, structure.g_phi * (nd.tau_nd**2 / nd.L_nd**2), **interp_kwargs)
            self.interpolations['Q'] = interp1d(r_nd, np.gradient(structure.L_conv/nd.lum_nd, r_nd)/(4*np.pi*r_nd**2), **interp_kwargs)
            self.interpolations['s0']  = lambda r: nd.Cp*((np.log(self.interpolations['pom0'](r)) + self.interpolations['ln_rho0'](r))*(1/nd.gamma1) - self.interpolations['ln_rho0'](r))
        else:
            raise ValueError("Specified equation formulation {} not supported".format(config.numerics['equations']))

    def _hydrostatic_nlbvp(self):
        """ 
        Solves for a hydrostatic equilibrium solution using a Dedalus nonlinear boundary value problem solver. 
        This function creates the self.sim_interpolations dictionary, which will be used for NCC construction.
        """
        self.sim_interpolations = deepcopy(self.interpolations)
        nd = SimpleNamespace(**self.nd)
        structure = SimpleNamespace(**self.reader.structure)
        r_nd = structure.r/nd.L_nd
        if config.numerics['equations'] == 'FC_HD':
            atmo = HSE_solve(self.coords, self.dist, self.bases,  self.interpolations['grad_ln_rho0'], self.N2_func, self.F_conv_func,
                            r_outer=self.r_bound_nd[-1], r_stitch=self.r_bound_nd[1:-1], \
                            R=self.nd['R'], gamma=self.nd['gamma1'], nondim_radius=self.nd['r_nd_coord'])
            self.sim_interpolations['ln_rho0']          = atmo['ln_rho']
            self.sim_interpolations['Q']                = atmo['Q']
            self.sim_interpolations['g']                = atmo['g']
            self.sim_interpolations['g_phi']            = atmo['g_phi']
            self.sim_interpolations['grad_s0']          = atmo['grad_s']
            self.sim_interpolations['s0']               = atmo['s0']
            self.sim_interpolations['pom0']             = atmo['pomega']
            self.sim_interpolations['grad_ln_pom0']     = atmo['grad_ln_pomega']
            self.sim_interpolations['nu_diff']          = interp1d(r_nd, structure.sim_nu_diff, **interp_kwargs) 
            self.sim_interpolations['chi_rad']          = interp1d(r_nd, structure.sim_rad_diff, **interp_kwargs)
            self.sim_interpolations['kappa_rad']        = interp1d(r_nd, np.exp(self.sim_interpolations['ln_rho0'](r_nd))*nd.Cp*structure.sim_rad_diff, **interp_kwargs)
            self.sim_interpolations['grad_kappa_rad']   = interp1d(r_nd, np.gradient(self.sim_interpolations['kappa_rad'](r_nd), r_nd), **interp_kwargs)
        else:
            raise ValueError("Specified equation formulation {} not supported".format(config.numerics['equations']))

    def _construct_nccs(self):
        """ Constructs simulation NCCs using Dedalus and interpolations. """
        ## Construct Dedalus NCCs
        for ncc in self.ncc_dict.keys():
            for i, bn in enumerate(self.bases.keys()):
                self.ncc_dict[ncc]['Nmax_{}'.format(bn)] = self.ncc_dict[ncc]['nr_max'][i]
                self.ncc_dict[ncc]['field_{}'.format(bn)] = None
            if ncc in self.sim_interpolations.keys():
                self.ncc_dict[ncc]['interp_func'] = self.sim_interpolations[ncc]
                self.ncc_dict[ncc]['mesa_interp_func'] = self.interpolations[ncc]
            else:
                self.ncc_dict[ncc]['interp_func'] = None
                self.ncc_dict[ncc]['mesa_interp_func'] = None

        #Loop over bases, then loop over the NCCs that need to be built for each basis
        for bn, basis in self.bases.items():
            for ncc in self.ncc_dict.keys():
                interp_func = self.ncc_dict[ncc]['interp_func']
                #If we have an interpolation function, build the NCC from the interpolator, 
                # unless we're using the Dedalus gradient of another field.
                if interp_func is not None and not self.ncc_dict[ncc]['from_grad']:
                    Nmax = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]
                    vector = self.ncc_dict[ncc]['vector']
                    grid_only = self.ncc_dict[ncc]['grid_only']
                    self.ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    if self.ncc_dict[ncc]['get_grad']: #If another NCC needs the gradient of this one, build it
                        name = self.ncc_dict[ncc]['grad_name']
                        logger.info('getting {}'.format(name))
                        grad_field = d3.grad(self.ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                        grad_field.change_scales((1,1,(Nmax+1)/self.resolutions[self.bases_keys == bn][2]))
                        grad_field.change_scales(basis.dealias)
                        self.ncc_dict[name]['field_{}'.format(bn)] = grad_field
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                    if self.ncc_dict[ncc]['get_inverse']: #If another NCC needs the inverse of this one, build it
                        name = 'inv_{}'.format(ncc)
                        inv_func = lambda r: 1/interp_func(r)
                        self.ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, self.coords, self.dist, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                        self.ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax

            # Special case for gravity; we build the NCC from the potential, then take the gradient, which is -g.
            if 'neg_g' in self.ncc_dict.keys():
                if 'g' not in self.ncc_dict.keys():
                    self.ncc_dict['g'] = OrderedDict()
                name = 'g'
                self.ncc_dict['g']['field_{}'.format(bn)] = (-self.ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
                self.ncc_dict['g']['vector'] = True
                self.ncc_dict['g']['interp_func'] = self.sim_interpolations['g']
                self.ncc_dict['g']['mesa_interp_func'] = self.interpolations['g']
                self.ncc_dict['g']['Nmax_{}'.format(bn)] = self.ncc_dict['neg_g']['Nmax_{}'.format(bn)]
                self.ncc_dict['g']['from_grad'] = True 

    def _save_star(self):
        """ Save NCC dictionary, nondimensionalization, and MESA profiles to a file """
        with h5py.File('{:s}'.format(self.out_file), 'w') as f:
            # Save Dedalus fields.
            # slicing preserves dimensionality
            f.create_group('dedalus')
            for bn, basis in self.bases.items():
                f['dedalus/r_{}'.format(bn)] = self.dedalus_r[bn]
                for ncc in self.ncc_dict.keys():
                    this_field = self.ncc_dict[ncc]['field_{}'.format(bn)]
                    if self.ncc_dict[ncc]['vector']:
                        f['dedalus/{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                        f['dedalus/{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
                    else:
                        f['dedalus/{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
                        f['dedalus/{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = self.ncc_dict[ncc]['Nmax_{}'.format(bn)]/self.resolutions[self.bases_keys == bn][2]
        
            # Save nondimensionalization quantities.
            f.create_group('scalars')
            for name, value in self.nd.items():
                if not isinstance(value, u.Quantity):
                    f['scalars/{}'.format(name)] = value
                    f['scalars/{}'.format(name)].attrs['units'] = 'dimensionless'
                else:
                    f['scalars/{}'.format(name)] = value.value
                    f['scalars/{}'.format(name)].attrs['units'] = str(value.unit)
            # Save simulation stitch points
            f['scalars/r_stitch'] = self.r_bound_nd[1:-1]
            f['scalars/r_outer']  = self.r_bound_nd[-1]
            
            # Save MESA profiles.
            f.create_group('mesa')
            for name, value in self.reader.structure.items():
                if not isinstance(value, u.Quantity):
                    f['mesa/{}'.format(name)] = value
                    f['mesa/{}'.format(name)].attrs['units'] = 'dimensionless'
                else:
                    f['mesa/{}'.format(name)] = value.value
                    f['mesa/{}'.format(name)].attrs['units'] = str(value.unit)

        logger.info('finished saving NCCs to {}'.format(self.out_file))
         
    def plot_nccs(self):
        """ Makes plots comparing the NCCs to the MESA model. """
        nd = SimpleNamespace(**self.nd)
        structure = SimpleNamespace(**self.reader.structure)
        r_nd = structure.r/nd.L_nd
        #Plot the NCCs
        for ncc in self.ncc_dict.keys():
            if self.ncc_dict[ncc]['interp_func'] is None:
                continue
            logger.info('plotting {}'.format(ncc))
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
            
            interp_func = self.ncc_dict[ncc]['mesa_interp_func']
            if config.numerics['equations'] == 'FC_HD':
                if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad', 'nu_diff']:
                    log = True
                if ncc == 'grad_s0': 
                    axhline = (nd.s_motions / nd.s_nd)
        
                if ncc in ['grad_T', 'grad_kappa_rad']:
                    this_interp_func = lambda r: -interp_func(r)
                    ylabel='-{}'.format(ncc)
                    for i in range(len(dedalus_yvals)):
                        dedalus_yvals[i] *= -1
                else:
                    ylabel = ncc
                    this_interp_func = interp_func

            plot_ncc_figure(rvals, this_interp_func, dedalus_yvals, Ns=nvals, \
                        ylabel=ylabel, fig_name=ncc, out_dir=self.out_dir, log=log, ylim=ylim, \
                        r_int=self.r_bound_nd[1:-1], axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])
        logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(self.out_dir))
   
    def _define_cz_bounds(self):
        """ Abstract class; must set self.r_bounds. Also defines a boolean array self.cz_bool that is the size of the MESA grid """
        pass

    def _nondimensionalize(self):
        """ Creates and fills the dictionary self.nd which contains scalar quantities used for nondimensionalization with astropy units.
         Also defines self.r_bounds_nd, which is the nondimensionalized radius bounds of the simulation bases. """
        pass
    
    def _construct_diffusivities(self):
        """ Define the diffusivities to use in the simulation. Adds the following keys to self.reader.structure: 'sim_nu_diff', 'sim_rad_diff'"""
        pass
    
    def _get_Fconv(self):
        """ defines a function self.F_conv_func which returns the convective flux at a given nondimensional simulation radius """
        pass
    
    def _get_stability(self):
        """ 
        Defines the stability profile in the simulation.
        If equation formulation is FC_HD, defines the the function self.N2_func
        """
        pass

class MassiveCoreStarBuilder(ConvectionSimStarBuilder):
    """ Builds the NCCs for a massive star's core convection zone and radiative envelope. """

    def __init__(self, *args, grad_s_transition_default=0.03, **kwargs):
        self.grad_s_transition_default = grad_s_transition_default
        super().__init__(*args, **kwargs)

    def _define_cz_bounds(self):
        """ Abstract class; must set self.r_bounds and self.r_bools. 
        Also defines two boolean arrays: self.sim_bool and self.cz_bool that are the size of the MESA grid """
        structure = SimpleNamespace(**self.reader.structure)
        self.core_cz_radius = find_core_cz_radius(self.mesa_file_path, dimensionless=False)
        
        # Specify fraction of total star to simulate
        self.r_bounds = list(config.star['r_bounds'])
        self.r_bools = []
        for i, rb in enumerate(self.r_bounds):
            if type(rb) == str:
                if 'R' in rb:
                    self.r_bounds[i] = float(rb.replace('R', ''))*structure.R_star
                elif 'L' in rb:
                    self.r_bounds[i] = float(rb.replace('L', ''))*self.core_cz_radius
                else:
                    try:
                        self.r_bounds[i] = float(self.r_bounds[i]) * u.cm
                    except:
                        raise ValueError("index {} ('{}') of r_bounds is poorly specified".format(i, rb))
                self.r_bounds[i] = self.core_cz_radius*np.around(self.r_bounds[i]/self.core_cz_radius, decimals=2)
        for i, rb in enumerate(self.r_bounds):
            if i < len(self.r_bounds) - 1:
                self.r_bools.append((structure.r > self.r_bounds[i])*(structure.r <= self.r_bounds[i+1]))
        logger.info('fraction of FULL star simulated: {:.2f}, up to r={:.3e}'.format(self.r_bounds[-1]/structure.R_star, self.r_bounds[-1]))
        self.reader.structure['sim_bool'] = self.sim_bool      = (structure.r > self.r_bounds[0])*(structure.r <= self.r_bounds[-1])
        self.reader.structure['cz_bool'] = self.cz_bool       = (structure.r <= self.core_cz_radius)
        logger.info('fraction of stellar mass simulated: {:.7f}'.format(structure.mass[self.sim_bool][-1]/structure.mass[-1]))

    def _nondimensionalize(self):
        """ Creates and fills the dictionary self.nd which contains scalar quantities used for nondimensionalization with astropy units. """
        structure = SimpleNamespace(**self.reader.structure)
        self.nd = OrderedDict()

        # Get some rough MLT values.
        mlt_u = ((structure.Luminosity / (4 * np.pi * structure.r**2 * structure.rho) )**(1/3)).cgs
        self.nd['avg_core_u'] = avg_core_u = np.sum((4*np.pi*structure.r**2*np.gradient(structure.r)*mlt_u)[structure.r < self.core_cz_radius]) / (4*np.pi*self.core_cz_radius**3 / 3)
        self.nd['avg_core_ma'] = avg_core_ma = np.sum((4*np.pi*structure.r**2*np.gradient(structure.r)*mlt_u/structure.csound)[structure.r < self.core_cz_radius]) / (4*np.pi*self.core_cz_radius**3 / 3)
        logger.info('avg core velocity: {:.3e} / ma: {:.3e}'.format(avg_core_u, avg_core_ma))

        #Get N2 info
        self.nd['N2max_sim']    = N2max_sim = structure.N2[self.sim_bool].max()
        shell_points = np.sum(self.sim_bool*(structure.r > self.core_cz_radius))
        self.nd['N2plateau']    = N2plateau = np.median(structure.N2[structure.r > self.core_cz_radius][int(shell_points*0.25):int(shell_points*0.75)])
        self.nd['f_brunt']      = f_brunt = np.sqrt(N2max_sim)/(2*np.pi)
    
        #Nondimensionalization
        self.nd['L_CZ']         = L_CZ    = self.core_cz_radius
        self.nd['m_core']       = m_core  = structure.rho[0] * L_CZ**3
        self.nd['T_core']       = T_core  = structure.T[0]
        self.nd['H0']           = H0  = (structure.rho*structure.eps_nuc)[0]
        self.nd['tau_heat']     = tau_heat  = ((H0*L_CZ/m_core)**(-1/3)).cgs #heating timescale
        self.nd['L_nd']         = L_nd    = L_CZ
        self.nd['r_nd_coord']   = structure.r[structure.r==L_nd][0]/L_nd
        self.nd['m_nd']         = m_nd    = structure.rho[structure.r==L_nd][0] * L_nd**3 #mass at core cz boundary
        self.nd['T_nd']         = T_nd    = structure.T[structure.r==L_nd][0] #temp at core cz boundary
        self.nd['tau_nd']       = tau_nd  = (1/f_brunt).cgs #timescale of max N^2
        self.nd['rho_nd']       = rho_nd  = m_nd/L_nd**3
        self.nd['u_nd']         = u_nd    = L_nd/tau_nd
        self.nd['s_nd']         = s_nd    = L_nd**2 / tau_nd**2 / T_nd
        self.nd['H_nd']         = H_nd    = (m_nd / L_nd) * tau_nd**-3
        self.nd['s_motions']    = s_motions    = L_nd**2 / tau_heat**2 / structure.T[0]
        self.nd['lum_nd']       = lum_nd  = L_nd**2 * m_nd / (tau_nd**2) / tau_nd
        self.nd['R']            = nondim_R_gas = (constants.R.cgs / structure.mu[0] / s_nd).cgs.value
        self.nd['gamma1']       = nondim_gamma1 = (structure.gamma1[0]).value
        self.nd['Cp']           = nondim_cp = nondim_R_gas * nondim_gamma1 / (nondim_gamma1 - 1)
        self.nd['u_heat_nd']    = u_heat_nd = (L_nd/tau_heat) / u_nd
        self.nd['Ma2_r0']       = Ma2_r0 = ((u_nd*(tau_nd/tau_heat))**2 / ((structure.gamma1[0]-1)*structure.cp[0]*structure.T[0])).cgs
        logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
        logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(nondim_cp, nondim_R_gas, nondim_gamma1))
        logger.info('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))
        logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(Ma2_r0), tau_heat))
        
        # Get some timestepping & wave frequency info
        self.nd['f_nyq'] = f_nyq = 2*tau_nd*np.sqrt(N2max_sim)/(2*np.pi)
        self.nd['nyq_dt'] = nyq_dt   = (1/f_nyq) 
        self.nd['kepler_tau'] = kepler_tau     = 30*60*u.s
        self.nd['max_dt_kepler'] = max_dt_kepler  = kepler_tau/tau_nd
        self.nd['max_dt'] = max_dt_kepler
        logger.info('needed nyq_dt is {} s / {} % of a nondimensional time (Kepler 30 min is {} %) '.format(nyq_dt*tau_nd, nyq_dt*100, max_dt_kepler*100))
        
        #MESA radial values at simulation joints & across full star in simulation units
        self.r_bound_nd = [(rb/L_nd).value for rb in self.r_bounds]
        self.r_nd = (structure.r/L_nd).cgs

        self.grad_s_width = self.grad_s_transition_default
        self.grad_s_width *= (L_CZ/L_nd).value
        self.grad_s_transition_point = self.r_bound_nd[1] - self.grad_s_width
        logger.info('using grad s transition point = {}'.format(self.grad_s_transition_point))
        logger.info('using grad s width = {}'.format(self.grad_s_width))

    def _interpolate_mesa_fields(self):
        #Adjust g_phi constant
        super()._interpolate_mesa_fields()
        structure = SimpleNamespace(**self.reader.structure)
        nd = SimpleNamespace(**self.nd)
        r_nd = structure.r/nd.L_nd
        self.interpolations['g_phi'] = interp1d(r_nd, (structure.g_phi - structure.g_phi[r_nd > self.r_bound_nd[-1]][0])* (nd.tau_nd**2 / nd.L_nd**2) , **interp_kwargs)

    def _construct_diffusivities(self):
        """ Define the diffusivities to use in the simulation. Adds the following keys to self.reader.structure: 'sim_nu_diff', 'sim_rad_diff'"""
        #construct diffusivity profiles which will be used in simulation.
        structure = SimpleNamespace(**self.reader.structure)
        nd = SimpleNamespace(**self.nd)
        rad_diff_nd = structure.rad_diff * (nd.tau_nd / nd.L_nd**2)
        self.nd['rad_diff_cutoff'] = rad_diff_cutoff = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((nd.L_CZ**2/nd.tau_heat) / (nd.L_nd**2/nd.tau_nd))
        self.reader.structure['sim_rad_diff'] = sim_rad_diff = np.copy(rad_diff_nd) + rad_diff_cutoff
        self.reader.structure['sim_nu_diff'] = sim_nu_diff = config.numerics['prandtl']*rad_diff_cutoff*np.ones_like(sim_rad_diff)
        logger.info('rad_diff cutoff: {:.3e}'.format(rad_diff_cutoff))
        logger.info('rad_diff cutoff (dimensional): {:.3e}'.format(rad_diff_cutoff * (nd.L_nd**2/nd.tau_nd)))
    
    def _get_Fconv(self):
        """ defines a function self.F_conv_func which returns the convective flux at a given nondimensional simulation radius """
        structure = SimpleNamespace(**self.reader.structure)
        nd = SimpleNamespace(**self.nd)
        # Construct convective flux function which determines how convection is driven
        if config.star['smooth_h']:
            #smooth CZ-RZ transition
            L_conv_sim = np.copy(structure.L_conv)
            L_conv_sim *= one_to_zero(structure.r, 0.9*self.core_cz_radius, width=0.05*self.core_cz_radius)
            L_conv_sim *= one_to_zero(structure.r, 0.95*self.core_cz_radius, width=0.05*self.core_cz_radius)
            L_conv_sim /= (structure.r/nd.L_nd)**2 * (4*np.pi)
            self.F_conv_func = interp1d(structure.r/nd.L_nd, L_conv_sim/nd.lum_nd, **interp_kwargs)
        else:
            raise NotImplementedError("must use smooth_h")
    
    def _get_stability(self):
        """ 
        Defines the stability profile in the simulation.
        If equation formulation is FC_HD, defines the the function self.N2_func
        """
        structure = SimpleNamespace(**self.reader.structure)
        nd = SimpleNamespace(**self.nd)
    
        #Build a nice function for our basis in the ball
        #have N^2 = A*r^2 + B; grad_N2 = 2 * A * r, so A = (grad_N2) / (2 * r_stitch) & B = stitch_value - A*r_stitch^2
        stitch_point = self.bases['B'].radius
        stitch_value = np.interp(stitch_point, structure.r/nd.L_nd, structure.N2)
        grad_N2_stitch = np.gradient(structure.N2, structure.r)[structure.r/nd.L_nd < stitch_point][-1]
        A = grad_N2_stitch / (2*self.bases['B'].radius * nd.L_nd)
        B = stitch_value - A* (self.bases['B'].radius * nd.L_nd)**2
        smooth_N2 = np.copy(structure.N2)
        smooth_N2[structure.r/nd.L_nd < stitch_point] = A*(structure.r[structure.r/nd.L_nd < stitch_point])**2 + B
        smooth_N2 *= zero_to_one(structure.r/nd.L_nd, self.grad_s_transition_point, width=self.grad_s_width)

        # Solve for hydrostatic equilibrium for background
        self.N2_func = interp1d(structure.r/nd.L_nd, nd.tau_nd**2 * smooth_N2, **interp_kwargs)

    def _construct_nccs(self):
        """ Constructs simulation NCCs using Dedalus and interpolations. """
        super()._construct_nccs()
        #Force more zeros in the CZ if requested. Uses all available coefficients in the expansion of grad s0.
        if config.star['reapply_grad_s_filter']:
            for bn, basis in self.bases.items():
                self.ncc_dict['grad_s0']['field_{}'.format(bn)]['g'] *= zero_to_one(self.dedalus_r[bn], self.grad_s_transition_point-5*self.grad_s_width, width=self.grad_s_width)
                self.ncc_dict['grad_s0']['field_{}'.format(bn)]['c'] *= 1
                self.ncc_dict['grad_s0']['field_{}'.format(bn)]['g'] 

def build_nccs(plot_nccs=True, grad_s_transition_default=0.03, reapply_grad_s_filter=False):
    builder = MassiveCoreStarBuilder(plot=plot_nccs, grad_s_transition_default=grad_s_transition_default)


def build_nccs_old(plot_nccs=False, grad_s_transition_default=0.03, reapply_grad_s_filter=False):
    """
    This function builds the NCCs for the star, then saves them to a file.
    TODO: This function is a bit of a mess, and should be cleaned up. It should be turned into a class.
    TODO: Generalize this function; it is currently specific to the massive star case.

    Arguments
    ---------
    plot_nccs : bool
        Whether to plot the NCCs
    grad_s_transition_default : float
        The default value for how far the entropy gradient transition should be away from the BallBasis outer boundary
    reapply_grad_s_filter : bool
        Whether to reapply the zero-to-one filter after expanding the entropy gradient field.
    """
    # Read in parameters and create output directory
    out_dir, out_file = name_star()
    ncc_dict = config.nccs

    # Find the path to the MESA profile file
    package_path = Path(compstar.__file__).resolve().parent
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

    # Read in the MESA profile
    reader = DimensionalMesaReader(mesa_file_path)
    dmr = SimpleNamespace(**reader.structure) # Turns the dictionary into a namespace so that fields can be accessed as attributes
    #make some commonly-used variables local.
    r, mass, rho, T = dmr.r, dmr.mass, dmr.rho, dmr.T
    N2, g, cp = dmr.N2, dmr.g, dmr.cp

    ### CORE CONVECTION LOGIC - lots of stuff here needs to be generalized for other types of stars.
    core_cz_radius = find_core_cz_radius(mesa_file_path, dimensionless=False)

    # Get some rough MLT values.
    mlt_u = ((dmr.Luminosity / (4 * np.pi * r**2 * rho) )**(1/3)).cgs
    avg_core_u = np.sum((4*np.pi*r**2*np.gradient(r)*mlt_u)[r < core_cz_radius]) / (4*np.pi*core_cz_radius**3 / 3)
    avg_core_ma = np.sum((4*np.pi*r**2*np.gradient(r)*mlt_u/dmr.csound)[r < core_cz_radius]) / (4*np.pi*core_cz_radius**3 / 3)
    logger.info('avg core velocity: {:.3e} / ma: {:.3e}'.format(avg_core_u, avg_core_ma))

    # Specify fraction of total star to simulate
    r_bounds = list(config.star['r_bounds'])
    r_bools = []
    for i, rb in enumerate(r_bounds):
        if type(rb) == str:
            if 'R' in rb:
                r_bounds[i] = float(rb.replace('R', ''))*dmr.R_star
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
    logger.info('fraction of FULL star simulated: {:.2f}, up to r={:.3e}'.format(r_bounds[-1]/dmr.R_star, r_bounds[-1]))
    sim_bool      = (r > r_bounds[0])*(r <= r_bounds[-1])
    logger.info('fraction of stellar mass simulated: {:.7f}'.format(mass[sim_bool][-1]/mass[-1]))

    #Get N2 info
    N2max_sim = N2[sim_bool].max()
    shell_points = np.sum(sim_bool*(r > core_cz_radius))
    N2plateau = np.median(N2[r > core_cz_radius][int(shell_points*0.25):int(shell_points*0.75)])
    f_brunt = np.sqrt(N2max_sim)/(2*np.pi)
 
    #Nondimensionalization
    L_CZ    = core_cz_radius
    m_core  = rho[0] * L_CZ**3
    T_core  = T[0]
    H0      = (rho*dmr.eps_nuc)[0]
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
    lum_nd  = L_nd**2 * m_nd / (tau_nd**2) / tau_nd
    nondim_R_gas = (dmr.R_gas / s_nd).cgs.value
    nondim_gamma1 = (dmr.gamma1[0]).value
    nondim_cp = nondim_R_gas * nondim_gamma1 / (nondim_gamma1 - 1)
    u_heat_nd = (L_nd/tau_heat) / u_nd
    Ma2_r0 = ((u_nd*(tau_nd/tau_heat))**2 / ((dmr.gamma1[0]-1)*cp[0]*T[0])).cgs
    logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
    logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(nondim_cp, nondim_R_gas, nondim_gamma1))
    logger.info('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))
    logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(Ma2_r0), tau_heat))

    #Gravitational potential, set to -1 at r = R_star
    g_phi = np.cumsum(g*np.gradient(r))  #gvec = -grad phi; 
    g_phi -= g_phi[-1] - u_nd**2 #set g_phi = -1 at r = dmr.R_star
    
    #construct diffusivity profiles which will be used in simulation.
    rad_diff_nd = dmr.rad_diff * (tau_nd / L_nd**2)
    rad_diff_cutoff = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((L_CZ**2/tau_heat) / (L_nd**2/tau_nd))
    sim_rad_diff = np.copy(rad_diff_nd) + rad_diff_cutoff
    sim_nu_diff = config.numerics['prandtl']*rad_diff_cutoff*np.ones_like(sim_rad_diff)
    Re_shift = ((L_nd**2/tau_nd) / (L_CZ**2/tau_heat))

    logger.info('u_heat_nd: {:.3e}'.format(u_heat_nd))
    logger.info('rad_diff cutoff: {:.3e}'.format(rad_diff_cutoff))
    logger.info('rad_diff cutoff: {:.3e}'.format(rad_diff_cutoff * (L_nd**2/tau_nd)))
    
    #MESA radial values at simulation joints & across full star in simulation units
    r_bound_nd = [(rb/L_nd).value for rb in r_bounds]
    r_nd = (r/L_nd).cgs
    
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

    # Construct convective flux function which determines how convection is driven
    if config.star['smooth_h']:
        #smooth CZ-RZ transition
        L_conv_sim = np.copy(dmr.L_conv)
        L_conv_sim *= one_to_zero(r, 0.9*core_cz_radius, width=0.05*core_cz_radius)
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)
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
    interpolations['grad_ln_rho0'] = interp1d(r_nd, dmr.dlogrhodr*L_nd, **interp_kwargs)
    interpolations['grad_ln_T0'] = interp1d(r_nd, dmr.dlogTdr*L_nd, **interp_kwargs)
    interpolations['T0'] = interp1d(r_nd, T/T_nd, **interp_kwargs)
    interpolations['nu_diff'] = interp1d(r_nd, sim_nu_diff, **interp_kwargs)
    interpolations['chi_rad'] = interp1d(r_nd, sim_rad_diff, **interp_kwargs)
    interpolations['grad_chi_rad'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd), **interp_kwargs)
    interpolations['g'] = interp1d(r_nd, -g * (tau_nd**2/L_nd), **interp_kwargs)
    interpolations['g_phi'] = interp1d(r_nd, g_phi * (tau_nd**2 / L_nd**2), **interp_kwargs)

    # construct N2 function 
    # TODO: I think some of this logic is happening inside the BVP; make sure it's all together.
    ### More core convection zone logic here
    grad_s_width = grad_s_transition_default
    grad_s_width *= (L_CZ/L_nd).value
    grad_s_transition_point = r_bound_nd[1] - grad_s_width
    logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
    logger.info('using default grad s width = {}'.format(grad_s_width))
 
    #Build a nice function for our basis in the ball
    #have N^2 = A*r^2 + B; grad_N2 = 2 * A * r, so A = (grad_N2) / (2 * r_stitch) & B = stitch_value - A*r_stitch^2
    stitch_point = 1
    stitch_point = bases['B'].radius
    stitch_value = np.interp(stitch_point, r/L_nd, N2)
    grad_N2_stitch = np.gradient(N2, r)[r/L_nd < stitch_point][-1]
    A = grad_N2_stitch / (2*bases['B'].radius * L_nd)
    B = stitch_value - A* (bases['B'].radius * L_nd)**2
    smooth_N2 = np.copy(N2)
    smooth_N2[r/L_nd < stitch_point] = A*(r[r/L_nd < stitch_point])**2 + B
    smooth_N2 *= zero_to_one(r/L_nd, grad_s_transition_point, width=grad_s_width)

    # Solve for hydrostatic equilibrium for background
    N2_func = interp1d(r_nd, tau_nd**2 * smooth_N2, **interp_kwargs)
    grad_ln_rho_func = interpolations['grad_ln_rho0']
    atmo = HSE_solve(c, d, bases,  grad_ln_rho_func, N2_func, F_conv_func,
              r_outer=r_bound_nd[-1], r_stitch=stitch_radii, \
              R=nondim_R_gas, gamma=nondim_gamma1, nondim_radius=1)

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

    ## Construct Dedalus NCCs
    for ncc in ncc_dict.keys():
        for i, bn in enumerate(bases.keys()):
            ncc_dict[ncc]['Nmax_{}'.format(bn)] = ncc_dict[ncc]['nr_max'][i]
            ncc_dict[ncc]['field_{}'.format(bn)] = None
        if ncc in interpolations.keys():
            ncc_dict[ncc]['interp_func'] = interpolations[ncc]
        else:
            ncc_dict[ncc]['interp_func'] = None

    #Loop over bases, then loop over the NCCs that need to be built for each basis
    for bn, basis in bases.items():
        rvals = dedalus_r[bn]
        for ncc in ncc_dict.keys():
            interp_func = ncc_dict[ncc]['interp_func']
            #If we have an interpolation function, build the NCC from the interpolator, 
            # unless we're using the Dedalus gradient of another field.
            if interp_func is not None and not ncc_dict[ncc]['from_grad']:
                Nmax = ncc_dict[ncc]['Nmax_{}'.format(bn)]
                vector = ncc_dict[ncc]['vector']
                grid_only = ncc_dict[ncc]['grid_only']
                ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, c, d, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                if ncc_dict[ncc]['get_grad']: #If another NCC needs the gradient of this one, build it
                    name = ncc_dict[ncc]['grad_name']
                    logger.info('getting {}'.format(name))
                    grad_field = d3.grad(ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                    grad_field.change_scales((1,1,(Nmax+1)/resolutions[bases_keys == bn][2]))
                    grad_field.change_scales(basis.dealias)
                    ncc_dict[name]['field_{}'.format(bn)] = grad_field
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                if ncc_dict[ncc]['get_inverse']: #If another NCC needs the inverse of this one, build it
                    name = 'inv_{}'.format(ncc)
                    inv_func = lambda r: 1/interp_func(r)
                    ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, c, d, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax

        # Special case for gravity; we build the NCC from the potential, then take the gradient, which is -g.
        if 'neg_g' in ncc_dict.keys():
            if 'g' not in ncc_dict.keys():
                ncc_dict['g'] = OrderedDict()
            name = 'g'
            ncc_dict['g']['field_{}'.format(bn)] = (-ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
            ncc_dict['g']['vector'] = True
            ncc_dict['g']['interp_func'] = interpolations['g']
            ncc_dict['g']['Nmax_{}'.format(bn)] = ncc_dict['neg_g']['Nmax_{}'.format(bn)]
            ncc_dict['g']['from_grad'] = True 
    
    #Force more zeros in the CZ if requested. Uses all available coefficients in the expansion of grad s0.
    if reapply_grad_s_filter:
        for bn, basis in bases.items():
            ncc_dict['grad_s0']['field_{}'.format(bn)]['g'] *= zero_to_one(dedalus_r[bn], grad_s_transition_point-5*grad_s_width, width=grad_s_width)
            ncc_dict['grad_s0']['field_{}'.format(bn)]['c'] *= 1
            ncc_dict['grad_s0']['field_{}'.format(bn)]['g']

    #reset ln_rho and ln_T interpolations for nice plots
    interpolations['ln_rho0'] = interp1d(r_nd, np.log(rho/rho_nd), **interp_kwargs)
    interpolations['ln_T0'] = interp1d(r_nd, np.log(T/T_nd), **interp_kwargs)

    #Fixup heating term to make simulation energy-neutral.       
    integral = 0
    for bn in bases.keys():
        integral += d3.integ(ncc_dict['Q']['field_{}'.format(bn)])
    C = integral.evaluate()['g']
    vol = (4/3) * np.pi * (r_bound_nd[-1])**3
    adj = C / vol
    logger.info('adjusting dLdt for energy conservation; subtracting {} from H'.format(adj))
    for bn in bases.keys():
        ncc_dict['Q']['field_{}'.format(bn)]['g'] -= adj 

    if plot_nccs:
        #Plot the NCCs
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
    
            interp_func = ncc_dict[ncc]['interp_func']
            if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad']:
                log = True
            if ncc == 'grad_s0': 
                axhline = (s_motions / s_nd)
            elif ncc in ['chi_rad', 'grad_chi_rad']:
                if ncc == 'chi_rad':
                    interp_func = interp1d(r_nd, (L_nd**2/tau_nd).value*rad_diff_nd, **interp_kwargs)
                    for ind in range(len(dedalus_yvals)):
                        dedalus_yvals[ind] *= (L_nd**2/tau_nd).value
                axhline = rad_diff_cutoff*(L_nd**2/tau_nd).value
    
            if ncc == 'grad_s0':
                interp_func = interp1d(r_nd, (L_nd/s_nd) * dmr.grad_s, **interp_kwargs)
            elif ncc in ['ln_T0', 'ln_rho0', 'grad_s0']:
                interp_func = interpolations[ncc]
    
            if ncc in ['grad_T', 'grad_kappa_rad']:
                interp_func = lambda r: -ncc_dict[ncc]['interp_func'](r)
                ylabel='-{}'.format(ncc)
                for i in range(len(dedalus_yvals)):
                    dedalus_yvals[i] *= -1
            elif ncc == 'chi_rad':
                ylabel = 'radiative diffusivity (cm^2/s)'
            else:
                ylabel = ncc

    
            plot_ncc_figure(rvals, interp_func, dedalus_yvals, Ns=nvals, \
                        ylabel=ylabel, fig_name=ncc, out_dir=out_dir, log=log, ylim=ylim, \
                        r_int=stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

    

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
        f['P_r0']  = dmr.P[0]
        f['P_r0'].attrs['units']  = str(dmr.P[0].unit)
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
        f['S1_mesa'] = dmr.lamb_freq(1)
        f['S1_mesa'].attrs['units'] = str(dmr.lamb_freq(1).unit)
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

    #Make some plots of stratification, hydrostatic equilibrium, etc.
    logger.info('Making final plots...')
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
    plt.plot(r_nd, tau_nd**2*N2, label='mesa', c='k')
    plt.plot(r_nd, -tau_nd**2*N2, c='k', ls='--')
#    plt.plot(r_nd, atmo['N2'](r_nd), label='atmosphere', c='b')
#    plt.plot(r_nd, -atmo['N2'](r_nd), c='b', ls='--')
    plt.plot(r_dedalus, N2_dedalus, label='dedalus', c='g')
    plt.plot(r_dedalus, -N2_dedalus, ls='--', c='g')
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