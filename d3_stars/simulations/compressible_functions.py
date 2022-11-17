from collections import OrderedDict

import h5py
import numpy as np
import dedalus.public as d3
from dedalus.core.operators import convert 

import logging
logger = logging.getLogger(__name__)

import d3_stars.defaults.config as config

def make_bases(resolutions, stitch_radii, radius, r_inner=0, dealias=3/2, dtype=np.float64, mesh=None):
    """ Makes a stitched together ball and shell domains """
    bases = OrderedDict()
    coords  = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
    bases_keys = []
    for i, resolution in enumerate(resolutions):
        if i == 0 and r_inner == 0:
            if len(resolutions) == 1:
                ball_radius = radius
            else:
                ball_radius = stitch_radii[i]
            bases['B']   = d3.BallBasis(coords, resolution, radius=ball_radius, dtype=dtype, dealias=dealias)
            bases_keys += ['B']
        else:
            if len(resolutions) == 1 and i == 0:
                shell_radii = (r_inner, radius)
            elif i == 0:
                shell_radii = (r_inner, stitch_radii[i])
            elif len(resolutions) == i+1:
                shell_radii = (stitch_radii[i-1], radius)
            else:
                shell_radii = (stitch_radii[i-1], stitch_radii[i])
            bases['S{}'.format(i)] = d3.ShellBasis(coords, resolution, radii=shell_radii, dtype=dtype, dealias=dealias)
            bases_keys += ['S{}'.format(i)]
    return coords, dist, bases, bases_keys

class SphericalCompressibleProblem():

    def __init__(self, resolutions, stitch_radii, radius, ncc_file, r_inner=0, dealias=3/2, dtype=np.float64, mesh=None, sponge=False, do_rotation=False, sponge_function=lambda r: r**2):
        self.stitch_radii = stitch_radii
        self.r_inner = r_inner
        self.radius = radius
        self.ncc_file = ncc_file
        self.sponge = sponge
        self.do_rotation = do_rotation
        self.sponge_function = sponge_function
        self.fields_filled = False

        coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, r_inner=r_inner, dealias=dealias, dtype=dtype, mesh=mesh)
        self.coords = coords
        self.dist = dist
        self.bases = bases
        self.bases_keys = bases_keys


        self.vec_fields = ['u', ]
        self.scalar_fields = ['ln_rho1', 's1', 'Q', 'ones'] 
        self.vec_taus = ['tau_u']
        self.scalar_taus =  ['tau_s']
        self.vec_nccs = ['grad_pom0', 'grad_ln_pom0', 'grad_ln_rho0', 'grad_s0', 'g', 'rvec', 'grad_nu_diff', 'grad_chi_rad', 'grad_kappa_rad', 'grad_T0_superad']
        self.scalar_nccs = ['pom0', 'rho0', 'ln_rho0', 'g_phi', 'nu_diff', 'kappa_rad', 'chi_rad', 's0', 'inv_pom0', 'L_heat']
        self.sphere_unit_vectors = ['ephi', 'etheta', 'er']
        self.cartesian_unit_vectors = ['ex', 'ey', 'ez']

        self.namespace = OrderedDict()

    def make_fields(self, vec_fields=[], scalar_fields=[], vec_nccs=[], scalar_nccs=[]):
        self.vec_fields += vec_fields
        self.scalar_fields += scalar_fields
        self.vec_nccs += vec_nccs
        self.scalar_nccs += scalar_nccs
        if self.sponge:
            self.scalar_nccs += ['sponge']

        self.namespace = OrderedDict()
        self.namespace['exp'] = np.exp
        self.namespace['log'] = np.log
        self.namespace['Grid'] = d3.Grid
        self.namespace['dt'] = dt =  d3.TimeDerivative
        one = self.dist.Field(name='one')
        one['g'] = 1
        self.namespace['one'] = one = d3.Grid(one)

        for basis_number, bn in enumerate(self.bases.keys()):
            basis = self.bases[bn]
            phi, theta, r = basis.local_grids(basis.dealias)
            phi1, theta1, r1 = basis.local_grids((1,1,1))
            self.namespace['phi_'+bn], self.namespace['theta_'+bn], self.namespace['r_'+bn] = phi, theta, r
            self.namespace['phi1_'+bn], self.namespace['theta1_'+bn], self.namespace['r1_'+bn] = phi1, theta1, r1

            #Define problem fields
            for fn in self.vec_fields:
                key = '{}_{}'.format(fn, bn)
                logger.debug('creating vector field {}'.format(key))
                self.namespace[key] = self.dist.VectorField(self.coords, name=key, bases=basis)
            for fn in self.scalar_fields:
                key = '{}_{}'.format(fn, bn)
                logger.debug('creating scalar field {}'.format(key))
                self.namespace[key] = self.dist.Field(name=key, bases=basis)

            #Taus
            S2_basis = basis.S2_basis()
            tau_names = []
            if type(basis) == d3.BallBasis:
                tau_names += ['',]
            else:
                tau_names += ['1', '2']
            for name in tau_names:
                for fn in self.vec_taus:
                    key = '{}{}_{}'.format(fn, name, bn)
                    logger.debug('creating vector tau {}'.format(key))
                    self.namespace[key] = self.dist.VectorField(self.coords, name=key, bases=S2_basis)
                for fn in self.scalar_taus:
                    key = '{}{}_{}'.format(fn, name, bn)
                    logger.debug('creating scalar tau {}'.format(key))
                    self.namespace[key] = self.dist.Field(name=key, bases=S2_basis)
            
            #Define problem NCCs
            for fn in self.vec_nccs:
                key = '{}_{}'.format(fn, bn)
                logger.debug('creating vector NCC {}'.format(key))
                self.namespace[key] = self.dist.VectorField(self.coords, name=key, bases=basis.radial_basis)
            self.namespace['rvec_{}'.format(bn)]['g'][2] = r1
            for fn in self.scalar_nccs:
                key = '{}_{}'.format(fn, bn)
                logger.debug('creating scalar NCC {}'.format(key))
                self.namespace[key] = self.dist.Field(name=key, bases=basis.radial_basis)

            #Define identity matrix
            logger.debug('creating identity matrix')
            self.namespace['eye_{}'.format(bn)] = self.dist.TensorField(self.coords, name='eye', bases=basis.radial_basis)
            for i in range(3):
                self.namespace['eye_{}'.format(bn)]['g'][i,i] = 1

            if self.sponge:
                self.namespace['sponge_{}'.format(bn)]['g'] = self.sponge_function(r1)

            #Define unit vectors
            for fn in self.sphere_unit_vectors:
                logger.debug('creating unit vector field {}'.format(key))
                self.namespace[fn] = self.dist.VectorField(self.coords, name=fn)
            self.namespace['er']['g'][2] = 1
            self.namespace['ephi']['g'][0] = 1
            self.namespace['etheta']['g'][1] = 1

            for fn in self.cartesian_unit_vectors:
                key = '{}_{}'.format(fn, bn)
                logger.debug('creating unit vector field {}'.format(key))
                self.namespace[key] = self.dist.VectorField(self.coords, name=key, bases=basis)


            self.namespace['ex_{}'.format(bn)]['g'][0] = -np.sin(phi1)
            self.namespace['ex_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.cos(phi1)
            self.namespace['ex_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.cos(phi1)

            self.namespace['ey_{}'.format(bn)]['g'][0] = np.cos(phi1)
            self.namespace['ey_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.sin(phi1)
            self.namespace['ey_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.sin(phi1)

            self.namespace['ez_{}'.format(bn)]['g'][0] = 0
            self.namespace['ez_{}'.format(bn)]['g'][1] = -np.sin(theta1)
            self.namespace['ez_{}'.format(bn)]['g'][2] =  np.cos(theta1)

            if basis_number == 0:
                self.namespace['gamma'] = gamma = self.dist.Field(name='gamma')
                self.namespace['R_gas'] = R_gas = self.dist.Field(name='R_gas')
                self.namespace['Cp'] = Cp = self.dist.Field(name='Cp')
                self.namespace['Cv'] = Cv = self.dist.Field(name='Cv')
            
            for k in ['ex', 'ey', 'ez']:
                self.namespace['{}_{}'.format(k, bn)] = d3.Grid(self.namespace['{}_{}'.format(k, bn)]).evaluate()
        return self.namespace

    def set_substitutions(self, EVP=False):
        """ Sets problem substitutions; must be run after self.fill_fields() """

        if not self.fields_filled:
            raise ValueError("Must fill fields before setting substitutions")

        for basis_number, bn in enumerate(self.bases.keys()):
            basis = self.bases[bn]

            #Grab various important fields
            u = self.namespace['u_{}'.format(bn)]
            ln_rho1 = self.namespace['ln_rho1_{}'.format(bn)]
            s1 = self.namespace['s1_{}'.format(bn)]
            s0 = self.namespace['s0_{}'.format(bn)]
            Q = self.namespace['Q_{}'.format(bn)]
            eye = self.namespace['eye_{}'.format(bn)]
            grad_ln_rho0 = self.namespace['grad_ln_rho0_{}'.format(bn)]
            grad_ln_pom0 = self.namespace['grad_ln_pom0_{}'.format(bn)]
            grad_pom0 = self.namespace['grad_pom0_{}'.format(bn)]
            rho0 = self.namespace['rho0_{}'.format(bn)]
            ln_rho0 = self.namespace['ln_rho0_{}'.format(bn)]
            pom0 = self.namespace['pom0_{}'.format(bn)]
            grad_s0 = self.namespace['grad_s0_{}'.format(bn)]
            g_phi = self.namespace['g_phi_{}'.format(bn)]
            gravity = self.namespace['g_{}'.format(bn)]
            nu_diff = self.namespace['nu_diff_{}'.format(bn)]
            grad_nu_diff = self.namespace['grad_nu_diff_{}'.format(bn)]
            chi_rad = self.namespace['chi_rad_{}'.format(bn)]
            grad_chi_rad = self.namespace['grad_chi_rad_{}'.format(bn)]
            kappa_rad = self.namespace['kappa_rad_{}'.format(bn)]
            grad_kappa_rad = self.namespace['grad_kappa_rad_{}'.format(bn)]
            inv_pom0 = self.namespace['inv_pom0_{}'.format(bn)]
            g = self.namespace['g_{}'.format(bn)]
            er = self.namespace['er']
            ephi = self.namespace['ephi']
            etheta = self.namespace['etheta']
            rvec = self.namespace['rvec_{}'.format(bn)]
            Cp = self.namespace['Cp']


            self.namespace['P0_{}'.format(bn)] = P0 = (rho0*pom0)

            #Make a 'ones' field for broadcasting scalars or radial fields to full basis domain
            ones = self.namespace['ones_{}'.format(bn)]
            ones['g'] = 1
            self.namespace['ones_{}'.format(bn)] = ones = d3.Grid(ones).evaluate()

            # Lift operators for boundary conditions
            self.namespace['grad_u_{}'.format(bn)] = grad_u = d3.grad(u)
            self.namespace['div_u_{}'.format(bn)] = div_u = d3.div(u)
            self.namespace['grad_s1_{}'.format(bn)] = grad_s1 = d3.grad(s1)
            self.namespace['grad_ln_rho1_{}'.format(bn)] = grad_ln_rho1 = d3.grad(ln_rho1)
            if type(basis) == d3.BallBasis:
                lift_basis = basis.derivative_basis(0)
                self.namespace['lift_{}'.format(bn)] = lift_fn = lambda A: d3.Lift(A, lift_basis, -1)
                self.namespace['taus_lnrho_{}'.format(bn)] = taus_lnrho = 0
                self.namespace['taus_u_{}'.format(bn)] = taus_u = lift_fn(self.namespace['tau_u_{}'.format(bn)])
                self.namespace['taus_s_{}'.format(bn)] = taus_s = lift_fn(self.namespace['tau_s_{}'.format(bn)])
            else:
                lift_basis = basis.derivative_basis(2)
                self.namespace['lift_{}'.format(bn)] = lift_fn = lambda A, n: d3.Lift(A, lift_basis, n)
                self.namespace['taus_lnrho_{}'.format(bn)] = taus_lnrho = (1/nu_diff)*rvec@lift_fn(self.namespace['tau_u2_{}'.format(bn)], -1)
                self.namespace['taus_u_{}'.format(bn)] = taus_u = lift_fn(self.namespace['tau_u1_{}'.format(bn)], -1) + lift_fn(self.namespace['tau_u2_{}'.format(bn)], -2)
                self.namespace['taus_s_{}'.format(bn)] = taus_s = lift_fn(self.namespace['tau_s1_{}'.format(bn)], -1) + lift_fn(self.namespace['tau_s2_{}'.format(bn)], -2)

            #Lock a bunch of fields onto the grid.

            #These fields are coming to the grid:
            grid_u = d3.Grid(u)
            grid_ln_rho1 = d3.Grid(ln_rho1)
            grid_s1 = d3.Grid(s1)
            grid_grad_u = d3.Grid(grad_u)
            grid_grad_ln_rho1 = d3.Grid(grad_ln_rho1)
            grid_grad_s1 = d3.Grid(grad_s1)
            grid_lap_ln_rho1 = d3.Grid(d3.lap(ln_rho1))
            grid_lap_s1 = d3.Grid(d3.lap(s1))
            grid_div_u = d3.Grid(div_u)

            #radial -> full domain; ***THIS SHOULD BE DONE AFTER FILLING FIELDS***
            self.namespace['grid_rho0_{}'.format(bn)] = grid_rho0 = d3.Grid(ones*rho0).evaluate()
            self.namespace['grid_ln_rho0_{}'.format(bn)] = grid_ln_rho0 = d3.Grid(ones*ln_rho0).evaluate()
            self.namespace['grid_grad_ln_rho0_{}'.format(bn)] = grid_grad_ln_rho0 = d3.Grid(ones*grad_ln_rho0).evaluate()
            self.namespace['grid_s0_{}'.format(bn)] = grid_s0 = d3.Grid(ones*s0).evaluate()
            self.namespace['grid_grad_s0_{}'.format(bn)] = grid_grad_s0 = d3.Grid(ones*grad_s0).evaluate()
            self.namespace['grid_pom0_{}'.format(bn)] = grid_pom0 = d3.Grid(ones*pom0).evaluate()
            self.namespace['grid_grad_pom0_{}'.format(bn)] = grid_grad_pom0 = d3.Grid(ones*grad_pom0).evaluate()
            self.namespace['grid_g_{}'.format(bn)] = grid_g = d3.Grid(g).evaluate()
            self.namespace['grid_chi_rad_{}'.format(bn)] = grid_chi_rad = d3.Grid(chi_rad*ones).evaluate()
            self.namespace['grid_grad_chi_rad_{}'.format(bn)] = grid_grad_chi_rad = d3.Grid(ones*grad_chi_rad).evaluate()
            self.namespace['grid_kappa_rad_{}'.format(bn)] = grid_kappa_rad = d3.Grid(kappa_rad*ones).evaluate()
            self.namespace['grid_grad_kappa_rad_{}'.format(bn)] = grid_grad_kappa_rad = d3.Grid(ones*grad_kappa_rad).evaluate()
            self.namespace['grid_inv_pom0_{}'.format(bn)] = grid_inv_pom0 = d3.Grid(inv_pom0).evaluate()
            self.namespace['grid_nu_diff_{}'.format(bn)] = grid_nu_diff = d3.Grid(nu_diff).evaluate()
            self.namespace['grid_neg_one_{}'.format(bn)] = neg_one = d3.Grid(-ones).evaluate()
            self.namespace['grid_eye_{}'.format(bn)] = grid_eye = d3.Grid(eye).evaluate()
            self.namespace['grid_P0_{}'.format(bn)] = grid_P0 = d3.Grid(P0*ones).evaluate()
            self.namespace['grid_g_phi_{}'.format(bn)] = grid_g_phi = d3.Grid(ones*g_phi)

            for fname in ['rho0', 'ln_rho0', 'grad_ln_rho0', 's0', 'grad_s0', 'pom0', 'grad_pom0', 'g', 'chi_rad', 'grad_chi_rad', 'kappa_rad', 'grad_kappa_rad', 'inv_pom0', 'nu_diff', 'neg_one', 'eye', 'P0']:
                self.namespace['grid_{}_{}'.format(fname, bn)].name = 'grid_{}_{}'.format(fname, bn)


            if basis_number == 0:
                gamma = self.namespace['gamma']
                R_gas = self.namespace['R_gas']
                Cp = self.namespace['Cp']
                Cv = self.namespace['Cv']
                self.namespace['grid_cp'] = grid_cp = d3.Grid(Cp).evaluate()
                self.namespace['grid_cp_div_R'] = grid_cp_div_R = d3.Grid(Cp/R_gas).evaluate() #= gamma/(gamma-1)
                self.namespace['grid_cv_div_R'] = grid_cv_div_R = d3.Grid(Cv/R_gas).evaluate() #= gamma/(gamma-1)
                self.namespace['grid_R_div_cp'] = grid_R_div_cp = d3.Grid(R_gas/Cp).evaluate() #= (gamma-1)/gamma
                self.namespace['grid_inv_cp'] = grid_inv_cp = d3.Grid(1/Cp).evaluate()
                self.namespace['grid_R'] = grid_R = d3.Grid(R_gas).evaluate()
                self.namespace['grid_gamma'] = grid_gamma = d3.Grid(gamma).evaluate()
                self.namespace['grid_inv_gamma'] = grid_inv_gamma = d3.Grid(1/gamma).evaluate()
                for fname in ['cp', 'cp_div_R', 'R_div_cp', 'inv_cp', 'R', 'gamma', 'inv_gamma']:
                    self.namespace['grid_{}'.format(fname)].name = 'grid_{}'.format(fname)

            lap_domain = d3.lap(s1).domain
            self.namespace['lap_C_{}'.format(bn)] = lap_C = lambda A: convert(A, lap_domain.bases)

            #Stress matrices & viscous terms
            self.namespace['E_{}'.format(bn)] = E = grad_u/2 + d3.trans(grad_u/2)
            self.namespace['sigma_{}'.format(bn)] = sigma = (E - div_u*eye/3)*2
            self.namespace['E_RHS_{}'.format(bn)] = E_RHS = (grid_grad_u + d3.trans(grid_grad_u))/2
            self.namespace['sigma_RHS_{}'.format(bn)] = sigma_RHS = (E_RHS - grid_div_u*grid_eye/3)*2
            self.namespace['visc_div_stress_L_{}'.format(bn)] = visc_div_stress_L = nu_diff*(d3.div(sigma) + sigma@grad_ln_rho0) + sigma@grad_nu_diff
            self.namespace['visc_div_stress_L_RHS_{}'.format(bn)] = visc_div_stress_L_RHS = grid_nu_diff*(d3.div(sigma) + sigma_RHS@grid_grad_ln_rho0) + sigma_RHS@d3.Grid(grad_nu_diff)
            self.namespace['visc_div_stress_R_{}'.format(bn)] = visc_div_stress_R = grid_nu_diff*(sigma_RHS@grid_grad_ln_rho1)
            self.namespace['VH_{}'.format(bn)] = VH = (grid_nu_diff)*(d3.trace(E_RHS@E_RHS) - (1/3)*grid_div_u**2)*2

            #Thermodynamics: rho, pressure, s 
            self.namespace['rho_full_{}'.format(bn)] = rho_full = grid_rho0*np.exp(ln_rho1)
            self.namespace['rho_fluc_{}'.format(bn)] = rho_fluc = rho_full - grid_rho0
            self.namespace['ln_rho_full_{}'.format(bn)] = ln_rho_full = (grid_ln_rho0 + ln_rho1)
            self.namespace['grad_ln_rho_full_{}'.format(bn)] = grad_ln_rho_full = grid_grad_ln_rho0 + grid_grad_ln_rho1
            self.namespace['s_full_{}'.format(bn)] = s_full = grid_s0 + grid_s1
#            self.namespace['P_full_{}'.format(bn)] = P_full = grid_P0*np.exp(grid_gamma*(s1*grid_inv_cp + grid_ln_rho1))
            self.namespace['P_full_{}'.format(bn)] = P_full = np.exp(grid_gamma*(s_full*grid_inv_cp + grid_ln_rho1 + grid_ln_rho0))
            self.namespace['grad_s_full_{}'.format(bn)] = grad_s_full = grid_grad_s0 + grid_grad_s1
            self.namespace['enthalpy_{}'.format(bn)] = enthalpy = grid_cp_div_R*P_full
            self.namespace['enthalpy_fluc_{}'.format(bn)] = enthalpy_fluc = enthalpy - d3.Grid(grid_cp_div_R*grid_P0)


            #Linear Pomega = R * T
            self.namespace['pom1_over_pom0_{}'.format(bn)] = pom1_over_pom0 = gamma*(s1/Cp + ((gamma-1)/gamma)*ln_rho1)
            self.namespace['grad_pom1_over_pom0_{}'.format(bn)] = grad_pom1_over_pom0 = gamma*(grad_s1/Cp + ((gamma-1)/gamma)*grad_ln_rho1)
            self.namespace['pom1_{}'.format(bn)] = pom1 = pom0 * pom1_over_pom0
            self.namespace['grad_pom1_{}'.format(bn)] = grad_pom1 = d3.grad(pom1)#grad_pom0*pom1_over_pom0 + pom0*grad_pom1_over_pom0

            #RHS terms
            self.namespace['pom1_over_pom0_RHS_{}'.format(bn)] = pom1_over_pom0_RHS = grid_gamma*(grid_s1*grid_inv_cp + grid_R_div_cp*grid_ln_rho1)
            self.namespace['pom1_RHS_{}'.format(bn)] = pom1_RHS = grid_pom0 * pom1_over_pom0_RHS
            self.namespace['grad_pom1_over_pom0_RHS_{}'.format(bn)] = grad_pom1_over_pom0_RHS = grid_gamma*(grad_s1*grid_inv_cp + grid_R_div_cp*grad_ln_rho1)
            self.namespace['lap_pom1_over_pom0_RHS_{}'.format(bn)] = lap_pom1_over_pom0_RHS = grid_gamma*(grid_lap_s1*grid_inv_cp + grid_R_div_cp*grid_lap_ln_rho1)
            self.namespace['grad_pom1_RHS_{}'.format(bn)] = grad_pom1_RHS = grid_grad_pom0*pom1_over_pom0_RHS + grid_pom0*grad_pom1_over_pom0_RHS
            self.namespace['inv_pom0_times_lap_pom1_RHS_{}'.format(bn)] = inv_pom0_times_lap_pom1_RHS = lap_pom1_over_pom0_RHS + 2 * grid_inv_pom0 * grid_grad_pom0 @ grad_pom1_over_pom0_RHS + d3.Grid(d3.lap(ones*pom0)) * grid_inv_pom0 *  pom1_over_pom0_RHS

            #Full pomega subs
            self.namespace['pom_fluc_over_pom0_{}'.format(bn)] = pom_fluc_over_pom0 = np.exp(pom1_over_pom0_RHS) + neg_one 
            self.namespace['pom_fluc_{}'.format(bn)] = pom_fluc = grid_pom0*pom_fluc_over_pom0
            self.namespace['grad_pom_fluc_{}'.format(bn)] = grad_pom_fluc = grid_grad_pom0*pom_fluc_over_pom0 + (pom_fluc_over_pom0 + ones)*grid_pom0*grad_pom1_over_pom0_RHS
            self.namespace['lap_pom_fluc_{}'.format(bn)] = lap_pom_fluc = d3.lap(pom_fluc)

            #Nonlinear Pomega = R*T
            self.namespace['pom2_over_pom0_{}'.format(bn)] = pom2_over_pom0 = pom_fluc_over_pom0 - pom1_over_pom0_RHS
            self.namespace['pom2_{}'.format(bn)] = pom2 = grid_pom0*pom2_over_pom0
            self.namespace['grad_pom2_{}'.format(bn)] = grad_pom2 = d3.grad(pom2)

            self.namespace['grad_pom_full_{}'.format(bn)] = grad_pom_full = (grid_grad_pom0 + grad_pom_fluc)
            self.namespace['pom_full_{}'.format(bn)] = pom_full = (grid_pom0 + pom_fluc)
            self.namespace['T_full_{}'.format(bn)] = T_full = pom_full / grid_R
            self.namespace['inv_pom_full_{}'.format(bn)] = inv_pom_full = d3.Grid(1/pom_full)
            self.namespace['grad_pom2_over_pom0_{}'.format(bn)] = grad_pom2_over_pom0 = grad_pom1_over_pom0_RHS*pom_fluc_over_pom0

            #Equation of state goodness
            self.namespace['EOS_{}'.format(bn)]    = EOS = (s_full)*grid_inv_cp - ( grid_inv_gamma * (np.log(pom_full) - np.log(grid_R)) - grid_R_div_cp * ln_rho_full )
            self.namespace['EOS_bg_{}'.format(bn)] = EOS_bg = d3.Grid(ones*(s0/Cp - ( grid_inv_gamma * (np.log(pom0) - np.log(R_gas)) - ((gamma-1)/(gamma)) * ln_rho0)))
            self.namespace['EOS_goodness_{}'.format(bn)]    = EOS_good_ = np.sqrt(EOS**2)
            self.namespace['EOS_goodness_bg_{}'.format(bn)] = EOS_good_bg = d3.Grid(np.sqrt(EOS_bg**2))

            #Momentum thermo / hydrostatic terms:
            self.namespace['gradP0_div_rho0_{}'.format(bn)]         = gradP0_div_rho0 = gamma*pom0*(grad_ln_rho0 + grad_s0*grid_inv_cp)
            self.namespace['background_HSE_{}'.format(bn)]          = background_HSE = gradP0_div_rho0 - g
            self.namespace['linear_gradP_div_rho_{}'.format(bn)]    = linear_gradP_div_rho    = gamma*pom0*(grad_ln_rho1 + grad_s1/Cp) + g*pom1_over_pom0
            self.namespace['nonlinear_gradP_div_rho_{}'.format(bn)] = nonlinear_gradP_div_rho = grid_gamma*pom_fluc*(grid_grad_ln_rho1 + grid_grad_s1*grid_inv_cp) + grid_g*pom2_over_pom0

            #Radiative diffusivity -- we model flux as kappa * grad T1 (not including nonlinear part of T; it's low mach so it's fine.
            self.namespace['F_cond_{}'.format(bn)] = F_cond = -1*kappa_rad*((grad_pom1_RHS)/R_gas)
          

            self.namespace['div_rad_flux_pt1_LHS_{}'.format(bn)] = div_rad_flux_pt1_LHS = grad_kappa_rad@(grad_pom1)
            self.namespace['div_rad_flux_pt2_LHS_{}'.format(bn)] = div_rad_flux_pt2_LHS = kappa_rad * d3.lap(pom1)
            self.namespace['div_rad_flux_pt1_{}'.format(bn)] = div_rad_flux_pt1 = grid_grad_kappa_rad@(grad_pom1_RHS)
            self.namespace['div_rad_flux_pt2_{}'.format(bn)] = div_rad_flux_pt2 = grid_kappa_rad * d3.lap(pom1_RHS)

            if EVP:
                diff_factor = 1
                self.namespace['div_rad_flux_L_{}'.format(bn)] = div_rad_flux_L = (diff_factor/P0) * (div_rad_flux_pt1_LHS + div_rad_flux_pt2_LHS)
            else:
                diff_factor = 1
                self.namespace['div_rad_flux_L_{}'.format(bn)] = div_rad_flux_L = (diff_factor/P0) * div_rad_flux_pt2_LHS 
            self.namespace['full_div_rad_flux_pt1_{}'.format(bn)] = full_div_rad_flux_pt1 =   d3.Grid(1/P_full) * (div_rad_flux_pt1) #technically wrong for EVP; also set to 0 so whatever.
            self.namespace['full_div_rad_flux_pt2_{}'.format(bn)] = full_div_rad_flux_pt2 =   d3.Grid(1/P_full + d3.Grid(diff_factor*neg_one/grid_P0)) * (div_rad_flux_pt2)

            # Rotation and damping terms
            if self.do_rotation:
                ez = self.namespace['ez_{}'.format(bn)]
                self.namespace['rotation_term_{}'.format(bn)] = -2*Omega*d3.cross(ez, u)
            else:
                self.namespace['rotation_term_{}'.format(bn)] = 0

            if self.sponge:
                self.namespace['sponge_term_{}'.format(bn)] = u*self.namespace['sponge_{}'.format(bn)]
            else:
                self.namespace['sponge_term_{}'.format(bn)] = 0

            sponge_term = self.namespace['sponge_term_{}'.format(bn)]
            rotation_term = self.namespace['rotation_term_{}'.format(bn)]


            #The sum order matters here based on ball or shell...weird.
            energy_terms_1 = d3.Grid(lap_C(-grid_u@grid_grad_s1 + d3.Grid(grid_R/P_full)*d3.Grid(Q) + d3.Grid(grid_R*inv_pom_full)*VH + full_div_rad_flux_pt1))
            energy_terms_2 = d3.Grid(lap_C(full_div_rad_flux_pt2))
            self.namespace['energy_RHS_{}'.format(bn)] = energy_terms_1 + energy_terms_2

            #output tasks
            er = self.namespace['er']
            self.namespace['r_vals_{}'.format(bn)] = r_vals = d3.Grid(er@(ones*rvec)).evaluate()
            self.namespace['ur_{}'.format(bn)] = er@u
            self.namespace['momentum_{}'.format(bn)] = momentum = rho_full * u
            self.namespace['u_squared_{}'.format(bn)] = u_squared = u@u
            self.namespace['KE_{}'.format(bn)] = KE = rho_full * u_squared / 2
            self.namespace['PE_{}'.format(bn)] = PE = rho_full * grid_g_phi
            self.namespace['IE_{}'.format(bn)] = IE = d3.Grid((P_full)*d3.Grid(Cv/R_gas).evaluate())
            self.namespace['PE0_{}'.format(bn)] = PE0 = d3.Grid(rho0 * g_phi)
            self.namespace['IE0_{}'.format(bn)] = IE0 = d3.Grid(grid_P0*(Cv/R_gas))
            self.namespace['PE1_{}'.format(bn)] = PE1 = PE + d3.Grid(-PE0*ones)
            self.namespace['IE1_{}'.format(bn)] = IE1 = IE + d3.Grid(-IE0*ones)
            self.namespace['TotE_{}'.format(bn)] = KE + PE + IE
            self.namespace['FlucE_{}'.format(bn)] = KE + PE1 + IE1
            self.namespace['FlucE_linear_{}'.format(bn)] = rho0*(Cv*pom1/R_gas + Cv*pom0*ln_rho1/R_gas + g_phi*ln_rho1)
            self.namespace['FlucE_linear_RHS_{}'.format(bn)] = grid_rho0*(grid_cv_div_R*pom1_RHS + d3.Grid(grid_g_phi + grid_cv_div_R*grid_pom0)*ln_rho1)
            self.namespace['Re_{}'.format(bn)] = np.sqrt(u_squared) * d3.Grid(1/nu_diff)
            self.namespace['Ma_{}'.format(bn)] = np.sqrt(u_squared) / np.sqrt(pom_full) 
            self.namespace['vorticity_{}'.format(bn)] = vorticity = d3.curl(u)
            self.namespace['enstrophy_{}'.format(bn)] = vorticity@vorticity
            self.namespace['L_{}'.format(bn)] = d3.cross(rvec, momentum)

            grad_T0_superad = d3.Grid(self.namespace['grad_T0_superad_{}'.format(bn)]*ones).evaluate()
            self.namespace['T_superad_z_{}'.format(bn)] = T_superad_z = grad_pom_fluc/R_gas + grad_T0_superad
            self.namespace['T_superad1_z_{}'.format(bn)] = T_superad1_z = grad_pom1_RHS/R_gas + grad_T0_superad
            self.namespace['F_cond_superad_{}'.format(bn)] = F_cond_superad = -1*kappa_rad*(T_superad1_z)

            #Fluxes
            self.namespace['F_KE_{}'.format(bn)] = F_KE = u * KE
            self.namespace['F_PE_{}'.format(bn)] = F_PE = u * PE
            self.namespace['F_enth_{}'.format(bn)] = F_enth = grid_cp_div_R * momentum * pom_full
            self.namespace['F_visc_{}'.format(bn)] = F_visc = d3.Grid(-nu_diff)*momentum@sigma_RHS
            self.namespace['F_conv_{}'.format(bn)] = F_conv = rho_full*T_full*s_full*u

            #Waves
            self.namespace['N2_{}'.format(bn)] = N2 = grad_s_full@d3.Grid(-g/Cp)

            #Source terms
            self.namespace['Q_source_{}'.format(bn)] = Q_source = self.namespace['Q_{}'.format(bn)]

            self.namespace['visc_source_KE_{}'.format(bn)] = visc_source_KE = momentum @ (visc_div_stress_L_RHS + visc_div_stress_R)
            self.namespace['visc_source_IE_{}'.format(bn)] = visc_source_IE = (P_full/grid_R)*d3.Grid(grid_R*inv_pom_full)*VH
            self.namespace['tot_visc_source_{}'.format(bn)] = tot_visc_source = visc_source_KE + visc_source_IE

            self.namespace['PdV_source_KE_{}'.format(bn)] = PdV_source_KE = momentum @ (-d3.grad(P_full)/rho_full) 
            self.namespace['PdV_source_IE_{}'.format(bn)] = PdV_source_IE =  - P_full*div_u
            self.namespace['PdV_source_anelastic_{}'.format(bn)] = PdV_source_anelastic = rho_full*(s1/Cp)*u@g
            self.namespace['tot_PdV_source_{}'.format(bn)] = tot_PdV_source = PdV_source_KE + PdV_source_IE

            self.namespace['divRad_source_{}'.format(bn)] = divRad_source = (P_full/grid_R)*(full_div_rad_flux_pt1 + full_div_rad_flux_pt2 + div_rad_flux_L)


            #Entropy dissipation terms
            flux_to_lum = 4*np.pi*r_vals**2
            L_heat = self.namespace['L_heat_{}'.format(bn)]*ones
            integ_by_parts_T = (1/T_full)**2 * (grad_pom_full/grid_R)
            self.namespace['F_entropy_{}'.format(bn)] = F_entropy = rho_full*s_full*u
            self.namespace['therm_visc_lum_{}'.format(bn)] = visc_source_IE  * (grid_R/pom_full) * flux_to_lum
            self.namespace['integ_by_parts_1_{}'.format(bn)] = (L_heat*er - flux_to_lum*F_cond) / T_full
            self.namespace['integ_by_parts_2_{}'.format(bn)] = L_heat*er@integ_by_parts_T 
            self.namespace['integ_by_parts_3_{}'.format(bn)] = -flux_to_lum*F_cond@integ_by_parts_T 



            self.namespace['source_KE_{}'.format(bn)] = visc_source_KE + PdV_source_KE #g term turns into dt(PE) + div(u*PE); do not include here while trying to solve for dt(KE) + div(u*KE).
            self.namespace['source_IE_{}'.format(bn)] = visc_source_IE + PdV_source_IE + Q_source + divRad_source
            self.namespace['tot_source_{}'.format(bn)] = self.namespace['source_KE_{}'.format(bn)] + self.namespace['source_IE_{}'.format(bn)]

            s_prime = s1/(pom_full/grid_R)
            self.namespace['therm_diss_1_{}'.format(bn)] = therm_diss_1 = F_cond@d3.grad(s_prime)
            self.namespace['therm_diss_2_{}'.format(bn)] = therm_diss_2 = s_prime*visc_source_IE
            self.namespace['therm_diss_3_{}'.format(bn)] = therm_diss_3 = s_prime*Q_source
        return self.namespace

    def fill_structure(self, scales=None):
        from scipy.interpolate import interp1d
        self.fields_filled = True
        logger.info('using NCC file {}'.format(self.ncc_file))
        max_dt = None
        t_buoy = None
        t_rot = None
        logger.info('collecing nccs for {}'.format(self.bases.keys()))
        for basis_number, bn in enumerate(self.bases.keys()):
            basis = self.bases[bn]
            ncc_scales = scales
            if ncc_scales is None:
                ncc_scales = basis.dealias
            phi, theta, r = basis.local_grids(ncc_scales)
            phi1, theta1, r1 = basis.local_grids((1,1,1))
            # Load MESA NCC file or setup NCCs using polytrope
            a_vector = self.namespace['{}_{}'.format(self.vec_fields[0], bn)]
            grid_slices  = self.dist.layouts[-1].slices(a_vector.domain, ncc_scales[-1])
            a_vector.change_scales(ncc_scales)
            local_vncc_size = self.namespace['{}_{}'.format(self.vec_nccs[0], bn)]['g'].size
            if self.ncc_file is not None:
                logger.info('reading NCCs from {}'.format(self.ncc_file))
                for k in self.vec_nccs + self.scalar_nccs + ['Q', 'rho0']:
                    self.namespace['{}_{}'.format(k, bn)].change_scales(ncc_scales)
                with h5py.File(self.ncc_file, 'r') as f:
                    self.namespace['Cp']['g'] = f['Cp'][()]
                    self.namespace['R_gas']['g'] = f['R_gas'][()]
                    self.namespace['gamma']['g'] = f['gamma1'][()]
                    self.namespace['Cv']['g'] = f['Cp'][()] - f['R_gas'][()]
                    logger.info('using Cp: {}, Cv: {}, R_gas: {}, gamma: {}'.format(self.namespace['Cp']['g'], self.namespace['Cv']['g'], self.namespace['R_gas']['g'], self.namespace['gamma']['g']))
                    for k in self.vec_nccs:
                        self.dist.comm_cart.Barrier()
                        if '{}_{}'.format(k, bn) not in f.keys():
                            logger.info('skipping {}_{}, not in file'.format(k, bn))
                            continue
                        if local_vncc_size > 0:
                            logger.info('reading {}_{}'.format(k, bn))
                            self.namespace['{}_{}'.format(k, bn)]['g'] = f['{}_{}'.format(k, bn)][:,:1,:1,grid_slices[-1]]
                    for k in self.scalar_nccs:
                        self.dist.comm_cart.Barrier()
                        if k == 'L_heat':
                            sim_lum_r = f['lum_r_vals'][()]
                            sim_lum   = f['sim_lum'][()]
                            lum_func = interp1d(sim_lum_r, sim_lum, bounds_error=False, fill_value='extrapolate')
                            self.namespace['{}_{}'.format(k, bn)]['g'] = lum_func(r)
                        if '{}_{}'.format(k, bn) not in f.keys():
                            logger.info('skipping {}_{}, not in file'.format(k, bn))
                            continue
                        logger.info('reading {}_{}'.format(k, bn))
                        self.namespace['{}_{}'.format(k, bn)]['g'] = f['{}_{}'.format(k, bn)][:,:,grid_slices[-1]]
                    self.namespace['Q_{}'.format(bn)]['g']         = f['Q_{}'.format(bn)][:,:,grid_slices[-1]]
                    self.namespace['rho0_{}'.format(bn)]['g']       = np.exp(f['ln_rho0_{}'.format(bn)][:,:,grid_slices[-1]])[None,None,:]


                    if max_dt is None:
                        max_dt = f['max_dt'][()]

                    if t_buoy is None:
                        t_buoy = f['tau_heat'][()]/f['tau_nd'][()]

                    if t_rot is None:
                        if self.do_rotation:
                            sim_tau_sec = f['tau_nd'][()]
                            sim_tau_day = sim_tau_sec / (60*60*24)
                            Omega = sim_tau_day * dimensional_Omega 
                            t_rot = 1/(2*Omega)
                        else:
                            t_rot = np.inf

                    if self.sponge:
                        f_brunt = f['tau_nd'][()]*np.sqrt(f['N2max_sim'][()])/(2*np.pi)
                        self.namespace['sponge_{}'.format(bn)]['g'] *= f_brunt
                for k in self.vec_nccs + self.scalar_nccs + ['rho0']:
                    self.namespace['{}_{}'.format(k, bn)].change_scales((1,1,1))

            else:
                raise NotImplementedError("Must supply star file")

            if self.do_rotation:
                logger.info("Running with Coriolis Omega = {:.3e}".format(Omega))

        return self.namespace, (max_dt, t_buoy, t_rot)

    def get_compressible_variables(self):
        problem_variables = []
        for field in ['ln_rho1', 'u', 's1']:
            for basis_number, bn in enumerate(self.bases_keys):
                problem_variables.append(self.namespace['{}_{}'.format(field, bn)])

        problem_taus = []
        for tau in ['tau_s']:
            for basis_number, bn in enumerate(self.bases_keys):
                if type(self.bases[bn]) == d3.BallBasis:
                    problem_taus.append(self.namespace['{}_{}'.format(tau, bn)])
                else:
                    problem_taus.append(self.namespace['{}1_{}'.format(tau, bn)])
                    problem_taus.append(self.namespace['{}2_{}'.format(tau, bn)])
        for tau in ['tau_u']:
            for basis_number, bn in enumerate(self.bases_keys):
                if type(self.bases[bn]) == d3.BallBasis:
                    problem_taus.append(self.namespace['{}_{}'.format(tau, bn)])
                else:
                    problem_taus.append(self.namespace['{}1_{}'.format(tau, bn)])
                    problem_taus.append(self.namespace['{}2_{}'.format(tau, bn)])

        return problem_variables + problem_taus

    def set_compressible_problem(self, problem, fixed_s=False):
        equations = OrderedDict()
        u_BCs = OrderedDict()
        T_BCs = OrderedDict()

        for basis_number, bn in enumerate(self.bases_keys):
            basis = self.bases[bn]

            #Standard Equations
            # Assumes background is in hse: -(grad T0 + T0 grad ln rho0) + gvec = 0.
            if config.numerics['equations'] == 'FC_HD':
                equations['continuity_{}'.format(bn)] = "dt(ln_rho1_{0}) + div_u_{0} + u_{0}@grad_ln_rho0_{0} + taus_lnrho_{0} = -(u_{0}@grad_ln_rho1_{0})".format(bn)
                equations['momentum_{}'.format(bn)] = "dt(u_{0}) + linear_gradP_div_rho_{0} - visc_div_stress_L_{0} + sponge_term_{0} + taus_u_{0} = (-(u_{0}@grad_u_{0}) - nonlinear_gradP_div_rho_{0} + visc_div_stress_R_{0})".format(bn)
                equations['energy_{}'.format(bn)] = "dt(s1_{0}) + u_{0}@grad_s0_{0} - div_rad_flux_L_{0} + taus_s_{0} = energy_RHS_{0}".format(bn)
            else:
                raise ValueError("Unknown equation choice, plesae use 'FC_HD'")

            constant_U = "u_{0}(r={2}) - u_{1}(r={2}) = 0 "
            constant_s = "s1_{0}(r={2}) - s1_{1}(r={2}) = 0"
            constant_gradT = "radial(grad_pom1_over_pom0_{0}(r={2}) - grad_pom1_over_pom0_{1}(r={2})) = 0" #this makes grad(pom_fluc) continuous if s1 and ln_rho1 are also continuous.
            constant_ln_rho = "ln_rho1_{0}(r={2}) - ln_rho1_{1}(r={2}) = 0"
            constant_momentum_ang = "angular(radial(sigma_{0}(r={2}) - sigma_{1}(r={2}))) = 0"



            #Boundary conditions
            if type(basis) == d3.BallBasis:
                if basis_number == len(self.bases_keys) - 1:
                    #No shell bases
                    u_BCs['BC_u1_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, basis.radius)
                    u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, basis.radius)
                    if fixed_s:
                        T_BCs['BC_T_outer_{}'.format(bn)] = "s1_{0}(r={1}) = 0".format(bn, basis.radius) #needed for energy conservation
                    else:
                        T_BCs['BC_T_outer_{}'.format(bn)] = "radial(grad_pom1_{0}(r={1})) = 0".format(bn, basis.radius) #needed for energy conservation
                else:
                    shell_name = self.bases_keys[basis_number+1] 
                    rval = self.stitch_radii[basis_number]
                    u_BCs['BC_u2_{}'.format(bn)] = constant_U.format(bn, shell_name, rval)
                    T_BCs['BC_T1_{}'.format(bn)] = constant_s.format(bn, shell_name, rval)
            else:
                if bn == 'S0':
                    #Bottom of domain
                    u_BCs['BC_u2_bot_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, basis.radii[0])
                    u_BCs['BC_u3_bot_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, basis.radii[0])
                    T_BCs['BC_T_bot_{}'.format(bn)] = "radial(grad_pom1_{0}(r={1})) = 0".format(bn, basis.radii[0])
                else:
                    #Stitch to basis below
                    below_name = self.bases_keys[basis_number - 1]
                    rval = self.stitch_radii[basis_number - 1]
                    u_BCs['BC_u1_vec_{}'.format(bn)] = constant_momentum_ang.format(bn, below_name, rval)
                    u_BCs['BC_u2_vec_{}'.format(bn)] = constant_ln_rho.format(bn, below_name, rval)
                    T_BCs['BC_T0_{}'.format(bn)] = constant_gradT.format(bn, below_name, rval)

                #Add upper BCs
                if basis_number != len(self.bases_keys) - 1:
                    shn = self.bases_keys[basis_number+1] 
                    rval = self.stitch_radii[basis_number]
                    u_BCs['BC_u3_vec_{}'.format(bn)] = constant_U.format(bn, shn, rval)
                    T_BCs['BC_T3_{}'.format(bn)] = constant_s.format(bn, shn, rval)
                else:
                    #top of domain
                    u_BCs['BC_u2_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, basis.radii[1])
                    u_BCs['BC_u3_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, basis.radii[1])
                    if fixed_s:
                        T_BCs['BC_T_outer_{}'.format(bn)] = "s1_{0}(r={1}) = 0".format(bn, basis.radii[1])
                    else:
                        T_BCs['BC_T_outer_{}'.format(bn)] = "radial(grad_pom1_{0}(r={1})) = 0".format(bn, basis.radii[1])


        for bn, basis in self.bases.items():
            continuity = equations['continuity_{}'.format(bn)]
            logger.info('adding eqn "{}"'.format(continuity))
            problem.add_equation(continuity)

        for bn, basis in self.bases.items():
            momentum = equations['momentum_{}'.format(bn)]
            logger.info('adding eqn "{}"'.format(momentum))
            problem.add_equation(momentum)

        for bn, basis in self.bases.items():
            energy = equations['energy_{}'.format(bn)]
            logger.info('adding eqn "{}"'.format(energy))
            problem.add_equation(energy)

        for BC in u_BCs.values():
            logger.info('adding BC "{}"'.format(BC))
            problem.add_equation(BC)

        FlucE_LHS = ""
        FlucE_RHS = ""
        for basis_number, bn in enumerate(self.bases_keys):
            if basis_number > 0:
                FlucE_LHS += " + "
                FlucE_RHS += " + "
            FlucE_LHS += "integ(FlucE_linear_{0})".format(bn)
            FlucE_RHS += "integ(FlucE_linear_RHS_{0}  - FlucE_{0})".format(bn)
        FlucE_LHS = eval(FlucE_LHS, dict(problem.namespace))
        FlucE_RHS = eval(FlucE_RHS, dict(problem.namespace))

        for name, BC in T_BCs.items():
            if 'outer' in name:
                if fixed_s:
                    logger.info('adding BC "{}"'.format(BC))
                    problem.add_equation(BC)
                else:
                    energy_constraint = (FlucE_LHS, FlucE_RHS)
                    logger.info('adding BC "{}" (ntheta != 0)'.format(BC))
                    logger.info('adding BC "{}" (ntheta == 0)'.format([str(o) for o in energy_constraint]))
                    problem.add_equation(BC, condition="ntheta != 0")
                    problem.add_equation(energy_constraint, condition="ntheta == 0")
            else:
                logger.info('adding BC "{}"'.format(BC))
                problem.add_equation(BC)

        return problem
