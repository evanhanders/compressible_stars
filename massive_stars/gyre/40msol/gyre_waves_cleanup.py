"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
from collections import OrderedDict
import re
import os

import h5py
import numpy as np
import pygyre as pg
import pymsg as pm
import tomso as tomso
from tomso import gyre
import mesa_reader as mr
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.interpolate import interp1d
import scipy.special as ss
from d3_stars.simulations.star_builder import find_core_cz_radius

Rsun_to_cm = 6.957e10
sigma_SB = 5.67037442e-05 #cgs
G = 6.67E-8
solar_z = 0.014


def natural_sort(iterable, reverse=False):
    """
    Sort alphanumeric strings naturally, i.e. with "1" before "10".
    Copied from dedalus; Based on http://stackoverflow.com/a/4836734.
    """

    convert = lambda sub: int(sub) if sub.isdigit() else sub.lower()
    key = lambda item: [convert(sub) for sub in re.split('([0-9]+)', str(item))]

    return sorted(iterable, key=key, reverse=reverse)

def calculate_optical_depths(eigenfrequencies, r, N2, S1, chi_rad, ell=1):
    #Calculate 'optical depths' of each mode.
    depths = []
#    chi_rad = chi_rad.min()
    for freq in eigenfrequencies.real:
        freq = np.abs(freq)
        om = 2*np.pi*freq
        lamb_freq = np.sqrt(ell*(ell+1) / 2) * S1
        wave_cavity = (2*np.pi*freq < np.sqrt(N2))*(2*np.pi*freq < lamb_freq)
        depth_integrand = np.zeros_like(lamb_freq)

        # from Lecoanet et al 2015 eqn 12. This is the more universal function
        Lambda = np.sqrt(ell*(ell+1))
        k_perp = Lambda/r
        kz = ((-1)**(3/4)/np.sqrt(2))*np.sqrt(-1j*2*k_perp**2 - (om/chi_rad) + np.sqrt(om**3 + 1j*4*k_perp**2*chi_rad*N2)/(chi_rad*np.sqrt(om)) )
#        kz = np.sqrt(-k_perp**2 + 1j*((2*np.pi*freq)/(2*chi_rad))*(1 - np.sqrt(1 + 1j*4*(N2*chi_rad*k_perp**2 / (2*np.pi*freq)**3))))
        depth_integrand[wave_cavity] = kz[wave_cavity].imag
#        plt.semilogy(r, depth_integrand)
#        plt.show()


        #Numpy integrate
        opt_depth = np.trapz(depth_integrand, x=r)
        depths.append(opt_depth)
    return depths



class GyreMSGPostProcessor:

    def __init__(self, ell, pos_summary, pos_details, neg_summary, neg_details, mesa_pulse_file, mesa_LOG_file, 
                 initial_z=0.006, specgrid=None, filters=['Red',], 
                 MSG_DIR = os.environ['MSG_DIR'], 
                 GRID_DIR=None,
                 PASS_DIR=os.path.join('..','gyre_phot','passbands')):
        self.ell = ell
        self.pos_summary = pos_summary
        self.neg_summary = neg_summary
        self.pos_details = pos_details
        self.neg_details = neg_details
        self.mesa_pulse_file = mesa_pulse_file
        self.mesa_LOG_file = mesa_LOG_file
        self.filters = filters
        self.core_cz_radius = find_core_cz_radius(self.mesa_LOG_file)
        self.data_dict = OrderedDict()

        # Load the MSG photometric grids (this code taken from the 
        # Python walkthrough in the MSG docs)
        if GRID_DIR is None:
            os.path.join(MSG_DIR, 'data', 'grids'),
        
        if specgrid == 'OSTAR2002':
            specgrid_file_name = os.path.join(GRID_DIR, 'sg-OSTAR2002-low.h5')
        else:
            specgrid_file_name = os.path.join(GRID_DIR, 'sg-demo.h5')

        self.photgrids = {}
        for filter in self.filters:
            passband_file_name = os.path.join(PASS_DIR, f'pb-TESS-TESS.{filter}-Vega.h5')
            print(specgrid_file_name, passband_file_name)
            self.photgrids[filter] = pm.PhotGrid(specgrid_file_name, passband_file_name)


            # Inspect grid parameters
            print('Grid parameters:')
            for label in self.photgrids[filter].axis_labels:
                print(f'  {label} [{self.photgrids[filter].axis_x_min[label]} -> {self.photgrids[filter].axis_x_max[label]}]')

        # Load the stellar model to figure out Teff and gravity
        self.model = pg.read_model(mesa_pulse_file)
        self.M = self.model.meta['M_star']
        self.R = self.model.meta['R_star']
        self.L = self.model.meta['L_star']
        self.Z = initial_z

        self.ZdZsol = self.Z/solar_z
        self.Teff = (self.L/(4*np.pi*self.R**2*sigma_SB))**0.25
        self.logg = np.log10(G*self.M/self.R**2)

        # Set up the atmosphere parameters dict (to be passed to MSG)

        self.model_x = {'Teff': self.Teff, 'log(g)': self.logg, 'Z/Zo': self.ZdZsol}

        print(f'Teff: {self.Teff}')
        print(f'log(g): {self.logg}')
        print(f'Z/Zo: {self.ZdZsol}')

        # Evaluate the intensity moments (eqn. 15 of Townsend 2003) and their
        # partials
        self.I_0 = {}
        self.I_l = {}
        self.dI_l_dlnTeff = {}
        self.dI_l_dlng = {}

        for filter in self.filters:

            self.I_0[filter] = self.photgrids[filter].D_moment(self.model_x, 0)
            self.I_l[filter] = self.photgrids[filter].D_moment(self.model_x, ell)
            print("I_0, I_l: {:.2e}, {:.2e}".format(self.I_0[filter], self.I_l[filter]))

            self.dI_l_dlnTeff[filter] = self.photgrids[filter].D_moment(self.model_x, 0, deriv={'Teff': True})*self.Teff
            self.dI_l_dlng[filter] = self.photgrids[filter].D_moment(self.model_x, 0, deriv={'log(g)': True})/np.log(10)
            print("dI_l_dlnTeff, dI_l_dlng: {:.2e}, {:.2e}".format(self.dI_l_dlnTeff[filter], self.dI_l_dlng[filter]))

    def evaluate_magnitudes(self):
        file_list = self.pos_details
        summary_file = self.pos_summary
        # Read summary file from GYRE
        summary = pg.read_output(summary_file)
        data = self.data_dict

        # Extract radial displacement and Lagrangian luminosity perturbation
        # amplitudes (note that these are complex quantities)
        data['Delta_R'] = data['xi_r_ref']  # xi_r/R
        data['Delta_L'] = data['lag_L_ref'] # deltaL/L

        # Evaluate the effective temperature perturbation (via
        # Stefan-Boltmann law)
        data['Delta_T'] = 0.25*(data['Delta_L'] - 2*data['Delta_R']) # deltaTeff/Teff

        # Evaluate the effective gravity perturbation (via equation 8
        # of Townsend 2003)
        omega = data['omega'].real
        data['Delta_g'] = -(2 + omega**2)*data['Delta_R']

        # Inspect the Delta's
        print('Delta_R: {}'.format(data['Delta_R']))
        print('Delta_T: {}'.format(data['Delta_T']))
        print('Delta_g: {}'.format(data['Delta_g']))

        # Evaluate the spherical harmonic at the observer location
        # (note that sph_harm has back-to-front angle labeling!)

        
        #get Y_l per eqn 8 of Townsend 2002
#        Ylm = ss.sph_harm(m, self.ell, phi_obs, theta_obs)
        ms  = np.linspace(-self.ell, self.ell, 2*self.ell + 1)[:,None,None]
        phi = np.linspace(0, np.pi, 100)[None,:,None]
        dphi = np.gradient(phi.ravel())[None,:,None]
        theta = np.linspace(0, np.pi, 100)[None,None,:]
        dtheta = np.gradient(theta.ravel())[None,None,:]
        Y_l = (1/(2*self.ell+1))*(1/(4*np.pi))*np.sum(np.sum(np.sum(dtheta*dphi*np.sin(theta)*np.abs(ss.sph_harm(ms, self.ell, phi, theta)),axis=1),axis=1),axis=0)
        print('this ell:', self.ell, Y_l)

        # Evaluate the differential flux functions (eqn. 14 of Townsend 2003)

        dff_R = {}
        dff_T = {}
        dff_G = {}
        dF = {}
        dF_mumag_dict = dict()

        for filter in self.filters:
            
            dff_R[filter] = (2+ell)*(1-ell)*self.I_l[filter]/self.I_0[filter]*Y_l
            dff_T[filter] = self.dI_l_dlnTeff[filter]/self.I_0[filter]*Y_l
            dff_G[filter] = self.dI_l_dlng[filter]/self.I_0[filter]*Y_l

            # Evaluate a light curve in each filter (eqn. 11 of Townsend 2003)
            dF[filter] = ((data['Delta_R']*dff_R[filter] +
                           data['Delta_T']*dff_T[filter] +
                           data['Delta_g']*dff_G[filter]))

            # Convert to micromag using Pogson's law
            dF_mumag = -2.5/np.log(10)*dF[filter]*1E6
            dF_mumag_dict[filter] = dF_mumag
        data['dF_mumags'] = dF_mumag_dict
        return self.data_dict


    def sort_eigenfunctions(self):
        #TODO: move these background info reading lines up to __init__()
        #get info about mesa background
        p = mr.MesaData(self.mesa_LOG_file)
        r_mesa = p.radius[::-1]*Rsun_to_cm #in cm
        bruntN2_mesa = p.brunt_N2[::-1] #rad^2/s^2
        lambS1_mesa  = p.lamb_S[::-1] #rad/s
        T       = p.temperature[::-1] #K
        rho     = 10**(p.logRho[::-1]) #g/cm^3
        opacity = p.opacity[::-1] #cm^2 / g
        cp      = p.cp[::-1] #erg / K / g
        chi_rad_mesa = 16 * sigma_SB * T**3 / (3 * rho**2 * cp * opacity)

        #Get pulsation & stratification information    
        summary = tomso.gyre.load_summary(self.pos_details[0])
        header = summary.header
        data_mode = summary.data
        self.rho = data_mode['rho'] #g / cm^3
        self.x = data_mode['x']
        self.r = self.x*self.R #cm

        V = data_mode['V_2']*self.x**2
        As = data_mode['As']
        c_1 = data_mode['c_1']
        Gamma_1 = data_mode['Gamma_1']

        bruntN2 = As/c_1
        lambS1 = np.sqrt(1*(1+1))*np.sqrt(Gamma_1/(V*c_1))
        chi_rad = 10**(interp1d(r_mesa, np.log10(chi_rad_mesa), bounds_error=False, fill_value='extrapolate')(self.r))

        #re-dimensionalize -- double-check this and compare omega, freq.
        mid_r = self.r[len(self.r)//2]
        self.gyre_tau_nd = 1/np.sqrt(10**(interp1d(r_mesa, np.log10(bruntN2_mesa))(mid_r)) / bruntN2[len(self.r)//2])
        bruntN2 /= self.gyre_tau_nd**2
        lambS1 /= self.gyre_tau_nd

        #data_dicts already has 'freq', 'omega', 'xi_r_ref', 'lag_L_ref', 'l', 'n_pg'
        file_list = self.pos_details
        summary_file = self.pos_summary
        data = self.data_dict
        for field in ['xi_r_eigfunc', 'xi_h_eigfunc', 'lag_L_eigfunc', 'u_r_eigfunc', 'u_h_eigfunc']:
            data[field] = np.zeros((len(file_list), len(self.x)), dtype=np.complex128) 
        for field in ['freq', 'omega', 'lag_L_ref', 'xi_r_ref']:
            data[field] = np.zeros((len(file_list)), dtype=np.complex128) 
        for field in ['depth', 'n_pg', 'l']:
            data[field] = np.zeros(len(file_list))
        for i,filename in enumerate(file_list):
            print('reading eigenfunctions from {}'.format(filename))
            summary = tomso.gyre.load_summary(filename)
            header = summary.header
            data_mode = summary.data

            data['n_pg'][i] = header['n_pg']
            data['l'][i] = header['l']

            data['freq'][i] = 1e-6*(header['Refreq'] + 1j*header['Imfreq']) #cgs
            data['omega'][i] = (header['Reomega'] + 1j*header['Imomega'])
            data['lag_L_ref'][i] = (header['Relag_L_ref'] + 1j*header['Imlag_L_ref'])
            data['xi_r_ref'][i] = (header['Rexi_r_ref'] + 1j*header['Imxi_r_ref'])

            data['depth'][i] = calculate_optical_depths(np.array([1e-6*data['freq'][i],]), self.r, bruntN2, lambS1, chi_rad, ell=self.ell)[0]
            data['xi_r_eigfunc'][i,:] = self.R*(data_mode['Rexi_r'] + 1j*data_mode['Imxi_r']) #arbitrary amplitude; cgs units.
            data['xi_h_eigfunc'][i,:] = self.R*(data_mode['Rexi_h'] + 1j*data_mode['Imxi_h']) #arbitrary amplitude; cgs units.
            data['lag_L_eigfunc'][i,:] = self.L*(data_mode['Relag_L'] + 1j*data_mode['Imlag_L']) #arbitrary amplitude; cgs units
#            print(2*np.pi*data['freq'], data['omega']/self.gyre_tau_nd) #these should be the same.
        print(data['n_pg'], data['l'], data['lag_L_ref'], data['lag_L_eigfunc'][:,-1])
      
        #u = dt(xi) = -i om u by defn.
        #eigenfunctions are dimensional but of arbitrary amplitude.
        data['u_r_eigfunc'] = 2*np.pi*data['freq'][:,None]*data['xi_r_eigfunc']
        data['u_h_eigfunc'] = 2*np.pi*data['freq'][:,None]*data['xi_h_eigfunc'] * (np.sqrt(self.ell*(self.ell+1))) #over r??
        data['delta_L_dL_top'] = data['lag_L_ref']#data['lag_L_eigfunc'][:,-1]/self.L
      
        smooth_oms = np.logspace(np.log10(np.abs(data['freq'].real).min())-3, np.log10(np.abs(data['freq'].real).max())+1, 100)
        smooth_depths = calculate_optical_depths(smooth_oms/(2*np.pi), self.r, bruntN2, lambS1, chi_rad, ell=self.ell)
        data['smooth_oms'] = smooth_oms
        data['smooth_depths'] = smooth_depths
        
        return self.data_dict

    def calculate_duals(self, max_cond=1e10):
        data = self.data_dict
        ur = data['u_r_eigfunc']
        uh = data['u_h_eigfunc']

        self.dr = np.gradient(self.r)
        self.dx = np.gradient(self.x)
        def IP(ur_1,ur_2,uh_1,uh_2):
            """
            Per daniel:
            for the inner product, you need the ell(ell+1) because what gyre calls uh is actually uh/sqrt(ell(ell+1)).
            (because the actual angular velocity has two components and is uh = xi_h * f * grad(Y_ell,m)) [so grad_h is the angular part of the gradient without any 1/r factor]
            but when you take <uh, uh> you can integrate-by-parts on one of the grad's to turn it into laplacian(Y_ell,m)=-(ell(ell+1)) Y_ell,m
            """
            dr = self.dr
            r = self.r
            return np.sum(dr*4*np.pi*r**2*self.rho*(np.conj(ur_1)*ur_2+np.conj(uh_1)*uh_2),axis=-1)
       
        n_modes = ur.shape[0]
        IP_matrix = np.zeros((ur.shape[0], ur.shape[0]),dtype=np.complex128)
        for i in range(ur.shape[0]):
            if i % 10 == 0: print(i)
            for j in range(ur.shape[0]):
                IP_matrix[i,j] = IP(ur[i],ur[j],uh[i],uh[j])
            cond = np.linalg.cond(IP_matrix[:i+1,:i+1])
            if max_cond is not None and i > 0:
                if cond > max_cond:
                    n_modes = i
                    IP_matrix = IP_matrix[:n_modes,:n_modes]
                    break
        print('dual IP matrix cond: {:.3e}; n_modes: {}/{}'.format(cond, n_modes, ur.shape[0]))
        cond = np.linalg.cond(IP_matrix[:i+1,:i+1]) 
        IP_inv = linalg.inv(IP_matrix)

        ur = ur[:n_modes]
        uh = uh[:n_modes]
        for k in self.data_dict.keys():
            if k == 'dF_mumags':
                for sk in self.data_dict[k]:
                    self.data_dict[k][sk] = self.data_dict[k][sk][:n_modes]
            elif 'smooth' not in k:
                self.data_dict[k] = self.data_dict[k][:n_modes]


        data['u_r_dual'] = u_r_dual = np.conj(IP_inv)@ur
        data['u_h_dual'] = u_h_dual = np.conj(IP_inv)@uh


        #Check that velocity duals were evaluated correctly
        IP_check = np.zeros_like(IP_matrix)
        for i in range(ur.shape[0]):
            for j in range(ur.shape[0]):
                IP_check[i,j] = IP(u_r_dual[i], ur[j], u_h_dual[i], uh[j])
        I_matrix = np.eye(IP_matrix.shape[0])

        if np.allclose(I_matrix.real, IP_check.real, rtol=1e-6, atol=1e-6):
            print('duals properly calculated')
        else:
            print('error in dual calculation')
            import sys
            sys.exit()
        return self.data_dict

Lmax = 5
ell_list = np.arange(1, Lmax+1)
for ell in ell_list:
    om_list = np.logspace(-8, -2, 1000) #Hz * 2pi

    pulse_file = 'LOGS/profile53.data.GYRE'
    mesa_LOG = 'LOGS/profile53.data'
    pos_mode_base = './gyre_output/mode_ell{:03d}_m+00_n{:06d}.txt'
    neg_mode_base = pos_mode_base.replace('pos', 'neg')
    pos_files = []
    neg_files = []

    max_n_pg = 100
    do_negative = False
    pos_summary_file='gyre_output/summary_ell01-05.txt'.format(ell)
    pos_summary = pg.read_output(pos_summary_file)
    neg_summary_file = None

    #sort eigenvalues by 1/freq
    sorting = np.argsort(pos_summary['freq'].real**(-1))
    pos_summary = pos_summary[sorting]



    good_freqs = []
    counted_n_pgs = []
    for row in pos_summary:
        this_ell = row['l']
        if this_ell != ell: continue
        n_pg = row['n_pg']
        if n_pg >= 0: continue
        #Check consistency...
        if np.abs(n_pg) > max_n_pg: continue
        if n_pg in counted_n_pgs: continue
        counted_n_pgs.append(n_pg)
        pos_files.append(pos_mode_base.format(ell, n_pg))
        good_freqs.append(complex(row['freq']))

    post = GyreMSGPostProcessor(ell, pos_summary_file, pos_files, neg_summary_file, neg_files, pulse_file, mesa_LOG,
                  specgrid='OSTAR2002', filters=['Red',], initial_z=solar_z,
                  MSG_DIR = os.environ['MSG_DIR'],
                  GRID_DIR=os.path.join('..','gyre-phot','specgrid'),
                  PASS_DIR=os.path.join('..','gyre-phot','passbands'))
    post.sort_eigenfunctions()
    data_dicts = post.evaluate_magnitudes()
    data_dict = post.calculate_duals()
#    print(data_dicts[0]['freq'], data_dicts[1]['freq'])

    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format('gyre_output', ell), 'w') as f:
        if do_negative:
            for k in data_dict.keys():
                if k == 'dF_mumags':
                    f[k+'_Red'] = np.concatenate((data_dict[k]['Red'], data_dict[k]['Red']), axis=0)
                else:
                    f[k] = np.concatenate((data_dict[k], data_dict[k]), axis=0)
        else:
            for k in data_dict.keys():
                if k == 'dF_mumags':
                    f[k+'_Red'] = data_dict[k]['Red']
                else:
                    f[k] = data_dict[k]

        f['r'] = post.r
        f['x'] = post.x
        f['rho'] = post.rho

    #Plot saved eigenvalue spectrum
    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format('gyre_output', ell), 'r') as f:
        freqs = f['freq'][()]

    fig = plt.figure()
    plt.scatter(freqs.real, -freqs.imag)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('f (Hz)')
    plt.ylabel(r'$\gamma/(2\pi)$ (Hz)')
    plt.savefig('evalue_spectrum.png', dpi=300)