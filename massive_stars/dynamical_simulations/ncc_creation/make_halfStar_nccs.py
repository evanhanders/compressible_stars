"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_coreCZ_nccs.py [options]

Options:
    --Nmax=<N>      Maximum radial coefficients [default: 127]
    --file=<f>      Path to MESA log file [default: MESA_Models_Dedalus_Full_Sphere/LOGS/6.data]
    --pre_log_folder=<f>  Folder name in which 'LOGS' sits [default: ]
"""
import os
import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
from dedalus.core import coords, distributor, basis, field, operators, arithmetic
import dedalus.public as de
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from numpy.polynomial import Chebyshev as Pfit

args = docopt(__doc__)

def plot_ncc_figure(r, mesa_y, dedalus_y, N, ylabel="", fig_name="", out_dir='.', zero_line=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
    ax1.plot(r, dedalus_y, label='dedalus', c='red')
    plt.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')

    ax2 = fig.add_subplot(2,1,2)
    difference = np.abs(1 - dedalus_y/mesa_y)
    ax2.plot(r, np.abs(difference).flatten())
    ax2.set_ylabel('abs(1 - dedalus/mesa)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')
    fig.suptitle('coeff bandwidth = {}'.format(N))
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)


#def load_data(nr1, nr2, r_int, get_dimensions=False):
Nmax = int(args['--Nmax'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
out_dir  = read_file.replace('/LOGS/', '_').replace('.data', '_halfStar')
if args['--pre_log_folder'] != '':
    out_dir = '{:s}_{:s}'.format(args['--pre_log_folder'], out_dir)
print('saving files to {}'.format(out_dir))
out_file = '{:s}/nccs_{}.h5'.format(out_dir, Nmax)
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))


p = mr.MesaData(read_file)


mass = p.mass[::-1] * u.M_sun
mass = mass.to('g')

r = p.radius[::-1] * u.R_sun
r = r.to('cm')

rho = 10**p.logRho[::-1] * u.g / u.cm**3
P = 10**p.logP[::-1] * u.g / u.cm / u.s**2
eps = p.eps_nuc[::-1] * u.erg / u.g / u.s
nablaT = p.gradT[::-1] #dlnT/dlnP
T = 10**p.logT[::-1] * u.K
cp = p.cp[::-1]  * u.erg / u.K / u.g
cv = p.cv[::-1]  * u.erg / u.K / u.g
opacity = p.opacity[::-1] * (u.cm**2 / u.g)
mu = p.mu[::-1]
N2 = p.brunt_N2[::-1] / u.s**2
N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
Luminosity = p.luminosity[::-1] * u.L_sun
Luminosity = Luminosity.to('erg/s')
L_conv = p.conv_L_div_L[::-1]*Luminosity
csound = p.csound[::-1] * u.cm / u.s

rad_diff = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)
rad_diff = rad_diff.cgs


cgs_G = constants.G.to('cm^3/(g*s^2)')
g = cgs_G*mass/r**2
gamma = cp/cv

#Thermo gradients
chiRho  = p.chiRho[::-1]
chiT    = p.chiT[::-1]
nablaT    =  p.gradT[::-1]
nablaT_ad = p.grada[::-1]
dlogPdr = -rho*g/P
gamma1  = dlogPdr/(-g/csound**2)
dlogrhodr = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
dlogTdr   = dlogPdr*(nablaT)
N2_therm_approx = g*(dlogPdr/gamma1 - dlogrhodr)
grad_s = cp*N2/g #includes composition terms



# Heating
#H = rho * eps
#C = (np.gradient(Luminosity-L_conv,r)/(4*np.pi*r**2))
#H_eff = H - C
H_eff = (np.gradient(L_conv,r)/(4*np.pi*r**2))

H0 = H_eff[0]
H_NCC = ((H_eff)  / H0)


#Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1])
core_cz_bound = 0.995*mass[cz_bool][-1] # 0.9 to avoid some of the cz->rz transition region.
bound_ind = np.argmin(np.abs(mass - core_cz_bound))


#fig = plt.figure()
#ax1 = fig.add_subplot(2,1,1)
#plt.plot(r, np.abs(N2), label='brunt2_full', c='k', lw=3)
#plt.plot(r, np.abs(N2_structure), label='brunt2_structure', c='r')
#plt.plot(r, np.abs(N2_composition), label='brunt2_comp', c='orange')
#plt.yscale('log')
#plt.ylabel(r'$N^2$')
#plt.legend(loc='best')
#ax2 = fig.add_subplot(2,1,2)
#plt.yscale('log')
#plt.plot(r, np.abs(N2), label='brunt2_full', c='k', lw=3)
#plt.plot(r, np.abs(N2_structure), label='brunt2_structure', c='r')
#plt.plot(r, np.abs(N2_composition), label='brunt2_comp', c='orange')
##plt.plot(r, np.abs(N2_therm_approx), label='brunt2_structure', c='r')
##plt.plot(r, np.abs(N2 - N2_therm_approx), label='brunt2_comp', c='orange', lw=0.5)
#plt.xlim(3*L.value/4, 5*L.value/4)
#plt.suptitle(filename)
#plt.ylabel(r'$N^2$')
#plt.xlabel(r'radius (cm)')
#plt.savefig('{:s}/{:s}_star_brunt_fig.png'.format(out_dir, filename.split('.data')[0]), dpi=200, bbox_inches='tight')


#Nondimensionalization
halfStar_r = r[-1]/2
L = L_CZ  = r[bound_ind]
g0 = g[bound_ind] 
rho0 = rho[0]
P0 = P[0]
T0 = T[0]
cp0 = cp[0]
gamma0 = gamma[0]
tau = (H0/L**2/rho0)**(-1/3)
tau = tau.cgs
print('one time unit is {:.2e}'.format(tau))
u_H = L/tau

Pe = u_H*L/rad_diff

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
plt.plot(r, N2, label='brunt2_full', c='k', lw=3)
plt.plot(r, N2_structure, label='brunt2_structure', c='r')
plt.plot(r, N2_composition, label='brunt2_comp', c='orange')
plt.yscale('log')
plt.ylabel(r'$N^2$')
plt.legend(loc='best')
ax2 = fig.add_subplot(2,1,2)
plt.yscale('log')
plt.axhline(1e3, c='k', lw=0.5)
plt.axhline(1e4, c='k', lw=0.5)
plt.axhline(1e5, c='k', lw=0.5)
plt.plot(r, Pe, c='k', lw=3)
plt.suptitle(filename)
plt.ylabel(r'Pe')
plt.xlabel(r'radius (cm)')
plt.savefig('{:s}/{:s}_brunt_and_Pe_fig.png'.format(out_dir, filename.split('.data')[0]), dpi=200, bbox_inches='tight')



Ma2 = u_H**2 / ((gamma0-1)*cp0*T0)
s_c = Ma2*(gamma0-1)*cp0

maxR = float(1.1)
sim_bool = r <= 1.1*L
#maxR = float(halfStar_r/L)
#sim_bool = r <= halfStar_r

r_sim = r[sim_bool]/L

c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
b = basis.BallBasis(c, (1, 1, Nmax+1), radius=maxR, dtype=np.float64, dealias=(1, 1, 1))
φg, θg, rg = b.global_grids((1, 1, 1))

grad = lambda A: operators.Gradient(A, c)
dot  = lambda A, B: arithmetic.DotProduct(A, B)
print(L, L_CZ, L_CZ/L, maxR)


r_vec  = field.Field(dist=d, bases=(b,), dtype=np.float64, tensorsig=(c,))
r_vec['g'][2,:] = 1


### Log Density
N = 16
ln_rho_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
ln_rho = np.log(rho/rho0)[sim_bool]
ln_rho_interp = np.interp(rg, r_sim, ln_rho)
ln_rho_field['g'] = ln_rho_interp
ln_rho_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_rho_interp.flatten(), (-1)+ln_rho_field['g'].flatten(), N, ylabel=r"$\ln\rho - 1$", fig_name="ln_rho", out_dir=out_dir)

N = 32
grad_ln_rho_field  = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
grad_ln_rho_interp = np.interp(rg, r_sim, dlogrhodr[sim_bool]*L)
grad_ln_rho_field['g'][2] = grad_ln_rho_interp
grad_ln_rho_field['c'][:,:,:,N:] = 0
plot_ncc_figure(rg.flatten(), grad_ln_rho_interp.flatten(), grad_ln_rho_field['g'][2].flatten(), N, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho", out_dir=out_dir)

### log (temp)
N = 16
ln_T_field  = field.Field(dist=d, bases=(b,), dtype=np.float64)
ln_T = np.log((T)[sim_bool]/T0)
ln_T_interp = np.interp(rg, r_sim, ln_T)
ln_T_field['g'] = ln_T_interp
ln_T_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_T_interp.flatten(), (-1)+ln_T_field['g'].flatten(), N, ylabel=r"$\ln(T) - 1$", fig_name="ln_T", out_dir=out_dir)

N = 32
grad_ln_T_field  = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=np.float64)
grad_ln_T_interp = np.interp(rg, r_sim, dlogTdr[sim_bool]*L)
grad_ln_T_field['g'][2] = grad_ln_T_interp 
grad_ln_T_field['c'][:, :, :, N:] = 0
plot_ncc_figure(rg.flatten(), grad_ln_T_interp.flatten(), grad_ln_T_field['g'][2].flatten(), N, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_T", out_dir=out_dir)

### Temperature
N = 10
T_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
T_nondim = (T)[sim_bool] / T0
T_interp = np.interp(rg, r_sim, T_nondim)
T_field['g'] = T_interp
T_field['c'][:, :, N:] = 0
plot_ncc_figure(rg.flatten(), T_interp.flatten(), T_field['g'].flatten(), N, ylabel=r"$T/T_c$", fig_name="T", out_dir=out_dir)



from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

if Nmax == 127:
    width = 0.045
    N = 32
    N_after = -1
    center =  0.99*(L_CZ/L).value
elif Nmax == 255:
    width = 0.02
    N = 127
    N_after = -1
    center = 0.99*(L_CZ/L).value
elif Nmax == 383:
    width = 0.015
    N = 192
    N_after = -1
    center = 0.99*(L_CZ/L).value
elif Nmax == 511:
    width = 0.01
    N = 192
    N_after = -1
    center = 0.99*(L_CZ/L).value

### effective heating / (rho * T)
H_field = field.Field(dist=d, bases=(b,), dtype=np.float64)
H_interp = np.interp(rg, r_sim, H_NCC[sim_bool])
H_interp_plot = np.interp(rg, r_sim, (H_NCC * rho0*T0/rho/T)[sim_bool])
H_field['g'] = H_interp / T_field['g'] / np.exp(ln_rho_field['g'])
plot_ncc_figure(rg.flatten(), H_interp_plot.flatten(), H_field['g'].flatten(), N, ylabel=r"$(H_{eff}/(\rho c_p T))$ (nondimensional)", fig_name="H_eff", out_dir=out_dir, zero_line=True)

### entropy gradient
grad_s_field  = field.Field(dist=d, bases=(b,), dtype=np.float64, tensorsig=(c,))
grad_s_interp = np.interp(rg, r_sim, grad_s[sim_bool]*L/s_c)
grad_s_base = np.copy(grad_s_interp)
flat_window = (rg.flatten() > 0.95*L_CZ/L)*(rg.flatten() < 1.015*L_CZ/L)
arg_flat     = np.argmin(np.abs(grad_s_interp.flatten() - grad_s_interp.flatten()[flat_window].max()))
grad_s_base[:,:,:arg_flat] = grad_s_interp[:,:,arg_flat]# grad_s_interp.flatten() > 0].flatten()[0]
#grad_s_base[:,:,grad_s_interp.flatten() <= 0] = grad_s_interp[:,:, grad_s_interp.flatten() > 0].flatten()[0]
grad_s_field['g'][2] = grad_s_base
grad_s_field['c'][:,:,:,N:] = 0
grad_s_field['g'][2] *= zero_to_one(rg, center, width=width*(L_CZ/L).value)
grad_s_field['c'][:,:,:,N_after:] = 0
plot_ncc_figure(rg.flatten(), grad_s_interp.flatten(), grad_s_field['g'][2].flatten(), N, ylabel=r"$L(\nabla s/s_c)$", fig_name="grad_s", out_dir=out_dir, zero_line=True)


plt.figure()
plt.plot(rg.flatten(), grad_s_field['g'][2].flatten())
plt.plot(rg.flatten(), -grad_s_field['g'][2].flatten(), ls='--')
plt.plot(rg.flatten(), grad_s_interp.flatten()) 
plt.plot(rg.flatten(), grad_s_base.flatten()) 
plt.yscale('log')
plt.show()

with h5py.File('{:s}'.format(out_file), 'w') as f:
    f['r']     = rg
    f['T']     = T_field['g']
    f['H_eff'] = H_field['g']
    f['ln_ρ']  = ln_rho_field['g'] 
    f['ln_T']  = ln_T_field['g']
    f['grad_ln_T']  = grad_ln_T_field['g']
    f['grad_ln_ρ']  = grad_ln_rho_field['g']
    f['grad_s0']    = grad_s_field['g']

    f['maxR']   = maxR
    f['L']   = L
    f['g0']  = g0
    f['ρ0']  = rho0
    f['P0']  = P0
    f['T0']  = T0
    f['H0']  = H0
    f['tau'] = tau 
    f['Ma2'] = tau 
