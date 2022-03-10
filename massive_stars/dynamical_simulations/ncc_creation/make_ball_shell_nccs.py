"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_ball_shell_nccs.py [options]

Options:
    --Re=<R>        simulation reynolds/peclet number [default: 4e3]
    --NB=<N>        Maximum radial degrees of freedom (ball) [default: 128]
    --NS=<N>        Maximum radial degrees of freedom (shell) [default: 128]
    --file=<f>      Path to MESA log file [default: ../../mesa_models/zams_15Msol/LOGS/profile47.data]
    --out_dir=<d>   output directory [default: nccs_15msol]
    --dealias=<n>   Radial dealiasing factor of simulation [default: 1.5]

    --no_plot       If flagged, don't output plots
"""
import os
import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
import dedalus.public as d3
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.integrate as si
from numpy.polynomial import Chebyshev as Pfit

args = docopt(__doc__)

smooth_H = True
plot=not(args['--no_plot'])

### Function definitions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

def plot_ncc_figure(mesa_r, mesa_y, dedalus_rs, dedalus_ys, Ns, ylabel="", fig_name="", out_dir='.', zero_line=False, log=False, r_int=None, ylim=None, axhline=None):
    """ Plots up a figure to compare a dedalus field to the MESA field it's based on. """
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    if axhline is not None:
        ax1.axhline(axhline, c='k')
    ax1.plot(mesa_r, mesa_y, label='mesa', c='k', lw=3)
    for r, y in zip(dedalus_rs, dedalus_ys):
        ax1.plot(r, y, label='dedalus', c='red')
    plt.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    if log:
        ax1.set_yscale('log')
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2 = fig.add_subplot(2,1,2)
    mesa_func = interp1d(mesa_r, mesa_y, bounds_error=False, fill_value='extrapolate') 
    for r, y in zip(dedalus_rs, dedalus_ys):
        diff = np.abs(1 - mesa_func(r)/y)
        ax2.plot(r, diff)
    ax2.axhline(1e-1, c='k', lw=0.5)
    ax2.axhline(1e-2, c='k', lw=0.5)
    ax2.axhline(1e-3, c='k', lw=0.5)
    ax2.set_ylabel('abs(1 - mesa/dedalus)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')

    ax2.set_ylim(1e-4, 1)
    fig.suptitle('coeff bandwidth = {}, {}'.format(Ns[0], Ns[1]))
    if r_int is not None:
        for ax in [ax1, ax2]:
            ax.axvline(r_int, c='k')
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

### Read in command line args & generate output path & file
nrB = NmaxB = int(args['--NB'])
nrS = NmaxS = int(args['--NS'])
dealias = float(args['--dealias'])
simulation_Re = float(args['--Re'])
read_file = args['--file']
filename = read_file.split('/LOGS/')[-1]
out_dir  = args['--out_dir'] + '/'
out_file = '{:s}/ballShell_nccs_B{}_S{}_Re{}_de{}.h5'.format(out_dir, nrB, nrS, args['--Re'], args['--dealias'])
print('saving output to {}'.format(out_file))
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))

### Read MESA file
p = mr.MesaData(read_file)
mass           = (p.mass[::-1] * u.M_sun).cgs
r              = (p.radius[::-1] * u.R_sun).cgs
rho            = 10**p.logRho[::-1] * u.g / u.cm**3
P              = p.pressure[::-1] * u.g / u.cm / u.s**2
T              = p.temperature[::-1] * u.K
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
grad_s          = cp*N2/g #entropy gradient, for NCC, includes composition terms
L_conv          = conv_L_div_L*Luminosity
dTdr            = (T)*dlogTdr

#True calculation of rad_diff, rad_cond
#rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs
#rad_cond        = rho*cp*rad_diff

#Calculate k_rad using luminosities and smooth things.
k_rad = rad_cond = -(Luminosity - L_conv)/(4*np.pi*r**2*dTdr)
rad_diff        = k_rad / (rho * cp)


### Split up the domain
# Find edge of core cz
cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1]) #rudimentary but works
core_index  = np.argmin(np.abs(mass - mass[cz_bool][-1]))
core_cz_radius = r[core_index]
r_inner_MESA   = r[core_index]*1.1 #outer radius of BallBasis; inner radius of SphericalShellBasis

# Specify fraction of total star to simulate
fracStar   = 0.975 #Simulate this much of the star, from r = 0 to r = R_*
r_outer_MESA    = fracStar*R_star
print('fraction of FULL star simulated: {}, up to r={:.3e}'.format(fracStar, r_outer_MESA))

#Set things up to slice out the star appropriately
ball_bool     = r <= r_inner_MESA
shell_bool    = (r > r_inner_MESA)*(r <= r_outer_MESA)
sim_bool      = r <= r_outer_MESA

# Calculate heating function
# Goal: H_eff= np.gradient(L_conv,r, edge_order=1)/(4*np.pi*r**2) # Heating, for ncc, H = rho*eps - portion carried by radiation
# (1/4pir^2) dL_conv/dr = rho * eps + (1/r^2)d/dr (r^2 k_rad dT/dr) -> chain rule
eo=2
H_eff = (1/(4*np.pi*r**2))*np.gradient(Luminosity, r, edge_order=eo) + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
H_eff_secondary = rho*eps_nuc + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
H_eff[:2] = H_eff_secondary[:2]


sim_H_eff = np.copy(H_eff)
L_conv_sim = np.zeros_like(L_conv)
L_eps = np.zeros_like(Luminosity)
for i in range(L_conv_sim.shape[0]):
    L_conv_sim[i] = np.trapz((4*np.pi*r**2*sim_H_eff)[:1+i], r[:1+i])
    L_eps[i] = np.trapz((4*np.pi*r**2*rho*eps_nuc)[:i+1], r[:i+1])
L_excess = L_conv_sim[-5] - Luminosity[-5]

#construct internal heating field
if smooth_H:
    #smooth CZ-RZ transition
    L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.15*core_cz_radius)
    L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)

    transition_region = (r > 0.5*core_cz_radius)
    sim_H_eff[transition_region] = ((1/(4*np.pi*r**2))*np.gradient(L_conv_sim, r, edge_order=eo))[transition_region]

#    plt.figure()
#    plt.axhline(0, c='k')
#    plt.plot(r, L_conv)
#    plt.plot(r, L_conv_sim, c='k', ls='--')
#    plt.figure()
#    plt.plot(r, H_eff)
#    plt.plot(r, sim_H_eff, ls='--', c='k')
#    plt.show()
else:
    sim_H_eff = H_eff


#Nondimensionalization
L_CZ    = core_cz_radius
L_nd    = L_CZ
#L_nd    = r_outer_MESA - r_inner_MESA
T_nd    = T[0]
m_nd    = rho[0] * L_nd**3
H0      = (rho*eps_nuc)[0]
tau_nd  = ((H0*L_nd/m_nd)**(-1/3)).cgs
rho_nd  = m_nd/L_nd**3
u_nd    = L_nd/tau_nd
s_nd    = L_nd**2 / tau_nd**2 / T_nd
rad_diff_nd = inv_Pe_rad = rad_diff * (tau_nd / L_nd**2)
print('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
print('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))

#Central values
rho_r0    = rho[0]
P_r0      = P[0]
T_r0      = T[0]
cp_r0     = cp[0]
gamma1_r0  = gamma1[0]
Ma2_r0 = (u_nd**2 / ((gamma1_r0-1)*cp_r0*T_r0)).cgs
print('estimated mach number: {:.3e}'.format(np.sqrt(Ma2_r0)))

cp_surf = cp[shell_bool][-1]

#MESA radial values, in simulation units
r_inner = (r_inner_MESA/L_nd).value
r_outer = (r_outer_MESA/L_nd).value
r_ball_nd  = (r[ball_bool]/L_nd).cgs
r_shell_nd = (r[shell_bool]/L_nd).cgs


# Get some timestepping & wave frequency info
N2max_ball = N2[ball_bool].max()
N2max_shell = N2[shell_bool].max()
shell_points = len(N2[shell_bool])
N2plateau = np.median(N2[int(shell_points*0.25):int(shell_points*0.75)])
f_nyq_ball  = np.sqrt(N2max_ball)/(2*np.pi)
f_nyq_shell = np.sqrt(N2max_shell)/(2*np.pi)
f_nyq    = 2*np.max((f_nyq_ball*tau_nd, f_nyq_shell*tau_nd))
nyq_dt   = (1/f_nyq) 

kepler_tau     = 30*60*u.s
max_dt_kepler  = kepler_tau/tau_nd

max_dt = max_dt_kepler
print('needed nyq_dt is {} s / {} % of a heating time (Kepler 30 min is {} %) '.format(nyq_dt*tau_nd, nyq_dt*100, max_dt_kepler*100))

### Make dedalus domain
c = d3.SphericalCoordinates('φ', 'θ', 'r')
d = d3.Distributor((c,), mesh=None, dtype=np.float64)
bB = d3.BallBasis(c, (8, 4, nrB), radius=r_inner, dtype=np.float64, dealias=(1,1,dealias))
bS = d3.ShellBasis(c, (8, 4, nrS), radii=(r_inner, r_outer), dtype=np.float64, dealias=(1,1,dealias))
φB, θB, rB = bB.global_grids((1, 1, dealias))
φS, θS, rS = bS.global_grids((1, 1, dealias))

fd = d.Field(bases=bB)
fd.preset_scales(bB.domain.dealias)
gradfd = d3.grad(fd).evaluate()

grad = lambda A: d3.Gradient(A, c)

def make_NCC(basis, interp_args, Nmax=32, vector=False, grid_only=False):
    if not grid_only:
        scales = (1, 1, Nmax/basis.radial_basis.radial_size)
    else:
        scales = basis.dealias
    rvals = basis.global_grid_radius(scales[2])
    interp = np.interp(rvals, *interp_args)
    if vector:
        this_field = d.VectorField(c, bases=basis)
        this_field.change_scales(scales)
        this_field['g'][2] = interp
    else:
        from dedalus.core import field
        this_field = field.Field(dist=d, bases=(basis,), dtype=np.float64)
        this_field.change_scales(scales)
        this_field['g'] = interp
    if not grid_only:
        this_field.change_scales(basis.dealias)
    return this_field, interp

### Log Density 
NmaxB, NmaxS = 32, 32
ln_rho_fieldB, ln_rho_interpB = make_NCC(bB, (r_ball_nd, np.log(rho/rho_nd)[ball_bool]), Nmax=NmaxB)
grad_ln_rho_fieldB, grad_ln_rho_interpB = make_NCC(bB, (r_ball_nd, dlogrhodr[ball_bool]*L_nd), Nmax=NmaxB, vector=True)
ln_rho_fieldS, ln_rho_interpS = make_NCC(bS, (r_shell_nd, np.log(rho/rho_nd)[shell_bool]), Nmax=NmaxS)
grad_ln_rho_fieldS, grad_ln_rho_interpS = make_NCC(bS, (r_shell_nd, dlogrhodr[shell_bool]*L_nd), Nmax=NmaxS, vector=True)

if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, np.log(rho/rho_nd)[sim_bool], (rB.flatten(), rS.flatten()), (ln_rho_fieldB['g'][:1,:1,:].flatten(), ln_rho_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\ln\rho$", fig_name="ln_rho", out_dir=out_dir, log=False, r_int=r_inner)
    plot_ncc_figure(r[sim_bool]/L_nd, (dlogrhodr*L_nd)[sim_bool], (rB.flatten(), rS.flatten()), (grad_ln_rho_fieldB['g'][2][:1,:1,:].flatten(), grad_ln_rho_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho", out_dir=out_dir, log=False, r_int=r_inner)

### Log Temperature
NmaxB, NmaxS = 16, 32
ln_T_fieldB, ln_T_interpB  = make_NCC(bB, (r_ball_nd, np.log(T/T_nd)[ball_bool]), Nmax=NmaxB)
grad_ln_T_fieldB, grad_ln_T_interpB  = make_NCC(bB, (r_ball_nd, dlogTdr[ball_bool]*L_nd), Nmax=NmaxB, vector=True)
ln_T_fieldS, ln_T_interpS  = make_NCC(bS, (r_shell_nd, np.log(T/T_nd)[shell_bool]), Nmax=NmaxS)
grad_ln_T_fieldS, grad_ln_T_interpS  = make_NCC(bS, (r_shell_nd, dlogTdr[shell_bool]*L_nd), Nmax=NmaxS, vector=True)

if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, np.log(T/T_nd)[sim_bool], (rB.flatten(), rS.flatten()), (ln_T_fieldB['g'][:1,:1,:].flatten(), ln_T_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\ln T$", fig_name="ln_T", out_dir=out_dir, log=False, r_int=r_inner)
    plot_ncc_figure(r[sim_bool]/L_nd, (dlogTdr*L_nd)[sim_bool], (rB.flatten(), rS.flatten()), (grad_ln_T_fieldB['g'][2][:1,:1,:].flatten(), grad_ln_T_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\nabla\ln T$", fig_name="grad_ln_T", out_dir=out_dir, log=False, r_int=r_inner)

### Temperature
NmaxB, NmaxS = 32, 32
T_fieldB, T_interpB = make_NCC(bB, (r_ball_nd, (T/T_nd)[ball_bool]), Nmax=NmaxB)
T_fieldS, T_interpS = make_NCC(bS, (r_shell_nd, (T/T_nd)[shell_bool]), Nmax=NmaxS)

if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, (T/T_nd)[sim_bool], (rB.flatten(), rS.flatten()), (T_fieldB['g'][:1,:1,:].flatten(), T_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$T$", fig_name="T", out_dir=out_dir, log=True, r_int=r_inner)

### Temperature gradient
NmaxB, NmaxS = 32, 32
grad_T_fieldB, grad_T_interpB = make_NCC(bB, (r_ball_nd,  (L_nd/T_nd)*dTdr[ball_bool]), Nmax=NmaxB, vector=True)
grad_T_fieldS, grad_T_interpS = make_NCC(bS, (r_shell_nd, (L_nd/T_nd)*dTdr[shell_bool]), Nmax=NmaxS, vector=True)
grad_T_fieldS['c']
grad_T_fieldS['g']

if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, -(L_nd/T_nd)*dTdr[sim_bool], (rB.flatten(), rS.flatten()), (-grad_T_fieldB['g'][2][:1,:1,:].flatten(), -grad_T_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$-\nabla T$", fig_name="grad_T", out_dir=out_dir, log=True, r_int=r_inner)

### Radiative diffusivity & gradient
#deg = 20
#shell_frac = 0.85
#fit = np.polyfit(r_shell_nd[r_shell_nd > r_shell_nd.max()*shell_frac].value, rad_diff_nd[shell_bool][r_shell_nd > r_shell_nd.max()*shell_frac].value, deg)
#polyfunc = np.poly1d(fit)
#plt.figure()
#plt.plot(r_shell_nd, rad_diff_nd[shell_bool])
#plt.plot(r_shell_nd, polyfunc(r_shell_nd.value))
#plt.yscale('log')
#plt.axhline(1/simulation_Re)
#



NmaxB, NmaxS = 8, 100#np.min((nrS - 1, 126))
transition = (r/L_nd)[inv_Pe_rad > 1/simulation_Re][0].value
gradPe_B_cutoff = 10
gradPe_S_cutoff = 120
inv_Pe_rad_fieldB, inv_Pe_rad_interpB = make_NCC(bB, (r_ball_nd,  inv_Pe_rad[ball_bool] ), Nmax=NmaxB)
inv_Pe_rad_fieldS, inv_Pe_rad_interpS = make_NCC(bS, (r_shell_nd, inv_Pe_rad[shell_bool]), Nmax=NmaxS)
inv_Pe_rad_fieldB['g'] = 1/simulation_Re
inv_Pe_rad_fieldS['g'][inv_Pe_rad_fieldS['g'] < (1/simulation_Re)] = (1/simulation_Re)

#Smooth inv_Pe transition
switch_ind = np.where(inv_Pe_rad_fieldS['g'][0,0,:] > 1/simulation_Re)[0][0]
roll = lambda array, i, n_roll: np.mean(array[i-int(n_roll/2):i+int(n_roll/2)])
n_roll = int(nrS/20)
for i in range(int(2*n_roll)):
    this_ind = switch_ind - n_roll + i
    inv_Pe_rad_fieldS['g'][:,:,this_ind] = roll(inv_Pe_rad_fieldS['g'][0,0,:], this_ind, n_roll) 
#inv_Pe_rad_fieldS['g'] += (1/simulation_Re)#[inv_Pe_rad_fieldS['g'] < (1/simulation_Re)] = (1/simulation_Re)


grad_inv_Pe_B = d3.Gradient(inv_Pe_rad_fieldB, c).evaluate()
grad_inv_Pe_B['c'][:,:,:,gradPe_B_cutoff:] = 0
grad_inv_Pe_B['g'] = 0
#
grad_inv_Pe_S = d3.grad(inv_Pe_rad_fieldS).evaluate()
grad_inv_Pe_S['c'][:,:,:,int(NmaxS/3):] = 0
grad_inv_Pe_S['g'][2,] *= zero_to_one(rS, rS[0,0,switch_ind], width=(r_outer-r_inner)/30)
grad_inv_Pe_S['c'][:,:,:,gradPe_S_cutoff:] = 0

grad_inv_Pe_rad = np.gradient(inv_Pe_rad, r)
if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, inv_Pe_rad[sim_bool], (rB.flatten(), rS.flatten()), (inv_Pe_rad_fieldB['g'][:1,:1,:].flatten(), inv_Pe_rad_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\mathrm{Pe}^{-1}$", fig_name="inv_Pe_rad", out_dir=out_dir, log=True, r_int=r_inner, axhline=1/simulation_Re)
    plot_ncc_figure(r[sim_bool]/L_nd, np.gradient(inv_Pe_rad, r/L_nd)[sim_bool], (rB.flatten(), rS.flatten()), (grad_inv_Pe_B['g'][2][:1,:1,:].flatten(), grad_inv_Pe_S['g'][2][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$\nabla\mathrm{Pe}^{-1}$", fig_name="grad_inv_Pe_rad", out_dir=out_dir, log=True, r_int=r_inner, ylim=(1e-4/simulation_Re, 1), axhline=1/simulation_Re)

### effective heating / (rho * T)
#Logic for smoothing heating profile at outer edge of CZ. Adjust outer edge of heating
H_NCC = ((sim_H_eff)  / H0) * (rho_nd*T_nd/rho/T)
H_NCC_true = ((H_eff)  / H0) * (rho_nd*T_nd/rho/T) * one_to_zero(r, 1.5*L_nd, width=0.1*L_nd)
NmaxB, NmaxS = 60, 10
H_fieldB, H_interpB = make_NCC(bB, (r_ball_nd, H_NCC[ball_bool]), Nmax=NmaxB, grid_only=True)
H_fieldS, H_interpS = make_NCC(bS, (r_shell_nd, H_NCC[shell_bool]), Nmax=NmaxS, grid_only=True)
if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, H_NCC_true[sim_bool], (rB.flatten(), rS.flatten()), (H_fieldB['g'][:1,:1,:].flatten(), H_fieldS['g'][:1,:1,:].flatten()), (NmaxB, NmaxS), ylabel=r"$H$", fig_name="heating", out_dir=out_dir, log=False, r_int=r_inner)

### entropy gradient
transition_point = 1.02
width = 0.02
center =  transition_point - 0.5*width
width *= (L_CZ/L_nd).value
center *= (L_CZ/L_nd).value

#Build a nice function for our basis in the ball
grad_s_smooth = np.copy(grad_s)
flat_value  = np.interp(transition_point, r/L_nd, grad_s)
grad_s_smooth[r/L_nd < transition_point] = flat_value

NmaxB, NmaxS = 31, 31
NmaxB_after = nrB - 1
grad_s_fieldB, grad_s_interpB = make_NCC(bB, (r_ball_nd, (grad_s_smooth*L_nd/s_nd)[ball_bool]), Nmax=NmaxB, vector=True)
grad_s_interpB = np.interp(rB, r_ball_nd, (grad_s*L_nd/s_nd)[ball_bool])
grad_s_fieldS, grad_s_interpS = make_NCC(bS, (r_shell_nd, (grad_s*L_nd/s_nd)[shell_bool]), Nmax=NmaxS, vector=True)
grad_s_fieldB['g'][2] *= zero_to_one(rB, center, width=width)
grad_s_fieldB['c'][:,:,:,NmaxB_after:] = 0

if plot:
    plot_ncc_figure(r[sim_bool]/L_nd, (grad_s*L_nd/s_nd)[sim_bool], (rB.flatten(), rS.flatten()), (grad_s_fieldB['g'][2][:1,:1,:].flatten(), grad_s_fieldS['g'][2][:1,:1,:].flatten()), (NmaxB_after, NmaxS), ylabel=r"$\nabla s$", fig_name="grad_s", out_dir=out_dir, log=True, r_int=r_inner, axhline=1)

with h5py.File('{:s}'.format(out_file), 'w') as f:
    # Save output fields.
    # slicing preserves dimensionality
    f['rB']          = rB
    f['TB']          = T_fieldB['g'][:1,:1,:]
    f['grad_TB']     = grad_T_fieldB['g'][:,:1,:1,:]
    f['H_effB']      = H_fieldB['g'][:1,:1,:]
    f['ln_ρB']       = ln_rho_fieldB['g'][:1,:1,:]
    f['ln_TB']       = ln_T_fieldB['g'][:1,:1,:]
    f['grad_ln_TB']  = grad_ln_T_fieldB['g'][:,:1,:1,:]
    f['grad_ln_ρB']  = grad_ln_rho_fieldB['g'][:,:1,:1,:]
    f['grad_s0B']    = grad_s_fieldB['g'][:,:1,:1,:]
    f['inv_Pe_radB'] = inv_Pe_rad_fieldB['g'][:1,:1,:]
    f['grad_inv_Pe_radB'] = grad_inv_Pe_B['g'][:,:1,:1,:]

    f['rS']          = rS
    f['TS']          = T_fieldS['g'][:1,:1,:]
    f['grad_TS']     = grad_T_fieldS['g'][:,:1,:1,:]
    f['H_effS']      = H_fieldS['g'][:1,:1,:]
    f['ln_ρS']       = ln_rho_fieldS['g'][:1,:1,:]
    f['ln_TS']       = ln_T_fieldS['g'][:1,:1,:]
    f['grad_ln_TS']  = grad_ln_T_fieldS['g'][:,:1,:1,:]
    f['grad_ln_ρS']  = grad_ln_rho_fieldS['g'][:,:1,:1,:]
    f['grad_s0S']    = grad_s_fieldS['g'][:,:1,:1,:]
    f['inv_Pe_radS'] = inv_Pe_rad_fieldS['g'][:1,:1,:]
    f['grad_inv_Pe_radS'] = grad_inv_Pe_S['g'][:,:1,:1,:]

    #Save properties of the star, with units.
    f['L_nd']   = L_nd
    f['L_nd'].attrs['units'] = str(L_nd.unit)
    f['rho_nd']  = rho_nd
    f['rho_nd'].attrs['units']  = str(rho_nd.unit)
    f['T_nd']  = T_nd
    f['T_nd'].attrs['units']  = str(T_nd.unit)
    f['tau_nd'] = tau_nd 
    f['tau_nd'].attrs['units'] = str(tau_nd.unit)
    f['m_nd'] = m_nd 
    f['m_nd'].attrs['units'] = str(m_nd.unit)
    f['s_nd'] = s_nd
    f['s_nd'].attrs['units'] = str(s_nd.unit)
    f['P_r0']  = P_r0
    f['P_r0'].attrs['units']  = str(P_r0.unit)
    f['H0']  = H0
    f['H0'].attrs['units']  = str(H0.unit)
    f['N2max_ball'] = N2max_ball
    f['N2max_ball'].attrs['units'] = str(N2max_ball.unit)
    f['N2max_shell'] = N2max_shell
    f['N2max_shell'].attrs['units'] = str(N2max_shell.unit)
    f['N2max'] = np.max((N2max_ball.value, N2max_shell.value))
    f['N2max'].attrs['units'] = str(N2max_ball.unit)
    f['N2plateau'] = N2plateau
    f['N2plateau'].attrs['units'] = str(N2plateau.unit)
    f['cp_surf'] = cp_surf
    f['cp_surf'].attrs['units'] = str(cp_surf.unit)
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

    f['r_inner']   = r_inner
    f['r_outer']   = r_outer
    f['max_dt'] = max_dt
    f['Ma2_r0'] = Ma2_r0
    for k in ['r_inner', 'r_outer', 'max_dt', 'Ma2_r0']:
        f[k].attrs['units'] = 'dimensionless'
print('finished saving NCCs to {}'.format(out_file))


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8,4))

for i in range(2):
    for j in range(3):
        axs[i][j].axvline(r_inner_MESA.value, c='k', lw=0.5)
        axs[i][j].axvline(r_outer_MESA.value, c='k', lw=0.5)


axs[0][0].plot(r, T)
axs[0][0].plot(rB.flatten()*L_nd, T_nd*T_fieldB['g'][0,0,:], c='k')
axs[0][0].plot(rS.flatten()*L_nd, T_nd*T_fieldS['g'][0,0,:], c='k')
axs[0][0].set_ylabel('T (K)')

axs[0][1].plot(r, np.log(rho/rho_nd))
axs[0][1].plot(rB.flatten()*L_nd, ln_rho_fieldB['g'][0,0,:], c='k')
axs[0][1].plot(rS.flatten()*L_nd, ln_rho_fieldS['g'][0,0,:], c='k')
axs[0][1].set_ylabel(r'$\ln(\rho/\rho_{\rm{nd}})$')

axs[1][0].plot(r, inv_Pe_rad)
axs[1][0].plot(rB.flatten()*L_nd, inv_Pe_rad_fieldB['g'][0,0,:], c='k')
axs[1][0].plot(rS.flatten()*L_nd, inv_Pe_rad_fieldS['g'][0,0,:], c='k')
axs[1][0].set_yscale('log')
axs[1][0].set_ylabel(r'$\chi_{\rm{rad}}\,L_{\rm{nd}}^{-2}\,\tau_{\rm{nd}}$')

#axs[0][1].plot(r, np.gradient(inv_Pe_rad, r))
#axs[0][1].plot(rB.flatten()*L_nd, grad_inv_Pe_B['g'][2,0,0,:]/L_nd, c='k')
#axs[0][1].plot(rS.flatten()*L_nd, grad_inv_Pe_S['g'][2,0,0,:]/L_nd, c='k')
#axs[0][1].set_yscale('log')


#axs[0][3].plot(r, np.log(T/T_nd))
#axs[0][3].plot(rB.flatten()*L_nd, ln_T_fieldB['g'][0,0,:], c='k')
#axs[0][3].plot(rS.flatten()*L_nd, ln_T_fieldS['g'][0,0,:], c='k')

#axs[1][0].plot(r, grad_T*T_nd/L_nd)
#axs[1][0].plot(rB.flatten()*L_nd, T_nd*grad_T_fieldB['g'][2,0,0,:]/L_nd, c='k')
#axs[1][0].plot(rS.flatten()*L_nd, T_nd*grad_T_fieldS['g'][2,0,0,:]/L_nd, c='k')



axs[1][1].plot(r, H_eff/(rho*T), c='b')
axs[1][1].plot(r, eps_nuc / T, c='r')
axs[1][1].plot(rB.flatten()*L_nd, (H0 / rho_nd / T_nd)*H_fieldB['g'][0,0,:], c='k')
axs[1][1].plot(rS.flatten()*L_nd, (H0 / rho_nd / T_nd)*H_fieldS['g'][0,0,:], c='k')
axs[1][1].set_ylim(-5e-4, 2e-3)
#axs[1][1].plot(rB.flatten()*L_nd, -(H0 / rho_nd / T_nd)*H_fieldB['g'][0,0,:], c='k', ls='--')
#axs[1][1].plot(rS.flatten()*L_nd, -(H0 / rho_nd / T_nd)*H_fieldS['g'][0,0,:], c='k', ls='--')
#axs[1][1].set_yscale('log')
axs[1][1].set_ylabel(r'$H/(\rho T)$ (units)')

axs[1][2].plot(r, grad_s)
axs[1][2].plot(rB.flatten()*L_nd, (s_nd/L_nd)*grad_s_fieldB['g'][2,0,0,:], c='k')
axs[1][2].plot(rS.flatten()*L_nd, (s_nd/L_nd)*grad_s_fieldS['g'][2,0,0,:], c='k')
axs[1][2].set_yscale('log')
axs[1][2].set_ylim(1e-6, 1e0)
axs[1][2].set_ylabel(r'$\nabla s$ (erg$\,\rm{g}^{-1}\rm{K}^{-1}\rm{cm}^{-1}$)')



plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig.savefig('dedalus_mesa_figure.png', dpi=200, bbox_inches='tight')

#Plot a propagation diagram
plt.figure()
plt.plot(r, np.sqrt(N2), label=r'$N$')
plt.plot(r, lamb_freq(1), label=r'$S_1$')
plt.plot(r, lamb_freq(10), label=r'$S_{10}$')
plt.plot(r, lamb_freq(100), label=r'$S_{100}$')
plt.xlim(0, r_outer_MESA.value)
plt.xlabel('r (cm)')
plt.ylabel('freq (1/s)')
plt.yscale('log')
plt.legend(loc='best')
plt.savefig('{}/propagation_diagram.png'.format(out_dir), dpi=300, bbox_inches='tight')


