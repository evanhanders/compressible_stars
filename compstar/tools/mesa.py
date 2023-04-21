from collections import OrderedDict
import numpy as np

import mesa_reader as mr
from astropy import units as u
from astropy import constants

import logging
logger = logging.getLogger(__name__)

aR = constants.sigma_sb.cgs * (4/constants.c.cgs)


class DimensionalMesaReader:
    """ Class to read in MESA profile and store it in a dictionary with astropy units """

    def __init__(self, filename):
        #TODO: figure out how to make MESA the file path w.r.t. stock model path w/o supplying full path here
        logger.info("Reading MESA file {}".format(filename))
        self.p = p = mr.MesaData(filename)
        self.structure = OrderedDict()
        self.structure['mass']           = mass           = (p.mass[::-1] * u.M_sun).cgs
        self.structure['r']              = r              = (p.radius[::-1] * u.R_sun).cgs
        self.structure['rho']            = rho            = 10**p.logRho[::-1] * u.g / u.cm**3
        self.structure['P']              = P              = p.pressure[::-1] * u.g / u.cm / u.s**2
        self.structure['T']              = T              = p.temperature[::-1] * u.K
        self.structure['nablaT']         = nablaT         = p.gradT[::-1] #dlnT/dlnP
        self.structure['nablaT_ad']      = nablaT_ad      = p.grada[::-1]
        self.structure['chiRho']         = chiRho         = p.chiRho[::-1]
        self.structure['chiT']           = chiT           = p.chiT[::-1]
        self.structure['cp']             = cp             = p.cp[::-1]  * u.erg / u.K / u.g
        self.structure['opacity']        = opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
        self.structure['Luminosity']     = Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
        self.structure['conv_L_div_L']   = conv_L_div_L   = p.lum_conv_div_L[::-1]
        self.structure['csound']         = csound         = p.csound[::-1] * u.cm / u.s
        self.structure['N2']             = N2 = N2_mesa   = p.brunt_N2[::-1] / u.s**2
        self.structure['N2_structure']   = N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
        self.structure['N2_composition'] = N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
        self.structure['eps_nuc']        = eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
        self.structure['mu']             = mu             = p.mu[::-1] * u.g / u.mol
        self.structure['lamb_freq_S1']    = lamb_freq_S1  = np.sqrt(1*(1 + 1)) * csound/r


        self.structure['R_star'] = R_star = (p.photosphere_r * u.R_sun).cgs

        self.structure['R_gas'] = R_gas                     = constants.R.cgs / mu[0]
        self.structure['g'] = g                             = constants.G.cgs*mass/r**2
        self.structure['dlogPdr'] = dlogPdr                 = -rho*g/P
        self.structure['gamma1'] = gamma1                   = dlogPdr/(-g/csound**2)
        self.structure['dlogrhodr'] = dlogrhodr             = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
        self.structure['dlogTdr'] = dlogTdr                 = dlogPdr*(nablaT)
        self.structure['grad_s_over_cp'] = grad_s_over_cp   = N2/g #entropy gradient, for NCC, includes composition terms
        self.structure['grad_s'] = grad_s                   = cp * grad_s_over_cp
        self.structure['L_conv'] = L_conv                   = conv_L_div_L*Luminosity
        self.structure['dTdr'] = dTdr                       = (T)*dlogTdr
        self.structure['dTdr_superad'] = dTdr_superad       = dTdr - (-g/cp)

        # Calculate k_rad and radiative diffusivity using luminosities and smooth things.
        self.structure['k_rad']    = k_rad    = rad_cond = -(Luminosity - L_conv)/(4*np.pi*r**2*dTdr)
        self.structure['rad_diff'] = rad_diff = k_rad / (rho * cp)
        #rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs # this is less smooth

        #Gravitational potential, set to -1 at r = R_star
        self.structure['g_phi'] = np.cumsum(g*np.gradient(r))  #gvec = -grad phi; 
        self.structure['g_phi'] -= (self.structure['g_phi'][-1] + 1*u.cm**2/u.s**2) #set g_phi = -1 at r = dmr.R_star

        # see appendix A of jermyn et al 2022 atlas of convection
        logLam = - 17.9 + 1.5*np.log(T/u.K) - 0.5*np.log(rho/(u.g/u.cm**3))
        logLam[T >= 4.2e5*u.K] = -11.5 + np.log(T[T >= 4.2e5*u.K]/u.K) - 0.5*np.log(rho[T >= 4.2e5*u.K]/(u.g/u.cm**3))
        nu_rad = 4*aR*T**4 / (15*constants.c.cgs*rho**2*opacity)
        nu_ie  = (2.21e-15 * u.cm**2 / u.s) * (T/u.K)**(5/2) / (logLam*(rho/(u.g/u.cm**3)))
        self.structure['nu_diff'] = nu_diff = (nu_rad + nu_ie).cgs


def find_core_cz_radius(mesa_file, dimensionless=True, L_conv_threshold=1):
    """ 
    Find the radius of the core convection zone in a MESA profile.
    The convective core is defined as the region where the convective luminosity is greater than a threshold
    and the mass coordinate is less than 90% of the total stellar mass.

    Arguments:
    ----------
    mesa_file : str
        Path to MESA profile file
    dimensionless : bool
        If True, return the radius as a float without astropy units attached.
    L_conv_threshold : float
        Threshold for determining if a region is convective.
    """
    p = mr.MesaData(mesa_file)
    r              = (p.radius[::-1] * u.R_sun).cgs
    mass           = (p.mass[::-1] * u.M_sun).cgs
    Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
    conv_L_div_L   = p.lum_conv_div_L[::-1]
    L_conv         = conv_L_div_L*Luminosity

    cz_bool = (L_conv.value > L_conv_threshold)*(mass < 0.9*mass[-1])
    core_index  = np.argmin(np.abs(mass - mass[cz_bool][-1]))
    core_cz_radius = r[core_index]
    if dimensionless: #no astropy units.
        return core_cz_radius.value
    else:
        return core_cz_radius