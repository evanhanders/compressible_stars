import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

rv = '1.25'
hz_to_invday = 24*60*60

with h5py.File('twoRcore_re3e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_512 = f['cgs_freqs'][()]
    ells_512  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_512 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re2e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_384 = f['cgs_freqs'][()]
    ells_384  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_384 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_256 = f['cgs_freqs'][()]
    ells_256  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_256 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re2e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_128_low = f['cgs_freqs'][()]
    ells_128_low  = f['ells'][()].ravel()
    lum_128_low = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re4e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_128 = f['cgs_freqs'][()]
    ells_128  = f['ells'][()].ravel()
    lum_128 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re1e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_96 = f['cgs_freqs'][()]
    ells_96  = f['ells'][()].ravel()
    lum_96 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]


freqs_512 *= hz_to_invday
freqs_384 *= hz_to_invday
freqs_256 *= hz_to_invday
freqs_128 *= hz_to_invday
freqs_128_low *= hz_to_invday
freqs_96 *= hz_to_invday



from palettable.colorbrewer.sequential import RdPu_6
for freq in np.array([3e-6, 5e-6, 1e-5])*hz_to_invday:
    print('f = {}'.format(freq))
    plt.loglog(ells_96, lum_96[freqs_96 > freq, :][0,:],                 color=RdPu_6.mpl_colors[0], label=r'Re $\sim$ 200')
    plt.loglog(ells_128_low, lum_128_low[freqs_128_low > freq, :][0,:],  color=RdPu_6.mpl_colors[1], label=r'Re $\sim$ 400')
    plt.loglog(ells_128, lum_128[freqs_128 > freq, :][0,:],              color=RdPu_6.mpl_colors[2], label=r'Re $\sim$ 800')
    plt.loglog(ells_256, lum_256[freqs_256 > freq, :][0,:],              color=RdPu_6.mpl_colors[3], label=r'Re $\sim$ 2000')
    plt.loglog(ells_384, lum_384[freqs_384 > freq, :][0,:],              color=RdPu_6.mpl_colors[4], label=r'Re $\sim$ 4000')
    plt.loglog(ells_512, lum_512[freqs_512 > freq, :][0,:],              color=RdPu_6.mpl_colors[5], label=r'Re $\sim$ 6000')
    kh = np.sqrt(ells_512*(ells_512+1))
    plt.loglog(ells_512, 3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlim(1e-2, 1e1)
    plt.ylabel('wave luminosity')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$f = $' + '{}'.format(freq))
    plt.savefig('wave_luminosity_comparison/freq{:0.2e}.png'.format(freq))
    plt.clf()

for ell in range(1, 4):
    print('ell = {}'.format(ell))
    plt.loglog(freqs_96,      np.abs(lum_96[:, ells_96 == ell]),           color=RdPu_6.mpl_colors[0], label=r'Re $\sim$ 200')
    plt.loglog(freqs_128_low, np.abs(lum_128_low[:, ells_128_low == ell]), color=RdPu_6.mpl_colors[1], label=r'Re $\sim$ 400')
    plt.loglog(freqs_128,     np.abs(lum_128[:, ells_128 == ell]),         color=RdPu_6.mpl_colors[2], label=r'Re $\sim$ 800')
    plt.loglog(freqs_256,     np.abs(lum_256[:, ells_256 == ell]),         color=RdPu_6.mpl_colors[3], label=r'Re $\sim$ 2000')
    plt.loglog(freqs_384,     np.abs(lum_384[:, ells_384 == ell]),         color=RdPu_6.mpl_colors[4], label=r'Re $\sim$ 4000')
    plt.loglog(freqs_512,     np.abs(lum_512[:, ells_512 == ell]),         color=RdPu_6.mpl_colors[5], label=r'Re $\sim$ 6000')
#    plt.loglog(freqs_256, 2.14e-28*freqs_256**(-6.5)*ell**2, c='k')
    kh = np.sqrt(ell*(ell+1))
    plt.loglog(freqs_256, 3e-11*(freqs_256/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlabel('freq')
    plt.ylabel('wave luminosity')
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e10, 1e30)
    plt.legend()
    plt.title(r'$\ell = $' + '{}'.format(ell))
    plt.savefig('wave_luminosity_comparison/ell{:03d}.png'.format(ell))
    plt.clf()
#    plt.show()

fig = plt.figure(figsize=(7.5, 3))
ax1 = fig.add_axes([0, 0, 0.45, 0.8])
ax2 = fig.add_axes([0.55, 0, 0.49, 0.8])
cax = fig.add_axes([0.25, 0.95, 0.50, 0.05])

bounds = [200, 400, 800, 2000, 4000, 6000]
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=6)
cmap = mpl.colors.ListedColormap(RdPu_6.mpl_colors)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

ell = 1
freq = 0.5
ax1.loglog(freqs_96,      np.abs(lum_96[:, ells_96 == ell]),           color=RdPu_6.mpl_colors[0], label=r'Re $\sim$ 200')
ax1.loglog(freqs_128_low, np.abs(lum_128_low[:, ells_128_low == ell]), color=RdPu_6.mpl_colors[1], label=r'Re $\sim$ 400')
ax1.loglog(freqs_128,     np.abs(lum_128[:, ells_128 == ell]),         color=RdPu_6.mpl_colors[2], label=r'Re $\sim$ 800')
ax1.loglog(freqs_256,     np.abs(lum_256[:, ells_256 == ell]),         color=RdPu_6.mpl_colors[3], label=r'Re $\sim$ 2000')
ax1.loglog(freqs_384,     np.abs(lum_384[:, ells_384 == ell]),         color=RdPu_6.mpl_colors[4], label=r'Re $\sim$ 4000')
ax1.loglog(freqs_512,     np.abs(lum_512[:, ells_512 == ell]),         color=RdPu_6.mpl_colors[5], label=r'Re $\sim$ 6000')
kh = np.sqrt(ell*(ell+1))
ax1.loglog(freqs_256, 3e-11*(freqs_256/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
ax1.set_ylim(1e8, 1e30)
ax1.set_xlim(1e-2, 2e1)
ax1.text(0.12, 4e28, r'$f^{-6.5}$', rotation=0)
ax1.set_xlabel(r'frequency (day$^{-1}$)')
ax1.set_ylabel(r'Wave Luminosity (erg$\,$s$^{-1}$)')

ax2.loglog(ells_96, lum_96[freqs_96 > freq, :][0,:],                 color=RdPu_6.mpl_colors[0], label=r'Re $\sim$ 200')
ax2.loglog(ells_128_low, lum_128_low[freqs_128_low > freq, :][0,:],  color=RdPu_6.mpl_colors[1], label=r'Re $\sim$ 400')
ax2.loglog(ells_128, lum_128[freqs_128 > freq, :][0,:],              color=RdPu_6.mpl_colors[2], label=r'Re $\sim$ 800')
ax2.loglog(ells_256, lum_256[freqs_256 > freq, :][0,:],              color=RdPu_6.mpl_colors[3], label=r'Re $\sim$ 2000')
ax2.loglog(ells_384, lum_384[freqs_384 > freq, :][0,:],              color=RdPu_6.mpl_colors[4], label=r'Re $\sim$ 4000')
ax2.loglog(ells_512, lum_512[freqs_512 > freq, :][0,:],              color=RdPu_6.mpl_colors[5], label=r'Re $\sim$ 6000')
kh = np.sqrt(ells_384*(ells_384+1))
ax2.loglog(ells_384, 3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
ax2.set_xlim(1, 100)
ax2.set_ylim(1e8, 1e30)
ax2.text(25, 3e27, r'$k_h^4=[\ell(\ell+1)]^2$', rotation=0,ha='left')
ax2.set_xlabel(r'$\ell$')

cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cax.text(-0.02, 0.5, 'Re', ha='right', va='center', transform=cax.transAxes)

plt.savefig('wave_luminosity_comparison/turbulence_waveflux_variation.png', dpi=300, bbox_inches='tight')
plt.savefig('wave_luminosity_comparison/turbulence_waveflux_variation.pdf', dpi=300, bbox_inches='tight')