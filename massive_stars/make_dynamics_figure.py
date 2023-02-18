import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap

from plotpal.slices import SlicePlotter
from plotpal.file_reader import match_basis

plt.rcParams['hatch.linewidth'] = 0.25

# Define smooth Heaviside functions
from scipy.special import erf 
def one_to_zero(x, x0, width=0.1):
        return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
        return -(one_to_zero(*args, **kwargs) - 1)


r_outer = 2
sponge_function = lambda r: zero_to_one(r, r_outer - 0.15, 0.07)

class MathTextSciFormatter():
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

sci_formatter = MathTextSciFormatter(fmt="%1.0e")

T_nd = 2.34736e7 #K/nondimensional
R_gas = 18.0285 #nondimensional 



turb_root_dir = 'twoRcore_re3e4_damping/'
turb_start_file = 5
wave_start_file = 1#140
wave_root_dir = 'compressible_re1e4_waves/'
file_dir='slices'
out_name='dynamics_figure'
n_files = 15
start_fig = 1

turb_plotter = SlicePlotter(turb_root_dir, file_dir=file_dir, out_name=out_name, start_file=turb_start_file, n_files=n_files)
wave_plotter = SlicePlotter(wave_root_dir, file_dir=file_dir, out_name=out_name, start_file=wave_start_file, n_files=n_files)

turb_tasks_cz = ['equator(u_B)', 'equator(u_S1)']
turb_tasks = ['equator(u_B)', 'equator(u_S1)']
wave_tasks = ['equator(pom_fluc_B)', 'equator(pom_fluc_S1)', 'equator(pom_fluc_S2)']
turb_coords_cz = dict()
turb_coords = dict()
wave_coords = dict()
for d in [turb_coords, turb_coords_cz, wave_coords]:
    for key in ['r', 'phi', 'rr', 'pp']:
        d[key] = []

with h5py.File('twoRcore_re3e4_damping/star/star_512+256_bounds0-2L_Re3.00e+04_de1.5_cutoff1.0e-10.h5', 'r') as f:
    L_nd = f['L_nd'][()]
    tau_nd = f['tau_nd'][()]
    u_nd = L_nd/tau_nd
    fracstar = 0.93 #for wave case.

#true_maxmin = 8e-3 * u_nd
true_maxmin = 5e4

first = True
with turb_plotter.my_sync:
    while turb_plotter.writes_remain() and wave_plotter.writes_remain():
        turb_dsets, turb_ni = turb_plotter.get_dsets(turb_tasks)
        wave_dsets, wave_ni = wave_plotter.get_dsets(wave_tasks)
        if first:
            for d, dsets, tasks in zip((turb_coords, turb_coords_cz, wave_coords), (turb_dsets, turb_dsets, wave_dsets), \
                                        (turb_tasks, turb_tasks_cz, wave_tasks)):
                for i, task in enumerate(tasks):
                    print(task)
                    d['r'].append(match_basis(dsets[task], 'r'))
                    d['phi'].append(match_basis(dsets[task], 'phi'))
                    d['phi'][-1] = np.append(d['phi'][-1], np.array([d['phi'][-1][0] + 2*np.pi]))
                    rr, pp = np.meshgrid(d['r'][-1], d['phi'][-1])
                    d['rr'].append(rr)
                    d['pp'].append(pp)
                full_rr = np.concatenate(d['rr'], axis=1)
                full_pp = np.concatenate(d['pp'], axis=1)
                d['xx'] = full_rr*np.cos(full_pp)
                d['yy'] = full_rr*np.sin(full_pp)

        fig = plt.figure(figsize=(7.5, 2.7))
        axCore = fig.add_axes([0.7, 0.06, 0.3, 0.88], polar=False)
        axDamp = fig.add_axes([0.35, 0.06, 0.3, 0.88], polar=False)
        axFull = fig.add_axes([0, 0.06, 0.3, 0.88], polar=False)
        caxCore = fig.add_axes([0.745, 0., 0.21, 0.03])
        caxDamp = fig.add_axes([0.395, 0., 0.21, 0.03])
        caxStar = fig.add_axes([0.045, 0., 0.21, 0.03])
        plots = []



        for ax, d, dsets, ni, tasks, i in zip((axCore, axDamp, axFull), (turb_coords_cz, turb_coords, wave_coords), \
                                    (turb_dsets, turb_dsets, wave_dsets), (turb_ni, turb_ni, wave_ni), \
                                    (turb_tasks_cz, turb_tasks, wave_tasks), (0, 1, 2)):
            data = []
            for t in tasks:
                if 'u_' in t:
                   data.append(dsets[t][ni,2,:].squeeze())
                else:
                   data.append(dsets[t][ni,:].squeeze())
            data = np.concatenate(data, axis=1)
            if i == 0:
                data *= u_nd
                vmin = -true_maxmin
                vmax = true_maxmin
            else:
                data -= np.mean(data, axis=0)[None,:]
                if i == 2:
                    data *= T_nd/R_gas
                    vmin = -30
                    vmax = 30
                    wavemin = -1
                    wavemax = 1
                else:
                    data /= np.std(data, axis=0)
                    vmin = -2
                    vmax = 2

            outline_r = d['r'][-1].max()
            if i == 0:
                outline_r = 1.3
            outline_phi = np.linspace(0, 2.1*np.pi, 1000)

            if i == 2:
                full_star_r = outline_r/fracstar
                star_phi = np.linspace(0, np.pi, 500, endpoint=True)
                x = full_star_r*np.cos(star_phi)
                y_bot = -full_star_r*np.sin(star_phi)
                y_top = full_star_r*np.sin(star_phi)
                ax.fill_between(x, y_bot, y_top, hatch='xxxx', facecolor='lightgrey', linewidth=0.5)
                ax.plot(full_star_r*np.cos(outline_phi), full_star_r*np.sin(outline_phi), c='k', lw=0.5)

            if i == 0:
                cmap = 'PuOr_r'
                norm = None
            elif i == 1:
                cmap = 'PiYG_r'
                norm = None
            elif i == 2:
                class TwoInnerPointsNormalize(matplotlib.colors.Normalize):
                    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False):
                        self.low = low
                        self.up = up
                        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

                    def __call__(self, value, clip=None):
                        x, y = [self.vmin, self.low, self.up, self.vmax], [0, 0.15, 0.85, 1]
                        return np.ma.masked_array(np.interp(value, x, y))
                teal = "#33FFFC"
                orange = "#FF5733"
                colors = plt.cm.RdBu_r(np.linspace(0, 1,128))
                colors = list(zip(np.linspace(0.15, 0.85, 128), colors))
                colors = [ (0., teal),] + colors + [(1, orange)]
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors)
                norm = TwoInnerPointsNormalize(vmin=vmin, vmax=vmax, low=wavemin, up=wavemax)


            good = d['xx']**2 + d['yy']**2 <= outline_r**2
            plot = ax.pcolormesh(d['xx'][:,good[0,:]], d['yy'][:,good[0,:]], data[:,good[0,:]], cmap=cmap, shading='nearest', rasterized=True, vmin=vmin, vmax=vmax, norm=norm)
            plots.append(plot)
            if i == 1:
                #add grey mask over damping region
                pmask = sponge_function(np.sqrt(d['xx']**2 +  d['yy']**2))
                t_cmap = np.ones([256, 4])*0.7
                t_cmap[:, 3] = np.linspace(0, 0.6, 256)
                t_cmap = ListedColormap(t_cmap)
                color2 = ax.pcolormesh(d['xx'], d['yy'], pmask, shading='auto', cmap=t_cmap, vmin=0, vmax=1, rasterized=True)
            if i == 0:
                cbar = plt.colorbar(plot, cax=caxCore, orientation='horizontal')
                caxCore.text(-0.01, 0.5, r'$u_r$ (cm$\,$s$^{-1}$)', transform=caxCore.transAxes, va='center', ha='right')
                cbar.set_ticks((vmin, 0, vmax))
                cbar.set_ticklabels([sci_formatter(vmin), '0', sci_formatter(vmax)])
            elif i == 1:
                cbar = plt.colorbar(plot, cax=caxDamp, orientation='horizontal')
                caxDamp.text(-0.01, 0.5, r'$u_r/\sigma(u_r)$', transform=caxDamp.transAxes, va='center', ha='right')
                cbar.set_ticks((vmin, 0, vmax))
                cbar.set_ticklabels(['{:.2f}'.format(vmin), '0', '{:.2f}'.format(vmax)])
            elif i == 2:
                display_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                cbar = matplotlib.colorbar.ColorbarBase(caxStar, cmap=cmap, norm=display_norm, orientation='horizontal')
                caxStar.text(-0.01, 0.5, r"$T'$ (K)", transform=caxStar.transAxes, va='center', ha='right')
                cbar.set_ticks((0, 0.15, 0.5, 0.85, 1))
                cbar.set_ticklabels(['{}'.format(vmin),'{}'.format(wavemin), '0', '{}'.format(wavemax), '{}'.format(vmax)])

            if i == 2:
                ax.text(0.5, 1.05, 'Near Surface Sim', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 1.05, 'Wave Flux Sim', ha='center', va='center', transform=ax.transAxes)



            ax.set_yticks([])
            ax.set_xticks([])
            for direction in ['left', 'right', 'bottom', 'top']:
                ax.spines[direction].set_visible(False)
            ax.plot(outline_r*np.cos(outline_phi), outline_r*np.sin(outline_phi), c='k', lw=0.5)
            if i == 0:
                axDamp.plot(outline_r*np.cos(outline_phi), outline_r*np.sin(outline_phi), c='k', lw=0.5)
                phi_1 = np.pi*0.55
#                phi_1 = np.pi*0.45
                xy1_top = outline_r*np.array((np.cos(phi_1), np.sin(phi_1)))
                xy2_top = xy1_top #(0, outline_r)
                xy1_bot = outline_r*np.array((np.cos(-phi_1), np.sin(-phi_1)))
                xy2_bot = xy1_bot #(0, -outline_r)
                for xy1, xy2 in zip((xy1_top, xy1_bot),(xy2_top, xy2_bot)):
                    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                                                  axesA=ax, axesB=axDamp, color="black", lw=0.5)
                    axDamp.add_artist(con)
            if i == 1:
                axFull.plot(outline_r*np.cos(outline_phi), outline_r*np.sin(outline_phi), c='k', lw=0.5, ls='--')
                phi_1 = np.pi*0.55
#                phi_1 = np.pi*0.45
                xy1_top = outline_r*np.array((np.cos(phi_1), np.sin(phi_1)))
                xy2_top = xy1_top #(0, outline_r)
                xy1_bot = outline_r*np.array((np.cos(-phi_1), np.sin(-phi_1)))
                xy2_bot = xy1_bot #(0, -outline_r)
                for xy1, xy2 in zip((xy1_top, xy1_bot),(xy2_top, xy2_bot)):
                    con = ConnectionPatch(xyA=xy1, xyB=xy2, coordsA="data", coordsB="data",
                                                  axesA=ax, axesB=axFull, color="black", lw=0.5, ls='--')
                    axFull.add_artist(con)

            if i == 2:
                ax.set_xlim(-outline_r*1.01/fracstar, outline_r*1.01/fracstar)
                ax.set_ylim(-outline_r*1.01/fracstar, outline_r*1.01/fracstar)
            else:
                ax.set_xlim(-outline_r*1.01, outline_r*1.01)
                ax.set_ylim(-outline_r*1.01, outline_r*1.01)

        write_num = turb_plotter.current_file_handle['scales/write_number'][turb_ni] 
        figname = '{:s}/{:s}_{:06d}.png'.format(turb_plotter.out_dir, turb_plotter.out_name, int(write_num+start_fig-1))
        fig.savefig(figname, dpi=1000, bbox_inches='tight')
        plt.close(fig)

        first = False


