"""
This script plots snapshots of entropy fluctuations in a shell domain through the equator and two meridions.
The goal is to see the entropy fluctuations of waves in the shell domain, masking out the ball domain.

Usage:
    ballShell_plot_shell_entropy.py <root_dir> --r_inner=<r> --r_outer=<r> [options]
    ballShell_plot_shell_entropy.py <root_dir> --mesa_file=<f> [options]
    ballShell_plot_shell_entropy.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: shell_entropy]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

if args['--r_inner'] is not None and args['--r_outer'] is not None:
    r_inner = float(args['--r_inner'])
    r_outer = float(args['--r_outer'])
elif args['--mesa_file'] is not None:
    import h5py
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    print('WARNING: using default r_inner = 1.1 and r_outer = 2.59')
    r_inner = 1.1
    r_outer = 2.59

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, data_dir, fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

# Just plot a single plot (1x1 grid) of the field "T eq"
# remove_x_mean option removes the (numpy horizontal mean) over phi
# divide_x_mean divides the radial mean(abs(T eq)) over the phi direction
plotter.setup_grid(num_rows=1, num_cols=3, polar=True, **plotter_kwargs)
kwargs = {'radial_basis' : 'r', 'r_inner' : r_inner, 'r_outer' : r_outer}
plotter.add_polar_colormesh('s1_S_eq', azimuth_basis='phi', remove_x_mean=True, **kwargs)
plotter.add_meridional_colormesh(left='s1_S(phi=0)', right='s1_S(phi=pi)', colatitude_basis='theta', remove_x_mean=True, **kwargs)
plotter.add_meridional_colormesh(left='s1_S(phi=0.5*pi)', right='s1_S(phi=1.5*pi)', colatitude_basis='theta', remove_x_mean=True, **kwargs)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
