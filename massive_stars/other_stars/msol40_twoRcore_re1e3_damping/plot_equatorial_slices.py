"""
This script plots snapshots of the evolution of a 2D slice through the equator of a BallBasis simulation.

Usage:
    plot_equatorial_slices.py [options]

Options:
    --root_dir=<s>                      Root directory [default: .]
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --out_name=<out_name>               Name of figure output directory & base name of saved figures [default: snapshots_equatorial]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --r_inner=<float>                   Inner shell radius [default: 1.05]
    --r_outer=<float>                   Outer shell radius [default: 1.82]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['--root_dir']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
out_name    = args['--out_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

r_inner = float(args['--r_inner'])
r_outer = float(args['--r_outer'])

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, out_name=out_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

plotter.setup_grid(num_rows=2, num_cols=2, polar=True, **plotter_kwargs)
plotter.add_ball_shell_polar_colormesh(ball='equator(s1_B)', shell='equator(s1_S1)', azimuth_basis='phi', radial_basis='r', remove_x_mean=True, divide_x_mean=True, r_inner=r_inner, r_outer=r_outer)
plotter.add_ball_shell_polar_colormesh(ball='equator(u_B)', shell='equator(u_S1)', vector_ind=0, azimuth_basis='phi', radial_basis='r', remove_x_mean=False, divide_x_mean=False, r_inner=r_inner, r_outer=r_outer)
plotter.add_ball_shell_polar_colormesh(ball='equator(u_B)', shell='equator(u_S1)', vector_ind=1, azimuth_basis='phi', radial_basis='r', remove_x_mean=False, divide_x_mean=False, r_inner=r_inner, r_outer=r_outer)
plotter.add_ball_shell_polar_colormesh(ball='equator(u_B)', shell='equator(u_S1)', vector_ind=2, azimuth_basis='phi', radial_basis='r', remove_x_mean=False, divide_x_mean=False, r_inner=r_inner, r_outer=r_outer)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
