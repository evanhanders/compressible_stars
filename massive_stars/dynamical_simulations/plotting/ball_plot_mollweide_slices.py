"""
This script plots snapshots of the evolution of 2D shell slices from a 3D ball simulation.
A mollweide projection is used for visualization.

Usage:
    ball_plot_mollweide_slices.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_mollweide]
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

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

plotter.setup_grid(num_cols=2, num_rows=2, mollweide=True, **plotter_kwargs)
kwargs = { 'azimuth_basis' : 'φ', 'colatitude_basis' : 'θ' }
plotter.add_mollweide_colormesh('s1(r=0.5)', remove_mean=True, **kwargs)
plotter.add_mollweide_colormesh('u(r=0.5)', vector_ind=2, cmap='PuOr_r', **kwargs)
plotter.add_mollweide_colormesh('s1(r=1)', remove_mean=True, **kwargs)
plotter.add_mollweide_colormesh('u(r=1)',vector_ind=2, cmap='PuOr_r', **kwargs)

plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
