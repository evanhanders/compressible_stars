"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.
There is a ball basis, and two shell bases.

Usage:
    build_star.py [options]

Options:
    --no_plot         If flagged, don't output plots
"""
import logging
logger = logging.getLogger(__name__)


from d3_stars.simulations.star_builder import build_nccs

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    build_nccs(plot_nccs=not(args['--no_plot']))
