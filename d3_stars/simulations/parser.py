import os, sys
from collections import OrderedDict
from pathlib import Path
from configparser import ConfigParser
import d3_stars.defaults.config as config

import logging
logger = logging.getLogger(__name__)

def name_star(star_dir='star'):
    star_file = '{:s}/star_'.format(star_dir)
    star_file += (len(config.star['nr'])*"{}+").format(*tuple(config.star['nr']))[:-1]
    if config.star['type'].lower() == 'massive':
        star_file += '_bounds{}-{}'.format(config.star['r_bounds'][0], config.star['r_bounds'][-1])
    elif config.star['type'].lower() == 'dwarf' or config.star['type'].lower() == 'envelope':
        star_file += '_nrho{}'.format(config.star['n_rho'])
    else:
        raise NotImplementedError()
    star_file += '_Re{:.2e}_de{}_cutoff{:.1e}.h5'.format(config.numerics['reynolds_target'], config.numerics['N_dealias'], config.numerics['ncc_cutoff'])
    logger.info('star file: {}'.format(star_file))
    if not os.path.exists('{:s}'.format(star_dir)):
        os.mkdir('{:s}'.format(star_dir))

    return star_dir, star_file





def parse_std_config(config_file, star_dir='star'):
    config = OrderedDict()
    raw_config = OrderedDict()
    config_file = Path(config_file)
    config_p = ConfigParser()
    config_p.read(str(config_file))
    for n, v in config_p.items('star'):
        if n.lower() == 'nr':
            config[n] = [int(n) for n in v.split(',')]
        elif n.lower() == 'r_bounds':
            config[n] = [n.replace(' ', '') for n in v.split(',')]
        elif v.lower() == 'true':
            config[n] = True
        elif v.lower() == 'false':
            config[n] = False
        else:
            config[n] = v
        raw_config[n] = v

    for n, v in config_p.items('numerics'):
        config[n] = v
        raw_config[n] = v

    for n, v in config_p.items('eigenvalue'):
        raw_config[n] = v
        if n in ['lmax',]:
            config[n] = int(v)
        elif n in ['hires_factor',]:
            config[n] = float(v)

    for n, v in config_p.items('dynamics'):
        raw_config[n] = v
        if n in ['ntheta',]:
            config[n] = int(v)
        elif n in ['safety', 'cfl_max_r', 'wall_hours', 'buoy_end_time', 'tau_factor', 'rotation_time', 'a0']:
            config[n] = float(v)
        elif n in ['sponge',]:
            config[n] = config_p.getboolean('dynamics', n)
        elif n == 'mesh':
            mesh = [int(m) for m in v.split(',')]
        else:
            config[n] = v


    for k in ['reynolds_target', 'prandtl', 'ncc_cutoff', 'n_dealias', 'l_dealias']:
        config[k] = float(config[k])

    if float(config['r_bounds'][0].lower()) != 0:
        raise ValueError("The inner basis must currently be a BallBasis; set the first value of r_bounds to zero.")

    star_file = '{:s}/star_'.format(star_dir)
    star_file += (len(config['nr'])*"{}+").format(*tuple(config['nr']))[:-1]
    star_file += '_bounds{}-{}'.format(config['r_bounds'][0], config['r_bounds'][-1])
    star_file += '_Re{}_de{}_cutoff{}.h5'.format(raw_config['reynolds_target'], raw_config['n_dealias'], raw_config['ncc_cutoff'])
    logger.info('star file: {}'.format(star_file))
    if not os.path.exists('{:s}'.format(star_dir)):
        os.mkdir('{:s}'.format(star_dir))

    return config, raw_config, star_dir, star_file


def parse_ncc_config(ncc_config_file):
    ncc_config_file = Path(ncc_config_file)
    ncc_config_p = ConfigParser()
    ncc_config_p.optionxform = str
    ncc_config_p.read(str(ncc_config_file))
    ncc_dict = OrderedDict()
    for ncc in ncc_config_p.keys():
        if ncc == 'defaults' or ncc == 'DEFAULT':
            continue
        ncc_dict[ncc] = OrderedDict()
        if 'nr_max' in ncc_config_p[ncc].keys():
            nr_max = ncc_config_p[ncc]['nr_max']
        else:
            nr_max = ncc_config_p['defaults']['nr_max']
        nr_max = [int(n) for n in nr_max.split(',')]
        ncc_dict[ncc]['nr_max'] = nr_max
        ncc_dict[ncc]['vector'] = ncc_config_p.getboolean(ncc, 'vector')
        ncc_dict[ncc]['grid_only'] = ncc_config_p.getboolean(ncc, 'grid_only')
        for k in ncc_config_p[ncc].keys():
            if k not in ncc_dict[ncc].keys() and k != 'nr_max':
                ncc_dict[ncc][k] = ncc_config_p[ncc][k]

    return ncc_dict
