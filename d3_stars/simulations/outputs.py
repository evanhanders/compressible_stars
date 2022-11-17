import functools
from operator import itemgetter
from collections import OrderedDict
import numpy as np
import dedalus.public as d3
from dedalus.core.future import FutureField
import h5py

from pathlib import Path
from configparser import ConfigParser

from .parser import name_star
from d3_stars.defaults import config

import logging
logger = logging.getLogger(__name__)

output_tasks = {}
flux_tags = ['cond', 'cond_superad', 'KE', 'PE', 'enth', 'visc', 'conv', 'entropy']
defaults = ['u', 'momentum', 'ur', 'u_squared', 'KE', 'PE', 'IE', 'TotE', 'PE1', 'IE1', 'FlucE', 'Re', 'Ma', 'ln_rho1', \
            'enstrophy','therm_visc_lum', 'L_heat', 'integ_by_parts_1', 'integ_by_parts_2', 'integ_by_parts_3',\
            'pom1', 'pom2', 'pom_fluc', 'pom_full', 'grad_s1', 'L', 's1', 'rho_full', 'rho_fluc', 'enthalpy_fluc', 'N2', \
            'Q_source', 'visc_source_KE', 'visc_source_IE', 'tot_visc_source', 'T_superad_z','T_superad1_z',\
            'therm_diss_1', 'therm_diss_2', 'therm_diss_3',\
            'divRad_source', 'PdV_source_KE', 'PdV_source_IE', 'tot_PdV_source', 'PdV_source_anelastic', \
            'source_KE', 'source_IE', 'tot_source',\
            'EOS_goodness', 'EOS_goodness_bg']

for k in defaults + ['F_{}'.format(t) for t in flux_tags]:
    output_tasks[k] = '{}'.format(k) + '_{0}'

for k in ['F_{}'.format(t) for t in flux_tags] + ['grad_s1',]:
    output_tasks[k+'_r'] = 'dot(Grid(er), ' + output_tasks[k] + ')' 

#angular momentum components
output_tasks['Lx'] = 'dot(ex_{0},L_{0})'
output_tasks['Ly'] = 'dot(ey_{0},L_{0})'
output_tasks['Lz'] = 'dot(ez_{0},L_{0})'
output_tasks['L_squared'] = 'dot(L_{0}, L_{0})'

for t in flux_tags:
    output_tasks['{}_lum'.format(t)] = 'Grid(4*np.pi*r_vals_{0}**2) * ( F_' + t + '_{0} )'
    output_tasks['{}_lum_r'.format(t)] = 'dot(Grid(er), ' + output_tasks['{}_lum'.format(t)] + ')'



class EvenTaskDict(OrderedDict):

    def __init__(self, solver, *args, sim_iter_wait=500, sim_time_wait=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.solver = solver
        self['output_dts'] = []
        self.handler_keys = []
        self.even_dt = np.inf
        self.slice_time = np.inf
        self.current_max_dt = np.inf
        self.start_iter = solver.iteration
        self.start_time = solver.sim_time
        self.sim_time_wait = sim_time_wait
        self.sim_iter_wait = sim_iter_wait
        self.sim_time = 0
        self.iter = 0
        self.max_dt_check = True
        self.evaluate = False
        self.first = True
        self.threshold = 1

    def add_handler(self, name, sim_dt, out_dir='./', **kwargs):
        logger.info('adding even tasks handler {}'.format(name))
        self['output_dts'].append(sim_dt)
        self.handler_keys.append(name)
        self[name] = self.solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, name), sim_dt=np.inf, iter=int(1e8), **kwargs)
        self[name].last_iter_div = self[name].last_wall_div = self[name].last_sim_div = np.inf #don't evaluate on first timestep
        return self[name]

    def compute_timestep(self, cfl):
        if self.first:
            if len(self['output_dts']) != 0:
                self.even_dt = np.min(self['output_dts'])
            self.first = False
            self.threshold = cfl.threshold

        self.iter = self.solver.iteration - self.start_iter
        self.sim_time = self.solver.sim_time - self.start_time
        timestep = cfl_dt = cfl.compute_timestep()

        if np.isfinite(self.even_dt):
            #throttle CFL max_dt once, after the transient.
            #Also, start outputting even analysis tasks.
            if self.max_dt_check and (timestep < self.even_dt*(1 + self.threshold)) and self.iter > self.sim_iter_wait and self.sim_time > self.sim_time_wait: #allow for warmup
                self.max_dt_check = False
                self.evaluate = True
                cfl.threshold = 0

            #Flag handler for evaluation
            #Adjust timestep only between outputs.
            if self.evaluate:
                self.evaluate = False
                for k in self.handler_keys:
                    self[k].last_iter_div = -1
                self.slice_time = self.solver.sim_time + self.even_dt
                num_steps = np.ceil(self.even_dt / timestep)
                timestep = self.current_max_dt = cfl.stored_dt = self.even_dt/num_steps
            elif self.max_dt_check:
                timestep = np.min((timestep, self.current_max_dt))
            else:
                cfl.stored_dt = timestep = self.current_max_dt

            t_future = self.solver.sim_time + timestep
            if t_future >= self.slice_time*(1-1e-8):
               self.evaluate = True

        return timestep

       

def initialize_outputs(solver, coords, namespace, bases, timescales, out_dir='./'):
    t_kepler, t_heat, t_rot = timescales
    locals().update(namespace)
    dist = solver.dist
    ## Analysis Setup
    # Cadence
    az_avg = lambda A: d3.Average(A, coords.coords[0])
    s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
    s2_std = lambda A: np.sqrt(s2_avg((A - s2_avg(A))**2))

    def integ(A):
        return d3.Integrate(A, coords)
    
    solver.problem.namespace['az_avg'] = az_avg
    solver.problem.namespace['s2_avg'] = s2_avg
    solver.problem.namespace['integ'] = integ
    namespace['az_avg'] = solver.problem.namespace['az_avg']
    namespace['s2_avg'] = solver.problem.namespace['s2_avg']
    namespace['integ'] = solver.problem.namespace['integ']

    star_dir, out_file = name_star()
    with h5py.File(out_file, 'r') as f:
        r_outer = f['r_outer'][()]

    analysis_tasks = OrderedDict()
    even_analysis_tasks = EvenTaskDict(solver)

    def vol_avg(A, volume):
        return d3.Integrate(A/volume, coords)

    for bn, basis in bases.items():
        if type(basis) == d3.BallBasis:
            vol = namespace['volume_{}'.format(bn)]  = (4/3)*np.pi*basis.radius**3
        else:
            Ri, Ro = basis.radii[0], basis.radii[1]
            vol = namespace['volume_{}'.format(bn)]  = (4/3)*np.pi*(Ro**3 - Ri**3)

        solver.problem.namespace['vol_avg_{}'.format(bn)] = functools.partial(vol_avg, volume=vol)
        namespace['vol_avg_{}'.format(bn)] = solver.problem.namespace['vol_avg_{}'.format(bn)]
        namespace['s2_std_{}'.format(bn)] = lambda A: np.sqrt(s2_avg((A - namespace['ones_{}'.format(bn)]*s2_avg(A))**2))
        solver.problem.namespace['s2_std_{}'.format(bn)] = namespace['s2_std_{}'.format(bn)]
    

    for h_name in config.handlers.keys():
        this_dict = config.handlers[h_name]
        max_writes = int(this_dict['max_writes'])
        time_unit = this_dict['time_unit']
        if time_unit == 'heating':
            t_unit = t_heat
        elif time_unit == 'kepler':
            t_unit = t_kepler
        else:
            logger.info('t unit not found; using t_unit = 1')
            t_unit = 1
        sim_dt = float(this_dict['dt_factor'])*t_unit
        if h_name == 'checkpoint':
            analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=sim_dt, max_writes=max_writes, parallel=this_dict['parallel'])
            analysis_tasks[h_name].add_tasks(solver.state, layout='g')
        else:
            if this_dict['even_outputs']:
                this_dict['handler'] = even_analysis_tasks.add_handler(h_name, sim_dt, out_dir=out_dir, max_writes=max_writes, parallel=this_dict['parallel'])
            else:
                this_dict['handler'] = analysis_tasks[h_name] = solver.evaluator.add_file_handler('{:s}/{:s}'.format(out_dir, h_name), sim_dt=sim_dt, max_writes=max_writes, parallel=this_dict['parallel'])

        tasks = this_dict['tasks']
        for this_task in tasks:
            handler = this_dict['handler']
            if this_task['type'] == 'full_integ':
                for fieldname in this_task['fields']:
                    task = None
                    for bn, basis in bases.items():
                        fieldstr = output_tasks[fieldname].format(bn)
                        tmp_task = d3.Grid(eval('integ({})'.format(fieldstr), dict(solver.problem.namespace)))
                        if task is None:
                            task = tmp_task
                        else:
                            task += tmp_task
                    handler.add_task(task, name='integ({})'.format(fieldname))
                continue

            for bn, basis in bases.items():
                if this_task['type'] == 'equator':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = eval('({})(theta=np.pi/2)'.format(fieldstr), dict(solver.problem.namespace))
                        handler.add_task(task, name='equator({}_{})'.format(fieldname, bn))
                elif this_task['type'] == 'meridian':
                    interps = this_task['interps']
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        for base_interp in interps:
                            task = d3.Grid(eval('({})(phi={})'.format(fieldstr, base_interp), dict(solver.problem.namespace)))
                            handler.add_task(task, name='meridian({}_{},phi={:.2f})'.format(fieldname, bn, base_interp))
                elif this_task['type'] == 'shell':
                    interps = this_task['interps']
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        for base_interp in interps:
                            if isinstance(base_interp, str) and 'R' in base_interp:
                                if base_interp == 'R':
                                    interp = '1'
                                else:
                                    interp = base_interp.replace('R', '')
                                interp = np.around(float(interp)*r_outer, decimals=2)
                            else:
                                interp = np.around(float(base_interp), decimals=2)
                            if type(basis) == d3.BallBasis and interp > basis.radius:
                                continue
                            elif type(basis) == d3.ShellBasis:
                                if interp <= basis.radii[0] or interp > basis.radii[1] :
                                    continue
                            task = d3.Grid(eval('({})(r={})'.format(fieldstr, interp), dict(solver.problem.namespace)))
                            handler.add_task(task, name='shell({}_{},r={})'.format(fieldname, bn, base_interp))
                elif this_task['type'] == 'vol_avg':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task = d3.Grid(eval('vol_avg_{}({})'.format(bn, fieldstr), dict(solver.problem.namespace)))
                        handler.add_task(task, name='vol_avg({}_{})'.format(fieldname, bn))
                elif this_task['type'] == 's2_avg':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task_str = 's2_avg({})'.format(fieldstr)
                        task = d3.Grid(eval(task_str, dict(solver.problem.namespace)))
                        handler.add_task(task, name='s2_avg({}_{})'.format(fieldname, bn))
                elif this_task['type'] == 's2_std':
                    for fieldname in this_task['fields']:
                        fieldstr = output_tasks[fieldname].format(bn)
                        task_str = 's2_std_{}({})'.format(bn, fieldstr)
                        task = d3.Grid(eval(task_str, dict(solver.problem.namespace)))
                        handler.add_task(task, name='s2_std_{}({}_{})'.format(bn, fieldname, bn))
                else:
                    raise NotImplementedError("Output task type not implemented: {}".format(this_task['type']))


    return analysis_tasks, even_analysis_tasks
