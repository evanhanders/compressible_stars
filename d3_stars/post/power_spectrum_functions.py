import gc
from collections import OrderedDict

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sys import stdout

hann_power_normalizer = 8/3
hann_amp_normalizer = 2


class FourierTransformer:

    def __init__(self, times, signal, window=np.hanning):
        #TODO: define f_nyq and f_sample
        self.times = times
        self.signal = np.array(signal)
        self.N = self.times.shape[0]
        self.dt = np.median(np.diff(self.times))
        if window is None:
            self.window = np.ones(self.N)
        else:
            self.window = window(self.N)
        while len(self.window.shape) < len(self.signal.shape):
            self.window = np.expand_dims(self.window, axis=-1)
        
        self.power_norm = 1
        self.amp_norm = 1
        if np.hanning == window:
            self.power_norm = hann_power_normalizer * (self.N/(self.N-1))
            self.amp_norm = hann_amp_normalizer * (self.N/(self.N-1))
       
        if self.signal.dtype == np.float64:
            self.complex = False
        else:
            self.complex = True

        self.freqs = None
        self.ft = None
        self.power = None
        self.power_freqs = None


    def take_transform(self):
        """
        Takes the Fourier transform of a time series of real data.
        Utilizes a Hann window & normalizes appropriately so that the integrated power spectrum has the same power as the time series.
        """
        if self.complex:
            self.clean_cfft()
        else:
            self.clean_rfft()
        self.power_interp = interp1d(self.power_freqs, self.power, axis=0)
        return self.freqs, self.ft

    def clean_rfft(self):
        self.freqs = np.fft.rfftfreq(self.times.shape[0], d=np.median(np.gradient(self.times.flatten()))) 
        self.ft = np.fft.rfft(self.window*self.signal, axis=0, norm="forward")
        self.power = self.normalize_rfft_power()
        self.power_freqs = self.freqs
        return self.freqs, self.ft

    def clean_cfft(self):
        self.freqs = np.fft.fftfreq(self.times.shape[0], d=np.median(np.gradient(self.times.flatten()))) 
        self.ft = np.fft.fft(self.window*self.signal, axis=0, norm="forward")
        self.power = self.normalize_cfft_power()
        return self.freqs, self.ft

    def get_power(self):
        """ returns power spectrum, accounting for window normalization so that parseval's theorem is satisfied"""
        return self.power * self.power_norm

    def get_power_freqs(self):
        """ returns power spectrum, accounting for window normalization so that parseval's theorem is satisfied"""
        return self.power_freqs

    def get_peak_power(self, freq):
        """ 
        returns the power of a peak at the given frequency as if that peak corresponded to a sine wave.
        So if your signal has a sine wave, A * sin(omega * t), we expect a peak at f = +/- omega/(2pi).
        Each peak will have amplitude (A/2), so the total power will be 2(A^2/4) = A^2/2.

        In the complex case, you can have a real and imaginary wave component with total amplitude A^2.
        So the complex case has a cos^2 + sin^2 ~ 1 thing going for it.
        The real case only has cos^2 ~ 1/2 and needs the extra factor of 2.
        """
        if self.complex:
            return self.power_interp(freq) * self.amp_norm**2
        else:
            return 2*self.power_interp(freq) * self.amp_norm**2

    def normalize_cfft_power(self):
        """
        Calculates the power spectrum of a complex fourier transform by collapsing negative and positive frequencies.
        """
        power = (self.ft*np.conj(self.ft)).real
        self.power_freqs = np.unique(np.abs(self.freqs))
        self.power = np.zeros((self.power_freqs.size,*tuple(power.shape[1:])))
        for i, f in enumerate(self.power_freqs):
            good = np.logical_or(self.freqs == f, self.freqs == -f)
            self.power[i] = np.sum(power[good], axis=0)
        return self.power

    def normalize_rfft_power(self):
        """
        Calculates the power spectrum of a real fourier transform accounting for its hermitian-ness
        """
        power = (self.ft*np.conj(self.ft)).real
        self.power_freqs = np.unique(np.abs(self.freqs))
        self.power = np.zeros((self.power_freqs.size,*tuple(power.shape[1:])))
        for i, f in enumerate(self.power_freqs):
            if f != 0:
                self.power[i] = 2*power[i] #account for negative frequencies which are conj(positive freqs)
            else:
                self.power[i] = power[i]
        return self.power


class ShortTimeFourierTransformer():

    def __init__(self, times, signal, min_freq, **kwargs):
        self.min_freq = min_freq
        self.stft_period = 1/min_freq
        self.times = times
        self.signal = np.array(signal)
        self.tot_N = self.times.size

        self.dt = np.median(np.diff(self.times))
        self.stft_N = int(self.stft_period/self.dt)
        if self.stft_N % 2 == 1:
            self.stft_N += 1 #make sure N is even.
        self.num_chunks = int(np.floor(self.tot_N/self.stft_N))
        slices = []
        for i in range(self.num_chunks):
            slices.append(slice(i*self.stft_N, (i+1)*self.stft_N, 1))

        self.time_chunks = []
        self.signal_chunks = []
        self.FT_list = []
        for sl in slices:
            self.time_chunks.append(self.times[sl])
            self.signal_chunks.append(self.signal[sl])
            self.FT_list.append(FourierTransformer(self.time_chunks[-1], self.signal_chunks[-1], **kwargs))

        self.freq_chunks = None
        self.transform_chunks = None

    def take_transforms(self):
        self.freq_chunks = []
        self.transform_chunks = []
        for FT in self.FT_list:
            freqs, transform = FT.take_transform()
            self.freq_chunks.append(freqs)
            self.transform_chunks.append(transform)
        return self.freq_chunks, self.transform_chunks

    def get_peak_evolution(self, freqs):
        """ Given a list of frequencies, return the evolution of the power in the peak at that frequency """
        self.evolution_times = []
        self.evolution_freqs = OrderedDict()
        for f in freqs:
            self.evolution_freqs[f] = []
        for FT in self.FT_list:
            for f in freqs:
                self.evolution_freqs[f].append(FT.get_peak_power(f))
            self.evolution_times.append(np.mean(FT.times))
        self.evolution_times = np.array(self.evolution_times)
        for f in freqs:
            self.evolution_freqs[f] = np.array(self.evolution_freqs[f])
        return self.evolution_times, self.evolution_freqs

    def get_power_evolution(self):
        """ Return the evolution of the summed power"""
        self.evolution_times = []
        self.evolution_power = []
        for FT in self.FT_list:
            self.evolution_power.append(FT.get_power())
            self.evolution_times.append(np.mean(FT.times))
        self.evolution_times = np.array(self.evolution_times)
        return self.evolution_times, self.evolution_power


class HarmonicTimeToFreq:
    """ 
    Transforms data from a DedalusShellSHTransformer (time, ell, m) into frequency space (freq, ell, m)
    Takes advantage of STFT logic.

    Relies on plotpal for file reading.
    """
    def __init__(self, root_dir, data_dir, **kwargs):
        from plotpal.file_reader import SingleTypeReader as SR
        self.root_dir = root_dir
        self.out_dir = 'FT_{}'.format(data_dir)
        self.reader = SR(root_dir, data_dir, self.out_dir, distribution='single', **kwargs)

        with h5py.File(self.reader.files[0], 'r') as f:
            self.fields = list(f['tasks'].keys())

    def write_transforms(self, min_freq=None):
        times = []
        print('getting times...')
        stdout.flush()
        first = True
        while self.reader.writes_remain():
            dsets, ni = self.reader.get_dsets([], verbose=False)
            times.append(self.reader.current_file_handle['time'][ni])
            if first:
                ells = self.reader.current_file_handle['ells'][()]
                ms = self.reader.current_file_handle['ms'][()]
                first = False

        times = np.array(times)

        with h5py.File('{}/transforms.h5'.format(self.out_dir), 'w') as wf: 
            wf['ells']  = ells
            wf['ms']  = ms


        for i, f in enumerate(self.fields):
            print('reading field {}'.format(f))
            stdout.flush()
            #Do one 'm' at a time for lighter memory usage.
            for j, m in enumerate(ms.squeeze()):

                print('loading m = {}...'.format(m))
                stdout.flush()
                writes = 0 
                while self.reader.writes_remain():
                    dsets, ni = self.reader.get_dsets([], verbose=False)
                    rf = self.reader.current_file_handle
                    this_task = rf['tasks'][f][ni,:].squeeze()
                    if writes == 0:  
                        if this_task.shape[0] == 3 and len(this_task.shape) == 3:
                            #vector
                            data_cube = np.zeros((times.shape[0], 3, ells.shape[1]), dtype=np.complex128)
                        else:
                            data_cube = np.zeros((times.shape[0], ells.shape[1]), dtype=np.complex128)
                    if this_task.shape[0] == 3:
                        data_cube[writes,:] = this_task[:,:,j]
                    else:
                        data_cube[writes,:] = this_task[:,j]
                    writes += 1

                if min_freq is None:
                    FT = FourierTransformer(times, data_cube)
                    freqs, transform = FT.take_transform()
                    #Logic for a single ell,m
                    #        transform = np.zeros(data_cube.shape, dtype=np.complex128)
                    #        for ell in range(data_cube.shape[-2]):
                    #            for m in range(data_cube.shape[-1]):
                    #                if len(data_cube.shape) == 3:
                    #                    input_data = data_cube[:,ell,m]
                    #                    FT = FourierTransformer(times, input_data)
                    #                    freqs, transform[:,ell,m] = FT.take_transform()
                    #                else:
                    #                    input_data = data_cube[:,:,ell,m]
                    #                    FT = FourierTransformer(times, input_data)
                    #                    freqs, transform[:,:,ell,m] = FT.take_transform()

                else:
                    FT = ShortTimeFourierTransformer(times, data_cube, min_freq)
                    freqs_chunks, transform_chunks = FT.take_transforms()

                del data_cube
                gc.collect()

                with h5py.File('{}/transforms.h5'.format(self.out_dir), 'r+') as wf:
                    if min_freq is None:
                        if j == 0: #create field
                            shape = transform.shape + (ms.shape[2],)
                            wf.create_dataset(name='{}_cft'.format(f), shape=shape, maxshape=shape, dtype=transform.dtype)
                        slices = tuple([slice(None) for sh in transform.shape] + [slice(j,j+1,1)])
                        wf['{}_cft'.format(f)][slices] = np.expand_dims(np.copy(transform), axis=-1)
                        if i == 0 and j == 0:
                            wf['freqs'] = np.copy(freqs)
                    else:
                        if j == 0: #create field
                            shape = transform.shape + (ms.shape[2],)
                            wf.create_dataset(name='{}_cft_chunks'.format(f), shape=shape, maxshape=shape, dtype=transform.dtype)
                        slices = tuple([slice(None) for sh in transform.shape] + [slice(j,j+1,1)])
                        wf['{}_cft_chunks'.format(f)][slices] = np.expand_dims(np.copy(transform), axis=-1)
                        if i == 0 and j == 0:
                            wf['freqs_chunks'] = np.copy(freqs_chunks)
