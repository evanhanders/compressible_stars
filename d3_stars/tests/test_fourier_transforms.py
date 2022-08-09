""" Tests of spherical harmonic transform operations. """

import pytest
import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt

import dedalus.public as d3
from d3_stars.post.power_spectrum_functions import FourierTransformer


@pytest.mark.parametrize('N', [1000, 10000])
@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('modes', [1, 2, 5])
def test_power(N, dtype, modes):
    t_final = 200
    time = np.linspace(0, t_final, N, endpoint=False)
    dt = t_final/N

    data = np.zeros_like(time, dtype=dtype)
    for f in np.arange(modes):
        omega = 10*np.pi * (f + 1)
        data += np.array(5*np.exp(1j*omega*(time/t_final)), dtype=dtype)
    time_series_power = np.sum(np.conj(data)*data*dt)/time.max()

    transformer = FourierTransformer(time, data)
    transformer.take_transform()
    power = transformer.get_power()
    transformed_power = np.sum(power)

    assert np.allclose(transformed_power, time_series_power, rtol=1e-4)


