# -*- coding: utf-8 -*-

'''
data_tools.signal
=================

Signal processing module.

Contents
--------
'''

__all__ = ['fconvolve']

import numpy as np


def fconvolve(u, v):
    '''
    Convolves two vectors or arrays using Fast Fourier Transform (FFT).
    According to Fourier theory, the convolution theorem states that:

    .. math::
      g(x)=u(x)\\ast v(x)=\\mathcal{F}^{-1}\\left\\{\\mathcal{F}\\{u(x)
      \\}\\mathcal{F}\\{v(x)\\}\\right\\}

    * Arguments:
        - *u* [numpy.ndarray]: First array to convolve.
        - *v* [numpy.ndarray]: The other array to be convolved.

    * Returns:
        - [numpy.ndarray]: The convolved array of *u* and *v*.
    '''

    U = np.fft.fftshift(np.fft.fftn(u))
    V = np.fft.fftshift(np.fft.fftn(v))

    return np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(U * V))))
