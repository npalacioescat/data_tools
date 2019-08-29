# -*- coding: utf-8 -*-

'''
data_tools.signal
=================

Signal processing module.

Contents
--------
'''

__all__ = ['fconvolve', 'gauss_kernel', 'gauss_noise']

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


def gauss_kernel(size, sd=1, ndim=2):
    '''
    Returns a N-dimensional Gaussian kernel. The kernel is defined as
    follows:

    .. math::
      k(\\vec{x})=\\frac{1}{(\\sqrt{2\\pi}\\sigma)^N}\\mathrm{e}^{-
      \\frac{||\\vec{x}||_2^2}{2\\sigma^2}}

    Where :math`N` is the number of dimensions and :math:`\\sigma` is
    the standard deviation of the kernel.

    * Arguments:
        - *size* [int]: The number of discrete points of the kernel
          (will be the same on each dimension).
        - *sd* [float]: Optional, ``1`` by default. The standard
          deviation of the gaussian kernel.
        - *ndim* [int]: Optional, ``2`` by default. Number of dimensions
          for the desired kernel.

    * Returns:
        - [numpy.ndarray]: The Gaussian kernel.
    '''

    s = int(size / 2)
    if size % 2 == 0:
        dims = np.mgrid[ndim * (slice(-s, s), )]

    else:
        dims = np.mgrid[ndim * (slice(-s, s + 1), )]

    f = (1 / ((np.sqrt(2 * np.pi) * sd) ** ndim)
         * np.exp(-(np.linalg.norm(dims, axis=0, ord=2) ** 2 / float(s))
                  / (2 * sd ** 2)))

    return f / f.sum()


def gauss_noise(x, sd=1):
    '''
    Applies additive Gaussian (white) noise to a given signal.

    * Arguments;
        - *x* [numpy.ndarray]: The signal, can have any number of
          dimensions.
        - *sd* [float]: Optional, ``1`` by default. The standard
          deviation of the noise to apply.

    * Returns:
        - [numpy.ndarray]: The signal *x* with the additive Gaussian
          noise applied.
    '''

    return np.random.normal(np.real(x), sd)
