import numpy as np


class Normalization(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M = x
        else:
            oldM = self._M.copy()
            self._M = oldM + (x - oldM) / self._n
            self._S = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class NormFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape: tuple, decorate_mean: bool = True, decorate_std: bool = True, clip=False):
        self.shape = shape
        self.decorate_mean = decorate_mean
        self.decorate_std = decorate_std
        self.clip = clip

        self.nml = Normalization(shape)

    def __call__(self, x, update=True):
        if update:
            self.nml.push(x)
        if self.decorate_mean:
            x = x - self.nml.mean
        if self.decorate_std:
            x = x / (self.nml.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def output_shape(self):
        return self.shape