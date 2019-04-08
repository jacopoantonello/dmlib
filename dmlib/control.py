#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import numpy as np

from numpy.random import normal
from numpy.linalg import norm, svd, pinv


h5_prefix = 'dmlib/control/'


def get_default_parameters():
    return {
        'ZernikeControl': ZernikeControl.get_default_parameters(),
        'SVDControl': SVDControl.get_default_parameters(),
        }


def get_parameters_info():
    return {
        'ZernikeControl': ZernikeControl.get_parameters_info(),
        'SVDControl': SVDControl.get_parameters_info(),
        }


class ZernikeControl:

    @staticmethod
    def get_default_parameters():
        return {
            'include': [],
            'exclude': [1, 2, 3, 4],
            'min': 5,
            'max': 6,
            'all': 1,
            'flipx': 0,
            'flipy': 0,
            'rotate': 0.0,
            'flat_on': 1,
            }

    @staticmethod
    def get_parameters_info():
        return {
            'include': (list, int, 'Zernike indices to include'),
            'exclude': (list, int, 'Zernike indices to include'),
            'min': (int, (1, None), 'Minimum Zernike index'),
            'max': (int, (1, None), 'Maximum Zernike index'),
            'all': (
                int, (0, 1), 'Use all Zernike available in calibration'),
            'flipx': (int, (0, 1), 'Flip pupil along x'),
            'flipy': (int, (0, 1), 'Flip pupil along y'),
            'rotate': (float, (None, None), 'Rotate pupil in degrees'),
            'flat_on': (int, (0, 1), 'Apply flattening offset'),
            }

    def __init__(
            self, dm, calib, pars={}, h5f=None):
        self.log = logging.getLogger(self.__class__.__name__)
        pars = {**self.get_default_parameters(), **pars}
        self.saturation = 0
        self.pars = pars
        self.P = None
        self.gui_callback = None

        nz = calib.H.shape[0]
        nz = calib.H.shape[0]
        nu = calib.H.shape[1]
        self.Cp = pinv(calib.C)

        # get controlled Zernike coefficients
        if pars['all']:
            indices = np.arange(1, nz + 1)
        else:
            indices = get_noll_indices(pars)
        assert(calib.get_rzern().nk == nz)
        ndof = indices.size

        self.nz = nz
        self.nu = nu
        self.ndof = ndof

        self.dm = dm
        self.calib = calib
        self.indices = indices
        self.h5f = h5f

        self.h5_save('indices', indices)
        self.h5_save('rad_to_nm', calib.get_rad_to_nm())

        # handle orthogonal pupil transform
        try:
            self.transform_pupil(pars['rotate'], pars['flipx'], pars['flipy'])
        except Exception:
            self.P = None
            self.pars['flipx'] = 0
            self.pars['flipy'] = 0
            self.pars['rotate'] = 0.0

        self.z = np.zeros((nz,))
        self.z1 = np.zeros((nz,))
        self.ab = np.zeros((ndof,))
        self.u = np.zeros((nu,))

        # handle flattening
        try:
            self.flat_on = bool(pars['flat_on'])
            self.uflat = np.array(pars['uflat'])
            assert(self.uflat.size == calib.uflat.size)
        except Exception:
            self.log.info('failed load uflat')
            self.flat_on = 1
            self.uflat = calib.uflat

        # handle initial value
        try:
            self.u[:] = np.array(pars['u'])
            self.z1[:] = self.u2z()
        except Exception:
            self.log.info('failed load u')

        if h5f:
            calib.save_h5py(h5f, prepend=h5_prefix)

        self.h5_make_empty('flat_on', (1,), np.bool)
        self.h5_make_empty('uflat', (nu,))
        self.h5_make_empty('x', (ndof,))
        self.h5_make_empty('u', (nu,))
        self.h5_save('name', self.__class__.__name__)
        self.h5_save('ab', self.ab)
        self.h5_save('P', np.eye(nz))
        self.h5_save('params', json.dumps(pars))

    def __str__(self):
        return (
            f'<dmlib.control.{self.__class__.__name__} ' +
            f'ndof={self.ndof} indices={self.indices}>')

    def save_parameters(self, merge={}, asflat=False):
        d = {**merge, **self.pars}
        if asflat:
            d['flat_on'] = 1
            d['uflat'] = self.u.tolist()
            d['u'] = self.u.tolist()
        else:
            d['flat_on'] = self.flat_on
            d['uflat'] = self.uflat.tolist()
            d['u'] = self.u.tolist()
        return d

    def h5_make_empty(self, name, shape, dtype=np.float):
        if self.h5f:
            name = h5_prefix + 'ZernikeControl/' + name
            if name in self.h5f:
                del self.h5f[name]
            self.h5f.create_dataset(
                name, shape + (0,), maxshape=shape + (None,),
                dtype=dtype)

    def h5_append(self, name, what):
        if self.h5f:
            name = h5_prefix + 'ZernikeControl/' + name
            self.h5f[name].resize((
                self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            name = h5_prefix + 'ZernikeControl/' + where
            if name in self.h5f:
                del self.h5f[name]
            self.h5f[name] = what

    def u2z(self):
        "Get current Zernike coefficients (without uflat) for GUIs"
        # z1 = P*z
        # u = C*z1
        # z1 = H*u
        if self.flat_on:
            tmp = self.u - self.uflat
        else:
            tmp = self.u
        z1 = np.dot(self.Cp, tmp)
        if self.P is None:
            return z1
        else:
            return np.dot(self.P.T, z1)

    def write(self, x):
        "Write Zernike coefficients"
        assert(x.shape == self.ab.shape)

        # z controlled Zernike degrees of freedom
        self.z[self.indices - 1] = x[:] + self.ab[:]

        # z1 transform all calibrated Zernike coefficients
        if self.P is not None:
            np.dot(self.P, self.z, self.z1)
        else:
            self.z1[:] = self.z[:]

        # apply control matrix
        np.dot(self.calib.C, self.z1, self.u)

        # apply flattening
        if self.flat_on:
            self.u += self.uflat

        # logging
        self.h5_append('uflat', self.uflat)
        self.h5_append('flat_on', self.flat_on)
        self.h5_append('x', x)
        self.h5_append('u', self.u)

        # handle saturation
        if norm(self.u, np.inf) > 1:
            self.log.warn(
                'Saturation {}'.format(str(np.abs(self.u).max())))
            self.u[self.u > 1.] = 1.
            self.u[self.u < -1.] = -1.
            self.saturation = 1
        else:
            self.saturation = 0
        assert(norm(self.u, np.inf) <= 1.)

        # write raw voltages
        self.dm.write(self.u)

        if self.gui_callback:
            self.gui_callback()

    def set_random_ab(self, rms=1.0):
        self.ab[:] = normal(size=self.ab.size)
        self.ab[:] /= norm(self.ab.size)
        if self.h5f:
            self.h5f[h5_prefix + 'ZernikeControl/ab'][:] = self.ab[:]

    def transform_pupil(self, alpha=0., flipx=False, flipy=False):
        rzern = self.calib.get_rzern()

        if alpha != 0.:
            R = rzern.make_rotation(alpha)
        else:
            R = 1

        if flipx:
            Fx = rzern.make_xflip()
        else:
            Fx = 1

        if flipy:
            Fy = rzern.make_yflip()
        else:
            Fy = 1

        tot = np.dot(Fy, np.dot(Fx, R))
        if tot.size == 1:
            return
        else:
            self.set_P(tot)

    def set_P(self, P):
        addr = h5_prefix + 'ZernikeControl/P'

        if P is None:
            self.P = None

            if self.h5f:
                del self.h5f[addr]
                self.h5f[addr][:] = np.eye(self.nz)
        else:
            assert(P.ndim == 2)
            assert(P.shape[0] == P.shape[1])
            assert(np.allclose(np.dot(P, P.T), np.eye(P.shape[0])))
            if self.P is None:
                self.P = P.copy()
            else:
                np.dot(P, self.P.copy(), self.P)

            if self.h5f:
                del self.h5f[addr]
                self.h5f[addr][:] = self.P[:]


class SVDControl(ZernikeControl):

    @staticmethod
    def get_default_parameters():
        return {
            'modes': 5,
            'zernike_exclude': 4,
            'flat_on': 1,
            }

    @staticmethod
    def get_parameters_info():
        return {
            'modes': (int, (1, None), 'Number of SVD modes'),
            'zernike_exclude': (
                int, (1, None),
                'Exclude Zernike indices up to (inclusive)'),
            'flat_on': (int, (0, 1), 'Apply flattening offset'),
            }

    def __init__(self, dm, calib, pars, h5=None):
        super().__init__(dm, calib, super().get_default_parameters(), h5)
        self.log = logging.getLogger(self.__class__.__name__)
        svd_pars = {**self.get_default_parameters(), **pars}

        svd_modes = self.svd_pars['modes']
        nignore = self.svd_pars['zernike_exclude'] - 1

        self.h5_save('svd_modes', svd_modes)

        H = self.calib.H
        Hl = H[:nignore, :]
        # Hh = H[nignore:, :]
        _, _, Vt = svd(Hl)
        Vl2 = Vt[nignore:, :].T
        test1 = np.dot(H, Vl2)
        assert(test1.shape[0] == H.shape[0])
        assert(test1.shape[1] == H.shape[1] - nignore)
        assert(np.allclose(test1[:nignore, :], 0))

        U, s, Vt = svd(np.dot(H, Vl2))
        U1 = U[:, :s.size]
        np.allclose(U1[:nignore, :], 0)

        nmodes = svd_pars.svd_modes
        V1 = Vt[:nmodes, :].T
        s1i = np.power(s[:nmodes], -1)
        S1i = np.diag(s1i)

        self.h5_make_empty('x', (nmodes,))
        self.K = Vl2@V1@S1i
        self.ndof = nmodes
        self.ab = np.zeros(nmodes)
        self.h5_save('ab', self.ab)

        def f(n, w):
            self.h5f[h5_prefix + 'ZernikeControl/SVDControl/' + n] = w

        if self.h5f:
            f('nignore', nignore)
            f('nmodes', nmodes)
            f('Vl2', Vl2)
            f('V1', V1)
            f('S1i', S1i)
            f('K', self.K)
            f('params', json.dumps(svd_pars))

    def write(self, x):
        assert(x.shape == self.ab.shape)
        z = x + self.ab
        np.dot(self.K, z, self.u)

        # apply flattening
        if self.flat_on:
            self.u += self.uflat

        # logging
        self.h5_append('uflat', self.uflat)
        self.h5_append('flat_on', self.flat_on)
        self.h5_append('x', x)
        self.h5_append('u', self.u)

        # handle saturation
        if norm(self.u, np.inf) > 1:
            self.log.warn(
                'Saturation {}'.format(str(np.abs(self.u).max())))
            self.u[self.u > 1.] = 1.
            self.u[self.u < -1.] = -1.
            self.saturation = 1
        else:
            self.saturation = 0

        assert(norm(self.u, np.inf) <= 1.)

        self.dm.write(self.u)

    def save_parameters(self, merge={}):
        d = {**merge, **self.pars}
        d['flat_on'] = self.flat_on
        d['uflat'] = self.uflat.tolist()

    def set_random_ab(self, rms=1.0):
        raise NotImplementedError()

    def set_P(self, P):
        raise NotImplementedError()


def get_noll_indices(pars):
    noll_min = np.array(pars['min'], dtype=np.int)
    noll_max = np.array(pars['max'], dtype=np.int)
    minclude = np.array(pars['include'], dtype=np.int)
    mexclude = np.array(pars['exclude'], dtype=np.int)

    mrange = np.arange(noll_min, noll_max + 1, dtype=np.int)
    zernike_indices = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))

    log = logging.getLogger('ZernikeControl')
    log.info(f'selected Zernikes {zernike_indices}')

    assert(zernike_indices.dtype == np.int)
    return zernike_indices


def get_controls():
    return {
        'ZernikeControl': ZernikeControl,
        'SVDControl': SVDControl,
    }


def new_control(dm, calib, name, pars={}, h5f=None):
    options = get_controls()
    if name not in options.keys():
        raise ValueError(
            f'name must be one of {", ".join(options.keys())}')

    return options[name](dm, calib, pars, h5f)
