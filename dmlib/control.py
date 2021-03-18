#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from copy import deepcopy

import numpy as np
from numpy.linalg import norm, pinv, svd
from numpy.random import normal

from dmlib.core import h5_store_str

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
            'include': (list, int, 'Zernike indices to include', 1),
            'exclude': (list, int, 'Zernike indices to include', 1),
            'min': (int, (1, None), 'Minimum Zernike index', 1),
            'max': (int, (1, None), 'Maximum Zernike index', 1),
            'all':
            (int, (0, 1), 'Use all Zernike available in calibration', 0),
            'flipx': (int, (0, 1), 'Flip pupil along x', 0),
            'flipy': (int, (0, 1), 'Flip pupil along y', 0),
            'rotate': (float, (None, None), 'Rotate pupil in degrees', 0),
            'flat_on': (int, (0, 1), 'Apply flattening offset', 0),
        }

    def __init__(self, dm, calib, pars={}, h5f=None):
        self.log = logging.getLogger(self.__class__.__name__)
        pars = {**deepcopy(self.get_default_parameters()), **deepcopy(pars)}
        self.saturation = 0
        self.pars = pars
        self.P = None
        self.R = None
        self.gui_callback = None

        nz = calib.H.shape[0]
        nz = calib.H.shape[0]
        nu = calib.H.shape[1]
        self.Cp = pinv(calib.C)

        try:
            enabled = self.pars['enabled']
        except KeyError:
            enabled = 1

        # get controlled Zernike coefficients
        if enabled:
            if pars['all']:
                indices = np.arange(1, nz + 1)
            else:
                indices = get_noll_indices(pars)
        else:
            indices = np.array([], dtype=np.int)
        assert (calib.get_rzern().nk == nz)
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
            self.transform_pupil()
        except Exception as ex:
            self.log.info(f'error in transform_pupil {str(ex)}')
            self.pars['flipx'] = 0
            self.pars['flipy'] = 0
            self.pars['rotate'] = 0.0
            self.transform_pupil()

        self.z = np.zeros((nz, ))
        self.z1 = np.zeros((nz, ))
        self.ab = np.zeros((ndof, ))
        self.u = np.zeros((nu, ))

        # handle flattening
        try:
            self.flat_on = bool(pars['flat_on'])
            self.uflat = np.array(pars['uflat'])
            assert (self.uflat.size == calib.uflat.size)
        except Exception as ex:
            self.log.info(f'failed loading uflat {str(ex)}')
            self.flat_on = 1
            self.calib.reflatten()
            self.uflat = calib.uflat.copy()

        # handle initial value
        try:
            self.u[:] = np.array(pars['u'])
            self.z1[:] = self.u2z()
        except Exception as ex:
            self.log.info(f'fail to load u {str(ex)}')
            self.u[:] = self.uflat[:]
            self.z1[:] = self.u2z()

        if h5f:
            calib.save_h5py(h5f, prepend=h5_prefix)

        self.h5_make_empty('flat_on', (1, ), np.bool)
        self.h5_make_empty('uflat', (nu, ))
        self.h5_make_empty('x', (ndof, ))
        self.h5_make_empty('u', (nu, ))
        self.h5_save('name', self.__class__.__name__)
        self.h5_save('ab', self.ab)
        self.h5_save('P', np.eye(nz))
        self.h5_save('R', np.eye(nz))
        self.h5_save('params', json.dumps(pars))

    def __str__(self):
        return (f'<dmlib.control.{self.__class__.__name__} ' +
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

    def h5_make_empty(self, name, shape, dtype=float):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + name
            if name in self.h5f:
                del self.h5f[name]
            self.h5f.create_dataset(name,
                                    shape + (0, ),
                                    maxshape=shape + (None, ),
                                    dtype=dtype)

    def h5_append(self, name, what):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + name
            self.h5f[name].resize(
                (self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            name = h5_prefix + self.__class__.__name__ + '/' + where
            if name in self.h5f:
                del self.h5f[name]
            if isinstance(what, str):
                h5_store_str(self.h5f, name, what)
            else:
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
        assert (x.shape == self.ab.shape)

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
            self.log.warn('Saturation {}'.format(str(np.abs(self.u).max())))
            self.u[self.u > 1.] = 1.
            self.u[self.u < -1.] = -1.
            self.saturation = 1
        else:
            self.saturation = 0
        assert (norm(self.u, np.inf) <= 1.)

        # write raw voltages
        self.dm.write(self.u)

        if self.gui_callback:
            self.gui_callback()

    def set_random_ab(self, rms=1.0):
        self.ab[:] = normal(size=self.ab.size)
        self.ab[:] /= norm(self.ab.size)
        if self.h5f:
            self.h5f[h5_prefix + self.__class__.__name__ +
                     '/ab'][:] = self.ab[:]

    def transform_pupil(self):
        "Make pupil rotation matrix"
        alpha = self.pars['rotate']
        flipx = self.pars['flipx']
        flipy = self.pars['flipy']
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
            self.R = None
        else:
            self.R = tot
        self.save_R()
        self.make_P()

    def make_P(self):
        "Total orthogonal matrix"
        if self.R is None:
            self.P = None
        else:
            self.P = self.R.copy()

        if self.P is None:
            P = np.eye(self.nz)
        else:
            P = self.P

        addr = h5_prefix + self.__class__.__name__ + '/P'
        if self.h5f:
            if addr in self.h5f:
                self.h5f[addr][:] = P[:]
            else:
                self.h5f[addr] = P

    def save_R(self):
        "Save pupil rotation matrix"
        addr = h5_prefix + self.__class__.__name__ + '/R'
        if self.R is None:
            R = np.eye(self.nz)
        else:
            R = self.R

        if self.h5f:
            if addr in self.h5f:
                self.h5f[addr][:] = R[:]
            else:
                self.h5f[addr] = R


class SVDControl(ZernikeControl):
    @staticmethod
    def get_default_parameters():
        return {
            'modes': 5,
            'zernike_exclude': [1, 2, 3, 4],
        }

    @staticmethod
    def get_parameters_info():
        return {
            'modes': (int, (1, None), 'Number of SVD modes', 1),
            'zernike_exclude':
            (list, int, 'Exclude Zernike indices up to (inclusive)', 1),
        }

    def __init__(self, dm, calib, pars, h5=None):
        super().__init__(dm, calib, super().get_default_parameters(), h5)
        self.log = logging.getLogger(self.__class__.__name__)
        svd_pars = {
            **deepcopy(self.get_default_parameters()),
            **deepcopy(pars)
        }

        self.svd_pars = svd_pars
        try:
            enabled = self.svd_pars['enabled']
        except KeyError:
            enabled = 1

        if enabled:
            svd_modes = self.svd_pars['modes']
        else:
            svd_modes = 0
        ignore = np.array(self.svd_pars['zernike_exclude'], dtype=np.int)
        nignore = ignore.size

        self.h5_save('svd_modes', svd_modes)
        self.h5_save('ignore', ignore)

        self.make_P()
        if self.P:
            H = np.dot(self.P.T, self.calib.H)
        else:
            H = self.calib.H
        smap = np.zeros(H.shape[0], dtype=np.bool)
        smap[ignore - 1] = 1
        O1 = np.eye(H.shape[0])
        O1 = np.hstack((O1[:, smap], O1[:, np.invert(smap)])).T
        test = np.zeros(H.shape[0])
        test[smap] = 1
        assert (np.dot(O1, test)[:ignore.size].sum() == ignore.size)
        self.h5_save('O1', O1)
        H = np.dot(O1, H)
        self.h5_save('Htot', H)

        Hl = H[:nignore, :]
        # Hh = H[nignore:, :]
        _, _, Vt = svd(Hl)
        Vl2 = Vt[nignore:, :].T
        test1 = np.dot(H, Vl2)
        assert (test1.shape[0] == H.shape[0])
        assert (test1.shape[1] == H.shape[1] - nignore)
        assert (np.allclose(test1[:nignore, :], 0))

        U, s, Vt = svd(np.dot(H, Vl2))
        U1 = U[:, :s.size]
        np.allclose(U1[:nignore, :], 0)

        V1 = Vt[:svd_modes, :].T
        s1i = np.power(s[:svd_modes], -1)
        S1i = np.diag(s1i)

        self.h5_make_empty('x', (svd_modes, ))
        self.K = Vl2 @ V1 @ S1i
        self.ndof = svd_modes
        self.ab = np.zeros(svd_modes)
        self.h5_save('ab', self.ab)

        def f(n, w):
            a = h5_prefix + self.__class__.__name__ + '/svd/' + n
            if isinstance(w, str):
                h5_store_str(self.h5f, a, w)
            else:
                self.h5f[a] = w

        if self.h5f:
            f('nignore', nignore)
            f('svd_modes', svd_modes)
            f('Vl2', Vl2)
            f('V1', V1)
            f('S1i', S1i)
            f('K', self.K)
            f('params', json.dumps(svd_pars))

    def write(self, x):
        assert (x.shape == self.ab.shape)
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
            self.log.warn('Saturation {}'.format(str(np.abs(self.u).max())))
            self.u[self.u > 1.] = 1.
            self.u[self.u < -1.] = -1.
            self.saturation = 1
        else:
            self.saturation = 0

        assert (norm(self.u, np.inf) <= 1.)

        self.dm.write(self.u)

        if self.gui_callback:
            self.gui_callback()

    def save_parameters(self, merge={}):
        d = {**merge, **self.pars}
        d['flat_on'] = self.flat_on
        d['uflat'] = self.uflat.tolist()

    def set_random_ab(self, rms=1.0):
        raise NotImplementedError()


def get_noll_indices(pars):
    noll_min = np.array(pars['min'], dtype=np.int)
    noll_max = np.array(pars['max'], dtype=np.int)
    minclude = np.array(pars['include'], dtype=np.int)
    mexclude = np.array(pars['exclude'], dtype=np.int)

    mrange = np.arange(noll_min, noll_max + 1, dtype=np.int)
    zernike_indices1 = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))
    zernike_indices = []
    for k in minclude:
        if k in zernike_indices1 and k not in zernike_indices:
            zernike_indices.append(k)
    remaining = np.setdiff1d(zernike_indices1, np.unique(zernike_indices))
    for k in remaining:
        zernike_indices.append(k)
    assert (len(zernike_indices) == zernike_indices1.size)
    zernike_indices = np.array(zernike_indices, dtype=np.int)

    log = logging.getLogger('ZernikeControl')
    log.info(f'selected Zernikes {zernike_indices}')

    assert (zernike_indices.dtype == np.int)
    return zernike_indices


def get_controls():
    return {
        'ZernikeControl': ZernikeControl,
        'SVDControl': SVDControl,
    }


def new_control(dm, calib, name, pars={}, h5f=None):
    options = get_controls()
    if name not in options.keys():
        raise ValueError(f'name must be one of {", ".join(options.keys())}')

    return options[name](dm, calib, pars, h5f)
