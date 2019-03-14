#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from numpy.random import normal
from numpy.linalg import norm, matrix_rank, svd


def merge_pars(dp, up):
    p = {}
    for k, v in dp.items():
        if type(v) == dict:
            options = list(v.keys())
            if k not in up:
                p[k] = {options[0]: dp[k][options[0]]}
            else:
                choice = list(up[k].keys())
                assert(len(choice) == 1)
                choice = choice[0]
                assert(choice in dp[k].keys())
                p[k] = {choice: merge_pars(dp[k][choice], up[k][choice])}
        else:
            if k in up:
                p[k] = up[k]
            else:
                p[k] = dp[k]
    return p


class ZernikeControl:

    saturation = 0

    def __init__(self, dm, calib, pars={}, h5f=None):
        pars = merge_pars(get_default_parameters(), pars)
        self.pars = pars
        self.log = logging.getLogger(self.__class__.__name__)

        nz = calib.H.shape[0]
        nu = calib.H.shape[1]

        if pars['control']['Zernike']['all']:
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

        self.h5_save('uflat', calib.uflat)
        self.h5_save('indices', indices)
        self.h5_save('rad_to_nm', calib.get_rad_to_nm())
        # NOTE P is supposed to be orthonormal
        self.P = None

        self.z = np.zeros((nz,))
        self.z1 = np.zeros((nz,))
        self.ab = np.zeros((ndof,))
        self.u = np.zeros((nu,))
        self.u0 = np.zeros((nu,))

        self.flat_on = 1

        if h5f:
            calib.save_h5py(h5f)

        if h5f:
            self.h5_make_empty('flat_on', (1,), np.bool)
            self.h5_make_empty('x', (ndof,))
            self.h5_make_empty('u', (nu,))

        self.h5_save('name', self.__class__.__name__)
        self.h5_save('ab', self.ab)
        self.h5_save('P', np.eye(nz))

    def h5_make_empty(self, name, shape, dtype=np.float):
        if self.h5f:
            name = 'ZernikeControl/' + name
            if name in self.h5f:
                del self.h5f[name]
            self.h5f.create_dataset(
                name, shape + (0,), maxshape=shape + (None,),
                dtype=dtype)

    def h5_append(self, name, what):
        if self.h5f:
            name = 'ZernikeControl/' + name
            self.h5f[name].resize((
                self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            name = 'ZernikeControl/' + where
            if name in self.h5f:
                del self.h5f[name]
            self.h5f[name] = what

    def u2z(self):
        # for GUI purposes & does not include flat
        # z1 = P*z
        # u = C*z1
        # z1 = H*u
        if self.flat_on:
            tmp = self.u - self.calib.uflat
        else:
            tmp = self.u
        tmp += self.u0
        z1 = np.dot(self.calib.H, tmp)
        if self.P is None:
            return z1
        else:
            return np.dot(self.P.T, z1)

    def write(self, x):
        assert(x.shape == self.ab.shape)
        self.z[self.indices - 1] = x[:] + self.ab[:]
        if self.P is not None:
            np.dot(self.P, self.z, self.z1)
        else:
            self.z1[:] = self.z[:]

        np.dot(self.calib.C, self.z1, self.u)
        if self.flat_on:
            self.u += self.calib.uflat

        self.u += self.u0

        if self.h5f:
            self.h5_append('flat_on', self.flat_on)
            self.h5_append('x', x)
            self.h5_append('u', self.u)

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

    def set_random_ab(self, rms=1.0):
        self.ab[:] = normal(size=self.ab.size)
        self.ab[:] /= norm(self.ab.size)
        if self.h5f:
            self.h5f['ZernikeControl/ab'][:] = self.ab[:]

    def transform_pupil(self, alpha=0., flipx=False, flipy=False):
        if alpha != 0.:
            R = self.make_rot_matrix(alpha)
        else:
            R = 1

        if flipx:
            Fx = self.make_xflip_matrix()
        else:
            Fx = 1

        if flipy:
            Fy = self.make_yflip_matrix()
        else:
            Fy = 1

        tot = np.dot(Fy, np.dot(Fx, R))
        if tot.size == 1:
            return
        else:
            self.set_P(tot)

    def set_P(self, P):
        if P is None:
            self.P = None

            if self.h5f:
                del self.h5f['P']
                self.h5f['P'][:] = np.eye(self.nz)
        else:
            assert(P.ndim == 2)
            assert(P.shape[0] == P.shape[1])
            assert(np.allclose(np.dot(P, P.T), np.eye(P.shape[0])))
            if self.P is None:
                self.P = P.copy()
            else:
                np.dot(P, self.P.copy(), self.P)

            if self.h5f:
                del self.h5f['P']
                self.h5f['P'][:] = self.P[:]

    def make_rot_matrix(self, alpha):
        cz = self.calib.get_rzern()
        nml = list(zip(cz.ntab.tolist(), cz.mtab.tolist()))
        R = np.zeros((cz.nk, cz.nk))
        for i, nm in enumerate(nml):
            n, m = nm[0], nm[1]
            if m == 0:
                R[i, i] = 1.0
            elif m > 0:
                R[i, i] = np.cos(m*alpha)
                R[i, nml.index((n, -m))] = np.sin(m*alpha)
            else:
                R[i, nml.index((n, -m))] = -np.sin(abs(m)*alpha)
                R[i, i] = np.cos(abs(m)*alpha)

        # checks
        assert(matrix_rank(R) == R.shape[0])
        assert(norm((np.dot(R, R.T) - np.eye(cz.nk)).ravel()) < 1e-11)
        assert(norm((np.dot(R.T, R) - np.eye(cz.nk)).ravel()) < 1e-11)
        return R

    def make_yflip_matrix(self):
        cz = self.calib.get_rzern()
        nml = list(zip(cz.ntab.tolist(), cz.mtab.tolist()))
        R = np.zeros((cz.nk, cz.nk))
        for i, nm in enumerate(nml):
            m = nm[1]
            if m < 0:
                R[i, i] = -1.0
            else:
                R[i, i] = 1.0

        # checks
        assert(matrix_rank(R) == R.shape[0])
        assert(norm((np.dot(R, R.T) - np.eye(cz.nk)).ravel()) < 1e-11)
        assert(norm((np.dot(R.T, R) - np.eye(cz.nk)).ravel()) < 1e-11)
        return R

    def make_xflip_matrix(self):
        cz = self.calib.get_rzern()
        nml = list(zip(cz.ntab.tolist(), cz.mtab.tolist()))
        R = np.zeros((cz.nk, cz.nk))
        for i, nm in enumerate(nml):
            m = nm[1]
            if abs(m) % 2 == 0 and m < 0:
                R[i, i] = -1.0
            elif abs(m) % 2 == 1 and m > 0:
                R[i, i] = -1.0
            else:
                R[i, i] = 1.0
        # checks
        assert(matrix_rank(R) == R.shape[0])
        assert(norm((np.dot(R, R.T) - np.eye(cz.nk)).ravel()) < 1e-11)
        assert(norm((np.dot(R.T, R) - np.eye(cz.nk)).ravel()) < 1e-11)
        return R

    def store_u0(self, u):
        self.u0[:] = u
        self.h5_save('u0', self.u0)


class SVDControl(ZernikeControl):

    def __init__(self, dm, calib, pars, h5=None):
        super().__init__(dm, calib, pars, h5)

        svd_modes = self.pars['control']['SVD']['modes']
        nignore = self.pars['control']['SVD']['zernike_exclude'] - 1

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
        # U2 = U[:, s.size:]
        np.allclose(U1[:nignore, :], 0)

        nmodes = pars.svd_modes
        V1 = Vt[:nmodes, :].T
        s1i = np.power(s[:nmodes], -1)
        S1i = np.diag(s1i)

        self.h5_make_empty('x', (nmodes,))
        self.K = Vl2@V1@S1i
        self.ndof = nmodes
        self.ab = np.zeros(nmodes)
        self.h5_save('ab', self.ab)

        def f(n, w):
            self.h5f['ZernikeControl/SVDControl/' + n] = w

        if self.h5f:
            f('nignore', nignore)
            f('nmodes', nmodes)
            f('Vl2', Vl2)
            f('V1', V1)
            f('S1i', S1i)
            f('K', self.K)

    def write(self, x):
        assert(x.shape == self.ab.shape)
        z = x + self.ab
        np.dot(self.K, z, self.u)
        if self.flat_on:
            self.u += self.calib.uflat

        self.u += self.u0

        if self.h5f:
            self.h5_append('flat_on', self.flat_on)
            self.h5_append('x', x)
            self.h5_append('u', self.u)

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

    def set_random_ab(self, rms=1.0):
        raise NotImplementedError()

    def set_P(self, P):
        raise NotImplementedError()


def get_noll_indices(params):
    p = params['control']
    if 'Zernike' in p:
        z = p['Zernike']
        noll_min = z['min']
        noll_max = z['max']
        minclude = z['include']
        mexclude = z['exclude']
    else:
        RuntimeError()

    mrange = np.arange(noll_min, noll_max + 1)
    zernike_indices = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))

    log = logging.getLogger('ZernikeControl')
    log.info(f'selected Zernikes {zernike_indices}')

    return zernike_indices


control_types = {
    'Zernike': ZernikeControl,
    'SVD': SVDControl,
    }


def get_default_parameters():
    return {
        'control': {
            'Zernike': {
                'include': [],
                'exclude': [1, 2, 3, 4],
                'min': 5,
                'max': 6,
                'all': 1,
                },
            'SVD': {
                'modes': 5,
                'exclude': [1, 2, 3, 4],
                },
            }
        }


def get_parameters_info():
    return {
        'control': {
            'Zernike': {
                'include': (list, int, 'Zernike indices to include'),
                'exclude': (list, int, 'Zernike indices to include'),
                'min': (int, (1, None), 'Minimum Zernike index'),
                'max': (int, (1, None), 'Maximum Zernike index'),
                'all': (
                    int, (0, 1), 'Use all Zernike available in calibration'),
                },
            'SVD': {
                'modes': (int, (1, None), 'Number of SVD modes'),
                'zernike_exclude': (
                    int, (1, None),
                    'Exclude Zernike indices up to (inclusive)'),
                },
            }
        }
