#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from numpy.random import normal
from numpy.linalg import norm, matrix_rank, svd


class ZernikeControl:

    saturation = 0

    def __init__(self, dm, calib, args=None, h5f=None):
        self.args = args
        self.log = logging.getLogger('ZernikeControl')

        nz = calib.H.shape[0]
        nu = calib.H.shape[1]

        if args is None:
            indices = np.arange(1, nz + 1)
        else:
            indices = get_noll_indices(args)
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

        def make_empty(name, shape, dtype=np.float):
            h5f.create_dataset(
                name, shape + (0,), maxshape=shape + (None,),
                dtype=dtype)
        if h5f:
            make_empty('ZernikeControl/flat_on', (1,), np.bool)
            make_empty('ZernikeControl/x', (ndof,))
            make_empty('ZernikeControl/u', (nu,))

        self.h5_save('name', self.__class__.__name__)
        self.h5_save('ab', self.ab)
        self.h5_save('P', np.eye(nz))

    def h5_append(self, name, what):
        if self.h5f:
            self.h5f[name].resize((
                self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            self.h5f['ZernikeControl/' + where] = what

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
            self.h5_append('ZernikeControl/flat_on', self.flat_on)
            self.h5_append('ZernikeControl/x', x)
            self.h5_append('ZernikeControl/u', self.u)

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
    def __init__(self, dm, calib, args, h5=None):
        super().__init__(dm, calib, dm, calib, args, h5)
        self.log = logging.getLogger('SVDControl')

        self.h5_save('svd_modes', args.svd_modes)

        H = self.calib.H
        nmax = (
            np.fromstring(args.noll_exclude, dtype=np.int, sep=',') - 1).max()
        Hl = H[:nmax, :]
        _, _, Vt = svd(Hl)
        Vl2 = Vt[nmax:, :].T
        self.ndof = min(args.svd_modes, Vl2.shape[1])

    def set_random_ab(self, rms=1.0):
        raise NotImplementedError()


def get_noll_indices(args):
    if args.noll_min > 0 and args.noll_max > 0:
        mrange = np.arange(args.noll_min, args.noll_max + 1)
    else:
        mrange = np.array([], dtype=np.int)

    if args.noll_include != '':
        minclude = np.fromstring(args.noll_include, dtype=np.int, sep=',')
        minclude = minclude[minclude > 0]
    else:
        minclude = np.array([], dtype=np.int)

    if args.noll_exclude != '':
        mexclude = np.fromstring(args.noll_exclude, dtype=np.int, sep=',')
        mexclude = mexclude[mexclude > 0]
    else:
        mexclude = np.array([], dtype=np.int)

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


def add_control_parameters(parser):
    parser.add_argument(
        '--noll-include', type=str, default='', metavar='INDICES',
        help='Comma separated list of Noll indices to include, eg 1,2')
    parser.add_argument(
        '--noll-exclude', type=str, default='1,2,3,4', metavar='INDICES',
        help='Comma separated list of Noll indices to exclude, eg 1,2')
    parser.add_argument(
        '--noll-min', type=int, default=5, metavar='MIN',
        help='Minimum Noll index to consider, use -1 to ignore')
    parser.add_argument(
        '--noll-max', type=int, default=6, metavar='MAX',
        help='Maximum Noll index to consider, use -1 to ignore')
    parser.add_argument(
        '--control', choices=list(control_types.keys()),
        default=list(control_types.keys())[0], help='DM control type')
    parser.add_argument(
        '--svd-modes', type=int, default=2, metavar='N',
        help='Correct N SVD modes if using SVD control')
