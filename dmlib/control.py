#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from numpy.random import normal
from numpy.linalg import norm, matrix_rank


class ZernikeControl:

    saturation = 0

    def __init__(self, dm, calib, indices=None, h5f=None):
        self.log = logging.getLogger('ZernikeControl')

        nz = calib.H.shape[0]
        nu = calib.H.shape[1]

        if indices is None:
            indices = np.arange(1, nz + 1)
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

        self.h5_save('ab', self.ab)
        self.h5_save('P', np.eye(nz, nz))

    def h5_append(self, name, what):
        if self.h5f:
            self.h5f[name].resize((
                self.h5f[name].shape[0], self.h5f[name].shape[1] + 1))
            self.h5f[name][:, -1] = what

    def h5_save(self, where, what):
        if self.h5f:
            self.h5f['ZernikeControl/' + where] = what

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
            if self.P is None:
                self.P = tot
            else:
                np.dot(tot, self.P.copy(), self.P)

        if self.h5f:
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


class SVDControl(ZernikeControl):
    def __init__(self, dm, calib, indices=None, h5f=None):
        super().__init__(dm, calib, indices, h5f)


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
        '--noll-exclude', type=str, default='', metavar='INDICES',
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
