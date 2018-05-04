#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from numpy.random import normal
from numpy.linalg import norm, matrix_rank


class ZernikeControl:

    def __init__(self, dm, calib, indices=None, h5f=None):
        nz = calib.H.shape[0]
        nu = calib.H.shape[1]

        if indices is None:
            indices = np.arange(2, nz + 1)
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
        self.P = None

        self.z = np.zeros((nz,))
        self.z1 = np.zeros((nz,))
        self.ab = np.zeros((ndof,))
        self.u = np.zeros((nu,))

        self.flat_on = 1
        self.scale_z = 1

        def make_empty(name, shape):
            h5f.create_dataset(
                name, shape + (0,), maxshape=shape + (None,),
                dtype=np.float)
        if h5f:
            make_empty('x', (ndof,))
            make_empty('u', (nu,))

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
        self.z[self.indices - 1] = x[:] + self.ab[:]
        if self.P is not None:
            np.dot(self.P, self.z, self.z1)
        else:
            self.z1[:] = self.z[:]
        self.z1 *= self.scale_z

        np.dot(self.calib.C, self.z1, self.u)
        if self.flat_on:
            self.u += self.calib.uflat

        if self.h5f:
            self.h5_append('x', x)
            self.h5_append('u', self.u)

        def trim(u):
            if norm(u, np.inf) > 1:
                logging.warn(
                    'Saturation {}'.format(str(np.abs(u).max())))
                u[u > 1.] = 1.
                u[u < -1.] = -1.

        trim(self.u)
        assert(norm(self.u, np.inf) <= 1.)

        self.dm.write(self.u)

    def set_random_ab(self, rms=1.0):
        self.ab[:] = normal(size=self.ab.size)
        self.ab[:] /= norm(self.ab.size)
        if self.h5f:
            self.h5f['ab'][:] = self.ab[:]

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


control_classes = [ZernikeControl]
control_names = [c.__name__ for c in control_classes]


def apply_control(name, optsys, args):
    if name not in control_names:
        raise ValueError(
            'Control name must be any of {}'.format(str(control_names)))
    return control_classes[control_names.index(name)](optsys, args)


def get_zernike_indeces_from_args(args):
    if args.z_min > 0 and args.z_max > 0:
        mrange = np.arange(args.z_min, args.z_max + 1)
    else:
        mrange = np.array([], dtype=np.int)

    if args.noll_include is not None:
        minclude = np.fromstring(args.noll_include, dtype=np.int, sep=',')
        minclude = minclude[minclude > 0]
    else:
        minclude = np.array([], dtype=np.int)

    if args.noll_exclude is not None:
        mexclude = np.fromstring(args.noll_exclude, dtype=np.int, sep=',')
        mexclude = mexclude[mexclude > 0]
    else:
        mexclude = np.array([], dtype=np.int)

    zernike_indices = np.setdiff1d(
        np.union1d(np.unique(mrange), np.unique(minclude)),
        np.unique(mexclude))

    logging.info('Selected Noll indices for the Zernike modes are:')
    logging.info(zernike_indices)

    return zernike_indices


def add_control_parameters(parser):
    parser.add_argument(
        '--dm-ab', type=float, default=0.0, metavar='RMS',
        help='Add random DM aberration of RMS [rad] (calibration lambda)')
    parser.add_argument(
        '--control', metavar='NAME',
        choices=control_names, default=control_names[0],
        help='Select a DM control')
    parser.add_argument(
        '--noll-include', type=str, default=None, metavar='INDICES',
        help='''
Comma separated list of Noll indices to include, e.g.,
1,2,3,4,5,6.
NB: DO NOT USE SPACES in the list!''')
    parser.add_argument(
        '--noll-exclude', type=str, default=None, metavar='INDICES',
        help='''
Comma separated list of Noll indices to exclude, e.g.,
1,5,6 to exclude piston and astigmatism.
NB: DO NOT USE SPACES in the list!''')
    parser.add_argument(
        '--z-min', type=int, default=5, metavar='MIN',
        help='Minimum Noll index to consider, use -1 to ignore')
    parser.add_argument(
        '--z-max', type=int, default=10, metavar='MAX',
        help='Maximum Noll index to consider, use -1 to ignore')
