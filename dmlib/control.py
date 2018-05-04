#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""dmcontrol - code to control the DM using Zernike polynomials

TODO merge high-NA defocus code

author: J. Antonello <jacopo.antonello@dpag.ox.ac.uk>
date: Wed Oct  4 12:16:36 BST 2017

"""

import logging
import numpy as np

from numpy.random import normal
from numpy.linalg import norm, matrix_rank
from scipy.io import loadmat

from czernike import RZern
from bmcs import BMC

# LabView convertion u to voltage:
# d = dmax*(u + 1)/2
# V = (-b + sqrt(b**2 + 4*a*D))/(2*a)

# MATLAB calibration
# d_f = f1(v_f)
# d_t = d_f + d
# u_t = 2*d_t/f1(300) - 1
# v_t = f2(d_t)

# SIGNAL VS VOL. DATABASE (Voltage vs Signal_Data Engine_V1)
# 1: 13RW018#054 Upper DM in optsys
# 0: 13RW023#017 Lower DM in optsys
# 0: 247.8239, 0.060365, -0.83694, 3500
# 1: 294.658, 0.03945, 0.2539, 3500
# Vmax, a, b, dmax


class DMCalib:

    flatOn = 1.

    def __init__(self, matfile):
        self.matfile = loadmat(matfile)
        self.uflat = self.matfile['u_flat_ls'].ravel()
        self.calibration_lambda = self.matfile['calibration_lambda'][0, 0]
        self.rad_to_nm = (self.calibration_lambda/1e-9)/(2*np.pi)

        self.Hf = np.ascontiguousarray(self.matfile['Hf'])
        self.Cf = np.ascontiguousarray(self.matfile['Cf'])
        self.H1 = np.ascontiguousarray(self.matfile['H1'])
        self.C1 = np.ascontiguousarray(self.matfile['C1'])

        self.noll_indices = set(range(1, self.Hf.shape[0] + 1))

    def rad_to_nm(self, rad):
        return rad*self.rad_to_nm

    def add_flat_to_u(self, u):
        u = self.flatOn*self.uflat + u
        return u

    def set_zernike_indices(self, noll_indices):
        # stored in Noll - 1
        if not isinstance(noll_indices, np.ndarray):
            noll_indices = np.array(noll_indices, dtype=np.int)
        self.zernike_indices = noll_indices - 1

    def zernike_to_u(self, x):
        assert(self.zernike_indices.size == x.size)
        xf = np.zeros((self.Cf.shape[1],))
        xf[self.zernike_indices] = x
        return np.dot(self.Cf, xf)

    def rotate_pupil(self, rad):
        if rad != 0.:
            T = self.make_rot_matrix(self.Hf.shape[0], rad)
            self.Hf = np.dot(T, self.Hf)
            self.Cf = np.dot(self.Cf, T.T)

    def flip_pupil(self, xflip, yflip):
        if xflip:
            T = self.make_xflip_matrix(self.Hf.shape[0])
            self.Hf = np.dot(T, self.Hf)
            self.Cf = np.dot(self.Cf, T)
        if yflip:
            T = self.make_yflip_matrix(self.Hf.shape[0])
            self.Hf = np.dot(T, self.Hf)
            self.Cf = np.dot(self.Cf, T)

    def make_rot_matrix(self, nz, alpha):
        # nz = (n + 1)*(n + 2)/2 = 1/2*n**2 + 3/2*n + 1
        n = int(-(3/2) + np.sqrt((3/2)**2 - 4/2*(1 - nz)))
        cz = RZern(n)
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

    def make_yflip_matrix(self, nz):
        n = int(-(3/2) + np.sqrt((3/2)**2 - 4/2*(1 - nz)))
        cz = RZern(n)
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

    def make_xflip_matrix(self, nz):
        n = int(-(3/2) + np.sqrt((3/2)**2 - 4/2*(1 - nz)))
        cz = RZern(n)
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


class Zernike:

    def __init__(self, optsys, args):
        self.optsys = optsys
        self.dm0 = DMCalib('calibration.mat')

        zernike_indices = get_zernike_indeces_from_args(args)

        if optsys.h5f:
            optsys.h5f['dmcontrol/zernike_indices'] = zernike_indices
            optsys.h5f['dmcontrol/rad_to_nm'] = self.dm0.rad_to_nm

        logging.info('dmcontrol: lambda = {} nm'.format(
            self.dm0.matfile['calibration_lambda'][0, 0]/1e-9))
        logging.info('dmcontrol: rad_to_nm = {}'.format(self.dm0.rad_to_nm))

        self.flag_dm0 = 1.0
        self.dm0.set_zernike_indices(zernike_indices)
        self.ndof0 = self.dm0.zernike_indices.size
        self.ndof = self.ndof0

        # initial DM aberration
        if args.dm_ab != 0.0:
            self.set_random_dm_ab_calib_rad(args.dm_ab)
        else:
            self.set_random_dm_ab_calib_rad(0.0)

        if optsys.h5f:
            optsys.h5f['dmcontrol/dm_ab_calib_rad'] = self.dm_ab_calib_rad

        self.log_write_count = 0

        self.bmc = BMC()
        self.bmc.open('aaaaaaaaaaa')

    def get_ndof(self):
        return self.ndof

    def set_random_dm_ab_calib_rad(self, rms=0.5):
        ab = normal(size=(self.ndof,))
        ab = (rms/norm(ab))*ab
        self.dm_ab_calib_rad = ab

    def write_settings(self, x):
        x = x.copy()

        if self.dm_ab_calib_rad is not None:
            x += self.dm_ab_calib_rad

        def trim(u, what):
            if norm(u, np.inf) > 1:
                logging.warn(
                    'Saturation {} {}'.format(what, str(np.abs(u).max())))
                u[u > 1.] = .99
                u[u < -1.] = -.99
            assert(norm(u, np.inf) <= 1.)
            return u

        u0 = trim(self.dm0.zernike_to_u(x), 'u0')
        u1 = trim(self.dm0.add_flat_to_u(u0), 'u1')
        v = np.sqrt((u1 + 1.0)/2.0)
        assert(np.all(np.isfinite(u0)))
        assert(np.all(np.isfinite(u1)))
        assert(np.all(np.isfinite(v)))

        if self.optsys.h5f:
            self.optsys.h5f['dmcontrol/x/{:09d}'.format(
                self.log_write_count)] = x
            self.optsys.h5f['dmcontrol/u/{:09d}'.format(
                self.log_write_count)] = u0
            self.optsys.h5f['dmcontrol/v/{:09d}'.format(
                self.log_write_count)] = v

        self.bmc.write(v)
        self.optsys.write_settings(x)

        # self.optsys.write_settings(
        #     self.dm0.add_flat_to_u(dm0),
        #     self.dm1.add_flat_to_u(dm1),
        #     zcomp)

        self.log_write_count += 1

    def close(self):
        if self.bmc:
            self.bmc.close()
            self.bmc = None


control_classes = [Zernike]
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
