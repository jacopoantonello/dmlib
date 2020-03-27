#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from h5py import File
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from numpy.linalg import lstsq, pinv
from scipy.linalg import cholesky, solve_triangular
from time import time
from skimage.restoration import unwrap_phase

from zernike import RZern

from dmlib.core import SquareRoot
from dmlib.interf import FringeAnalysis

LOG = logging.getLogger('calibration')
HDF5_options = {
    'chunks': True,
    'shuffle': True,
    'fletcher32': True,
    'compression': 'gzip',
    'compression_opts': 6}


def u2v(u, vmin, vmax, sqrt=False):
    u = np.array(u, copy=True)
    u[u > 1.] = 1.
    u[u < -1.] = -1.
    v = (u + 1.)/2.
    if sqrt:
        v = np.sqrt(v)
    return v*(vmax - vmin) + vmin


def v2u(v, vmin, vmax, sqrt=False):
    v = np.array(v, copy=True)
    v[v > vmax] = vmax
    v[v < vmin] = vmin
    u = (v - vmin)/(vmax - vmin)
    if sqrt:
        u = np.power(u, 2)
    return 2*u - 1


def make_normalised_input_matrix(nacts, nsteps, mag):
    if nsteps < 3:
        raise RuntimeError('nsteps < 3')
    elif nsteps % 2 != 1:
        raise RuntimeError('nsteps % 2 != 1')

    return np.hstack((
        np.zeros((nacts, 1)),
        np.kron(np.eye(nacts), np.linspace(-mag, mag, nsteps)),
        np.zeros((nacts, 1))
        ))


def fix_principal_val(U, phases):
    ns = phases.shape[0]
    norms = np.square(U).sum(axis=0)
    assert(norms.size == ns)
    inds = np.where(norms <= 1e-6)[0]

    nodes0 = [phases[i, :].mean() for i in inds]
    nodes1 = unwrap_phase(np.array(nodes0))
    pistons1 = phases.mean(axis=1)

    xq = np.linspace(-1, 1, ns)
    yq = interp1d(xq[inds], nodes1, copy=False, assume_sorted=True)(xq)

    pistons2 = np.zeros_like(pistons1)
    for i in range(ns):
        k = round((yq[i] - pistons1[i])/(2*np.pi))
        pistons2[i] = pistons1[i] + 2*k*np.pi - yq[i]
    phases += (pistons2 - pistons1).reshape(-1, 1)

    return inds


# https://stackoverflow.com/questions/10117073
class PhaseExtract:

    def __init__(self, fringe):
        self.fringe = fringe
        self.mask = np.invert(self.fringe.mask)

    def __call__(self, img):
        self.fringe.analyse(img)
        unwrapped = self.fringe.unwrapped
        return unwrapped[self.mask]


class RegLSCalib:
    """Compute a DM calibration using regularised least-squares."""

    def __init__(self):
        self.zfA1 = None
        self.zfA2 = None

    def _make_zfAs(self):
        if not hasattr(self.cart, 'ZZ'):
            xx, yy, _ = self.fringe.get_unit_aperture()
            self.cart.make_cart_grid(xx, yy)

        mask = np.invert(self.zfm)
        zfA1 = np.zeros((self.zfm.sum(), self.cart.nk))
        zfA2 = np.zeros_like(self.cart.ZZ)
        for i in range(zfA1.shape[1]):
            tmp = self.cart.matrix(self.cart.ZZ[:, i])
            zfA1[:, i] = tmp[self.zfm].ravel()
            zfA2[:, i] = tmp.ravel()

        self.zfA1 = zfA1
        self.zfA2 = zfA2
        self.zfA1TzfA1 = np.dot(zfA1.T, zfA1)
        self.chzfA1TzfA1 = cholesky(self.zfA1TzfA1, lower=False)

        return zfA1, zfA2, mask

    def calibrate(
            self, U, images, fringe, wavelength, cam_pixel_size,
            cam_serial='',
            dname='', dm_serial='', dmplot_txs=(0, 0, 0),
            dm_transform=SquareRoot.name, hash1='',
            n_radial=25, alpha=.75, lambda1=5e-3, status_cb=False):

        if status_cb:
            status_cb('Computing Zernike polynomials ...')
        t1 = time()
        nu, ns = U.shape
        xx, yy, shape = fringe.get_unit_aperture()
        assert(xx.shape == shape)
        assert(yy.shape == shape)
        cart = RZern(n_radial)
        cart.make_cart_grid(xx, yy)
        LOG.info(
            f'calibrate(): Computing Zernike polynomials {time() - t1:.1f}')

        if status_cb:
            status_cb('Computing masks ...')
        t1 = time()
        zfm = cart.matrix(np.isfinite(cart.ZZ[:, 0]))
        self.cart = cart
        self.zfm = zfm
        zfA1, zfA2, mask = self._make_zfAs()
        LOG.info(
            f'calibrate(): Computing masks {time() - t1:.1f}')

        # TODO remove me
        mask1 = np.sqrt(xx**2 + yy**2) >= 1.
        assert(np.allclose(mask, mask1))
        assert(np.allclose(fringe.mask, mask1))

        if status_cb:
            status_cb('Computing phases 00.00% ...')
        t1 = time()

        def make_progress():
            prevts = [time()]

            def f(pc):
                t = time()
                dt = t - prevts[0]
                prevts[0] = t
                if dt > 1.5 or pc > 99:
                    status_cb(f'Computing phases {pc:05.2f}% ...')
            return f

        with Pool() as p:
            if status_cb:
                chunksize = ns//(4*cpu_count())
                if chunksize < 4:
                    chunksize = 4
                phases = []
                progress_fun = make_progress()
                for i, phi in enumerate(p.imap(
                        PhaseExtract(fringe),
                        [images[i, ...] for i in range(ns)],
                        chunksize), 1):
                    phases.append(phi)
                    progress_fun(100*i/ns)
            else:
                phases = p.map(
                    PhaseExtract(fringe), [images[i, ...] for i in range(ns)])
            phases = np.array(phases)

        inds0 = fix_principal_val(U, phases)
        inds1 = np.setdiff1d(np.arange(ns), inds0)
        assert(np.allclose(np.arange(ns), np.sort(np.hstack((inds0, inds1)))))
        phi0 = phases[inds0, :].mean(axis=0)
        z0 = lstsq(np.dot(zfA1.T, zfA1), np.dot(zfA1.T, phi0), rcond=None)[0]
        phases -= phi0.reshape(1, -1)
        LOG.info(
            f'calibrate(): Computing phases {time() - t1:.1f}')

        if status_cb:
            status_cb('Computing least-squares matrices ...')
        t1 = time()
        nphi = phases.shape[1]
        uiuiT = np.zeros((nu, nu))
        phiiuiT = np.zeros((nphi, nu))
        for i in inds1:
            uiuiT += np.dot(U[:, [i]], U[:, [i]].T)
            phiiuiT += np.dot(phases[[i], :].T, U[:, [i]].T)
        LOG.info(
            f'calibrate(): Computing least-squares matrices {time() - t1:.1f}')
        if status_cb:
            status_cb('Solving least-squares ...')
        t1 = time()
        A = np.dot(zfA1.T, zfA1)
        C = np.dot(zfA1.T, phiiuiT)
        B = uiuiT
        U1 = cholesky(A, lower=False, overwrite_a=True)
        Y = solve_triangular(U1, C, trans='T', lower=False)
        D = solve_triangular(U1, Y, trans='N', lower=False)
        U2 = cholesky(B, lower=False, overwrite_a=True)
        YT = solve_triangular(U2, D.T, trans='T', lower=False)
        XT = solve_triangular(U2, YT, trans='N', lower=False)
        H = XT.T

        def vaf(y, ye):
            return 100*(1 - np.var(y - ye, axis=1)/np.var(y, axis=1))

        mvaf = vaf(phases.T, zfA1@H@U)
        LOG.info(
            f'calibrate(): Solving least-squares {time() - t1:.1f}')

        if status_cb:
            status_cb('Applying regularisation ...')
        t1 = time()
        if alpha > 0.:
            # weighted least squares
            rr = np.sqrt(xx**2 + yy**2)
            win = .5*(1 + np.cos(np.pi*((2*rr/(alpha) - 2/alpha + 1))))
            win[rr < 1 - alpha/2] = 1
            win[rr >= 1] = 0

            stds = np.zeros(nu)
            for i in range(nu):
                ind = np.where(U[i, :] == U.max())[0][0]
                stds[i] = np.std(phases[ind]*win[zfm])
            stds -= stds.min()
            stds /= stds.max()
            assert(stds.min() == 0.)
            assert(stds.max() == 1.)

            C = np.dot(pinv(lambda1*np.diag(1 - stds) + np.dot(H.T, H)), H.T)
        else:
            C = np.linalg.pinv(H)
        uflat = -np.dot(C, z0)

        self.fringe = fringe
        self.shape = shape

        self.H = H
        self.mvaf = mvaf
        self.phi0 = phi0
        self.z0 = z0
        self.uflat = uflat
        self.C = C
        self.alpha = alpha
        self.lambda1 = lambda1

        self.wavelength = wavelength
        self.dm_serial = dm_serial
        self.dm_transform = dm_transform
        self.cam_pixel_size = cam_pixel_size
        self.cam_serial = cam_serial
        self.dmplot_txs = dmplot_txs
        self.dname = dname
        self.hash1 = hash1

        LOG.info(
            f'calibrate(): Applying regularisation {time() - t1:.1f}')

    def reflatten(self, exclude_zernike_noll=4):
        tmp = self.z0.copy()
        tmp[:exclude_zernike_noll] = 0
        self.uflat = -np.dot(self.C, tmp)

    def get_rzern(self):
        return self.cart

    def zernike_eval(self, z):
        if self.zfA2 is None:
            self._make_zfAs()

        return np.dot(self.zfA2, z).reshape(self.shape)

    def zernike_fit(self, phi):
        if self.zfA2 is None:
            self._make_zfAs()

        Y = solve_triangular(
            self.chzfA1TzfA1, np.dot(self.zfA1.T, phi[self.zfm]),
            trans='T', lower=False)
        s2 = solve_triangular(self.chzfA1TzfA1, Y, trans='N', lower=False)

        # s1 = lstsq(
        #     self.zfA1TzfA1,
        #     np.dot(self.zfA1.T, phi[self.zfm]), rcond=None)[0]
        # assert(np.allclose(s1, s2))

        return s2

    def apply_aperture_mask(self, phi):
        phi[np.invert(self.zfm)] = -np.inf
        phi[self.zfm] -= phi[self.zfm].mean()

    def get_radius(self):
        return self.fringe.radius

    def get_rad_to_nm(self):
        return (self.wavelength)/(2*np.pi)

    @classmethod
    def query_calibration(cls, f):
        with File(f, 'r') as f:
            if 'RegLSCalib' not in f:
                raise ValueError(f.filename + ' is not a calibration file')
            else:
                return (
                    f['RegLSCalib/dm_serial'][()],
                    f['RegLSCalib/dm_transform'][()],
                    f['RegLSCalib/dmplot_txs'][()],
                    f['RegLSCalib/H'].shape,
                    )

    @classmethod
    def load_h5py(cls, f, prepend=None, lazy_cart_grid=False):
        """Load object contents from an opened HDF5 file object."""
        z = cls()

        prefix = cls.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        z.cart = RZern.load_h5py(f, prefix + 'cart/')
        z.fringe = FringeAnalysis.load_h5py(f, prefix + 'fringe/')
        z.zfA1 = None
        z.zfA2 = None
        z.zfm = f[prefix + 'zfm'][()]
        z.shape = f[prefix + 'shape'][()]

        z.H = f[prefix + 'H'][()]
        z.mvaf = f[prefix + 'mvaf'][()]
        z.phi0 = f[prefix + 'phi0'][()]
        z.z0 = f[prefix + 'z0'][()]
        z.uflat = f[prefix + 'uflat'][()]
        z.C = f[prefix + 'C'][()]
        z.alpha = f[prefix + 'alpha'][()][0]
        z.lambda1 = f[prefix + 'lambda1'][()][0]

        z.wavelength = f[prefix + 'wavelength'][()]
        z.dm_serial = f[prefix + 'dm_serial'][()]
        z.dm_transform = f[prefix + 'dm_transform'][()]
        z.cam_pixel_size = f[prefix + 'cam_pixel_size'][()]
        z.cam_serial = f[prefix + 'cam_serial'][()]
        z.dmplot_txs = f[prefix + 'dmplot_txs'][()]
        z.dname = f[prefix + 'dname'][()]
        z.hash1 = f[prefix + 'hash1'][()]

        if not lazy_cart_grid:
            xx, yy, _ = z.fringe.get_unit_aperture()
            z.cart.make_cart_grid(xx, yy)

        z.zfA1TzfA1 = None
        z.chzfA1TzfA1 = None

        return z

    def save_h5py(self, f, prepend=None, params=HDF5_options):
        """Dump object contents into an opened HDF5 file object."""
        prefix = self.__class__.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        try:
            ZZ = self.cart.ZZ
            del self.cart.ZZ
        except AttributeError:
            ZZ = None
        self.cart.save_h5py(f, prefix + 'cart/', params=HDF5_options)
        if ZZ is not None:
            self.cart.ZZ = ZZ
        self.fringe.save_h5py(f, prefix + 'fringe/', params=HDF5_options)

        params['data'] = self.zfm
        f.create_dataset(prefix + 'zfm', **params)
        params['data'] = self.shape
        f.create_dataset(prefix + 'shape', **params)

        params['data'] = self.H
        f.create_dataset(prefix + 'H', **params)
        params['data'] = self.mvaf
        f.create_dataset(prefix + 'mvaf', **params)
        params['data'] = self.phi0
        f.create_dataset(prefix + 'phi0', **params)
        params['data'] = self.z0
        f.create_dataset(prefix + 'z0', **params)
        params['data'] = self.uflat
        f.create_dataset(prefix + 'uflat', **params)
        params['data'] = self.C
        f.create_dataset(prefix + 'C', **params)
        f.create_dataset(prefix + 'alpha', data=np.array([self.alpha]))
        f.create_dataset(prefix + 'lambda1', data=np.array([self.lambda1]))

        f[prefix + 'wavelength'] = self.wavelength
        f[prefix + 'dm_serial'] = self.dm_serial
        f[prefix + 'dm_transform'] = self.dm_transform
        f[prefix + 'cam_pixel_size'] = self.cam_pixel_size
        f[prefix + 'cam_serial'] = self.cam_serial
        f[prefix + 'dmplot_txs'] = self.dmplot_txs
        f[prefix + 'dname'] = self.dname
        f[prefix + 'hash1'] = self.hash1
