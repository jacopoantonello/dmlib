#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from h5py import File
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from numpy.linalg import lstsq, pinv
from scipy.linalg import cholesky, solve_triangular
from time import time
from skimage.restoration import unwrap_phase

from zernike.czernike import RZern

from dmlib.interf import FringeAnalysis

HDF5_options = {
    'chunks': True,
    'shuffle': True,
    'fletcher32': True,
    'compression': 'gzip',
    'compression_opts': 6}


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

    def __call__(self, img):
        self.fringe.analyse(img)
        unwrapped = self.fringe.unwrapped
        return unwrapped[np.invert(self.fringe.mask)]


class WeightedLSCalib:

    def __init__(self):
        pass

    def calibrate(
            self, U, images, fringe, wavelength, dm_serial, dm_transform,
            cam_pixel_size, cam_serial, dmplot_txs, dname, hash1,
            n_radial=25, alpha=.75, lambda1=5e-3, status_cb=False):

        if status_cb:
            status_cb('Computing Zernike polynomials ...')
        nu, ns = U.shape
        xx, yy, shape = fringe.get_unit_aperture()
        assert(xx.shape == shape)
        assert(yy.shape == shape)
        cart = RZern(n_radial)
        cart.make_cart_grid(xx, yy)

        if status_cb:
            status_cb('Computing masks ...')
        zfm = cart.matrix(np.isfinite(cart.ZZ[:, 0]))
        mask = np.invert(zfm)
        zfA1 = np.zeros((zfm.sum(), cart.nk))
        zfA2 = np.zeros_like(cart.ZZ)
        for i in range(zfA1.shape[1]):
            tmp = cart.matrix(cart.ZZ[:, i])
            zfA1[:, i] = tmp[np.invert(mask)].ravel()
            zfA2[:, i] = tmp.ravel()

        # TODO remove me
        mask1 = np.sqrt(xx**2 + yy**2) >= 1.
        assert(np.allclose(mask, mask1))
        assert(np.allclose(fringe.mask, mask1))

        if status_cb:
            status_cb('Computing phases 0.00% ...')

        def make_progress():
            prevts = [time()]

            def f(pc):
                t = time()
                dt = t - prevts[0]
                prevts[0] = t
                if dt > 1.5 or pc > 99:
                    status_cb(f'Computing phases {pc:.2f}% ...')
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

        if status_cb:
            status_cb('Computing least-squares matrices ...')
        nphi = phases.shape[1]
        uiuiT = np.zeros((nu, nu))
        phiiuiT = np.zeros((nphi, nu))
        for i in inds1:
            uiuiT += np.dot(U[:, [i]], U[:, [i]].T)
            phiiuiT += np.dot(phases[[i], :].T, U[:, [i]].T)
        if status_cb:
            status_cb('Solving least-squares ...')
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

        if status_cb:
            status_cb('Applying regularisation ...')
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
        self.cart = cart
        self.zfA1 = zfA1
        self.zfA2 = zfA2
        self.zfm = zfm
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

        self.zfA1TzfA1 = np.dot(self.zfA1.T, self.zfA1)
        self.chzfA1TzfA1 = cholesky(self.zfA1TzfA1, lower=False)

    def get_rzern(self):
        return self.cart

    def zernike_eval(self, z):
        return np.dot(self.zfA2, z).reshape(self.shape)

    def zernike_fit(self, phi):
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
        return (self.wavelength/1e-9)/(2*np.pi)

    @classmethod
    def query_calibration(cls, f):
        with File(f, 'r') as f:
            if 'WeightedLSCalib' not in f:
                raise ValueError(f.filename + ' is not a calibration file')
            else:
                return (
                    f['WeightedLSCalib/dm_serial'][()],
                    f['WeightedLSCalib/dm_transform'][()],
                    f['WeightedLSCalib/dmplot_txs'][()],
                    f['WeightedLSCalib/H'].shape,
                    )

    @classmethod
    def load_h5py(cls, f, prepend=None):
        """Load object contents from an opened HDF5 file object."""
        z = cls()

        prefix = cls.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        z.cart = RZern.load_h5py(f, prefix + 'cart/')
        z.fringe = FringeAnalysis.load_h5py(f, prefix + 'fringe/')
        z.zfA1 = f[prefix + 'zfA1'][()]
        z.zfA2 = f[prefix + 'zfA2'][()]
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

        z.zfA1TzfA1 = np.dot(z.zfA1.T, z.zfA1)
        z.chzfA1TzfA1 = cholesky(z.zfA1TzfA1, lower=False)

        return z

    def save_h5py(self, f, prepend=None, params=HDF5_options):
        """Dump object contents into an opened HDF5 file object."""
        prefix = self.__class__.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        self.cart.save_h5py(f, prefix + 'cart/', params=HDF5_options)
        self.fringe.save_h5py(f, prefix + 'fringe/', params=HDF5_options)

        params['data'] = self.zfA1
        f.create_dataset(prefix + 'zfA1', **params)
        params['data'] = self.zfA2
        f.create_dataset(prefix + 'zfA2', **params)
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
