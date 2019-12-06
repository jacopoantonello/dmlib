#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# interf
# Copyright 2018 J. Antonello <jacopo@antonello.org>


import numpy as np
from numpy.linalg import norm
from numpy.fft import fft, ifft, fftshift
from skimage import morphology
from skimage import measure
from skimage.restoration import unwrap_phase
from scipy.signal import tukey


HDF5_options = {
    'chunks': True,
    'shuffle': True,
    'fletcher32': True,
    'compression': 'gzip',
    'compression_opts': 9}


def mgcentroid(xx, yy, img, thr=0.0):
    assert(img.dtype == np.float)

    if thr > 0.0:
        img = img.copy()
        img[img < thr] = 0

    xsum = (xx*img).sum()
    ysum = (yy*img).sum()
    mass = img.sum()

    return xsum/mass, ysum/mass


def make_cam_grid(sh, ps):
    ny, nx = sh  # grids are stored as (height, width) matrices
    hP, wP = ps  # pixel dimensions are stored as (height, width)

    yv = np.arange(-ny//2, ny//2, 1)*hP
    xv = np.arange(-nx//2, nx//2, 1)*wP
    # matplotlib's extent is (left, right, bottom, top) with origin='lower'
    ext1 = (xv.min(), xv.max(), yv.min(), yv.max())

    return xv, yv, ext1


def make_ft_grid(sh, ps, pad=2):
    ny, nx = sh
    hP, wP = ps

    fy = np.arange(-ny//2, ny//2, 1)/(pad*ny*hP)
    fx = np.arange(-nx//2, nx//2, 1)/(pad*nx*wP)
    ext2 = (fx.min(), fx.max(), fy.min(), fy.max())

    return fx, fy, ext2


def ft(f, pad=2, alpha=.25):
    ny, nx = f.shape

    if alpha > 0:
        wy = tukey(ny, alpha, True)
        wx = tukey(nx, alpha, True)
        f = f*(wy.reshape(-1, 1)*wx.reshape(1, -1))

    f = np.hstack((np.zeros((ny, nx//pad)), f, np.zeros((ny, nx//pad))))
    f = fftshift(fft(
        fftshift(f, axes=1), axis=1), axes=1)[:, (nx//pad):(nx + nx//pad)]
    f = np.vstack((np.zeros((ny//pad, nx)), f, np.zeros((ny//pad, nx))))
    f = fftshift(fft(
        fftshift(f, axes=0), axis=0), axes=0)[(ny//pad):(ny + ny//pad), :]

    return f


def ift(g, pad=2):
    ny, nx = g.shape

    g = np.hstack((np.zeros((ny, nx//pad)), g, np.zeros((ny, nx//pad))))
    g = fftshift(
        ifft(fftshift(g, axes=1), axis=1), axes=1)[:, (nx//pad):(nx + nx//pad)]
    g = np.vstack((np.zeros((ny//pad, nx)), g, np.zeros((ny//pad, nx))))
    g = fftshift(
        ifft(fftshift(g, axes=0), axis=0), axes=0)[(ny//pad):(ny + ny//pad), :]

    return g


def find_orders(fx, fy, img, mul=.9, maxcount=50):
    selem = np.ones((6, 6))
    count = 0
    while count < maxcount:
        thresh = img >= mul*img.max()
        ind1 = img.shape[0]//2 - 4
        ind2 = img.shape[0]//2 + 4
        ind3 = img.shape[1]//2 - 4
        ind4 = img.shape[1]//2 + 4
        thresh[ind1:ind2, ind3:ind4] = 1
        binop = morphology.binary_opening(thresh, selem)
        binop = morphology.binary_closing(binop, selem)
        labels = measure.label(binop)
        unilabs = np.unique(labels)
        # print(nlabs, np.unique(labels))
        if unilabs.size >= 4:
            break

        # extent = (fx.min(), fx.max(), fy.min(), fy.max())
        # plt.figure(1)
        # plt.subplot(2, 2, 1)
        # plt.imshow(img, extent=ext, origin='lower')
        # plt.subplot(2, 2, 2)
        # plt.imshow(thresh, extent=ext, origin='lower')
        # plt.subplot(2, 2, 3)
        # plt.imshow(
        #     labels, cmap='nipy_spectral', extent=ext,
        #     origin='lower')
        # plt.title(str(np.unique(labels).size))
        # plt.show()

        mul -= .05
        count += 1

    # plt.subplot(2, 2, 1)
    # plt.imshow(
    #     img, extent=ext, origin='lower')
    # plt.subplot(2, 2, 2)
    # plt.imshow(
    #     thresh, extent=ext, origin='lower')
    # plt.subplot(2, 2, 3)
    # plt.imshow(
    #     labels, cmap='nipy_spectral', extent=ext, origin='lower')
    # plt.title(str(np.unique(labels).size))
    # plt.show()

    if count >= maxcount:
        raise ValueError()
    else:
        ffx, ffy = np.meshgrid(fx, fy)
        fs = []
        norms = []

        for i in range(1, unilabs.size):
            img2 = img.copy()
            img2[labels != np.unique(labels)[i]] = 0
            fi = np.array(mgcentroid(ffx, ffy, img2))
            fs.append(fi)
            norms.append(norm(fi))

        isort = np.array(norms).argsort()
        fxc = (fs[isort[-1]][0] - fs[isort[-2]][0])/2
        fyc = (fs[isort[-1]][1] - fs[isort[-2]][1])/2

        # plt.figure(5)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img2, extent=ext, origin='lower')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img, extent=ext, origin='lower')
        # plt.plot(fxc, fyc, 'go', markersize=12)
        # plt.plot(-fxc, -fyc, 'go', markersize=12)
        # plt.show()

        return fxc, fyc


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def extract_order(fi, fx, fy, fxc, fyc, ps):
    # TODO check ext3 and the axes
    hP, wP = ps
    ny, nx = fi.shape

    iy = int(np.round(fyc*(2*ny*hP)))
    ix = int(np.round(fxc*(2*nx*wP)))
    fi = np.roll(fi, (iy, ix), axis=(0, 1))

    if ix > 0:
        ixa = nx//2 - ix
        ixb = nx//2 + ix
    else:
        ixa = nx//2 + ix
        ixb = nx//2 - ix
    if iy > 0:
        iya = ny//2 - iy
        iyb = ny//2 + iy
    else:
        iya = ny//2 + iy
        iyb = ny//2 - iy

    fi = fi[iya:iyb, ixa:ixb]
    ext3 = (fx[ixa], fx[ixb - 1], fy[iya], fy[iyb - 1])

    return fi, ext3


def repad_order(f3, fx, fy, pad=2, alpha=.25):
    dfx = np.diff(fx)[0]
    dfy = np.diff(fy)[0]
    wx = tukey(f3.shape[1], alpha, True)
    wy = tukey(f3.shape[0], alpha, True)
    w = wy.reshape(-1, 1)*wx.reshape(1, -1)

    x0 = int(nextpow2(f3.shape[1]))
    y0 = int(nextpow2(f3.shape[0]))

    offxa = (x0 - f3.shape[1])//2
    offxb = offxa + f3.shape[1]
    offya = (y0 - f3.shape[0])//2
    offyb = offya + f3.shape[0]
    f4 = np.zeros((y0, x0), dtype=np.complex)
    f4[offya:offyb, offxa:offxb] = f3*w

    yy = np.arange(-y0//2, y0//2, 1)/(pad*y0*dfy)
    xx = np.arange(-x0//2, x0//2, 1)/(pad*x0*dfx)
    ext4 = (xx.min(), xx.max(), yy.min(), yy.max())

    return f4, xx, yy, ext4


def estimate_aperture_centre(
        xv, yv, zero_mag, zero_phi, centre_mag, centre_phi, radius):

    def make_diff(a):
        d1 = np.vstack([np.zeros((1, a.shape[1])), np.diff(a, axis=0)])
        d2 = np.hstack([np.zeros((a.shape[0], 1)), np.diff(a, axis=1)])
        return d1**2 + d2**2

    def make_delta(a, b):
        cm = a.mean() - b.mean()
        k = round(cm/(2*np.pi))
        c1 = a - b - 2*k*np.pi
        c2 = c1 - c1.mean()
        return c2

    def make_rr(c, xx1, yy1, radius):
        rr = np.sqrt(xx1**2 + yy1**2) < radius
        rr1 = np.sqrt(xx1**2 + yy1**2) < .1*radius
        rr2 = np.sqrt(xx1**2 + yy1**2) > .9*radius
        s1 = c[np.logical_and(rr, rr1)].mean()
        s2 = c[np.logical_and(rr, rr2)].mean()
        if s1 < s2:
            c = -c
        return c, rr

    delta = make_delta(centre_phi, zero_phi)
    dc = make_diff(delta)

    dd = np.linspace(-1, 1, 16)
    xx1, yy1 = np.meshgrid(dd, dd)
    rr = np.sqrt(xx1**2 + yy1**2)
    selem = rr < 1
    mask0 = morphology.binary_opening(dc, selem)
    mask = morphology.convex_hull_image(mask0)

    xx, yy = np.meshgrid(xv, yv)
    cc1 = mgcentroid(xx, yy, zero_mag*mask)

    xx1, yy1 = np.meshgrid(xv - cc1[0], yv - cc1[1])
    delta2, rr = make_rr(delta, xx1, yy1, radius)

    delta2 = delta2[rr]
    delta2 -= delta2.min()
    _, e = np.histogram(delta2.ravel(), bins=10)
    cc2 = mgcentroid(xx[rr], yy[rr], delta2, thr=e[7])

    return np.array(cc2)


def call_unwrap(phase, mask=None, seed=None):
    if mask is not None:
        assert(mask.shape == phase.shape)
        masked = np.ma.masked_array(phase, mask)
        phi = np.array(unwrap_phase(masked, seed=seed))
        # phi[mask] = phi[np.invert(mask)].mean()
        phi[mask] = 0
        return phi
    else:
        return np.array(unwrap_phase(phase, seed=seed))


class FringeAnalysis:

    def __init__(self, shape, P):
        self.order_shape = None

        self.fxcfyc = None
        self.img = None
        self.ext3 = None
        self.xv = None
        self.yv = None
        self.ext4 = None
        self.gp = None
        self.unwrapped = None
        self.mask = None

        self.centre = None
        self.radius = 0.

        self.shape = shape
        self.P = P

        self.cam_grid = make_cam_grid(shape, P)
        self.ft_grid = make_ft_grid(shape, P)

    def analyse(
            self, img, auto_find_orders=False, store_logf2=False,
            store_logf3=False, store_gp=False, store_mag=False,
            store_wrapped=False, do_unwrap=True, use_mask=True,
            seed=None):

        self.img = img
        fimg = ft(img)

        if store_logf2 or auto_find_orders or self.fxcfyc is None:
            self.logf2 = np.log(np.abs(fimg))
        else:
            self.logf2 = None

        if self.fxcfyc is None or auto_find_orders:
            self.fxcfyc = find_orders(
                self.ft_grid[0], self.ft_grid[1], self.logf2)

        f3, self.ext3 = extract_order(
            fimg, self.ft_grid[0], self.ft_grid[1],
            self.fxcfyc[0], self.fxcfyc[1], self.P)
        if auto_find_orders or self.order_shape is None:
            self.order_shape = f3.shape
        else:
            assert(f3.shape == self.order_shape)

        if store_logf3:
            self.logf3 = np.log(np.abs(f3))
        else:
            self.logf3 = None

        f4, self.xv, self.yv, self.ext4 = repad_order(
            f3, self.ft_grid[0], self.ft_grid[1])

        gp = ift(f4)
        mag = np.abs(gp)
        wrapped = np.arctan2(gp.imag, gp.real)

        if store_gp:
            self.gp = gp
        else:
            self.gp = None

        if store_mag:
            self.mag = mag
        else:
            self.mag = None

        if store_wrapped:
            self.wrapped = wrapped
        else:
            self.wrapped = None

        if do_unwrap:
            if not use_mask or self.mask is None:
                _, edges = np.histogram(mag.ravel(), bins=100)
                mask = (mag < edges[1]).reshape(mag.shape)
            else:
                mask = self.mask
            self.unwrapped = call_unwrap(wrapped, mask, seed=seed)

    @classmethod
    def load_h5py(cls, f, prepend=None):
        """Load object contents from an opened HDF5 file object."""

        prefix = cls.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        shape = f[prefix + 'shape'][()]
        P = f[prefix + 'P'][()]
        z = cls(shape, P)
        z.cam_grid = (
            f[prefix + 'cam_grid0'][()],
            f[prefix + 'cam_grid1'][()],
            f[prefix + 'cam_grid2'][()])
        z.ft_grid = (
            f[prefix + 'ft_grid0'][()],
            f[prefix + 'ft_grid1'][()],
            f[prefix + 'ft_grid2'][()])

        if prefix + 'fxcfyc' in f:
            z.fxcfyc = f[prefix + 'fxcfyc'][()]
        if prefix + 'img' in f:
            z.img = f[prefix + 'img'][()]
        if prefix + 'ext3' in f:
            z.ext3 = f[prefix + 'ext3'][()]
        if prefix + 'xv' in f:
            z.xv = f[prefix + 'xv'][()]
        if prefix + 'yv' in f:
            z.yv = f[prefix + 'yv'][()]
        if prefix + 'ext4' in f:
            z.ext4 = f[prefix + 'ext4'][()]
        if prefix + 'gp' in f:
            z.gp = f[prefix + 'gp'][()]
        if prefix + 'unwrapped' in f:
            z.unwrapped = f[prefix + 'unwrapped'][()]

        if prefix + 'centre' in f and prefix + 'radius' in f:
            z.centre = f[prefix + 'centre'][()]
            z.radius = f[prefix + 'radius'][()]
            z._make_mask()

        return z

    def save_h5py(self, f, prepend=None, params=HDF5_options):
        """Dump object contents into an opened HDF5 file object."""
        prefix = self.__class__.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        params['data'] = self.shape
        f.create_dataset(prefix + 'shape', **params)
        params['data'] = self.P
        f.create_dataset(prefix + 'P', **params)

        params['data'] = self.cam_grid[0]
        f.create_dataset(prefix + 'cam_grid0', **params)
        params['data'] = self.cam_grid[1]
        f.create_dataset(prefix + 'cam_grid1', **params)
        params['data'] = self.cam_grid[2]
        f.create_dataset(prefix + 'cam_grid2', **params)

        params['data'] = self.ft_grid[0]
        f.create_dataset(prefix + 'ft_grid0', **params)
        params['data'] = self.ft_grid[1]
        f.create_dataset(prefix + 'ft_grid1', **params)
        params['data'] = self.ft_grid[2]
        f.create_dataset(prefix + 'ft_grid2', **params)

        if self.fxcfyc is not None:
            params['data'] = self.fxcfyc
            f.create_dataset(prefix + 'fxcfyc', **params)
        if self.img is not None:
            params['data'] = self.img
            f.create_dataset(prefix + 'img', **params)
        if self.ext3 is not None:
            params['data'] = self.ext3
            f.create_dataset(prefix + 'ext3', **params)
        if self.xv is not None:
            params['data'] = self.xv
            f.create_dataset(prefix + 'xv', **params)
        if self.yv is not None:
            params['data'] = self.yv
            f.create_dataset(prefix + 'yv', **params)
        if self.ext4 is not None:
            params['data'] = self.ext4
            f.create_dataset(prefix + 'ext4', **params)
        if self.gp is not None:
            params['data'] = self.gp
            f.create_dataset(prefix + 'gp', **params)
        if self.unwrapped is not None:
            params['data'] = self.unwrapped
            f.create_dataset(prefix + 'unwrapped', **params)

        if self.centre is not None:
            f[prefix + 'centre'] = self.centre
        if self.radius is not None:
            f[prefix + 'radius'] = self.radius

    def get_unit_aperture(self):
        xx, yy = np.meshgrid(
            (self.xv - self.centre[0])/self.radius,
            (self.yv - self.centre[1])/self.radius)
        assert(xx.shape[1] == self.xv.size)
        assert(xx.shape[0] == self.yv.size)
        assert(yy.shape[1] == self.xv.size)
        assert(yy.shape[0] == self.yv.size)
        return xx, yy, (self.yv.size, self.xv.size)

    def _make_mask(self):
        if self.radius > 0.:
            xx, yy = np.meshgrid(
                self.xv - self.centre[0],
                self.yv - self.centre[1])
            self.xx = xx
            self.yy = yy
            self.mask = np.sqrt(xx**2 + yy**2) >= self.radius
        else:
            self.mask = None
            self.radius = 0.

    def clear_aperture(self):
        self.centre = None
        self.mask = None
        self.radius = 0.

    def estimate_aperture(self, img_zero, img_centre, radius):
        "radius [um]"
        if radius <= 0.:
            self.clear_aperture()
        else:
            self.analyse(
                img_zero, auto_find_orders=False, do_unwrap=True,
                use_mask=False, store_mag=True)
            mag_zero = self.mag
            phi_zero = self.unwrapped
            self.analyse(
                img_centre, auto_find_orders=False, do_unwrap=True,
                use_mask=False, store_mag=True)
            mag_centre = self.mag
            phi_centre = self.unwrapped

            centre = estimate_aperture_centre(
                self.xv, self.yv, mag_zero, phi_zero, mag_centre,
                phi_centre, radius)

            self.set_aperture(centre, radius)

    def set_aperture(self, centre, radius):
        "[um]"
        self.centre = centre
        self.radius = radius
        self._make_mask()
