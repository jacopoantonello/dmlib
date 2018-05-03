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


def mgcentroid(xx, yy, img, mythr=0.0):
    assert(img.dtype == np.float)

    mysum1 = np.sum((xx*img).ravel())
    mysum2 = np.sum((yy*img).ravel())
    mymass = np.sum(img.ravel())
    return mysum1/mymass, mysum2/mymass


def make_cam_grid(sh, ps):
    mm, nn = sh
    wP, hP = ps
    dd0 = np.arange(-mm//2, mm//2, 1)*wP
    dd1 = np.arange(-nn//2, nn//2, 1)*hP
    ext1 = (dd1.min(), dd1.max(), dd0.min(), dd0.max())
    return dd0, dd1, ext1


def make_ft_grid(sh, ps, pad=2):
    mm, nn = sh
    wP, hP = ps
    ff0 = np.arange(-mm//2, mm//2, 1)/(pad*mm*wP)
    ff1 = np.arange(-nn//2, nn//2, 1)/(pad*nn*hP)
    ext2 = (ff1.min(), ff1.max(), ff0.min(), ff0.max())
    return ff0, ff1, ext2


def ft(x, pad=2, alpha=.25):
    mm, nn = x.shape

    if alpha > 0:
        w0 = tukey(x.shape[0], alpha, True)
        w1 = tukey(x.shape[1], alpha, True)
        x = x*(w0.reshape(-1, 1)*w1.reshape(1, -1))

    x = np.hstack((np.zeros((mm, nn//pad)), x, np.zeros((mm, nn//pad))))
    x = fftshift(fft(
        fftshift(x, axes=1), axis=1), axes=1)[:, (nn//pad):(nn + nn//pad)]
    x = np.vstack((np.zeros((mm//pad, nn)), x, np.zeros((mm//pad, nn))))
    x = fftshift(fft(
        fftshift(x, axes=0), axis=0), axes=0)[(mm//pad):(mm + mm//pad), :]
    return x


def ift(y, pad=2):
    l0, l1 = y.shape
    y = np.hstack((np.zeros((l0, l1//pad)), y, np.zeros((l0, l1//pad))))
    y = fftshift(
        ifft(fftshift(y, axes=1), axis=1), axes=1)[:, (l1//pad):(l1 + l1//pad)]
    y = np.vstack((np.zeros((l0//pad, l1)), y, np.zeros((l0//pad, l1))))
    y = fftshift(
        ifft(fftshift(y, axes=0), axis=0), axes=0)[(l0//pad):(l0 + l0//pad), :]
    return y


def find_orders(ff0, ff1, img, mul=.9, maxcount=50):
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

        # plt.subplot(2, 2, 1)
        # plt.imshow(img)
        # plt.subplot(2, 2, 2)
        # plt.imshow(thresh)
        # plt.subplot(2, 2, 3)
        # plt.imshow(labels, cmap='nipy_spectral')
        # plt.title(str(np.unique(labels).size))
        # plt.show()

        mul -= .05
        count += 1

    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.subplot(2, 2, 2)
    # plt.imshow(thresh)
    # plt.subplot(2, 2, 3)
    # plt.imshow(labels, cmap='nipy_spectral')
    # plt.title(str(np.unique(labels).size))
    # plt.show()

    if count >= maxcount:
        raise ValueError()
    else:
        xx, yy = np.meshgrid(ff1, ff0)
        fs = []
        norms = []

        for i in range(1, unilabs.size):
            img2 = img.copy()
            img2[labels != np.unique(labels)[i]] = 0
            fi = np.array(mgcentroid(xx, yy, img2))
            fs.append(fi)
            norms.append(norm(fi))

        isort = np.array(norms).argsort()
        f0 = (fs[isort[-1]][0] - fs[isort[-2]][0])/2
        f1 = (fs[isort[-1]][1] - fs[isort[-2]][1])/2

        # plt.figure(5)
        # plt.subplot(1, 2, 1)
        # plt.imshow(img2, extent=ext2, origin='lower')
        # plt.subplot(1, 2, 2)
        # plt.imshow(img, extent=ext2, origin='lower')
        # plt.plot(f0, f1, 'go', markersize=12)
        # plt.plot(-f0, -f1, 'go', markersize=12)
        # plt.show()

        return f0, f1


def nextpow2(n):
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def extract_order(fi, ff0, ff1, f0, f1, ps):
    # TODO check ext3 and the axes
    wP, hP = ps
    mm, nn = fi.shape

    i1 = int(np.round(f0*(2*nn*hP)))
    i0 = int(np.round(f1*(2*mm*wP)))
    fi = np.roll(fi, (i0, i1), axis=(0, 1))

    l0 = int(i0)
    l1 = int(i1)

    if l0 > 0:
        i0a = mm//2 - l0
        i0b = mm//2 + l0
    else:
        i0a = mm//2 + l0
        i0b = mm//2 - l0
    if l1 > 0:
        i1a = nn//2 - l1
        i1b = nn//2 + l1
    else:
        i1a = nn//2 + l1
        i1b = nn//2 - l1

    fi = fi[i0a:i0b, i1a:i1b]
    ext3 = (ff1[i1a], ff1[i1b], ff0[i0a], ff0[i0b])

    return fi, ext3


def repad_order(f3, ff0, ff1, pad=2, alpha=.25):
    dff0 = np.diff(ff0)[0]
    dff1 = np.diff(ff1)[0]
    w1 = tukey(f3.shape[0], alpha, True)
    w2 = tukey(f3.shape[1], alpha, True)
    w = w1.reshape(-1, 1)*w2.reshape(1, -1)

    mm0 = nextpow2(f3.shape[0])
    nn0 = nextpow2(f3.shape[1])
    l0 = int(mm0)
    l1 = int(nn0)
    off1 = (l0 - f3.shape[0])//2
    off2 = off1 + f3.shape[0]
    off3 = (l1 - f3.shape[1])//2
    off4 = off3 + f3.shape[1]
    f4 = np.zeros(((l0, l1)), dtype=np.complex)
    f4[off1:off2, off3:off4] = f3*w

    dd0 = np.arange(-mm0//2, mm0//2, 1)/(pad*mm0*dff0)
    dd1 = np.arange(-nn0//2, nn0//2, 1)/(pad*nn0*dff1)
    ext4 = (dd1.min(), dd1.max(), dd0.min(), dd0.max())

    return f4, dd0, dd1, ext4


def estimate_aperture_centre(
        dd0, dd1, zero_mag, zero_phi, centre_mag, centre_phi):
    _, e0 = np.histogram(zero_mag, bins=10)
    mask0 = zero_mag > e0[1]
    _, ec = np.histogram(centre_mag, bins=10)
    maskc = centre_mag > ec[1]
    mask = mask0*maskc

    dd = np.linspace(-1, 1, 64)
    selem = np.ones((dd.size, dd.size))
    [xx, yy] = np.meshgrid(dd, dd)
    rr = np.sqrt(xx**2 + yy**2)
    selem = rr < 1
    mask = morphology.binary_erosion(mask, selem)
    mask = morphology.binary_opening(mask, selem)

    phi0 = zero_phi*mask
    phic = centre_phi*mask

    delta = phi0 - phic
    delta[mask] = delta[mask] - delta[mask].mean()
    delta[np.invert(mask)] = 0
    delta = np.abs(delta)
    _, e2 = np.histogram(delta, bins=10)
    delta[delta < e2[3]] = 0

    xx, yy = np.meshgrid(dd1, dd0)
    cross = np.array(mgcentroid(xx, yy, delta))
    return cross


def call_unwrap(phase, mask=None):
    if mask is not None:
        masked = np.ma.masked_array(phase, mask)
        phi = np.array(unwrap_phase(masked))
        # phi[mask] = phi[np.invert(mask)].mean()
        phi[mask] = 0
        return phi
    else:
        return np.array(unwrap_phase(phase))


class FringeAnalysis:

    f0f1 = None
    img = None
    ext3 = None
    dd0 = None
    dd1 = None
    ext4 = None
    gp = None
    unwrapped = None

    cross = None
    radius = None

    def __init__(self, shape, P):
        self.shape = shape
        self.P = P

        self.cam_grid = make_cam_grid(shape, P)
        self.ft_grid = make_ft_grid(shape, P)

    def analyse(
            self, img, auto_find_orders=False, store_logf2=False,
            store_logf3=False, store_gp=False, store_mag=False,
            store_wrapped=False, do_unwrap=True, mask=None):

        fimg = ft(img)
        self.img = img

        if store_logf2 or auto_find_orders:
            self.logf2 = np.log(np.abs(fimg))
        else:
            self.logf2 = None

        if find_orders or self.f0f1 is None:
            self.f0f1 = find_orders(
                self.ft_grid[0], self.ft_grid[1], self.logf2)

        f3, self.ext3 = extract_order(
            fimg, self.ft_grid[0], self.ft_grid[1], self.f0f1[0],
            self.f0f1[0], self.P)

        if store_logf3:
            self.logf3 = np.log(np.abs(f3))
        else:
            self.logf3 = None

        f4, self.dd0, self.dd1, self.ext4 = repad_order(
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
            if mask is None:
                _, edges = np.histogram(mag.ravel(), bins=100)
                mask = (mag < edges[1]).reshape(mag.shape)

            self.unwrapped = call_unwrap(wrapped, mask)

    @classmethod
    def load_h5py(cls, f, prepend=None):
        """Load object contents from an opened HDF5 file object."""
        z = cls(1)

        prefix = cls.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        z.shape = f[prefix + 'shape'][()]
        z.P = f[prefix + 'P'][()]
        z.cam_grid = (
            f[prefix + 'cam_grid0'][()],
            f[prefix + 'cam_grid1'][()],
            f[prefix + 'cam_grid2'][()])
        z.ft_grid = (
            f[prefix + 'ft_grid0'][()],
            f[prefix + 'ft_grid1'][()],
            f[prefix + 'ft_grid2'][()])

        if prefix + 'f0f1' in f:
            z.f0f1 = f[prefix + 'f0f1'][()]
        if prefix + 'img' in f:
            z.img = f[prefix + 'img'][()]
        if prefix + 'ext3' in f:
            z.ext3 = f[prefix + 'ext3'][()]
        if prefix + 'dd0' in f:
            z.dd0 = f[prefix + 'dd0'][()]
        if prefix + 'dd1' in f:
            z.dd1 = f[prefix + 'dd1'][()]
        if prefix + 'ext4' in f:
            z.ext4 = f[prefix + 'ext4'][()]
        if prefix + 'gp' in f:
            z.gp = f[prefix + 'gp'][()]
        if prefix + 'unwrapped' in f:
            z.unwrapped = f[prefix + 'unwrapped'][()]

        if prefix + 'cross' in f:
            z.cross = f[prefix + 'cross'][()]
        if prefix + 'radius' in f:
            z.radius = f[prefix + 'radius'][()]

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

        if self.f0f1:
            f.create_dataset(prefix + 'f0f1', **params)
        if self.img:
            f.create_dataset(prefix + 'img', **params)
        if self.ext3:
            f.create_dataset(prefix + 'ext3', **params)
        if self.dd0:
            f.create_dataset(prefix + 'dd0', **params)
        if self.dd1:
            f.create_dataset(prefix + 'dd1', **params)
        if self.ext4:
            f.create_dataset(prefix + 'ext4', **params)
        if self.gp:
            f.create_dataset(prefix + 'gp', **params)
        if self.unwrapped:
            f.create_dataset(prefix + 'unwrapped', **params)

        if self.cross:
            f.create_dataset(prefix + 'cross', **params)
        if self.radius:
            f.create_dataset(prefix + 'radius', **params)
