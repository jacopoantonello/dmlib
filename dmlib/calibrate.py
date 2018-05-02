#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.interpolate import interp1d
from multiprocessing import Pool
from numpy.linalg import lstsq, pinv
from scipy.linalg import cholesky, solve_triangular
from skimage.restoration import unwrap_phase

from interf import ft, ift, repad_order, extract_order, call_unwrap
from sensorless.czernike import RZern


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

    def __init__(self, ft_grid, f0, f1, P, mask):
        self.ft_grid = ft_grid
        self.f0 = f0
        self.f1 = f1
        self.P = P
        self.mask = mask

    def __call__(self, img):
        fimg = ft(img)
        f3, ext3 = extract_order(
            fimg, self.ft_grid[0], self.ft_grid[1],
            self.f0, self.f1, self.P)
        f4, _, _, _ = repad_order(
            f3, self.ft_grid[0], self.ft_grid[1])
        gp = ift(f4)
        wrapped = np.arctan2(gp.imag, gp.real)
        unwrapped = call_unwrap(wrapped, self.mask)
        return unwrapped[np.invert(self.mask)]


def calibrate(
        ft_grid, f0, f1, P, dd0, dd1, cross, radius, U, images, n_radial=25,
        alpha=.75, lambda1=5e-3):
    nu, ns = U.shape
    dd0 = dd0 - cross[1]
    dd1 = dd1 - cross[0]
    dd0 = dd0/radius
    dd1 = dd1/radius
    K = dd0.size
    L = dd1.size
    xx, yy = np.meshgrid(dd1, dd0)
    cart = RZern(n_radial)
    cart.make_cart_grid(xx, yy)
    zfm = np.isfinite(cart.ZZ[:, 0]).reshape((L, K), order='F')
    mask = np.invert(zfm)
    mask1 = np.sqrt(xx**2 + yy**2) >= 1.
    zfA1 = np.zeros((zfm.sum(), cart.nk))
    zfA2 = np.zeros_like(cart.ZZ)
    for i in range(zfA1.shape[1]):
        tmp = cart.ZZ[:, i].reshape((L, K), order='F')
        zfA1[:, i] = tmp[np.invert(mask)].ravel()
        zfA2[:, i] = tmp.ravel()
    assert(np.allclose(mask, mask1))

    with Pool() as p:
        phases = np.array(p.map(PhaseExtract(
            ft_grid, f0, f1, P, mask), [images[i, ...] for i in range(ns)]))

    inds0 = fix_principal_val(U, phases)
    inds1 = np.setdiff1d(np.arange(ns), inds0)
    assert(np.allclose(np.arange(ns), np.sort(np.hstack((inds0, inds1)))))
    phi0 = phases[inds0, :].mean(axis=0)
    z0 = lstsq(np.dot(zfA1.T, zfA1), np.dot(zfA1.T, phi0), rcond=None)[0]
    phases -= phi0.reshape(1, -1)

    nphi = phases.shape[1]
    uiuiT = np.zeros((nu, nu))
    phiiuiT = np.zeros((nphi, nu))
    for i in inds1:
        uiuiT += np.dot(U[:, [i]], U[:, [i]].T)
        phiiuiT += np.dot(phases[[i], :].T, U[:, [i]].T)
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

    return H, mvaf, phi0, z0, C
