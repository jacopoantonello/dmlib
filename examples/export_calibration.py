#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np

import names

from numpy.random import normal
from numpy.linalg import norm
from h5py import File
"""

This script shows how to export the control matrix C computed using dmlib
for use by other applications. It generates a JSON file that can be loaded for
example in LabVIEW using Unflatten From JSON.

The JSON has the following components:
    - C; the control matrix
    - uflat; the DM flattening
    - ztest; a random vector of Zernike polynomials
    - utest0; the control variable to apply to obtain ztest
    - utest1; same as above, but including the flattening
    - rtest0; the real voltage that you must apply to the DM driver to
              obtain ztest
    - rtest1; same as above, but including the flattening
    - znames; names of the Zernike coefficients if znames = 1 below
    - nolls; Noll indices if nolls = 1 below
    - nms; m and n indices if nms = 1 below

If you are applying correctly all the conversions outlined below, your code
must send rtest0 or rtest1 to the DM driver.

z; Zernike coefficients in rad relative to the calibration wavelength;
   Use lambda/(2pi) to convert to nm;
u; [-1, 1]; control variable used by dmlib
v; [-1, 1]; linearised voltage used by dmlib
r; [rmin, rmax]; real voltage accepted by your DM driver

dmtx; linearisation function

For linear DMs, dmtx is simply v = u. For electrostatically actuated DMs the
displacement of a single actuator varies quadratically with respect to the
applied voltage. This non-linearity can be approximately removed by using the
square root of the voltage as the control variable u. So in this case dmtx is
2*sqrt((u + 1)/2) - 1.

u = uflat + C*z
v = dmtx(u)
r = rmin + (rmax - rmin)*(v + 1)/2
"""

rmin = 0  # minimum voltage accepted by your DM driver
rmax = 1  # maximum voltage accepted by your DM driver

# set to one to include the names of the Zernike polynomials, the Noll indices,
# and n m indices
znames = 0
nolls = 0
nms = 0

# Noll indices to exclude from the flattening
noll_exclude = np.array([1, 2, 3, 4])

# Usually you don't want to flatten including the piston, tip, tilt, and
# defocus as seen from the interferometer. As there will be a non-common path
# error between the interferometer and the microscope objective.

# calibration file to use
calib_file = r'./data/calib-example.h5'


def reflatten(C, z0, exclude):
    z0[exclude - 1] = 0
    return -np.dot(C, z0)


def make_test(z0, rms=1):
    zt = np.zeros_like(z0)
    nmax = min(zt.size, 6)
    u = normal(size=nmax)
    u *= rms/norm(u)
    zt[:nmax] = u
    return zt


def apply_dmtx(dmtx, u):
    d = {'u': u, 'np': np}
    exec(dmtx, d)
    return d['v']


def real_voltage(v, rmin, rmax):
    v01 = (v + 1)/2
    return rmin + (rmax - rmin)*v01


with File(calib_file, 'r') as f:
    C = f['/RegLSCalib/C'][()]                # control matrix
    z0 = f['/RegLSCalib/z0'][()]              # interferometer flat
    dmtx = f['/RegLSCalib/dm_transform'][()]  # voltage transformation
    n = int(f['/RegLSCalib/cart/RZern/n'][()])

# remove indices from noll_exclude from the flattening
uflat = reflatten(C, z0, noll_exclude)


# draw a random Zernike vector
ztest = make_test(z0)

utest0 = np.dot(C, ztest)  # without flat
utest1 = utest0 + uflat    # with flat

vtest0 = apply_dmtx(dmtx, utest0)
vtest1 = apply_dmtx(dmtx, utest1)

rtest0 = real_voltage(rmin, rmax, vtest0)
rtest1 = real_voltage(rmin, rmax, vtest1)

with open('export_calibration.json', 'w') as f:
    d = {
        'C': C.tolist(),
        'uflat': uflat.tolist(),
        'ztest': ztest.tolist(),
        'utest0': utest0.tolist(),
        'utest1': utest1.tolist(),
        'rtest0': rtest0.tolist(),
        'rtest1': rtest1.tolist(),
    }
    if znames or nolls or nms:
        t = names.make_names(n)
        if znames:
            d['znames'] = t[0]
        if nolls:
            d['nolls'] = t[1]
        if nms:
            d['nms'] = t[2]

    json.dump(d, f, sort_keys=True, indent=4)
