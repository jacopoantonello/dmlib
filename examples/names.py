#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from zernike import RZern


def default_name(j, n, m):
    if abs(j) == 1:
        s = 'piston'
    elif abs(j) == 2:
        s = 'tip'
    elif abs(j) == 3:
        s = 'tilt'
    elif abs(j) == 4:
        s = 'defocus'
    elif m == 0:
        s = 'spherical'
    elif abs(m) == 1:
        s = 'coma'
    elif abs(m) == 2:
        s = 'astigmatism'
    elif abs(m) == 3:
        s = 'trefoil'
    elif abs(m) == 4:
        s = 'quadrafoil'
    elif abs(m) == 5:
        s = 'pentafoil'
    else:
        s = ''
    return s


def make_names(n):
    r = RZern(n)
    ntab = r.ntab
    mtab = r.mtab
    nolls = []
    names = []
    nms = []
    for i in range(r.nk):
        j = i + 1
        n = ntab[i]
        m = mtab[i]
        names.append(default_name(j, ntab[i], mtab[i]))
        nolls.append(j)
        nms.append([int(n), int(m)])
    return names, nolls, nms
