#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from glob import glob
from os import path

import numpy as np
from matplotlib.cm import get_cmap
from PyQt5.QtWidgets import QInputDialog


def load_layout(name):
    if path.isfile(name):
        fname = name
    else:
        fname = path.join(path.dirname(__file__), 'dmlayouts', name + '.json')
    with open(fname, 'r') as f:
        d = json.load(f)
    return d


def get_layouts():
    base = path.join(path.dirname(__file__), 'dmlayouts', '*.json')
    return [path.basename(g).replace('.json', '') for g in glob(base)]


def dmplot_from_layout(name):
    d = load_layout(name)
    locations = np.array(d['locations'], dtype=float)
    loc2ind = np.array(d['loc2ind']).ravel()
    scale_shapes = d['scale_shapes']
    shapes = [np.array(s) for s in d['shapes']]
    presets = {}
    for k, v in d['presets'].items():
        presets[k] = np.array(v)
    return DMPlot(locations, loc2ind, scale_shapes, shapes, presets)


class DMPlot():
    def __init__(self,
                 locations,
                 loc2ind,
                 scale_shapes,
                 shapes,
                 presets,
                 txs=[0, 0, 0]):
        self.locations = locations
        self.loc2ind = loc2ind
        self.scale_shapes = scale_shapes
        self.shapes = shapes
        self.presets = presets

        self.ax = None

        self.T = np.eye(2)
        self.arts = []
        self.make_xys()
        self.cmap = get_cmap()
        self.txs = txs
        self.abs_cmap = 1

        if self.locations.shape[0] != self.loc2ind.size:
            raise ValueError('locations.shape[0] != loc2ind.size')
        for i, t in enumerate(self.loc2ind):
            if t < 0 or t >= len(self.shapes):
                raise ValueError(f'loc2ind[{i}] = {t} not in bounds ' +
                                 f'(0, {self.locations.shape[0]})')
        for i, s in enumerate(self.shapes):
            if s.ndim != 2:
                raise ValueError(f'shapes[{i}].ndim != 2')

    def clone(self):
        d = DMPlot(self.locations.copy(), self.loc2ind.copy(),
                   self.scale_shapes, [d.copy() for d in self.shapes])
        d.update_txs(self.txs)
        return d

    def make_xys(self):
        self.xys = []
        for i in range(self.locations.shape[0]):
            off = self.locations[i, :].reshape(1, -1)
            self.xys.append(self.scale_shapes * self.shapes[self.loc2ind[i]] +
                            off)

    def update_txs(self, txs=None):
        self.make_xys()
        if txs is None:
            txs = self.txs
        else:
            self.txs[:] = txs[:]

        alpha = txs[2]
        if abs(alpha) > 0:
            T = np.array([[np.cos(alpha), -np.sin(alpha)],
                          [np.sin(alpha), np.cos(alpha)]])
        else:
            T = np.eye(2)

        if txs[0]:
            T = np.array(([[-1, 0], [0, 1]])).dot(T)
        if txs[1]:
            T = np.array([[1, 0], [0, -1]]).dot(T)

        self.xys = [(T.dot(xy.T)).T for xy in self.xys]
        if self.ax:
            self.setup_pattern(self.ax)

    def flipx(self, b):
        self.txs[0] = b
        self.update_txs()

    def flipy(self, b):
        self.txs[1] = b
        self.update_txs()

    def rotate(self, b):
        self.txs[2] = b
        self.update_txs()

    def size(self):
        return self.locations.shape[0]

    def set_abs_cmap(self, b):
        self.abs_cmap = b

    def update_pattern(self, u):

        m1 = u.min()
        m2 = u.max()
        if self.abs_cmap or m1 == m2:
            inds = np.round(
                (len(self.cmap.colors) - 1) * (u + 1) / 2).astype(int)
        else:
            inds = u - u.min()
            inds /= inds.max()
            inds = np.round((len(self.cmap.colors) - 1) * inds).astype(int)

        np.clip(inds, 0, len(self.cmap.colors) - 1, inds)

        for i in range(len(self.arts)):
            col = self.cmap.colors[inds[i]]
            self.arts[i].set_facecolor(col)

        self.ax.figure.canvas.draw()

    def setup_pattern(self, ax):
        ax.axis('equal')
        ax.axis('off')
        for a in self.arts:
            a.remove()
            del a
        self.arts.clear()
        for xy in self.xys:
            self.arts.append(
                ax.fill(xy[:, 0],
                        xy[:, 1],
                        color=self.cmap.colors[0],
                        edgecolor=None)[0])
        self.ax = ax
        self.ax.figure.canvas.draw()

    def index_actuator(self, x, y):
        rhos = np.sqrt(
            np.sum(np.square(self.locations - np.array([x, y]).reshape(1, -1)),
                   axis=1))
        ind = rhos.argmin()
        m1 = rhos[ind]
        if m1 < self.scale_shapes:
            return ind
        else:
            return -1

    def install_select_callback(self, ax, u, parent, write=None):
        def f(e):
            if e.inaxes is not None:
                ind = self.index_actuator(e.xdata, e.ydata)
                if ind != -1:
                    val, ok = QInputDialog.getDouble(
                        parent, f'Actuator {ind}',
                        f'Actuator {ind}; Range [-1, 1]', u[ind], -1., 1., 4)
                    if ok:
                        u[ind] = val
                        self.update(u)
                        if write:
                            write(u)

        ax.figure.canvas.callbacks.connect('button_press_event', f)

    def update(self, u):
        self.update_pattern(u)

    @classmethod
    def load_h5py(cls, f, prepend=None, lazy_cart_grid=False):
        """Load object contents from an opened HDF5 file object."""

        prefix = cls.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        if prefix not in f:
            raise ValueError('No DMPlot information')

        locations = f[prefix + 'locations'][()]
        loc2ind = f[prefix + 'loc2ind'][()]
        scale_shapes = f[prefix + 'scale_shapes'][()]

        nshapes = f[prefix + 'nshapes'][()]
        shapes = []
        for i in range(nshapes):
            shapes.append(f[prefix + f'shapes/{i}'][()])
        shapes = shapes
        txs = f[prefix + 'txs'][()]

        d = {}
        for k in f[prefix + 'presets']:
            d[k] = f[prefix + 'presets/' + k][()]
        presets = d

        z = cls(locations, loc2ind, scale_shapes, shapes, presets, txs)
        return z

    def save_h5py(self, f, prepend=None, params={}):
        """Dump object contents into an opened HDF5 file object."""
        prefix = self.__class__.__name__ + '/'

        if prepend is not None:
            prefix = prepend + prefix

        params['data'] = self.locations
        f.create_dataset(prefix + 'locations', **params)

        params['data'] = self.loc2ind
        f.create_dataset(prefix + 'loc2ind', **params)

        f[prefix + 'scale_shapes'] = self.scale_shapes
        nshapes = len(self.shapes)
        f[prefix + 'nshapes'] = nshapes
        for i in range(nshapes):
            params['data'] = self.shapes[i]
            f.create_dataset(prefix + f'shapes/{i}', **params)
        f[prefix + 'txs'] = self.txs

        count = 0
        for k, v in self.presets.items():
            params['data'] = v
            f.create_dataset(prefix + f'presets/{count}', **params)
            count += 1
