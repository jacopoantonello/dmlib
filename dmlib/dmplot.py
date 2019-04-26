#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from PyQt5.QtWidgets import QInputDialog


class DMPlot():

    def __init__(self, sampling=128, nact=12, pitch=.3, roll=2, mapmul=.3):
        self.txs = [0, 0, 0]
        self.floor = -1.5
        self.make_grids(sampling, nact, pitch, roll, mapmul)

    def update_txs(self, txs):
        self.txs[:] = txs[:]
        self.make_grids(
            self.sampling, self.nact, self.pitch, self.roll, self.mapmul)

    def flipx(self, b):
        self.txs[0] = b
        self.make_grids(
            self.sampling, self.nact, self.pitch, self.roll, self.mapmul)

    def flipy(self, b):
        self.txs[1] = b
        self.make_grids(
            self.sampling, self.nact, self.pitch, self.roll, self.mapmul)

    def rotate(self, p):
        self.txs[2] = p
        self.make_grids(
            self.sampling, self.nact, self.pitch, self.roll, self.mapmul)

    def make_grids(
            self, sampling=128, nact=12, pitch=.3, roll=2, mapmul=.3,
            txs=[0, 0, 0]):
        self.sampling = sampling
        self.nact = nact
        self.pitch = pitch
        self.roll = roll
        self.mapmul = mapmul
        self.txs = txs

        d = np.linspace(-1, 1, nact)
        d *= pitch/np.diff(d)[0]
        x, y = np.meshgrid(d, d)
        if txs[2]:
            x = np.rot90(x, txs[2])
            y = np.rot90(y, txs[2])
        if txs[0]:
            if txs[2] % 2:
                x = np.flipud(x)
            else:
                x = np.fliplr(x)
        if txs[1]:
            if txs[2] % 2:
                y = np.fliplr(y)
            else:
                y = np.flipud(y)

        dd = np.linspace(d.min() - pitch, d.max() + pitch, sampling)
        xx, yy = np.meshgrid(dd, dd)

        maps = []
        acts = []
        index = []
        exclude = [(0, 0), (0, 11), (11, 0), (11, 11)]
        count = 1
        patvis = []
        for i in range(x.shape[1]):
            for j in range(y.shape[0]):
                if (i, j) in exclude:
                    continue

                r = np.sqrt((xx - x[i, j])**2 + (yy - y[i, j])**2)
                z = np.exp(-roll*r/pitch)
                acts.append(z.reshape(-1, 1))

                mp = np.logical_and(
                    np.abs(xx - x[i, j]) < mapmul*pitch,
                    np.abs(yy - y[i, j]) < mapmul*pitch)
                maps.append(mp)
                index.append(count*mp.reshape(-1, 1))
                patvis.append(mp.reshape(-1, 1).astype(np.float))
                count += 1

        self.sampling = sampling
        self.A_shape = xx.shape
        self.A = np.hstack(acts)
        self.maps = maps
        self.layout = np.sum(np.dstack(maps), axis=2)
        self.pattern = np.hstack(index)
        self.index = np.sum(np.hstack(index), axis=1)
        self.patvis = np.hstack(patvis)
        self.mappatvis = np.invert(self.layout.astype(np.bool)).ravel()

    def size(self):
        return self.A.shape[1]

    def compute_gauss(self, u):
        pat = np.dot(self.A, u)
        return pat.reshape(self.A_shape)

    def compute_pattern(self, u):
        pat = np.dot(self.patvis, u)
        pat[self.mappatvis] = self.floor
        return pat.reshape(self.A_shape)

    def index_actuator(self, x, y):
        return self.index[int(y)*self.sampling + int(x)] - 1

    def install_select_callback(self, ax, u, parent, write=None):
        def f(e):
            if e.inaxes is not None:
                ind = self.index_actuator(e.xdata, e.ydata)
                if ind != -1:
                    val, ok = QInputDialog.getDouble(
                        parent, 'Actuator ' + str(ind), 'range [-1, 1]',
                        u[ind], -1., 1., 4)
                    if ok:
                        u[ind] = val
                        self.draw(ax, u)
                        ax.figure.canvas.draw()
                        if write:
                            write(u)

        ax.figure.canvas.callbacks.connect('button_press_event', f)

    def draw(self, ax, u):
        return ax.imshow(self.compute_pattern(u), vmin=self.floor, vmax=1)
