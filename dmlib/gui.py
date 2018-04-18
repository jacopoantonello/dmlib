#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time

from multiprocessing import Process, Queue, Array, Value
from datetime import datetime, timezone
from numpy.linalg import norm
from numpy.random import uniform

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import (  # noqa: F401
    QImage, QPainter, QDoubleValidator, QIntValidator, QKeySequence,
    )
from PyQt5.QtWidgets import (  # noqa: F401
    QMainWindow, QDialog, QTabWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QCheckBox, QVBoxLayout, QFrame,
    QApplication, QShortcut, QSlider, QDoubleSpinBox, QToolBox,
    QWidget, QFileDialog, QScrollArea, QMessageBox, QSplitter,
    QInputDialog, QStyleFactory, QSizePolicy
    )

from interf import (
    make_cam_grid, make_ft_grid, ft, ift, find_orders, repad_order,
    extract_order, call_unwrap)


class Control(QMainWindow):

    closing = False

    def __init__(self, worker, shared, settings={}, parent=None):
        super().__init__()

        self.worker = worker
        self.shared = shared
        self.shared.make_static()

        self.setWindowTitle('DM control')
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        central = QSplitter(Qt.Horizontal)

        self.toolbox = QToolBox()
        self.make_toolbox()
        central.addWidget(self.toolbox)

        self.tabs = QTabWidget()
        self.make_panel_align()
        self.make_panel_dataacq()
        central.addWidget(self.tabs)

        self.setCentralWidget(central)

    def make_toolbox(self):
        self.make_tool_cam()
        self.make_tool_dm()

    def make_tool_cam(self):
        tool_cam = QFrame()
        layout = QVBoxLayout()
        tool_cam.setLayout(layout)

        def cam_get(cmd):
            def f():
                self.shared.iq.put((cmd,))
                return self.shared.oq.get()
            return f

        def cam_set(cmd):
            def f(x):
                self.shared.iq.put((cmd, x))
                return self.shared.oq.get()
            return f

        def up(l, s, txt, r, v):
            rg = r()
            l.setText('min: {}<br>max: {}<br>step: {}'.format(
                rg[0], rg[1], rg[2]))
            s.setRange(rg[0], rg[1])
            s.setSingleStep(rg[2])
            s.blockSignals(True)
            s.setValue(v())
            s.blockSignals(False)

        g1 = QGroupBox('Exposure [ms]')
        gl1 = QVBoxLayout()
        l1 = QLabel()
        s1 = QDoubleSpinBox()
        s1.setDecimals(6)
        up(
            l1, s1, 'Exposure [ms]',
            cam_get('get_exposure_range'),
            cam_get('get_exposure'))
        gl1.addWidget(l1)
        gl1.addWidget(s1)
        g1.setLayout(gl1)
        layout.addWidget(g1)

        g2 = QGroupBox('FPS')
        gl2 = QGridLayout()
        l2 = QLabel()
        s2 = QDoubleSpinBox()
        s2.setDecimals(6)
        up(
            l2, s2, 'FPS',
            cam_get('get_framerate_range'),
            cam_get('get_framerate'))
        gl2.addWidget(l2)
        gl2.addWidget(s2)
        g2.setLayout(gl2)
        layout.addWidget(g2)

        def f1(fun, sa, lb, sb, txtb, rb, gb):
            def f():
                x = sa.value()
                x = fun(x)
                sa.setValue(x)
                sa.blockSignals(True)
                sa.setValue(x)
                sa.blockSignals(False)

                up(lb, sb, txtb, rb, gb)

            return f

        s1.editingFinished.connect(f1(
            cam_set('set_exposure'), s1, l2, s2, 'FPS',
            cam_get('get_framerate_range'),
            cam_get('get_framerate')))
        s2.editingFinished.connect(f1(
            cam_set('set_framerate'), s2, l1, s1, 'exposure',
            cam_get('get_exposure_range'),
            cam_get('get_exposure')))

        self.toolbox.addItem(tool_cam, 'camera')

    def write_dm(self, u=None):
        if u is not None:
            self.shared.u[:] = u[:]
        self.dmplot.draw(self.dm_ax, self.shared.u)
        self.dm_ax.figure.canvas.draw()
        self.shared.iq.put(('write',))
        self.shared.oq.get()

    def make_tool_dm(self):
        tool_dm = QFrame()
        layout = QGridLayout()
        tool_dm.setLayout(layout)

        self.dm_fig = FigureCanvas(Figure(figsize=(3, 2)))
        self.dm_ax = self.dm_fig.figure.add_subplot(1, 1, 1)
        layout.addWidget(self.dm_fig, 0, 0, 1, 0)
        self.dmplot = DMPlot()
        self.dmplot.install_select_callback(
            self.dm_ax, self.shared.u, self, self.write_dm)
        self.dm_fig.figure.subplots_adjust(
            left=.125, right=.9,
            bottom=.1, top=.9,
            wspace=0.45, hspace=0.45)

        reset = QPushButton('reset')
        layout.addWidget(reset, 1, 0)
        flipx = QPushButton('flipx')
        layout.addWidget(flipx, 2, 0)
        flipy = QPushButton('flipy')
        layout.addWidget(flipy, 2, 1)
        rotate1 = QPushButton('rotate cw')
        layout.addWidget(rotate1, 3, 0)
        rotate2 = QPushButton('rotate acw')
        layout.addWidget(rotate2, 3, 1)

        def f4(n):
            def f(p):
                if p:
                    d = .8
                else:
                    d = .0
                self.shared.iq.put(('preset', n, d))
                self.shared.oq.get()
                self.write_dm(None)
            return f

        i = 4
        j = 0
        for name in ('centre', 'cross', 'x', 'rim', 'checker'):
            b = QPushButton(name)
            b.setCheckable(True)
            layout.addWidget(b, i, j)
            if j == 1:
                i += 1
                j = 0
            else:
                j += 1
            b.clicked[bool].connect(f4(name))

        def f2():
            def f():
                self.shared.u[:] = 0
                self.write_update_dm()
            return f

        reset.clicked.connect(f2())

        def f3(sign):
            ind = [0]

            def f():
                ind[0] += sign
                ind[0] %= 4
                self.dmplot.rotate(ind[0])
                self.dm_ax.figure.canvas.draw()
            return f

        flipx.setCheckable(True)
        flipy.setCheckable(True)
        flipx.clicked.connect(self.dmplot.flipx)
        flipy.clicked.connect(self.dmplot.flipy)
        rotate1.clicked.connect(f3(1))
        rotate2.clicked.connect(f3(-1))

        self.dm_ax.axis('off')
        self.write_dm(None)

        self.toolbox.addItem(tool_dm, 'dm')

    def make_panel_align(self):
        frame = QFrame()
        self.align_fig = FigureCanvas(Figure(figsize=(7, 5)))
        layout = QGridLayout()
        frame.setLayout(layout)
        layout.addWidget(self.align_fig, 0, 0, 1, 0)

        self.tabs.addTab(frame, 'align')

        self.align_axes = self.align_fig.figure.subplots(2, 3)
        self.align_fig.figure.subplots_adjust(
            left=.125, right=.9,
            bottom=.1, top=.9,
            wspace=0.45, hspace=0.45)
        self.align_axes[0, 0].set_title('camera')
        self.align_axes[0, 1].set_title('FT')
        self.align_axes[0, 2].set_title('1st order')
        self.align_axes[1, 0].set_title('magnitude')
        self.align_axes[1, 1].set_title('wrapped phi')
        self.align_axes[1, 2].set_title('unwrapped phi')

        brun = QPushButton('run')
        bstop = QPushButton('stop')
        layout.addWidget(brun, 1, 0)
        layout.addWidget(bstop, 1, 1)
        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 2, 0, 1, 2)

        bauto = QCheckBox('auto')
        bauto.setChecked(True)
        layout.addWidget(bauto, 3, 0)
        brepeat = QCheckBox('repeat')
        layout.addWidget(brepeat, 3, 1)

        listener = Listener(self.shared)

        bpoke = QCheckBox('poke')
        bsleep = QPushButton('sleep')
        layout.addWidget(bpoke, 4, 0)
        layout.addWidget(bsleep, 4, 1)
        bunwrap = QCheckBox('unwrap')
        bunwrap.setChecked(True)
        layout.addWidget(bunwrap, 5, 0)

        def f1():
            def f():
                val, ok = QInputDialog.getDouble(
                    self, '', 'sleep [s]', listener.sleep, decimals=4)
                if ok:
                    listener.sleep = val
            return f

        def f2():
            def f(p):
                listener.poke = p
            return f

        def f3():
            def f(p):
                listener.unwrap = p
            return f

        bsleep.clicked.connect(f1())
        bpoke.stateChanged.connect(f2())
        bunwrap.stateChanged.connect(f3())

        def f1():
            def f():
                status.setText('working...')
                brun.setEnabled(False)
                bauto.setEnabled(False)
                brepeat.setEnabled(False)
                bpoke.setEnabled(False)
                bsleep.setEnabled(False)
                bunwrap.setEnabled(False)
                self.align_nav.setEnabled(False)
                listener.start()
            return f

        def f2():
            def f():
                listener.auto = not listener.auto
            return f

        def f3():
            def f(p):
                listener.repeat = p
            return f

        def f20():
            def f(msg):
                print(msg)
                a1 = self.align_axes[0, 0]
                a2 = self.align_axes[0, 1]
                a3 = self.align_axes[0, 2]
                a4 = self.align_axes[1, 0]
                a5 = self.align_axes[1, 1]
                a6 = self.align_axes[1, 2]

                a1.clear()
                a2.clear()
                a3.clear()
                a4.clear()
                a5.clear()
                a6.clear()

                a1.imshow(
                    self.shared.cam, extent=self.shared.cam_ext,
                    origin='lower')
                a1.set_xlabel('mm')
                if self.shared.cam_sat:
                    a1.set_title('cam SAT')
                else:
                    a1.set_title('cam {: 3d} {: 3d}'.format(
                        self.shared.cam.min(), self.shared.cam.max()))
                
                a2.imshow(
                    self.shared.ft, extent=self.shared.ft_ext,
                    origin='lower')
                a2.set_xlabel('1/mm')
                a2.set_title('FT')

                if msg != 'OK':
                    status.setText(x)
                    return

                a2.plot(
                    self.shared.f0f1[0]*1e3, self.shared.f0f1[1]*1e3,
                    'rx', markersize=6)

                a2.plot(
                    -self.shared.f0f1[0]*1e3, -self.shared.f0f1[1]*1e3,
                    'rx', markersize=6)

                fstord, mag, wrapped, unwrapped = self.shared.get_phase()

                a3.imshow(
                    fstord, extent=self.shared.fstord_ext, origin='lower')
                a3.set_xlabel('1/mm')
                a3.set_title('1st order')

                a4.imshow(
                    mag, extent=self.shared.mag_ext, origin='lower')
                a4.set_xlabel('mm')
                a4.set_title('magnitude')

                a5.imshow(
                    wrapped, extent=self.shared.mag_ext, origin='lower')
                a5.set_xlabel('mm')
                a5.set_title('wrapped phi')

                a6.imshow(
                    unwrapped, extent=self.shared.mag_ext, origin='lower')
                a6.set_xlabel('mm')
                a6.set_title('unwrapped phi')

                a6.figure.canvas.draw()

                status.setText('')
                brun.setEnabled(True)
                bauto.setEnabled(True)
                brepeat.setEnabled(True)
                bpoke.setEnabled(True)
                bsleep.setEnabled(True)
                bunwrap.setEnabled(True)
                self.align_nav.setEnabled(True)
            return f

        listener.sig_update.connect(f20())
        brun.clicked.connect(f1())
        bauto.stateChanged.connect(f2())
        brepeat.stateChanged.connect(f3())
        self.align_nav = NavigationToolbar2QT(self.align_fig, frame)

    def make_panel_dataacq(self):
        frame = QFrame()
        self.da_fig = FigureCanvas(Figure(figsize=(7, 5)))
        layout = QGridLayout()
        frame.setLayout(layout)
        layout.addWidget(self.da_fig, 0, 0, 1, 0)

        self.tabs.addTab(frame, 'calibration')

        self.da_axes = self.da_fig.figure.subplots(2, 2)
        self.da_fig.figure.subplots_adjust(
            left=.125, right=.9,
            bottom=.1, top=.9,
            wspace=0.45, hspace=0.45)
        self.da_axes[0, 0].set_title('camera')
        self.da_axes[0, 1].set_title('mag')
        self.da_axes[1, 0].set_title('wrapped phi')
        self.da_axes[1, 1].set_title('unwrapped phi')

        brun = QPushButton('run')
        bstop = QPushButton('stop')
        layout.addWidget(brun, 1, 0)
        layout.addWidget(bstop, 1, 1)
        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 2, 0, 1, 2)

        return
        dataacq = DataAcq(self.dm, self.cam, self.dmplot)

        def f1():
            def f():
                status.setText('working...')
                brun.setEnabled(False)
                dataacq.start()
            return f

        def f2():
            def f():
                dataacq.stop = True
            return f

        def f3():
            a = self.da_axes[0, 0]
            def f(t):
                print(t[2:])
                mul = 1e-3
                lab = 'mm'
                ext = (t[1][0]*mul, t[1][1]*mul, t[1][2]*mul, t[1][3]*mul)
                a.imshow(t[0], extent=ext, origin='lower')
                a.set_xlabel(lab)
                if t[0].max() == self.cam.get_image_max():
                    a.set_title('cam SAT')
                else:
                    a.set_title('cam {: 3d} {: 3d}'.format(
                        t[0].min(), t[0].max()))
                a.figure.canvas.draw()
            return f

        brun.clicked.connect(f1())
        bstop.clicked.connect(f2())
        dataacq.sig_cam.connect(f3())


class FakeCamera():
    exp = 0.06675 + 5*0.06675
    fps = 4

    def grab_image(self):
        img = plt.imread('../test/data/int3.tif')
        img = np.roll(img, int(np.round(uniform(-200, 200))), axis=0)
        img = np.roll(img, int(np.round(uniform(-200, 200))), axis=1)
        return img

    def shape(self):
        return (1024, 1280)

    def get_pixel_size(self):
        return (5.20, 5.20)

    def get_exposure(self):
        return self.exp

    def get_exposure_range(self):
        return (0.06675, 99.92475, 0.06675)

    def set_exposure(self, e):
        self.exp = e
        return e

    def get_framerate_range(self):
        return (4, 13, 4)

    def get_framerate(self):
        return self.fps

    def set_framerate(self, f):
        self.frate = f
        return f

    def get_serial_number(self):
        return 'cam0'

    def get_camera_info(self):
        return 'camera info'

    def get_sensor_info(self):
        return 'sensor info'

    def get_image_dtype(self):
        return 'uint8'

    def get_image_max(self):
        return 0xff

    def close(self):
        pass


class FakeDM():

    def size(self):
        return 140

    def write(self, v):
        print('FakeDM', v)

    def preset(self, name):
        return uniform(-1., 1., size=(140,))

    def get_transform(self):
        return None

    def get_serial_number(self):
        return 'dm0'

    def close(self):
        pass

    def preset(self, name, mag=0.7):
        u = np.zeros((140,))
        if name == 'centre':
            u[63:65] = mag
            u[75:77] = mag
        elif name == 'cross':
            u[58:82] = mag
            u[4:6] = mag
            u[134:136] = mag
            for i in range(10):
                off = 15 + 12*i
                u[off:(off + 2)] = mag
        elif name == 'x':
            inds = np.array([
                11, 24, 37, 50, 63, 76, 89, 102, 115, 128,
                20, 31, 42, 53, 64, 75, 86, 97, 108, 119])
            u[inds] = mag
        elif name == 'rim':
            u[0:10] = mag
            u[130:140] = mag
            for i in range(10):
                u[10 + 12*i] = mag
                u[21 + 12*i] = mag
        elif name == 'checker':
            c = 0
            s = mag
            for i in range(10):
                u[c] = s
                c += 1
                s *= -1
            for j in range(10):
                for i in range(12):
                    u[c] = s
                    c += 1
                    s *= -1
                s *= -1
            s *= -1
            for i in range(10):
                u[c] = s
                c += 1
                s *= -1
        else:
            raise NotImplementedError(name)
        return u


class VoltageTransform():

    def __init__(self, dm):
        self.dm = dm

    def size(self):
        return self.dm.size()

    def write(self, u):
        assert(np.all(np.isfinite(u)))

        if norm(u, np.inf) > 1.:
            print('Saturation')
            u[u > 1.] = 1.
            u[u < -1.] = -1.
        assert(norm(u, np.inf) <= 1.)

        v = 2*np.sqrt((u + 1.0)/2.0) - 1.
        assert(np.all(np.isfinite(v)))
        assert(norm(v, np.inf) <= 1.)
        del u

        self.dm.write(v)

    def get_transform(self):
        return 'v = 2*np.sqrt((u + 1.0)/2.0) - 1.'

    def get_serial_number(self):
        return self.dm.get_serial_number()

    def preset(self, name, mag):
        return self.dm.preset(name, mag)


class DMPlot():

    txs = [0, 0, 0]
    floor = -1.5

    def __init__(self, sampling=128, nact=12, pitch=.3, roll=2, mapmul=.3):
        self.make_grids(sampling, nact, pitch, roll, mapmul)

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
                        parent, 'Actuator ' + str(ind), 'Range [-1, 1]',
                        u[ind], -1., 1., 4)
                    if ok:
                        u[ind] = val
                        self.draw(ax, u)
                        ax.figure.canvas.draw()
                        if write:
                            write(u)

        ax.figure.canvas.callbacks.connect('button_press_event', f)

    def draw(self, ax, u):
        ax.imshow(self.compute_pattern(u), vmin=self.floor, vmax=1)


# https://stackoverflow.com/questions/41794635/
# https://stackoverflow.com/questions/38666078/

class Listener(QThread):

    auto = True
    repeat = False
    poke = False
    sleep = .1
    unwrap = True

    sig_update = pyqtSignal(str)

    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def run(self):
        while True:
            self.shared.iq.put((
                'align', self.auto, self.repeat, self.poke, self.sleep,
                self.unwrap))
            result = self.shared.oq.get()
            self.sig_update.emit(result)
            if result == 'OK':
                self.shared.iq.put(('checkstop', self.repeat))
                self.shared.oq.get()
            if self.repeat == False:
                return


class Snapshot(QThread):

    f0f1 = None
    sleep = 0.1

    sig_error = pyqtSignal(str)
    sig_cam = pyqtSignal(tuple)
    sig_ft = pyqtSignal(tuple)
    sig_f0f1 = pyqtSignal(tuple)
    sig_1ord = pyqtSignal(tuple)
    sig_magwrapped = pyqtSignal(tuple)
    sig_unwrapped = pyqtSignal(tuple)
    sig_dm = pyqtSignal(np.ndarray)

    def __init__(self, dm_size, cam):
        super().__init__()
        self.cam = cam
        self.use_last = False
        self.repeat = False
        self.poke = False
        self.unwrap = True
        self.last_poke = 0
        self.u = np.zeros((dm_size,))
        self.cam_grid = make_cam_grid(cam.shape(), cam.get_pixel_size())
        self.ft_grid = make_ft_grid(cam.shape(), cam.get_pixel_size())

    def run(self):
        while True:
            if self.poke:
                self.u[:] = 0.
                self.u[self.last_poke] = .7
                self.sig_dm.emit(self.u)
                self.last_poke += 1
                self.last_poke %= self.u.size
                time.sleep(self.sleep)

            img = self.cam.grab_image()
            self.sig_cam.emit((img, self.cam_grid[2]))

            fimg = ft(img)
            logf2 = np.log(np.abs(fimg))
            self.sig_ft.emit((logf2, self.ft_grid[2]))

            if self.use_last and self.f0f1:
                f0, f1 = self.f0f1
            else:
                try:
                    f0, f1 = find_orders(
                        self.ft_grid[0], self.ft_grid[1], logf2)
                except ValueError:
                    self.sig_error.emit('Failed to find orders')
                    if self.repeat:
                        continue
                    else:
                        return
                self.sig_f0f1.emit((f0, f1))
                self.f0f1 = (f0, f1)

            try:
                f3, ext3 = extract_order(
                    fimg, self.ft_grid[0], self.ft_grid[1], f0, f1,
                    self.cam.get_pixel_size())
            except Exception as ex:
                self.sig_error.emit('Failed to extract order: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
                return
            self.sig_1ord.emit((np.log(np.abs(f3)), ext3))

            try:
                f4, _, _, ext4 = repad_order(
                    f3, self.ft_grid[0], self.ft_grid[1])
            except Exception as ex:
                self.sig_error.emit('Failed to repad order: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return

            try:
                gp = ift(f4)
                mag = np.abs(gp)
                wrapped = np.arctan2(gp.imag, gp.real)
            except Exception as ex:
                self.sig_error.emit('Failed to extract phase: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
            self.sig_magwrapped.emit((mag, wrapped, ext4))

            if self.unwrap:
                try:
                    unwrapped = call_unwrap(wrapped)
                except Exception as ex:
                    self.sig_error.emit('Failed to unwrap phase: ' + str(ex))
                    if self.repeat:
                        continue
                    else:
                        return
                self.sig_unwrapped.emit((unwrapped, ext4))

            if not self.repeat:
                break


class DataAcq(QThread):

    sleep = .1
    wavelength = 0
    fname = None
    stop = False
    sig_error = pyqtSignal(str)
    sig_cam = pyqtSignal(tuple)
    sig_h5 = pyqtSignal(str)

    def __init__(self, dm, cam, dmplot):
        super().__init__()
        self.dm = dm
        self.cam = cam
        self.dmplot = dmplot

        self.cam_grid = make_cam_grid(cam.shape(), cam.get_pixel_size())

        Ualign = []
        for name in ('centre', 'cross', 'x', 'rim', 'checker'):
            try:
                Ualign.append(dm.preset(name).reshape(-1, 1))
            except Exception:
                pass
        if len(Ualign) > 0:
            self.Ualign = np.hstack(Ualign)
        else:
            self.Ualign = np.zeros((self.dm.size(), 0))

    def run(self):
        self.stop = False

        libver = 'latest'
        now = datetime.now(timezone.utc)
        h5fn = now.astimezone().strftime('%Y%m%d_%H%M%S.h5')
        dmsn = self.dm.get_serial_number()
        if dmsn:
            h5fn = dmsn + '_' + h5fn
        self.fname = h5fn

        U = np.hstack((
                np.zeros((dm.size(), 1)),
                np.kron(np.eye(dm.size()), np.linspace(-.7, .7, 5)),
                np.zeros((dm.size(), 1))
                ))

        with h5py.File(h5fn, 'w', libver=libver) as h5f:
            h5f['datetime'] = now.isoformat()

            # save HDF5 library info
            h5f['h5py/libver'] = libver
            h5f['h5py/api_version'] = h5py.version.api_version
            h5f['h5py/version'] = h5py.version.version
            h5f['h5py/hdf5_version'] = h5py.version.hdf5_version
            h5f['h5py/info'] = h5py.version.info

            # save dmlib info
            h5f['dmlib/__date__'] = ''
            h5f['dmlib/__version__'] = ''
            h5f['dmlib/__commit__'] = ''

            h5f['dmplot/txs'] = self.dmplot.txs

            h5f['cam/serial'] = self.cam.get_serial_number()
            h5f['cam/camera_info'] = str(self.cam.get_camera_info())
            h5f['cam/sensor_info'] = str(self.cam.get_sensor_info())
            h5f['cam/pixel_size'] = self.cam.get_pixel_size()
            h5f['cam/pixel_size'].attrs['units'] = 'um'
            h5f['cam/exposure'] = self.cam.get_exposure()
            h5f['cam/exposure'].attrs['units'] = 'ms'
            h5f['cam/dtype'] = self.cam.get_image_dtype()
            h5f['cam/max'] = self.cam.get_image_max()

            h5f['wavelength'] = self.wavelength
            h5f['wavelength'].attrs['units'] = 'nm'
            h5f['sleep'] = self.sleep
            h5f['sleep'].attrs['units'] = 's'

            h5f['dm/serial'] = self.dm.get_serial_number()
            h5f['dm/transform'] = self.dm.get_transform()

            h5f['data/U'] = U
            h5f['data/U'].dims[0].label = 'actuators'
            h5f['data/U'].dims[1].label = 'step'
            h5f.create_dataset(
                'data/images', (U.shape[1],) + cam.shape(),
                dtype=self.cam.get_image_dtype())
            h5f['data/images'].dims[0].label = 'step'
            h5f['data/images'].dims[1].label = 'height'
            h5f['data/images'].dims[1].label = 'width'

            h5f['align/U'] = self.Ualign
            h5f['align/U'].dims[0].label = 'actuators'
            h5f['align/U'].dims[1].label = 'step'
            h5f.create_dataset(
                'align/images', (U.shape[1],) + cam.shape(),
                dtype=self.cam.get_image_dtype())
            h5f['align/images'].dims[0].label = 'step'
            h5f['align/images'].dims[1].label = 'height'
            h5f['align/images'].dims[1].label = 'width'

            tot = U.shape[1] + self.Ualign.shape[1]
            count = 0

            for i in range(self.Ualign.shape[1]):
                self.dm.write(self.Ualign[:, i])
                time.sleep(self.sleep)
                img = self.cam.grab_image()
                h5f['align/images'][i, ...] = img
                self.sig_cam.emit((img, self.cam_grid[2], count, tot))
                count += 1
                if self.stop:
                    return

            for i in range(U.shape[1]):
                self.dm.write(U[:, i])
                time.sleep(self.sleep)
                img = self.cam.grab_image()
                h5f['data/images'][i, ...] = img
                self.sig_cam.emit((img, self.cam_grid[2], count, tot))
                count += 1
                if self.stop:
                    return

        self.sig_h5.emit(self.fname)


def open_hardware(args):
    if args.cam == 'sim':
        cam = FakeCamera()
    elif args.cam == 'thorcam':
        from thorcam import ThorCam
        cam = ThorCam()
        cam.open(args.cam_name)
    else:
        raise NotImplementedError(args.cam)

    if args.dm == 'sim':
        dm = FakeDM()
    elif args.dm == 'bmc':
        from bmc import BMC
        dm = BMC()
        dm.open(args.dm_name)
    elif args.dm == 'ciusb':
        from ciusb import CIUsb
        dm = CIUsb()
        dm.open(args.dm_index)
    else:
        raise NotImplementedError(args.dm)

    return cam, dm


class Shared:

    def __init__(self, cam, dm):
        dbl_dtsize = np.dtype('float').itemsize
        cam_dtsize = np.dtype(cam.get_image_dtype()).itemsize
        cam_shape = cam.shape()
        totpixs = cam_shape[0]*cam_shape[1]
        dm_size = dm.size()

        self.cam_buf = Array('c', cam_dtsize*totpixs, lock=False)
        self.cam_ext = Array('d', 4, lock=False)
        self.cam_sat = Value('i', lock=False)

        self.ft_buf = Array('d', totpixs, lock=False)
        self.ft_ext = Array('d', 4, lock=False)

        self.f0f1 = Array('d', 2, lock=False)

        self.fstord_buf = Array('c', dbl_dtsize*totpixs, lock=False)
        self.fstord_ext = Array('d', 4, lock=False)
        self.fstord_shape = Array('i', 2, lock=False)

        self.mag_buf = Array('c', dbl_dtsize*totpixs, lock=False)
        self.wrapped_buf = Array('c', dbl_dtsize*totpixs, lock=False)
        self.unwrapped_buf = Array('c', dbl_dtsize*totpixs, lock=False)
        self.mag_ext = Array('d', 4, lock=False)
        self.mag_shape = Array('i', 2, lock=False)

        self.cam_dtype = cam.get_image_dtype()
        self.cam_shape = cam_shape
        self.dm_size = dm.size()

        self.dm = Array('d', dm.size(), lock=False)
        self.iq = Queue()
        self.oq = Queue()

    def make_static(self):
        self.u = np.frombuffer(self.dm, np.float)
        self.cam = np.frombuffer(
            self.cam_buf, self.cam_dtype).reshape(self.cam_shape)
        self.ft = np.frombuffer(
            self.ft_buf, np.float).reshape(self.cam_shape)

    def get_phase(self):
        nsum1 = self.fstord_shape[0]*self.fstord_shape[1]
        fstord = np.frombuffer(
            self.fstord_buf, np.float, count=nsum1).reshape(self.fstord_shape)
        nsum2 = self.mag_shape[0]*self.mag_shape[1]
        mag = np.frombuffer(
            self.mag_buf, np.float, count=nsum2).reshape(self.mag_shape)
        wrapped = np.frombuffer(
            self.wrapped_buf, np.float, count=nsum2).reshape(self.mag_shape)
        unwrapped = np.frombuffer(
            self.unwrapped_buf, np.float, count=nsum2).reshape(self.mag_shape)
        return fstord, mag, wrapped, unwrapped


class ArrayQueue:

    def __init__(self, maxbytes, maxlen):
        self.maxbytes = maxbytes
        self.maxlen = maxlen

        self.qfree = Queue(maxlen)
        self.qbuzy = Queue(maxlen)
        self.bufs = []
        for i in range(maxlen):
            self.bufs.append(Array('c', self.maxbytes, lock=False))
            self.qfree.append(i)

    def put(self, item, *args, **kwargs):
        if type(item) is np.ndarray:
            if item.nbytes > self.maxbytes:
                raise ValueError('item.nbytes > self.maxbytes')
            bufid = self.qfree.get()
            self.bufs[bufid][:item.nbytes] = item.tobytes()
            self.qbuzy.put(item.shape + (bufid, item.dtype.name))
        else:
            raise NotImplementedError()

    def get(self, *args, **kwargs):
        item = self.q.get(*args, **kwargs)
        if type(item) is tuple:
            return np.frombuffer(
                self.bufs[item[2]], dtype=item[3]).copy().reshape(item[:2])
        else:
            raise NotImplementedError()


def worker(shared, args):
    cam, dm = open_hardware(args)
    dm = VoltageTransform(dm)
    P = cam.get_pixel_size()

    shared.make_static()
    shared.f0f1[0] = 0.
    shared.f0f1[1] = 0.

    cam_grid = make_cam_grid(cam.shape(), P)
    ft_grid = make_ft_grid(cam.shape(), P)
    for i in range(4):
        shared.cam_ext[i] = cam_grid[2][i]/1000
        shared.ft_ext[i] = ft_grid[2][i]*1000

    # f0f1, lastpoke
    state = [None, 0]

    def run_align(auto, repeat, poke, sleep, unwrap):
        while True:
            if poke:
                shared.u[:] = 0.
                shared.u[last_poke] = .7
                state[1] += 1
                state[1] %= shared.u.size
                sleep(sleep)

            img = cam.grab_image()
            if img.max == cam.get_image_max():
                shared.cam_sat = 1
            else:
                shared.cam_sat = 0
            shared.cam[:] = img[:]

            fimg = ft(img)
            logf2 = np.log(np.abs(fimg))
            shared.ft[:] = logf2[:]

            if state[0] is None or auto:
                try:
                    f0, f1 = find_orders(ft_grid[0], ft_grid[1], logf2)
                except ValueError:
                    shared.oq.put('Failed to find orders')
                    if repeat:
                        continue
                    else:
                        return
            else:
                f0, f1 = state[0]
            state[0] = (f0, f1)
            shared.f0f1[0] = f0
            shared.f0f1[1] = f1

            try:
                f3, ext3 = extract_order(
                    fimg, ft_grid[0], ft_grid[1], f0, f1, P)
            except Exception as ex:
                shared.oq.put('Failed to extract order: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
                return
            logf3 = np.log(np.abs(f3))
            shared.fstord_buf[:logf3.nbytes] = logf3.tobytes()
            for i in range(4):
                shared.fstord_ext[i] = ext3[i]*1000
            shared.fstord_shape[:] = logf3.shape[:]

            try:
                f4, dd0, dd1, ext4 = repad_order(f3, ft_grid[0], ft_grid[1])
            except Exception as ex:
                shared.oq.put('Failed to repad order: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
                return
            try:
                gp = ift(f4)
                mag = np.abs(gp)
                wrapped = np.arctan2(gp.imag, gp.real)
            except Exception as ex:
                shared.oq.put('Failed to extract phase: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
            shared.mag_buf[:mag.nbytes] = mag.tobytes()
            shared.wrapped_buf[:wrapped.nbytes] = wrapped.tobytes()
            for i in range(4):
                shared.mag_ext[i] = ext4[i]/1000
            shared.mag_shape[:] = mag.shape[:]

            if unwrap:
                try:
                    unwrapped = call_unwrap(wrapped)
                except Exception as ex:
                    shared.oq.put('Failed to unwrap phase: ' + str(ex))
                    if self.repeat:
                        continue
                    else:
                        return
            shared.unwrapped_buf[:unwrapped.nbytes] = unwrapped.tobytes()

            shared.oq.put('OK')
            print('RAN align')

            checkstop = shared.iq.get()[0]
            shared.oq.put('')

            if not repeat or checkstop:
                break

    for cmd in iter(shared.iq.get, 'STOP'):
        print(cmd)
        if cmd[0] == 'get_exposure':
            shared.oq.put(cam.get_exposure())
        elif cmd[0] == 'get_exposure_range':
            shared.oq.put(cam.get_exposure_range())
        elif cmd[0] == 'set_exposure':
            shared.oq.put(cam.set_exposure(cmd[1]))
        elif cmd[0] == 'get_framerate':
            shared.oq.put(cam.get_framerate())
        elif cmd[0] == 'get_framerate_range':
            shared.oq.put(cam.get_framerate_range())
        elif cmd[0] == 'set_framerate':
            shared.oq.put(cam.set_framerate(cmd[1]))
        elif cmd[0] == 'write':
            dm.write(shared.u)
            shared.oq.put('OK')
        elif cmd[0] == 'preset':
            shared.u[:] = dm.preset(cmd[1], cmd[2])
            shared.oq.put('OK')
        elif cmd[0] == 'align':
            run_align(*cmd[1:])
        else:
            raise NotImplementedError(cmd)

    print('STOP CMD')
    while not shared.iq.empty():
        print(shared.iq.get())

    print('STOPPED')


class Snapshot(QThread):

    def run(self):
        while True:
            if self.poke:
                self.u[:] = 0.
                self.u[self.last_poke] = .7
                self.sig_dm.emit(self.u)
                self.last_poke += 1
                self.last_poke %= self.u.size
                sleep(self.sleep)

            img = self.cam.grab_image()
            self.sig_cam.emit((img, self.cam_grid[2]))

            fimg = ft(img)
            logf2 = np.log(np.abs(fimg))
            self.sig_ft.emit((logf2, self.ft_grid[2]))

            if self.use_last and self.f0f1:
                f0, f1 = self.f0f1
            else:
                try:
                    f0, f1 = find_orders(
                        self.ft_grid[0], self.ft_grid[1], logf2)
                except ValueError:
                    self.sig_error.emit('Failed to find orders')
                    if self.repeat:
                        continue
                    else:
                        return
                self.sig_f0f1.emit((f0, f1))
                self.f0f1 = (f0, f1)

            try:
                f3, ext3 = extract_order(
                    fimg, self.ft_grid[0], self.ft_grid[1], f0, f1,
                    self.cam.get_pixel_size())
            except Exception as ex:
                self.sig_error.emit('Failed to extract order: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
                return
            self.sig_1ord.emit((np.log(np.abs(f3)), ext3))

            try:
                f4, _, _, ext4 = repad_order(
                    f3, self.ft_grid[0], self.ft_grid[1])
            except Exception as ex:
                self.sig_error.emit('Failed to repad order: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return

            try:
                gp = ift(f4)
                mag = np.abs(gp)
                wrapped = np.arctan2(gp.imag, gp.real)
            except Exception as ex:
                self.sig_error.emit('Failed to extract phase: ' + str(ex))
                if self.repeat:
                    continue
                else:
                    return
            self.sig_magwrapped.emit((mag, wrapped, ext4))

            if self.unwrap:
                try:
                    unwrapped = call_unwrap(wrapped)
                except Exception as ex:
                    self.sig_error.emit('Failed to unwrap phase: ' + str(ex))
                    if self.repeat:
                        continue
                    else:
                        return
                self.sig_unwrapped.emit((unwrapped, ext4))

            if not self.repeat:
                break



if __name__ == '__main__':
    app = QApplication(sys.argv)
    if platform.system() == 'Windows':
        print(QStyleFactory.keys())
        try:
            app.setStyle(QStyleFactory.create('Fusion'))
        except Exception:
            pass

    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='DM calibration GUI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dm', choices=['sim', 'bmc', 'ciusb'], default='sim')
    parser.add_argument(
        '--cam', choices=['sim', 'thorcam'], default='sim')
    parser.add_argument('--dm-name', type=str, default='C17W005#050')
    parser.add_argument('--dm-index', type=int, default=0)
    parser.add_argument('--cam-name', type=str, default=None)
    args = parser.parse_args(args[1:])

    cam, dm = open_hardware(args)
    shared = Shared(cam, dm)
    dm.close()
    cam.close()

    p = Process(target=worker, args=(shared, args))
    p.start()

    control = Control(p, shared)
    control.show()

    exit = app.exec_()
    shared.iq.put('STOP')
    sys.exit(exit)
