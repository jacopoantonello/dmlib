#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

from multiprocessing import Process, Queue, Array
from datetime import datetime, timezone
from time import sleep
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
        self.shared.make_u()

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
        layout.addWidget(brun, 1, 0)
        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 2, 0, 1, 2)

        bauto = QCheckBox('auto')
        bauto.setChecked(True)
        layout.addWidget(bauto, 3, 0)
        brepeat = QCheckBox('repeat')
        layout.addWidget(brepeat, 3, 1)

        return
        snapshot = Snapshot(self.shared.u.size, self.cam)

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
                    self, '', 'sleep [s]', snapshot.sleep, decimals=4)
                if ok:
                    snapshot.sleep = val
            return f

        def f2():
            def f(p):
                snapshot.poke = p
            return f

        def f3():
            def f(p):
                snapshot.unwrap = p
            return f

        bsleep.clicked.connect(f1())
        bpoke.stateChanged.connect(f2())
        bunwrap.stateChanged.connect(f3())

        def f1():
            def f():
                snapshot.repeat = brepeat.isChecked()
                status.setText('working...')
                brun.setEnabled(False)
                self.align_nav.setEnabled(False)
                snapshot.start()
            return f

        def f2(a, txt, mm=True, satcheck=False):
            def f(t):
                a.clear()
                if mm:
                    mul = 1e-3
                    lab = 'mm'
                else:
                    mul = 1e3
                    lab = '1/mm'
                ext = (t[1][0]*mul, t[1][1]*mul, t[1][2]*mul, t[1][3]*mul)
                a.imshow(t[0], extent=ext, origin='lower')
                a.set_xlabel(lab)
                if satcheck:
                    # FIXME
                    if t[0].max() == self.cam.get_image_max():
                        a.set_title(txt + ' SAT')
                    else:
                        a.set_title('{} {: 3d} {: 3d}'.format(
                            txt, t[0].min(), t[0].max()))
                else:
                    a.set_title(txt)
                a.figure.canvas.draw()
            return f

        def f3(a):
            def f(t):
                a.plot(t[0]*1e3, t[1]*1e3, 'rx', markersize=6)
                a.plot(-t[0]*1e3, -t[1]*1e3, 'rx', markersize=6)
                a.figure.canvas.draw()
            return f

        def f4(a1, a2, txt1, txt2):
            def f(t):
                mul = 1e-3
                ext = (t[2][0]*mul, t[2][1]*mul, t[2][2]*mul, t[2][3]*mul)
                a1.clear()
                a1.imshow(t[0], extent=ext, origin='lower')
                a1.set_xlabel('mm')
                a1.set_title(txt1)

                a2.clear()
                a2.imshow(t[1], extent=ext, origin='lower')
                a2.set_xlabel('mm')
                a2.set_title(txt2)

                a1.figure.canvas.draw()
            return f

        def f5():
            def f():
                if status.text() == 'working...':
                    status.setText('')
                brun.setEnabled(True)
                self.align_nav.setEnabled(True)
            return f

        def f6():
            def f(errstr):
                self.align_axes[0, 2].clear()
                self.align_axes[1, 0].clear()
                self.align_axes[1, 1].clear()
                self.align_axes[1, 2].clear()

                status.setText(errstr)
                self.dm_fig.figure.canvas.draw()
            return f

        def f7():
            def f():
                snapshot.use_last = not snapshot.use_last
            return f

        def f8():
            def f(p):
                snapshot.repeat = p
            return f

        def f9():
            def f(u):
                self.write_dm(u)
            return f

        def f10():
            def f():
                snapshot.repeat = False
            return f

        snapshot.finished.connect(f5())
        snapshot.sig_error.connect(f6())
        snapshot.sig_cam.connect(f2(
            self.align_axes[0, 0], 'camera', True, True))
        snapshot.sig_ft.connect(f2(self.align_axes[0, 1], 'FT', False))
        snapshot.sig_f0f1.connect(f3(self.align_axes[0, 1]))
        snapshot.sig_1ord.connect(f2(
            self.align_axes[0, 2], '1st order', False))
        snapshot.sig_magwrapped.connect(f4(
            self.align_axes[1, 0], self.align_axes[1, 1],
            'magnitude',  'wrapped phi'))
        snapshot.sig_unwrapped.connect(
            f2(self.align_axes[1, 2], 'unwrapped phi'))
        snapshot.sig_dm.connect(f9())

        brun.clicked.connect(f1())
        bauto.stateChanged.connect(f7())
        brepeat.stateChanged.connect(f8())

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
                sleep(self.sleep)
                img = self.cam.grab_image()
                h5f['align/images'][i, ...] = img
                self.sig_cam.emit((img, self.cam_grid[2], count, tot))
                count += 1
                if self.stop:
                    return

            for i in range(U.shape[1]):
                self.dm.write(U[:, i])
                sleep(self.sleep)
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
        cam_dtsize = np.dtype(cam.get_image_dtype()).itemsize
        cam_shape = cam.shape()
        totpixs = cam_shape[0]*cam_shape[1]
        dm_size = dm.size()

        self.cam = Array('c', cam_dtsize*totpixs, lock=False)
        self.camext = Array('d', 4, lock=False)

        self.ft = Array('d', totpixs, lock=False)
        self.ftext = Array('d', 4, lock=False)

        self.f0f1 = Array('d', 2, lock=False)

        self.fstord = Array('d', totpixs, lock=False)
        self.fstordext = Array('d', 4, lock=False)

        self.mag = Array('d', totpixs, lock=False)
        self.magext = Array('d', 4, lock=False)

        self.wrapped = Array('d', totpixs, lock=False)
        self.wrappedext = Array('d', 4, lock=False)

        self.unwrapped = Array('d', totpixs, lock=False)
        self.unwrappedext = Array('d', 4, lock=False)

        self.unwrapped = Array('d', totpixs, lock=False)
        self.unwrappedext = Array('d', 4, lock=False)

        self.cam_dtsize = cam_dtsize
        self.cam_shape = cam_shape
        self.dm_size = dm.size()

        self.dm = Array('d', dm.size(), lock=False)
        self.iq = Queue()
        self.oq = Queue()

    def make_u(self):
        self.u = np.frombuffer(self.dm, np.float)


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
    shared.make_u()
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
        else:
            raise NotImplementedError(cmd)

    print('STOP CMD')
    while not shared.iq.empty():
        print(shared.iq.get())

    print('STOPPED')


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
