#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import multiprocessing

from os import path
from collections import namedtuple
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
    QInputDialog, QStyleFactory, QSizePolicy, QErrorMessage,
    )

import version
from interf import (
    make_cam_grid, make_ft_grid, ft, ift, find_orders, repad_order,
    extract_order, call_unwrap, estimate_aperture_centre)

from calibrate import calibrate


class Control(QMainWindow):

    closing = False

    def __init__(self, worker, shared, settings={}, parent=None):
        super().__init__()

        self.worker = worker
        self.shared = shared
        self.shared.make_static()

        self.setWindowTitle('DM calibration ' + version.__version__)
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

    def update_dm_gui(self):
        self.dmplot.draw(self.dm_ax, self.shared.u)
        self.dm_ax.figure.canvas.draw()

    def write_dm(self, u=None):
        if u is not None:
            self.shared.u[:] = u[:]
        self.update_dm_gui()
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

        flipx = QPushButton('flipx')
        layout.addWidget(flipx, 1, 0)
        flipy = QPushButton('flipy')
        layout.addWidget(flipy, 1, 1)
        rotate1 = QPushButton('rotate cw')
        layout.addWidget(rotate1, 2, 0)
        rotate2 = QPushButton('rotate acw')
        layout.addWidget(rotate2, 2, 1)

        def f4(n):
            def f():
                self.shared.iq.put(('preset', n, 0.8))
                self.shared.oq.get()
                self.write_dm(None)
            return f

        reset = QPushButton('reset')
        layout.addWidget(reset, 3, 0)
        i = 4
        j = 0
        for name in ('centre', 'cross', 'x', 'rim', 'checker', 'arrows'):
            b = QPushButton(name)
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
                self.write_dm()
            return f

        reset.clicked.connect(f2())

        def f3(sign):
            ind = [0]

            def f():
                ind[0] += sign
                ind[0] %= 4
                self.dmplot.rotate(ind[0])
                self.update_dm_gui()
            return f

        def f4(cb, b):
            def f():
                cb(b.isChecked())
                self.update_dm_gui()
            return f

        flipx.setCheckable(True)
        flipy.setCheckable(True)
        flipx.clicked.connect(f4(self.dmplot.flipx, flipx))
        flipy.clicked.connect(f4(self.dmplot.flipy, flipy))
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

        listener = AlignListener(self.shared)

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
                    self, 'Delay', 'write/read delay [s]',
                    listener.sleep, decimals=4)
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
                ind = self.tabs.indexOf(frame)
                for i in range(self.tabs.count()):
                    if i != ind:
                        self.tabs.setTabEnabled(i, False)
                self.toolbox.setEnabled(False)
                brun.setEnabled(False)
                bauto.setEnabled(False)
                brepeat.setEnabled(False)
                bpoke.setEnabled(False)
                bsleep.setEnabled(False)
                bunwrap.setEnabled(False)
                self.align_nav.setEnabled(False)
                listener.repeat = brepeat.isChecked()
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

        def f4():
            def f():
                listener.repeat = False
                if not listener.isFinished():
                    status.setText('stopping...')
            return f

        def f20():
            def finish():
                for i in range(self.tabs.count()):
                    self.tabs.setTabEnabled(i, True)
                self.toolbox.setEnabled(True)
                brun.setEnabled(True)
                bauto.setEnabled(True)
                brepeat.setEnabled(True)
                bpoke.setEnabled(True)
                bsleep.setEnabled(True)
                bunwrap.setEnabled(True)
                self.align_nav.setEnabled(True)

            def f(msg):
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
                if self.shared.cam_sat.value:
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
                    status.setText(msg)
                    finish()
                    a2.figure.canvas.draw()
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

                self.update_dm_gui()

                if not listener.repeat:
                    status.setText('stopped')
                    finish()

            return f

        listener.sig_update.connect(f20())
        brun.clicked.connect(f1())
        bstop.clicked.connect(f4())
        bauto.stateChanged.connect(f2())
        brepeat.stateChanged.connect(f3())
        self.align_nav = NavigationToolbar2QT(self.align_fig, frame)

    def make_panel_dataacq(self):
        frame = QFrame()
        self.dataacq_fig = FigureCanvas(Figure(figsize=(7, 5)))
        layout = QGridLayout()
        frame.setLayout(layout)
        layout.addWidget(self.dataacq_fig, 0, 0, 1, 0)

        self.tabs.addTab(frame, 'calibration')

        self.dataacq_axes = self.dataacq_fig.figure.subplots(2, 2)
        self.dataacq_fig.figure.subplots_adjust(
            left=.125, right=.9,
            bottom=.1, top=.9,
            wspace=0.45, hspace=0.45)
        self.dataacq_axes[0, 0].set_title('camera')
        self.dataacq_axes[0, 1].set_title('dm')
        self.dataacq_axes[1, 0].set_title('wrapped phi')
        self.dataacq_axes[1, 1].set_title('unwrapped phi')

        brun = QPushButton('run')
        bstop = QPushButton('stop')
        bwavelength = QPushButton('wavelength')
        layout.addWidget(brun, 1, 0)
        layout.addWidget(bstop, 1, 1)
        layout.addWidget(bwavelength, 1, 2)
        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 2, 0, 1, 3)

        bplot = QPushButton('open')
        bprev = QPushButton('prev')
        bnext = QPushButton('next')
        layout.addWidget(bplot, 3, 0)
        layout.addWidget(bprev, 3, 1)
        layout.addWidget(bnext, 3, 2)

        baperture = QPushButton('aperture')
        layout.addWidget(baperture, 4, 0)
        bcalibrate = QPushButton('calibrate')
        layout.addWidget(bcalibrate, 4, 1)
        bclear = QPushButton('clear')
        layout.addWidget(bclear, 4, 2)

        disables = [
            self.toolbox, brun, bwavelength, bplot,
            bprev, bnext, baperture, bcalibrate, bclear]

        wavelength = []
        dataset = []
        lastind = []
        centre = []
        radius = [.0]
        listener = DataAcqListener(self.shared, wavelength, self.dmplot)

        def clearup(clear_status=False):
            wavelength.clear()
            dataset.clear()
            lastind.clear()
            centre.clear()
            radius[0] = .0

            if clear_status:
                status.setText('')
                self.dataacq_axes[0, 0].clear()
                self.dataacq_axes[0, 1].clear()
                self.dataacq_axes[1, 0].clear()
                self.dataacq_axes[1, 1].clear()
                self.dataacq_axes[1, 1].figure.canvas.draw()

        def disable():
            ind = self.tabs.indexOf(frame)
            for i in range(self.tabs.count()):
                if i != ind:
                    self.tabs.setTabEnabled(i, False)
            for b in disables:
                b.setEnabled(False)

        def enable():
            for i in range(self.tabs.count()):
                self.tabs.setTabEnabled(i, True)
            for b in disables:
                b.setEnabled(True)

        def f0():
            def f():
                if wavelength:
                    wl = wavelength[0]
                else:
                    wl = 775.
                val, ok = QInputDialog.getDouble(
                    self, 'Wavelength', 'wavelength [nm]', wl, decimals=1)
                if ok:
                    if wavelength:
                        wavelength[0] = val
                    else:
                        wavelength.append(val)
            return f

        def f1():
            askwl = f0()

            def f():
                clearup()

                while not wavelength:
                    askwl()

                self.dataacq_axes[0, 0].clear()
                self.dataacq_axes[0, 1].clear()
                self.dataacq_axes[1, 0].clear()
                self.dataacq_axes[1, 1].clear()
                self.dataacq_axes[1, 1].figure.canvas.draw()
                disable()

                listener.run = True
                listener.start()
            return f

        def check_err():
            reply = self.shared.oq.get()
            if reply[0] != 'OK':
                status.setText(reply[0])
                return -1
            else:
                return reply[1:]

        def bootstrap():
            if not dataset:
                fileName, _ = QFileDialog.getOpenFileName(
                    self, 'Select dataset', '',
                    'H5 (*.h5);;All Files (*)')
                if not fileName:
                    return False
                else:
                    dataset.append(fileName)
                    return True
            else:
                return True

        def f3(offset=None):
            theta = np.linspace(0, 2*np.pi, 96)

            def f():
                if not bootstrap():
                    return

                if lastind:
                    last = lastind[0]
                else:
                    last = 0
                self.shared.iq.put(('query', dataset[0]))

                ndata = check_err()
                if ndata == -1:
                    clearup()
                    return
                else:
                    self.dmplot.update_txs(ndata[1])

                if offset is None or not lastind:
                    val, ok = QInputDialog.getInt(
                        self, 'Select an index to plot',
                        'time step [{}, {}]'.format(0, ndata[0] - 1),
                        last, 0, ndata[0] - 1)
                    if not ok:
                        return
                else:
                    val = lastind[0] + offset

                if val < 0:
                    val = 0
                elif val >= ndata[0]:
                    val = ndata[0] - 1
                if lastind:
                    lastind[0] = val
                else:
                    lastind.append(val)

                self.shared.iq.put((
                    'plot', dataset[0], val, centre, radius[0]))
                if check_err() == -1:
                    return

                a1 = self.dataacq_axes[0, 0]
                a2 = self.dataacq_axes[0, 1]
                a3 = self.dataacq_axes[1, 0]
                a4 = self.dataacq_axes[1, 1]

                a1.clear()
                a2.clear()
                a3.clear()
                a4.clear()

                a1.imshow(
                    self.shared.cam, extent=self.shared.cam_ext,
                    origin='lower')
                a1.set_xlabel('mm')
                if self.shared.cam_sat.value:
                    a1.set_title('cam SAT')
                else:
                    a1.set_title('cam {: 3d} {: 3d}'.format(
                        self.shared.cam.min(), self.shared.cam.max()))

                data = self.shared.get_phase()
                wrapped, unwrapped = data[2:]

                self.dmplot.draw(a2, self.shared.u)
                a2.axis('off')
                a2.set_title('dm')

                a3.imshow(
                    wrapped, extent=self.shared.mag_ext,
                    origin='lower')
                a3.set_xlabel('mm')
                a3.set_title('wrapped phi')

                a4.imshow(
                    unwrapped, extent=self.shared.mag_ext,
                    origin='lower')
                a4.set_xlabel('mm')
                a4.set_title('unwrapped phi')
                if centre:
                    a4.plot(centre[0]/1000, centre[1]/1000, 'rx')
                if radius[0] > 0. and centre:
                    a4.plot(
                        centre[0]/1000 + radius[0]/1000*np.cos(theta),
                        centre[1]/1000 + radius[0]/1000*np.sin(theta), 'r')

                a4.figure.canvas.draw()

                # self.update_dm_gui()
                status.setText('{} {}/{}'.format(
                    dataset[0], val, ndata[0] - 1))
            return f

        def f4():
            drawf = f3(0)

            def f():
                if not bootstrap():
                    return False

                if radius:
                    rad = radius[0]/1000
                else:
                    rad = 2.1

                if self.shared.cam_ext[1] > 0:
                    radmax = min((
                        self.shared.cam_ext[1], self.shared.cam_ext[3]))
                else:
                    radmax = 10.

                val, ok = QInputDialog.getDouble(
                    self, 'Aperture radius', 'radius in mm', rad, 0.,
                    radmax, 3)
                if ok and val > 0.:
                    if radius:
                        radius[0] = val*1000
                    else:
                        radius.append(val*1000)

                    self.shared.iq.put(('centre', dataset[0]))
                    ndata = check_err()
                    if ndata == -1:
                        return False
                    if centre:
                        centre[:] = ndata[:]
                    else:
                        centre.append(ndata[0])
                        centre.append(ndata[1])

                    drawf()
                    return True
                else:
                    return False

            return f

        def f6():
            def f():
                ndata = check_err()
                if ndata != -1:
                    status.setText('{} {:.2f}%'.format(*ndata))
                enable()
                bstop.setEnabled(True)
            return f

        clistener = CalibListener(self.shared, dataset, centre, radius)
        clistener.finished.connect(f6())

        def f5():
            setup_aperture = f4()

            def f():
                disable()
                bstop.setEnabled(False)

                ok = True
                if radius[0] <= 0 or not centre:
                    ok = setup_aperture()

                if ok and radius[0] > 0 and centre:
                    status.setText('working...')
                    clistener.start()
                else:
                    enable()
                    bstop.setEnabled(True)

            return f

        def f2():
            def f():
                listener.run = False
                if not listener.isFinished():
                    status.setText('stopping...')
            return f

        def f20():
            def f(msg):
                listener.busy = True

                a1 = self.dataacq_axes[0, 0]
                a2 = self.dataacq_axes[0, 1]

                a1.clear()

                a1.imshow(
                    self.shared.cam, extent=self.shared.cam_ext,
                    origin='lower')
                a1.set_xlabel('mm')
                if self.shared.cam_sat.value:
                    a1.set_title('cam SAT')
                else:
                    a1.set_title('cam {: 3d} {: 3d}'.format(
                        self.shared.cam.min(), self.shared.cam.max()))

                if msg[0] == 'OK':
                    status.setText('{}/{}'.format(msg[1] + 1, msg[2]))
                elif msg[0] == 'finished':
                    if dataset:
                        dataset[0] = msg[1]
                    else:
                        dataset.append(msg[1])
                    status.setText('finished ' + msg[1])
                    enable()
                else:
                    status.setText(msg[0])
                    enable()

                self.dmplot.draw(a2, self.shared.u)
                a2.axis('off')
                a2.set_title('dm')
                a2.figure.canvas.draw()

                listener.busy = False

            return f

        def f6():
            def f():
                clearup(True)
            return f

        brun.clicked.connect(f1())
        bstop.clicked.connect(f2())
        bwavelength.clicked.connect(f0())
        bplot.clicked.connect(f3())
        bnext.clicked.connect(f3(1))
        bprev.clicked.connect(f3(-1))
        baperture.clicked.connect(f4())
        bcalibrate.clicked.connect(f5())
        bclear.clicked.connect(f6())

        listener.sig_update.connect(f20())
        self.dataacq_nav = NavigationToolbar2QT(self.dataacq_fig, frame)
        disables.append(self.dataacq_nav)


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

    def get_devices(self):
        return ['cam0']


class FakeDM():

    def get_devices(self):
        return ['dm0']

    def size(self):
        return 140

    def write(self, v):
        print('FakeDM', v)

    def get_transform(self):
        return 'v = u'

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
        elif name == 'arrows':
            inds = np.array([
                20, 31, 42, 53, 64, 75, 86, 97, 108, 119,
                16, 17, 18, 19,
                29, 30,
                32, 44, 56, 68,
                43, 55,
                34, 23, 12, 25, 38, 24, 36, 48, 60,
                89, 102, 115, 128, 101, 113, 90, 91,
                ])
            u[inds] = mag
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

    def preset(self, name, mag=0.7):
        return self.dm.preset(name, mag)


class DMPlot():

    txs = [0, 0, 0]
    floor = -1.5

    def __init__(self, sampling=128, nact=12, pitch=.3, roll=2, mapmul=.3):
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
        ax.imshow(self.compute_pattern(u), vmin=self.floor, vmax=1)


# https://stackoverflow.com/questions/41794635/
# https://stackoverflow.com/questions/38666078/

class AlignListener(QThread):

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
        self.shared.iq.put((
            'align', self.auto, self.repeat, self.poke, self.sleep,
            self.unwrap))
        while True:
            result = self.shared.oq.get()
            self.sig_update.emit(result)
            if result == 'OK':
                self.shared.iq.put(('stopcmd', not self.repeat))
                self.shared.oq.get()
            if not self.repeat:
                return


class CalibListener(QThread):

    def __init__(self, shared, dset, centre, radius):
        super().__init__()
        self.shared = shared
        self.dset = dset
        self.centre = centre
        self.radius = radius

    def run(self):
        self.shared.iq.put((
            'calibrate', self.dset[0], self.centre, self.radius[0]))


class DataAcqListener(QThread):

    busy = False
    run = True
    sig_update = pyqtSignal(tuple)

    def __init__(self, shared, wavelength, dmplot):
        super().__init__()
        self.shared = shared
        self.wavelength = wavelength
        self.dmplot = dmplot

    def run(self):
        self.shared.iq.put(('dataacq', self.wavelength[0], self.dmplot.txs))
        while True:
            result = self.shared.oq.get()
            print('listener result', result)
            if result[0] == 'OK':
                self.shared.iq.put(('stopcmd', not self.run))
                self.shared.oq.get()
                if not self.run:
                    self.sig_update.emit(('stopped',))
                    print('DataAcqListener stopped')
                    return
                elif not self.busy:
                    self.sig_update.emit(result)
            else:
                self.sig_update.emit(result)
                print('DataAcqListener terminates')
                return


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

        self.totpixs = totpixs
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


def run_worker(shared, args):
    p = Worker(shared, args)
    p.run()


class Worker:

    dfname = None
    dset = None

    # f0f1, lastpoke
    run_align_state = [None, 0]

    def __init__(self, shared, args):
        cam, dm = open_hardware(args)
        cam.set_exposure(cam.get_exposure_range()[0])
        dm = VoltageTransform(dm)

        shared.make_static()
        shared.f0f1[0] = 0.
        shared.f0f1[1] = 0.

        P = cam.get_pixel_size()
        cam_grid = make_cam_grid(cam.shape(), P)
        ft_grid = make_ft_grid(cam.shape(), P)
        for i in range(4):
            shared.cam_ext[i] = cam_grid[2][i]/1000
            shared.ft_ext[i] = ft_grid[2][i]*1000

        self.cam = cam
        self.dm = dm
        self.shared = shared
        self.cam_grid = cam_grid
        self.ft_grid = ft_grid
        self.P = P

    def run(self):
        cam = self.cam
        dm = self.dm
        shared = self.shared

        for cmd in iter(shared.iq.get, 'STOP'):
            print('worker', 'cmd', cmd)
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
                self.run_align(*cmd[1:])
            elif cmd[0] == 'dataacq':
                self.run_dataacq(*cmd[1:])
            elif cmd[0] == 'query':
                self.run_query(*cmd[1:])
            elif cmd[0] == 'plot':
                self.run_plot(*cmd[1:])
            elif cmd[0] == 'centre':
                self.run_centre(*cmd[1:])
            elif cmd[0] == 'calibrate':
                self.run_calibrate(*cmd[1:])
            else:
                raise NotImplementedError(cmd)

        print('worker', 'STOP CMD')
        while not shared.iq.empty():
            print(shared.iq.get())

        print('worker', 'STOPPED')

    def run_align(self, auto, repeat, poke, sleep, unwrap):
        cam = self.cam
        dm = self.dm
        shared = self.shared
        state = self.run_align_state

        ft_grid = self.ft_grid
        P = self.P

        while True:
            if poke:
                shared.u[:] = 0.
                shared.u[state[1]] = .7
                state[1] += 1
                state[1] %= shared.u.size
                dm.write(shared.u)
                time.sleep(sleep)

            try:
                img = cam.grab_image()
                if img.max() == cam.get_image_max():
                    shared.cam_sat.value = 1
                else:
                    shared.cam_sat.value = 0
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

                f3, ext3 = extract_order(
                    fimg, ft_grid[0], ft_grid[1], f0, f1, P)

                logf3 = np.log(np.abs(f3))
                shared.fstord_buf[:logf3.nbytes] = logf3.tobytes()
                for i in range(4):
                    shared.fstord_ext[i] = ext3[i]*1000
                shared.fstord_shape[:] = logf3.shape[:]

                f4, dd0, dd1, ext4 = repad_order(f3, ft_grid[0], ft_grid[1])

                gp = ift(f4)
                mag = np.abs(gp)
                wrapped = np.arctan2(gp.imag, gp.real)

                shared.mag_buf[:mag.nbytes] = mag.tobytes()
                shared.wrapped_buf[:wrapped.nbytes] = wrapped.tobytes()
                for i in range(4):
                    shared.mag_ext[i] = ext4[i]/1000
                shared.mag_shape[:] = mag.shape[:]

                if unwrap:
                    try:
                        _, edges = np.histogram(mag.ravel(), bins=100)
                        mask = (mag < edges[1]).reshape(mag.shape)
                        unwrapped = call_unwrap(wrapped, mask)
                    except Exception as ex:
                        shared.oq.put('Failed to unwrap phase: ' + str(ex))
                        if repeat:
                            continue
                        else:
                            return
                    shared.unwrapped_buf[:unwrapped.nbytes] = \
                        unwrapped.tobytes()
                else:
                    shared.unwrapped_buf[:] = \
                        np.zeros(shared.totpixs).tobytes()
            except Exception as ex:
                shared.oq.put('Error: ' + str(ex))
                if repeat:
                    continue
                else:
                    return

            shared.oq.put('OK')
            print('run_align', 'iteration')

            stopcmd = shared.iq.get()[1]
            shared.oq.put('')

            if not repeat or stopcmd:
                print('run_align', 'break', repeat, stopcmd)
                break
            else:
                print('run_align', 'continue')

    def open_dset(self, dname):
        if self.dfname is None or self.dfname != dname:
            if self.dset is not None:
                self.dset.close()
            self.dset = h5py.File(dname, 'r')
            self.dfname = dname

            if 'data/images' not in self.dset:
                self.shared.oq.put((dname + ' does not look like a dataset',))
                return -1

            try:
                img = self.dset['data/images'][0, ...]
                P = self.dset['cam/pixel_size'][()]
                shape = img.shape
                cam_grid = make_cam_grid(shape, P)
                ft_grid = make_ft_grid(shape, P)
                fimg = ft(img)
                logf2 = np.log(np.abs(fimg))
                f0, f1 = find_orders(ft_grid[0], ft_grid[1], logf2)
                f3, ext3 = extract_order(
                    fimg, ft_grid[0], ft_grid[1], f0, f1, P)
                f4, dd0, dd1, ext4 = repad_order(f3, ft_grid[0], ft_grid[1])
                gp = ift(f4)
                mag = np.abs(gp)
                wrapped = np.arctan2(gp.imag, gp.real)
                unwrapped = call_unwrap(wrapped)
            except Exception as e:
                self.shared.oq.put((str(e),))
                return -1

            DSetPars = namedtuple(
                'DSetPars', (
                    'img P shape cam_grid ft_grid fimg logf2 f0 f1 f3 ' +
                    'ext3 f4 dd0 dd1 ext4 gp mag wrapped unwrapped'))
            self.dsetpars = DSetPars(
                img, P, shape, cam_grid, ft_grid, fimg, logf2, f0, f1, f3,
                ext3, f4, dd0, dd1, ext4, gp, mag, wrapped, unwrapped)
            return 0

    def run_query(self, dname):
        if self.open_dset(dname):
            return

        dmplot_txs = self.dset['dmplot/txs'][()].tolist()

        self.shared.oq.put((
            'OK',
            self.dset['align/U'].shape[1] + self.dset['data/U'].shape[1],
            dmplot_txs))

    def pull(self, addr, ind, centre=None, radius=.0):
        img = self.dset[addr + '/images'][ind, ...]
        fimg = ft(img)
        f3, ext3 = extract_order(
            fimg, self.dsetpars.ft_grid[0], self.dsetpars.ft_grid[1],
            self.dsetpars.f0, self.dsetpars.f1, self.dsetpars.P)
        f4, dd0, dd1, ext4 = repad_order(
            f3, self.dsetpars.ft_grid[0], self.dsetpars.ft_grid[1])
        if radius > 0 and centre:
            [xx, yy] = np.meshgrid(dd1 - centre[0], dd0 - centre[1])
            mask = np.sqrt(xx**2 + yy**2) >= radius
        else:
            mask = None
        gp = ift(f4)
        mag = np.abs(gp)
        wrapped = np.arctan2(gp.imag, gp.real)
        unwrapped = call_unwrap(wrapped, mask)

        return img, mag, ext4, wrapped, unwrapped

    def run_calibrate(self, dname, centre, radius):
        if self.open_dset(dname):
            return

        try:
            print('run_centre', centre, radius, self.dsetpars.dd0.max())
            H, mvaf, phi0, z0, C, alpha, lambda1, cart = calibrate(
                self.dsetpars.ft_grid, self.dsetpars.f0, self.dsetpars.f1,
                self.dsetpars.P, self.dsetpars.dd0, self.dsetpars.dd1,
                centre, radius, self.dset['data/U'][()],
                self.dset['data/images'])

            now = datetime.now(timezone.utc)
            libver = 'latest'
            h5fn = path.join(
                path.dirname(dname), path.basename(dname).rstrip('.h5') +
                '-{:.3f}mm'.format(radius/1000) + '.h5')

            wavelength = self.dset['wavelength'][()]
            dm_serial = self.dset['dm/serial'][()]
            dm_transform = self.dset['dm/transform'][()]
            dm_transform = self.dset['dm/transform'][()]
            cam_pixel_size = self.dset['cam/pixel_size'][()]
            cam_serial = self.dset['cam/serial'][()]
            dmplot_txs = self.dset['dmplot/txs'][()]
            with h5py.File(h5fn, 'w', libver=libver) as h5f:
                self.write_h5_header(h5f, libver, now)

                h5f['data/name'] = dname
                h5f['data/md5'] = version.hash_file(dname)
                h5f['data/wavelength'] = wavelength
                h5f['data/wavelength'].attrs['units'] = 'nm'
                h5f['dm/serial'] = dm_serial
                h5f['dm/transform'] = dm_transform
                h5f['cam/serial'] = cam_serial
                h5f['cam/pixel_size'] = cam_pixel_size
                h5f['cam/pixel_size'].attrs['units'] = 'um'
                h5f['dmplot/txs'] = dmplot_txs

                h5f['interf/img'] = self.dsetpars.img
                h5f['interf/P'] = self.dsetpars.P
                h5f['interf/shape'] = self.dsetpars.shape
                for k, p in enumerate(self.dsetpars.cam_grid):
                    h5f['interf/cam_grid{}'.format(k)] = p
                for k, p in enumerate(self.dsetpars.ft_grid):
                    h5f['interf/ft_grid{}'.format(k)] = p
                h5f['interf/fimg'] = self.dsetpars.fimg
                h5f['interf/logf2'] = self.dsetpars.logf2
                h5f['interf/f0'] = self.dsetpars.f0
                h5f['interf/f1'] = self.dsetpars.f1
                h5f['interf/f3'] = self.dsetpars.f3
                h5f['interf/ext3'] = self.dsetpars.ext3
                h5f['interf/f4'] = self.dsetpars.f4
                h5f['interf/dd0'] = self.dsetpars.dd0
                h5f['interf/dd1'] = self.dsetpars.dd1
                h5f['interf/ext4'] = self.dsetpars.ext4
                h5f['interf/gp'] = self.dsetpars.gp
                h5f['interf/mag'] = self.dsetpars.mag
                h5f['interf/wrapped'] = self.dsetpars.wrapped
                h5f['interf/unwrapped'] = self.dsetpars.unwrapped

                h5f['calib/H'] = H
                h5f['calib/mvaf'] = mvaf
                h5f['calib/phi0'] = phi0
                h5f['calib/z0'] = z0
                h5f['calib/C'] = C
                h5f['calib/alpha'] = alpha
                h5f['calib/lambda1'] = lambda1
                cart.save_h5py(h5f, 'cart')

            self.shared.oq.put(('OK', h5fn, mvaf.mean()))
        except Exception as e:
            self.shared.oq.put((str(e),))

    def run_centre(self, dname):
        if self.open_dset(dname):
            return

        names = self.dset['align/names'][()]
        if 'centre' not in names.split(','):
            self.shared.oq.put(('centre measurement is missing',))
            return

        try:
            centre = self.pull('align', names.index('centre'))
            zero = self.pull('data', 0)

            cross = estimate_aperture_centre(
                self.dsetpars.dd0, self.dsetpars.dd1,
                zero[-4], zero[-1], centre[-4], centre[-1])
            self.shared.oq.put(('OK', cross[0], cross[1]))
        except Exception as e:
            self.shared.oq.put((str(e),))

    def run_plot(self, dname, ind, centre, radius):
        if self.open_dset(dname):
            return

        t1 = self.dset['align/U'].shape[1]
        t2 = self.dset['data/U'].shape[1]
        if ind < 0 or ind > t1 + t2:
            self.shared.oq.put((
                'index must be within {} and {}'.format(0, t1 + t2 - 1),))
        if ind < t1:
            addr = 'align'
        else:
            addr = 'data'
            ind -= t1
        try:
            img, mag, ext4, wrapped, unwrapped = self.pull(
                addr, ind, centre, radius)

            if img.max() == self.cam.get_image_max():
                self.shared.cam_sat.value = 1
            else:
                self.shared.cam_sat.value = 0
            self.shared.cam[:] = img[:]
            self.shared.u[:] = self.dset[addr + '/U'][:, ind]
            self.shared.wrapped_buf[:wrapped.nbytes] = wrapped.tobytes()
            for i in range(4):
                self.shared.mag_ext[i] = ext4[i]/1000
            self.shared.mag_shape[:] = mag.shape[:]
            self.shared.unwrapped_buf[:unwrapped.nbytes] = unwrapped.tobytes()
            self.shared.oq.put(('OK',))
        except Exception as e:
            self.shared.oq.put((str(e),))

    def write_h5_header(self, h5f, libver, now):
        libver = 'latest'

        h5f['datetime'] = now.isoformat()

        # save HDF5 library info
        h5f['h5py/libver'] = libver
        h5f['h5py/api_version'] = h5py.version.api_version
        h5f['h5py/version'] = h5py.version.version
        h5f['h5py/hdf5_version'] = h5py.version.hdf5_version
        h5f['h5py/info'] = h5py.version.info

        # save dmlib info
        h5f['dmlib/__date__'] = version.__date__
        h5f['dmlib/__version__'] = version.__version__
        h5f['dmlib/__commit__'] = version.__commit__

    def run_dataacq(self, wavelength, dmplot_txs, sleep=.1):
        cam = self.cam
        dm = self.dm
        shared = self.shared

        Ualign = []
        align_names = []
        for name in ('centre', 'cross', 'x', 'rim', 'checker', 'arrows'):
            try:
                Ualign.append(dm.preset(name, 0.7).reshape(-1, 1))
                align_names.append(name)
            except Exception:
                pass
        if len(Ualign) > 0:
            Ualign = np.hstack(Ualign)
        else:
            Ualign = np.zeros((dm.size(), 0))

        libver = 'latest'
        now = datetime.now(timezone.utc)
        h5fn = now.astimezone().strftime('%Y%m%d_%H%M%S.h5')
        dmsn = dm.get_serial_number()
        if dmsn:
            h5fn = dmsn + '_' + h5fn

        U = np.hstack((
                np.zeros((dm.size(), 1)),
                np.kron(np.eye(dm.size()), np.linspace(-.7, .7, 5)),
                np.zeros((dm.size(), 1))
                ))

        with h5py.File(h5fn, 'w', libver=libver) as h5f:
            self.write_h5_header(h5f, libver, now)

            h5f['dmplot/txs'] = dmplot_txs

            h5f['cam/serial'] = cam.get_serial_number()
            h5f['cam/camera_info'] = str(cam.get_camera_info())
            h5f['cam/sensor_info'] = str(cam.get_sensor_info())
            h5f['cam/pixel_size'] = cam.get_pixel_size()
            h5f['cam/pixel_size'].attrs['units'] = 'um'
            h5f['cam/exposure'] = cam.get_exposure()
            h5f['cam/exposure'].attrs['units'] = 'ms'
            h5f['cam/dtype'] = cam.get_image_dtype()
            h5f['cam/max'] = cam.get_image_max()

            h5f['wavelength'] = wavelength
            h5f['wavelength'].attrs['units'] = 'nm'
            h5f['sleep'] = sleep
            h5f['sleep'].attrs['units'] = 's'

            h5f['dm/serial'] = dm.get_serial_number()
            h5f['dm/transform'] = dm.get_transform()

            h5f['data/U'] = U
            h5f['data/U'].dims[0].label = 'actuators'
            h5f['data/U'].dims[1].label = 'step'
            h5f.create_dataset(
                'data/images', (U.shape[1],) + cam.shape(),
                dtype=cam.get_image_dtype())
            h5f['data/images'].dims[0].label = 'step'
            h5f['data/images'].dims[1].label = 'height'
            h5f['data/images'].dims[1].label = 'width'

            h5f['align/U'] = Ualign
            h5f['align/U'].dims[0].label = 'actuators'
            h5f['align/U'].dims[1].label = 'step'
            h5f.create_dataset(
                'align/images', (Ualign.shape[1],) + cam.shape(),
                dtype=cam.get_image_dtype())
            h5f['align/images'].dims[0].label = 'step'
            h5f['align/images'].dims[1].label = 'height'
            h5f['align/images'].dims[1].label = 'width'
            h5f['align/names'] = ','.join(align_names)

            tot = U.shape[1] + Ualign.shape[1]
            count = [0]

            todo = ((Ualign, 'align/images'), (U, 'data/images'))
            for U1, imaddr in todo:
                for i in range(U1.shape[1]):
                    try:
                        dm.write(U1[:, i])
                        time.sleep(sleep)
                        img = cam.grab_image()
                    except Exception as e:
                        shared.oq.put((str(e),))
                        return
                    if img.max() == cam.get_image_max():
                        shared.cam_sat.value = 1
                    else:
                        shared.cam_sat.value = 0
                    h5f[imaddr][i, ...] = img
                    shared.u[:] = U1[:, i]
                    shared.cam[:] = img
                    shared.oq.put(('OK', count[0], tot))
                    print('run_dataacq', 'iteration')

                    stopcmd = shared.iq.get()[1]
                    shared.oq.put('')

                    if stopcmd:
                        print('run_dataacq', 'stopcmd')
                        h5f.close()
                        try:
                            os.remove(h5fn)
                        except OSError:
                            pass
                        return
                    else:
                        print('run_dataacq', 'continue')

                    count[0] += 1

        print('run_dataacq', 'finished')
        shared.oq.put(('finished', h5fn))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

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
    parser.add_argument('--dm-name', type=str, default=None)
    parser.add_argument('--cam-name', type=str, default=None)
    args = parser.parse_args(args[1:])

    cam, dm = open_hardware(args)

    def set_dm(t):
        args.dm_name = t

    def set_cam(t):
        args.cam_name = t

    loops = zip(
        (cam, dm), ('camera', 'dm'),
        (args.cam_name, args.dm_name),
        (set_cam, set_dm))
    for dev, name, def1, set1 in loops:
        devs = dev.get_devices()
        if len(devs) == 0:
            e = QErrorMessage()
            e.showMessage('no {} found'.format(name))
            sys.exit(app.exec_())
        elif def1 is None or def1 not in devs:
            if len(devs) == 1:
                set1(devs[0])
            else:
                item, ok = QInputDialog.getItem(
                    None, '', 'select ' + name + ':', devs, 0, False)
                if ok and item:
                    set1(item)
                else:
                    sys.exit(0)

    shared = Shared(cam, dm)
    dm.close()
    cam.close()

    p = Process(name='worker', target=run_worker, args=(shared, args))
    p.start()

    control = Control(p, shared)
    control.show()

    exit = app.exec_()

    shared.iq.put('STOP')
    p.join()
    sys.exit(exit)
