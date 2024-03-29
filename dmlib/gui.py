#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Array, Process, Queue, Value
from os import path

import h5py
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import (FigureCanvas,
                                                NavigationToolbar2QT)
from matplotlib.figure import Figure
from numpy.linalg import norm
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog, QDoubleSpinBox,
                             QFileDialog, QFrame, QGridLayout, QGroupBox,
                             QInputDialog, QLabel, QLineEdit, QMainWindow,
                             QMessageBox, QPushButton, QShortcut, QSizePolicy,
                             QSplitter, QStyleFactory, QTabWidget, QToolBox,
                             QVBoxLayout)

from dmlib.calibration import RegLSCalib, make_normalised_input_matrix
from dmlib.control import ZernikeControl, get_noll_indices
from dmlib.core import (add_cam_parameters, add_dm_parameters,
                        add_log_parameters, get_suitable_dmplot, h5_read_str,
                        h5_store_str, hash_file, open_cam, open_dm,
                        setup_logging, spawn_file, write_h5_header)
from dmlib.dmplot import DMPlot
from dmlib.interf import FringeAnalysis
from dmlib.version import __version__
from dmlib.zpanel import MyQIntValidator, ZernikePanel


def conf_mismatch_spawn(h5):
    thisfile = path.join(path.dirname(path.abspath(__file__)), 'gui.py')
    cmds = [
        sys.executable,
        thisfile,
        '--config-like',
        h5,
        '--dm-driver',
        'sim',
        '--dm-name',
        'simdm0',
        '--cam-driver',
        'sim',
        '--cam-name',
        'simcam0',
    ]
    logging.getLogger('conf_mismatch_spawn').info(' '.join(cmds))
    subprocess.Popen(cmds)


class GetNollIndices(QDialog):
    def update_noll(self):
        indices = get_noll_indices(self.pars)
        indices.sort()
        self.indices = indices
        self.lenoll.setText(','.join([str(s) for s in indices]))

    def __init__(self, pars, max_zernike, parent=None):
        super().__init__(parent)

        self.pars = pars
        self.max_zernike = max_zernike

        lay = QGridLayout()
        self.setLayout(lay)

        lenoll = QLineEdit()
        lenoll.setFixedWidth(300)
        lenoll.setReadOnly(True)
        self.lenoll = lenoll

        def make_val_int(le, val, name):
            def f():
                newval = int(le.text())
                self.pars[name] = newval
                val.setFixup(newval)
                self.update_noll()

            return f

        def help_int(le, name):
            vv = MyQIntValidator()
            vv.setFixup(int(self.pars[name]))
            vv.setBottom(1)
            vv.setTop(max_zernike)
            le.setValidator(vv)
            le.editingFinished.connect(make_val_int(le, vv, name))

        def make_val_list(le, name):
            def f():
                old = self.pars[name]
                try:
                    tmp = [int(s) for s in le.text().split(',') if s != '']
                    tmp = [s for s in tmp if s >= 1 and s <= max_zernike]
                except Exception:
                    tmp = old
                self.pars[name] = tmp
                le.blockSignals(True)
                le.setText(', '.join([str(c) for c in tmp]))
                le.blockSignals(False)
                self.update_noll()

            return f

        def help_list(le, name):
            le.setText(', '.join([str(c) for c in pars[name]]))
            le.editingFinished.connect(make_val_list(le, name))

        leinc = QLineEdit()
        help_list(leinc, 'include')

        leexc = QLineEdit()
        help_list(leexc, 'exclude')

        lemin = QLineEdit()
        lemin.setText(str(pars['min']))
        help_int(lemin, 'min')

        lemax = QLineEdit()
        lemax.setText(str(pars['max']))
        help_int(lemax, 'max')

        lay.addWidget(QLabel('Noll #'), 0, 0)
        lay.addWidget(lenoll, 0, 1)
        lay.addWidget(QLabel('include'), 1, 0)
        lay.addWidget(leinc, 1, 1)
        lay.addWidget(QLabel('exclude'), 2, 0)
        lay.addWidget(leexc, 2, 1)
        lay.addWidget(QLabel('min'), 3, 0)
        lay.addWidget(lemin, 3, 1)
        lay.addWidget(QLabel('max'), 4, 0)
        lay.addWidget(lemax, 4, 1)

        self.update_noll()


class Control(QMainWindow):
    def __init__(self,
                 worker,
                 shared,
                 cam_name,
                 dm_name,
                 dmplot,
                 min_delay,
                 parent=None):
        super().__init__()
        self.log = logging.getLogger(self.__class__.__name__)

        self.min_delay = min_delay
        self.align_bauto = None
        self.zernikePanel = None
        self.can_close = True

        self.worker = worker
        self.shared = shared
        self.shared.make_static()
        self.cam_name = cam_name
        self.dm_name = dm_name
        self.dmplot = dmplot

        self.setWindowTitle('DM calibration ' + __version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        central = QSplitter(Qt.Horizontal)

        self.toolbox = QToolBox()
        self.make_toolbox()
        central.addWidget(self.toolbox)

        self.tabs = QTabWidget()
        self.make_panel_align()
        self.make_panel_dataacq()
        self.make_panel_test()

        def change_tab():
            def f(ind):
                if ind == 0 and self.toolbox.count() == 1:
                    self.toolbox.addItem(self.tool_cam, self.tool_cam_name)
                    for g in self.tool_dm_toggle:
                        g.setEnabled(1)
                elif self.toolbox.count() == 2:
                    self.toolbox.removeItem(1)
                    for g in self.tool_dm_toggle:
                        g.setEnabled(0)

            return f

        self.tabs.currentChanged.connect(change_tab())
        central.addWidget(self.tabs)

        self.setCentralWidget(central)

    def closeEvent(self, event):
        if self.can_close:
            if self.zernikePanel:
                self.zernikePanel.close()
            event.accept()
        else:
            event.ignore()

    def make_toolbox(self):
        self.make_tool_dm()
        self.make_tool_cam()

    def make_tool_cam(self):
        tool_cam = QFrame()
        layout = QVBoxLayout()
        tool_cam.setLayout(layout)

        def cam_get(cmd):
            def f():
                self.shared.iq.put((cmd, ))
                return self.shared.oq.get()

            return f

        def cam_set(cmd):
            def f(x):
                self.shared.iq.put((cmd, x))
                return self.shared.oq.get()

            return f

        def up(l1, s, txt, r, v):
            rg = r()
            l1.setText(f'min: {rg[0]}<br>max: {rg[1]}<br>step: {rg[2]}')
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
        up(l1, s1, 'Exposure [ms]', cam_get('get_exposure_range'),
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
        up(l2, s2, 'FPS', cam_get('get_framerate_range'),
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

        s1.editingFinished.connect(
            f1(cam_set('set_exposure'), s1, l2, s2, 'FPS',
               cam_get('get_framerate_range'), cam_get('get_framerate')))
        s2.editingFinished.connect(
            f1(cam_set('set_framerate'), s2, l1, s1, 'exposure',
               cam_get('get_exposure_range'), cam_get('get_exposure')))

        self.tool_cam = tool_cam
        self.tool_cam_name = 'cam: ' + self.cam_name
        self.toolbox.addItem(self.tool_cam, self.tool_cam_name)

    def update_tool_dm(self):
        self.dmplot.update(self.shared.u)

    def write_dm(self, u=None):
        if u is not None:
            self.shared.u[:] = u[:]
        self.update_tool_dm()
        self.shared.iq.put(('write', ))
        self.shared.oq.get()

    def make_tool_dm(self):
        tool_dm = QFrame()
        central = QSplitter(Qt.Vertical)
        layout = QVBoxLayout()

        self.dm_fig = FigureCanvas(Figure(figsize=(3, 2)))
        self.dm_ax = self.dm_fig.figure.add_axes([0.0, 0.0, 0.70, 1.0])
        ax2 = self.dm_fig.figure.add_axes([0.70, 0.2, 0.05, 0.5])
        self.dmplot.setup_pattern(self.dm_ax, ax2)
        self.dmplot.install_select_callback(self.dm_ax, self.shared.u, self,
                                            self.write_dm)
        self.dm_fig.figure.subplots_adjust(left=.125,
                                           right=.9,
                                           bottom=.1,
                                           top=.9,
                                           wspace=0.45,
                                           hspace=0.45)
        central.addWidget(self.dm_fig)

        g1 = QGroupBox('Plot transforms')
        gl1 = QGridLayout()
        flipx = QPushButton('flipx')
        gl1.addWidget(flipx, 0, 0)
        flipy = QPushButton('flipy')
        gl1.addWidget(flipy, 0, 1)
        brotate = QPushButton('rotate')
        gl1.addWidget(brotate, 1, 0)
        bcmap = QPushButton('cmap')
        gl1.addWidget(bcmap, 1, 1)
        g1.setLayout(gl1)
        central.addWidget(g1)

        def f4(n):
            def f():
                if self.align_bauto:
                    self.align_bauto.setChecked(False)
                self.shared.iq.put(('preset', n, self.pokemag[0]))
                self.shared.oq.get()
                self.write_dm(None)

            return f

        g2 = QGroupBox('Actuators')
        gl2 = QGridLayout()
        reset = QPushButton('reset')
        gl2.addWidget(reset, 0, 0)
        setall = QPushButton('set all')
        gl2.addWidget(setall, 1, 0)
        loadflat = QPushButton('load flat')
        gl2.addWidget(loadflat, 1, 1)
        i = 2
        j = 0
        for name in ('centre', 'cross', 'x', 'rim', 'checker', 'arrows'):
            b = QPushButton(name)
            gl2.addWidget(b, i, j)
            if j == 1:
                i += 1
                j = 0
            else:
                j += 1
            b.clicked[bool].connect(f4(name))
        g2.setLayout(gl2)
        central.addWidget(g2)

        def f2():
            def f():
                self.shared.u[:] = 0
                self.write_dm()

            return f

        def f3():
            def f():
                val, ok = QInputDialog.getDouble(
                    self, 'Set all actuators',
                    'Set all actuators; Range [-1, 1]', 0., -1., 1., 4)
                if ok:
                    self.shared.u[:] = val
                    self.write_dm()

            return f

        def f4():
            def f():
                fileName, _ = QFileDialog.getOpenFileName(
                    self,
                    'Select factory flat file',
                    filter='TXT (*.txt);;All Files (*)')
                if fileName:
                    try:
                        uflat = np.loadtxt(fileName, delimiter='\n')
                        uflat = 2 * (uflat**2) - 1
                        assert (self.shared.u.size == uflat.size)
                        self.shared.u[:] = uflat
                        self.write_dm()
                    except Exception:
                        pass

            return f

        reset.clicked.connect(f2())
        setall.clicked.connect(f3())
        loadflat.clicked.connect(f4())

        def fbrotate():
            prev = [.0]

            def f():
                val, ok = QInputDialog.getDouble(
                    self, 'Rotate actuators plot',
                    'Rotate actuators plot [deg]', 180 / np.pi * prev[0])
                if ok:
                    prev[0] = np.pi / 180 * val
                    self.dmplot.rotate(prev[0])
                    self.update_tool_dm()

            return f

        def f4(cb, b):
            def f():
                cb(b.isChecked())
                self.update_tool_dm()

            return f

        def fcmap():
            prev = [True]

            def f():
                prev[0] = not prev[0]
                self.dmplot.set_abs_cmap(prev[0])
                self.update_tool_dm()

            return f

        flipx.setCheckable(True)
        flipy.setCheckable(True)
        flipx.clicked.connect(f4(self.dmplot.flipx, flipx))
        flipy.clicked.connect(f4(self.dmplot.flipy, flipy))
        brotate.clicked.connect(fbrotate())
        bcmap.clicked.connect(fcmap())

        tool_dm.setLayout(layout)
        layout.addWidget(central)

        self.write_dm(None)

        self.tool_dm = tool_dm
        self.tool_dm_name = 'dm: ' + self.dm_name
        self.tool_dm_toggle = (g1, g2)
        self.toolbox.addItem(self.tool_dm, self.tool_dm_name)

    def make_panel_align(self):
        frame = QFrame()
        self.align_fig = FigureCanvas(Figure(figsize=(7, 5)))
        self.align_nav = NavigationToolbar2QT(self.align_fig, frame)
        layout = QGridLayout()
        frame.setLayout(layout)
        layout.addWidget(self.align_nav, 0, 0, 1, 0)
        layout.addWidget(self.align_fig, 1, 0, 1, 0)

        self.tabs.addTab(frame, 'align')
        self.tabs.setTabToolTip(
            0,
            'Align the DM, test actuators, and change the DM plot orientation')

        self.align_axes = self.align_fig.figure.subplots(2, 3)
        self.align_fig.figure.subplots_adjust(left=.125,
                                              right=.9,
                                              bottom=.1,
                                              top=.9,
                                              wspace=0.45,
                                              hspace=0.45)
        self.align_axes[0, 0].set_title('cam')
        self.align_axes[0, 1].set_title('FT')
        self.align_axes[0, 2].set_title('1st order')
        self.align_axes[1, 0].set_title('magnitude')
        self.align_axes[1, 1].set_title('wrapped phi')
        self.align_axes[1, 2].set_title('unwrapped phi')

        brun = QPushButton('run')
        bstop = QPushButton('stop')
        layout.addWidget(brun, 2, 0)
        layout.addWidget(bstop, 2, 1)
        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 3, 0, 1, 2)

        botrow = QFrame()
        botlay = QGridLayout()
        botrow.setLayout(botlay)
        botrow.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        bauto = QCheckBox('auto')
        bauto.setToolTip('Lock first order position automatically')
        bauto.setChecked(True)
        self.align_bauto = bauto
        botlay.addWidget(bauto, 0, 0)
        brepeat = QCheckBox('repeat')
        brepeat.setToolTip('Acquire data continuously or one time only')
        botlay.addWidget(brepeat, 0, 2)

        bsleep = QPushButton('delay')
        bsleep.setToolTip(
            'Interval between setting the DM and acquiring an image')
        bpoke = QPushButton('poke')
        bpoke.setToolTip('Set a custom magnitude for the pokes')
        botlay.addWidget(bsleep, 1, 0)
        botlay.addWidget(bpoke, 1, 1)
        bunwrap = QCheckBox('unwrap')
        bunwrap.setChecked(True)
        bunwrap.setToolTip('Perform phase extraction & unwrapping')
        botlay.addWidget(bunwrap, 0, 1)

        layout.addWidget(botrow, 4, 0, 1, 2)

        disables = [
            self.toolbox, brun, bauto, brepeat, bsleep, bunwrap,
            self.align_nav, bpoke
        ]
        pokemag = [.7]
        self.pokemag = pokemag
        sleepmag = [.5]
        self.sleepmag = sleepmag

        listener = AlignListener(self.shared, self.sleepmag)

        def disable():
            self.can_close = False
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
            self.can_close = True

        def f1():
            def f():
                val, ok = QInputDialog.getDouble(
                    self,
                    'Delay between DM write and camera read',
                    f'DM write / camera read delay [{self.min_delay} sec, ]',
                    value=self.sleepmag[0],
                    min=self.min_delay,
                    decimals=4)
                if ok:
                    self.sleepmag[0] = val

            return f

        def f3():
            def f(p):
                listener.unwrap = p

            return f

        bsleep.clicked.connect(f1())
        bunwrap.stateChanged.connect(f3())

        def f1():
            def f():
                disable()
                status.setText('Working...')
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
                    status.setText('Stopping...')

            return f

        def f20():
            def f(result):
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

                if result[0] != 'ERR1':
                    a1.imshow(self.shared.cam,
                              extent=self.shared.cam_ext,
                              origin='lower')
                    a1.set_xlabel('mm')
                    if self.shared.cam_sat.value:
                        a1.set_title('cam SAT')
                    else:
                        a1.set_title(f'cam {self.shared.cam.min(): 3d} ' +
                                     f'{self.shared.cam.max(): 3d}')

                if listener.unwrap and result[0] not in ('ERR1', 'ERR2'):
                    a2.imshow(self.shared.ft,
                              extent=self.shared.ft_ext,
                              origin='lower')
                    a2.set_xlabel('1/mm')
                    a2.set_title('FT')

                    a2.plot(self.shared.fxcfyc[0] * 1e3,
                            self.shared.fxcfyc[1] * 1e3,
                            'rx',
                            markersize=6)

                    a2.plot(-self.shared.fxcfyc[0] * 1e3,
                            -self.shared.fxcfyc[1] * 1e3,
                            'rx',
                            markersize=6)

                if listener.unwrap and result[0] not in ('ERR1', 'ERR2',
                                                         'ERR3'):
                    fstord, mag, wrapped, unwrapped = self.shared.get_phase()

                    a3.imshow(fstord,
                              extent=self.shared.fstord_ext,
                              origin='lower')
                    a3.set_xlabel('1/mm')
                    a3.set_title(f'1st order {fstord.shape}')

                    a4.imshow(mag, extent=self.shared.mag_ext, origin='lower')
                    a4.set_xlabel('mm')
                    a4.set_title('magnitude')

                    a5.imshow(wrapped,
                              extent=self.shared.mag_ext,
                              origin='lower')
                    a5.set_xlabel('mm')
                    a5.set_title('wrapped phi')

                    a6.imshow(unwrapped,
                              extent=self.shared.mag_ext,
                              origin='lower')
                    a6.set_xlabel('mm')
                    a6.set_title('unwrapped phi')

                a6.figure.canvas.draw()
                self.update_tool_dm()
                if listener.repeat:
                    if result[0] == 'OK':
                        status.setText('Working...')
                    else:
                        status.setText(f'Working... {result[2]}')
                else:
                    if result[0] == 'OK':
                        status.setText('Stopped')
                    else:
                        status.setText(result[2])
                    enable()

            return f

        def fpoke():
            def f():
                val, ok = QInputDialog.getDouble(
                    self,
                    'Maximum poke amplitude',
                    'Maximum poke amplitude [0, 1]',
                    value=self.pokemag[0],
                    min=0.,
                    max=1.,
                    decimals=2)
                if ok:
                    self.pokemag[0] = val

            return f

        listener.sig_update.connect(f20())
        brun.clicked.connect(f1())
        bstop.clicked.connect(f4())
        bauto.stateChanged.connect(f2())
        brepeat.stateChanged.connect(f3())
        bpoke.clicked.connect(fpoke())

    def make_panel_dataacq(self):
        frame = QFrame()
        self.dataacq_fig = FigureCanvas(Figure(figsize=(7, 5)))
        self.dataacq_nav = NavigationToolbar2QT(self.dataacq_fig, frame)
        layout = QGridLayout()
        frame.setLayout(layout)
        layout.addWidget(self.dataacq_nav, 0, 0, 1, 0)
        layout.addWidget(self.dataacq_fig, 1, 0, 1, 0)

        self.tabs.addTab(frame, 'calibration')
        self.tabs.setTabToolTip(
            1, ('Acquire calibration data, define the DM pupil, ' +
                'and compute calibrations'))

        self.dataacq_axes = self.dataacq_fig.figure.subplots(2, 2)
        self.dataacq_fig.figure.subplots_adjust(left=.125,
                                                right=.9,
                                                bottom=.1,
                                                top=.9,
                                                wspace=0.45,
                                                hspace=0.45)
        self.dataacq_axes[0, 0].set_title('cam')
        self.dataacq_axes[1, 0].set_title('wrapped phi')
        self.dataacq_axes[1, 1].set_title('unwrapped phi')

        brun = QPushButton('run')
        brun.setToolTip('Collect new calibration data')
        bstop = QPushButton('stop')
        bwavelength = QPushButton('wavelength')
        bwavelength.setToolTip('Calibration laser wavelength')
        bpoke = QPushButton('poke')
        bpoke.setToolTip('Set a custom magnitude for the pokes')
        bsleep = QPushButton('delay')
        bsleep.setToolTip(
            'Interval between setting the DM and acquiring an image')
        layout.addWidget(brun, 2, 0)
        layout.addWidget(bstop, 2, 1)
        layout.addWidget(bwavelength, 2, 2)
        layout.addWidget(bpoke, 2, 3)
        layout.addWidget(bsleep, 2, 4)

        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 3, 0, 1, 3)

        bplot = QPushButton('open')
        bplot.setToolTip((
            'Open an existing calibration file or plot the n-th measurement ' +
            'of the current dataset'))
        bprev = QPushButton('prev')
        bprev.setToolTip('Plot previous measurement')
        bnext = QPushButton('next')
        bnext.setToolTip('Plot next measurement')
        layout.addWidget(bplot, 4, 0)
        layout.addWidget(bprev, 4, 1)
        layout.addWidget(bnext, 4, 2)

        baperture = QPushButton('aperture')
        baperture.setToolTip('Define the pupil size over the DM surface')
        layout.addWidget(baperture, 5, 0)
        bcalibrate = QPushButton('calibrate')
        bcalibrate.setToolTip('Compute a calibration file')
        layout.addWidget(bcalibrate, 5, 1)
        bclear = QPushButton('clear')
        layout.addWidget(bclear, 5, 4)

        disables = [
            self.toolbox, brun, bwavelength, bplot, self.dataacq_nav, bprev,
            bnext, baperture, bcalibrate, bclear, bpoke, bsleep
        ]

        wavelength = []
        dataset = []
        lastind = []
        centre = [None]
        radius = [.0]
        listener = DataAcqListener(self.shared, wavelength, self.dmplot,
                                   self.pokemag, self.sleepmag)

        def clearup(clear_status=False):
            wavelength.clear()
            dataset.clear()
            lastind.clear()
            centre[0] = None
            radius[0] = .0

            if clear_status:
                status.setText('')
                status.setToolTip('')
                self.dataacq_axes[0, 0].clear()
                self.dataacq_axes[0, 1].clear()
                self.dataacq_axes[1, 0].clear()
                self.dataacq_axes[1, 1].clear()
                self.dataacq_fig.figure.canvas.draw()

        def disable():
            self.can_close = False
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
            self.can_close = True

        def fpoke():
            def f():
                val, ok = QInputDialog.getDouble(
                    self,
                    'Maximum poke amplitude',
                    'Maximum poke amplitude [0, 1]',
                    value=self.pokemag[0],
                    min=0.,
                    max=1.,
                    decimals=2)
                if ok:
                    self.pokemag[0] = val

            return f

        def f0():
            def f():
                if wavelength:
                    wl = wavelength[0]
                else:
                    wl = 775.
                val, ok = QInputDialog.getDouble(self,
                                                 'Wavelength',
                                                 'wavelength [nm]',
                                                 wl,
                                                 decimals=1)
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
                self.dataacq_fig.figure.canvas.draw()
                disable()

                listener.run = True
                listener.start()

            return f

        def check_err():
            reply = self.shared.oq.get()
            if reply[0].startswith('Configuration mismatch;'):
                try:
                    conf_mismatch_spawn(reply[1])
                except Exception as e:
                    self.log.error(str(e))
                status.setText(reply[0])
                return -1
            elif reply[0] != 'OK':
                status.setText(reply[0])
                return -1
            else:
                return reply[1:]

        def bootstrap():
            if not dataset:
                fileName, _ = QFileDialog.getOpenFileName(
                    self, 'Select dataset', filter='H5 (*.h5);;All Files (*)')
                if not fileName:
                    return False
                else:
                    dataset.append(fileName)
                    status.setToolTip(path.abspath(fileName))
                    return True
            else:
                return True

        def f3(offset=None):
            theta = np.linspace(0, 2 * np.pi, 96)

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
                        f'time step [0, {ndata[0] - 1}]', last, 0,
                        ndata[0] - 1)
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

                self.shared.iq.put(('plot', dataset[0], val, radius[0]))
                if check_err() == -1:
                    return

                a1 = self.dataacq_axes[0, 0]
                a2 = self.dataacq_axes[0, 1]
                a3 = self.dataacq_axes[1, 0]
                a4 = self.dataacq_axes[1, 1]

                a1.clear()
                a3.clear()
                a4.clear()

                a1.imshow(self.shared.cam,
                          extent=self.shared.cam_ext,
                          origin='lower')
                a1.set_xlabel('mm')
                if self.shared.cam_sat.value:
                    a1.set_title('cam SAT')
                else:
                    a1.set_title(f'cam {self.shared.cam.min(): 3d} ' +
                                 f'{self.shared.cam.max(): 3d}')

                data = self.shared.get_phase()
                wrapped, unwrapped = data[2:]

                a2.clear()
                a2.plot(self.shared.u)
                a2.set_title(f'acts [{self.shared.u.min():+.1f}, ' +
                             f'{self.shared.u.max():+.1f}]')
                a2.set_ylim([-1, 1])
                self.dmplot.update(self.shared.u)

                a3.imshow(wrapped, extent=self.shared.mag_ext, origin='lower')
                a3.set_xlabel('mm')
                a3.set_title('wrapped phi')

                a4.imshow(unwrapped,
                          extent=self.shared.mag_ext,
                          origin='lower')
                a4.set_xlabel('mm')
                a4.set_title('unwrapped phi')
                if centre[0] is not None:
                    a4.plot(centre[0][0] / 1000, centre[0][1] / 1000, 'rx')
                if radius[0] > 0. and centre[0] is not None:
                    a4.plot(
                        centre[0][0] / 1000 + radius[0] / 1000 * np.cos(theta),
                        centre[0][1] / 1000 + radius[0] / 1000 * np.sin(theta),
                        'r')

                a4.figure.canvas.draw()

                status.setText(
                    path.basename(dataset[0]) + f' {val}/{ndata[0] - 1}')

            return f

        def f4():
            drawf = f3(0)

            def f():
                if not bootstrap():
                    return False

                if radius:
                    rad = radius[0] / 1000
                else:
                    rad = 2.1

                if self.shared.cam_ext[1] > 0:
                    radmax = min(
                        (self.shared.cam_ext[1], self.shared.cam_ext[3]))
                else:
                    radmax = 10.

                val, ok = QInputDialog.getDouble(
                    self, 'Aperture radius',
                    f'Radius [mm] (max {radmax:.3f} mm)<br>' +
                    'As seen by the camera (including magnification)', rad, 0.,
                    radmax, 6)
                if ok and val >= 0.:
                    if radius:
                        radius[0] = val * 1000
                    else:
                        radius.append(val * 1000)

                    self.shared.iq.put(('aperture', dataset[0], radius[0]))
                    ndata = check_err()
                    if ndata == -1:
                        radius[0] = 0.
                        centre[0] = None
                        return False
                    centre[0] = ndata[0]

                    drawf()
                    return True
                else:
                    return False

            return f

        clistener = CalibListener(self.shared, dataset, centre, radius,
                                  self.dmplot)

        def f6():
            def f(reply):
                status.setText(reply[1])
                if reply[0] == 'OK' or reply[0] == 'ERR':
                    if reply[0] == 'OK':
                        try:
                            spawn_file(path.abspath(reply[2]))
                        except Exception:
                            pass
                    enable()
                    bstop.setEnabled(True)

            return f

        clistener.sig_update.connect(f6())

        def f5():
            setup_aperture = f4()

            def f():
                disable()
                bstop.setEnabled(False)

                ok = True
                if radius[0] <= 0 or centre[0] is None:
                    ok = setup_aperture()

                if ok and radius[0] > 0 and centre[0] is not None:
                    status.setText(
                        'Computing calibration (this can take long) ...')
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
                a1.clear()
                a1.imshow(self.shared.cam,
                          extent=self.shared.cam_ext,
                          origin='lower')
                a1.set_xlabel('mm')
                if self.shared.cam_sat.value:
                    a1.set_title('cam SAT')
                else:
                    a1.set_title(f'cam {self.shared.cam.min(): 3d} ' +
                                 f'{self.shared.cam.max(): 3d}')

                a2 = self.dataacq_axes[0, 1]
                a2.clear()
                a2.set_title('acts')
                a2.plot(self.shared.u)
                self.dmplot.update(self.shared.u)
                a2.set_ylim([-1, 1])

                if msg[0] == 'OK':
                    status.setText(f'{msg[1] + 1}/{msg[2]}')
                elif msg[0] == 'finished':
                    if dataset:
                        dataset[0] = msg[1]
                    else:
                        dataset.append(msg[1])
                    status.setText('Saved calibration data file ' + msg[1])
                    try:
                        spawn_file(path.abspath(msg[1]))
                    except Exception:
                        QMessageBox.information(self,
                                                'Saved calibration data file',
                                                path.abspath(msg[1]))
                    status.setToolTip(path.abspath(msg[1]))
                    enable()
                else:
                    status.setText(msg[0])
                    enable()

                a1.figure.canvas.draw()
                listener.busy = False

            return f

        def f7():
            def f():
                clearup(True)

            return f

        def fsleep():
            def f():
                val, ok = QInputDialog.getDouble(
                    self,
                    'Delay between DM write and camera read',
                    f'DM write / camera read delay [{self.min_delay} sec, ]',
                    value=self.sleepmag[0],
                    min=self.min_delay,
                    decimals=4)
                if ok:
                    self.sleepmag[0] = val

            return f

        brun.clicked.connect(f1())
        bstop.clicked.connect(f2())
        bwavelength.clicked.connect(f0())
        bpoke.clicked.connect(fpoke())
        bplot.clicked.connect(f3())
        bnext.clicked.connect(f3(1))
        bprev.clicked.connect(f3(-1))
        baperture.clicked.connect(f4())
        bcalibrate.clicked.connect(f5())
        bclear.clicked.connect(f7())
        bsleep.clicked.connect(fsleep())

        listener.sig_update.connect(f20())

    def make_panel_test(self):
        frame = QFrame()
        self.test_fig = FigureCanvas(Figure(figsize=(7, 5)))
        self.test_nav = NavigationToolbar2QT(self.test_fig, frame)
        layout = QGridLayout()
        frame.setLayout(layout)
        layout.addWidget(self.test_nav, 0, 0, 1, 0)
        layout.addWidget(self.test_fig, 1, 0, 1, 0)

        self.tabs.addTab(frame, 'test')
        self.tabs.setTabToolTip(
            2, ('Test a calibration file interferometrically'))

        gs = gridspec.GridSpec(2, 1)
        self.test_axes = [
            self.test_fig.figure.add_subplot(gs[0]),
            self.test_fig.figure.add_subplot(gs[1]),
        ]
        self.test_fig.figure.subplots_adjust(left=.125,
                                             right=.9,
                                             bottom=.1,
                                             top=.9,
                                             wspace=0.45,
                                             hspace=0.45)

        brun = QPushButton('run')
        bstop = QPushButton('stop')
        bsleep = QPushButton('delay')
        bsleep.setToolTip(
            'Interval between setting the DM and acquiring an image')
        bzsize = QPushButton('# Zernike')
        bzsize.setToolTip(
            ('Select Zernike coefficients to plot; Blue set by the DM; ' +
             'Orange measured by the interferometer'))
        layout.addWidget(brun, 2, 0)
        layout.addWidget(bstop, 2, 1)
        layout.addWidget(bsleep, 2, 2)
        layout.addWidget(bzsize, 2, 3)
        status = QLabel('')
        status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(status, 3, 0, 1, 3)

        bzernike = QPushButton('open')
        bzernike.setToolTip('Open a calibration file')
        layout.addWidget(bzernike, 4, 0)

        bflat = QCheckBox('flat')
        bnoflat = QPushButton('exclude flat')
        bnoflat.setToolTip('Exclude some Zernike modes from the DM flattening')
        bflat.setChecked(True)
        bflat.setToolTip(
            'Enable or disable the flattening computed at calibration time')
        layout.addWidget(bflat, 4, 1)
        layout.addWidget(bnoflat, 4, 2)
        bclear = QPushButton('clear')
        layout.addWidget(bclear, 4, 3)

        disables = [
            self.toolbox, brun, bflat, bnoflat, bzernike, bclear,
            self.test_nav, bzernike, bsleep, bzsize
        ]
        llistener = LoopListener(self.shared, status, self.sleepmag)
        calib = []
        noll_sel_pars = {
            'min': 1,
            'max': 15,
            'include': [],
            'exclude': [],
        }
        arts = []
        noflat_index = [0]

        def clearup(clear_status=False):
            for c in arts:
                c[1].remove()
            arts.clear()
            calib.clear()
            if self.zernikePanel:
                self.zernikePanel.close()
                self.zernikePanel = None

            if clear_status:
                status.setText('')
                for ax in self.test_axes:
                    ax.clear()
                self.test_fig.figure.canvas.draw()

        def disable():
            self.can_close = False
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
            self.can_close = True

        def check_err():
            reply = self.shared.oq.get()
            if reply[0].startswith('Configuration mismatch;'):
                try:
                    conf_mismatch_spawn(reply[1])
                except Exception as e:
                    self.log.error(str(e))
                status.setText(reply[0])
                return -1
            elif reply[0] != 'OK':
                status.setText(reply[0])
                return -1
            else:
                return reply[1:]

        def bootstrap():
            if not calib:
                fileName, _ = QFileDialog.getOpenFileName(
                    self,
                    'Select a calibration',
                    filter='H5 (*.h5);;All Files (*)')
                if not fileName:
                    return False
                else:
                    calib.append(fileName)
                    return True
            else:
                return True

        def f2():
            def cb(z):
                nmax = min(self.shared.z_sp.size, z.size)
                self.shared.z_sp[:nmax] = z[:nmax]

            def f():
                disable()
                if not calib or len(calib) == 0:
                    if not bootstrap():
                        enable()
                        return False
                status.setText(f'Loading {path.basename(calib[0])} ...')
                status.repaint()
                self.shared.iq.put(('query_calib', calib[0]))
                ndata = check_err()
                if ndata == -1:
                    clearup()
                    enable()
                    return False
                else:
                    self.dmplot.update_txs(ndata[3])
                    if self.zernikePanel:
                        self.zernikePanel.close()
                    self.zernikePanel = ZernikePanel(ndata[0],
                                                     ndata[1],
                                                     callback=cb)
                    self.zernikePanel.show()
                    status.setText(
                        f'{path.basename(calib[0])} {ndata[2] / 1000:.3f} mm')
                    enable()
                    return True

            return f

        def f3():
            def f():
                clearup(True)

            return f

        def f1():
            redo = f2()

            def f():
                status.setText('Starting (can take some minutes)...')
                if not calib or len(calib) == 0:
                    if not redo():
                        return
                disable()
                self.shared.z_sp *= 0
                llistener.busy = False
                llistener.run = True
                llistener.calib = calib[0]
                llistener.flat = bflat.isChecked()
                llistener.noflat_index = noflat_index[0]
                llistener.start()

            return f

        def make_cb():
            def f():
                llistener.busy = True

                noll_inds = get_noll_indices(noll_sel_pars) - 1
                noll_inds.sort()
                keep_inds = noll_inds < self.shared.z_size.value
                noll_inds = noll_inds[keep_inds]
                zx = noll_inds + 1

                ax2 = self.test_axes[0]
                ax2.clear()
                noll_sp = self.shared.z_sp[noll_inds]
                noll_ms = self.shared.z_ms[noll_inds]
                ax2.plot(zx, noll_sp, zx, noll_ms, marker='.')
                txt = f'rms={norm(noll_sp - noll_ms):.2f} [rad]; '
                if self.shared.dm_sat:
                    txt += 'DM SAT'
                ax2.set_title(txt)

                phi_ms = self.shared.get_phase()[-1]

                if len(arts) != 1:
                    ax3 = self.test_axes[1]
                    im = ax3.imshow(phi_ms,
                                    extent=self.shared.mag_ext,
                                    origin='lower')
                    ax3.axis('off')
                    cb = ax3.figure.colorbar(im, ax=ax3)
                    cb.locator = ticker.MaxNLocator(nbins=5)
                    cb.update_ticks()
                    arts.append((im, cb))
                else:
                    what = phi_ms
                    m1 = what[np.isfinite(what)].min()
                    m2 = what[np.isfinite(what)].max()
                    arts[0][0].set_data(what)
                    arts[0][0].set_clim(m1, m2)

                self.update_tool_dm()

                self.test_fig.figure.canvas.draw()

                llistener.busy = False

            return f

        def f4():
            def f():
                llistener.run = False
                enable()

            return f

        llistener.sig_update.connect(make_cb())

        def fs1():
            def f():
                val, ok = QInputDialog.getDouble(
                    self,
                    'Delay between DM write and camera read',
                    f'DM write / camera read delay [{self.min_delay} sec, ]',
                    value=self.sleepmag[0],
                    min=self.min_delay,
                    decimals=4)
                if ok:
                    self.sleepmag[0] = val

            return f

        def fbzernike():
            def f():
                w = GetNollIndices(noll_sel_pars, self.shared.z_size.value - 1)
                w.exec_()

            return f

        def f5():
            def f():
                val, ok = QInputDialog.getInt(
                    self, 'Exclude Noll indices',
                    'Noll indices to exclude from the flattening (inclusive):',
                    noflat_index[0], 0, self.shared.z_sp.size)
                if not ok:
                    return
                else:
                    noflat_index[0] = val

            return f

        brun.clicked.connect(f1())
        bstop.clicked.connect(f4())
        bzernike.clicked.connect(f2())
        bclear.clicked.connect(f3())
        bsleep.clicked.connect(fs1())
        bzsize.clicked.connect(fbzernike())
        bnoflat.clicked.connect(f5())


# https://stackoverflow.com/questions/41794635/
# https://stackoverflow.com/questions/38666078/


class AlignListener(QThread):

    sig_update = pyqtSignal(tuple)

    def __init__(self, shared, sleepmag):
        super().__init__()
        self.auto = True
        self.repeat = False
        self.poke = False
        self.sleepmag = sleepmag
        self.unwrap = True
        self.shared = shared
        self.log = logging.getLogger('AlignListener')

    def run(self):
        self.shared.iq.put(('align', self.auto, self.repeat, self.poke,
                            self.sleepmag[0], self.unwrap))
        while True:
            result = self.shared.oq.get()
            self.sig_update.emit(result)
            if result[0] != 'OK' and result[1] == 'STOP':
                self.repeat = False
            self.shared.iq.put(('stopcmd', not self.repeat))
            self.shared.oq.get()
            if not self.repeat:
                return
        self.log.info('dies')


class CalibListener(QThread):

    sig_update = pyqtSignal(tuple)

    def __init__(self, shared, dset, centre, radius, dmplot):
        super().__init__()
        self.shared = shared
        self.dset = dset
        self.centre = centre
        self.radius = radius
        self.dmplot = dmplot

    def run(self):
        self.shared.iq.put(
            ('calibrate', self.dset[0], self.radius[0], self.dmplot.clone()))
        while True:
            result = self.shared.oq.get()
            self.sig_update.emit(result)
            if result[0] == 'OK' or result[0] == 'ERR':
                break


class DataAcqListener(QThread):

    sig_update = pyqtSignal(tuple)

    def __init__(self, shared, wavelength, dmplot, pokemag, sleepmag):
        super().__init__()
        self.busy = False
        self.run = True
        self.shared = shared
        self.wavelength = wavelength
        self.dmplot = dmplot
        self.log = logging.getLogger('DataAcqListener')
        self.pokemag = pokemag
        self.sleepmag = sleepmag

    def run(self):
        self.shared.iq.put(('dataacq', self.wavelength[0], self.dmplot.clone(),
                            self.pokemag[0], self.sleepmag[0]))
        while True:
            result = self.shared.oq.get()
            if result[0] == 'OK':
                self.shared.iq.put(('stopcmd', not self.run))
                self.shared.oq.get()
                if not self.run:
                    self.sig_update.emit(('stopped', ))
                    self.log.info('dies')
                    return
                elif not self.busy:
                    self.sig_update.emit(result)
            else:
                self.sig_update.emit(result)
                self.log.info('dies')
                return


class LoopListener(QThread):

    sig_update = pyqtSignal(tuple)

    def __init__(self, shared, status, sleepmag):
        super().__init__()
        self.busy = False
        self.run = True
        self.calib = False
        self.flat = True
        self.noflat_index = 0
        self.closed_loop = True
        self.shared = shared
        self.log = logging.getLogger('LoopListener')
        self.status = status
        self.sleepmag = sleepmag

    def run(self):
        self.shared.iq.put(('loop', self.calib, self.flat, self.noflat_index,
                            self.closed_loop, self.sleepmag[0]))
        while True:
            result = self.shared.oq.get()
            self.status.setText('')
            if result[0] == 'OK':
                self.shared.iq.put(('stopcmd', not self.run))
                self.shared.oq.get()
                if not self.run:
                    self.sig_update.emit(('stopped', ))
                    self.log.info('dies')
                    return
                elif not self.busy:
                    self.sig_update.emit(result)
                else:
                    self.log.debug('throttling')
            else:
                self.sig_update.emit(result)
                self.log.info('dies')
                return


class Shared:
    def __init__(self, cam, dm):
        dbl_dtsize = np.dtype('float').itemsize
        cam_dtsize = np.dtype(cam.get_image_dtype()).itemsize
        cam_shape = cam.shape()
        cam_shape = (int(cam_shape[0]), int(cam_shape[1]))
        totpixs = cam_shape[0] * cam_shape[1]

        self.cam_buf = Array('c', cam_dtsize * totpixs, lock=False)
        self.cam_ext = Array('d', 4, lock=False)
        self.cam_sat = Value('i', lock=False)
        self.dm_sat = Value('i', lock=False)
        self.z_size = Value('i', lock=False)

        self.ft_buf = Array('d', totpixs, lock=False)
        self.ft_ext = Array('d', 4, lock=False)

        self.fxcfyc = Array('d', 2, lock=False)

        self.fstord_buf = Array('c', dbl_dtsize * totpixs, lock=False)
        self.fstord_ext = Array('d', 4, lock=False)
        self.fstord_shape = Array('i', 2, lock=False)

        self.mag_buf = Array('c', dbl_dtsize * totpixs, lock=False)
        self.wrapped_buf = Array('c', dbl_dtsize * totpixs, lock=False)
        self.unwrapped_buf = Array('c', dbl_dtsize * totpixs, lock=False)
        self.mag_ext = Array('d', 4, lock=False)
        self.mag_shape = Array('i', 2, lock=False)

        self.totpixs = totpixs
        self.cam_dtype = cam.get_image_dtype()
        self.cam_shape = cam_shape

        self.dm = Array('d', int(dm.size()), lock=False)
        self.z_sp_buf = Array('d', 1024, lock=False)
        self.z_ms_buf = Array('d', 1024, lock=False)
        self.z_er_buf = Array('d', 1024, lock=False)
        self.iq = Queue()
        self.oq = Queue()

    def make_static(self):
        self.u = np.frombuffer(self.dm, float)
        self.z_sp = np.frombuffer(self.z_sp_buf, float)
        self.z_ms = np.frombuffer(self.z_ms_buf, float)
        self.z_er = np.frombuffer(self.z_er_buf, float)
        self.cam = np.frombuffer(self.cam_buf,
                                 self.cam_dtype).reshape(self.cam_shape)
        self.ft = np.frombuffer(self.ft_buf, float).reshape(self.cam_shape)

    def get_phase(self):
        nsum1 = self.fstord_shape[0] * self.fstord_shape[1]
        fstord = np.frombuffer(self.fstord_buf, float,
                               count=nsum1).reshape(self.fstord_shape)
        nsum2 = self.mag_shape[0] * self.mag_shape[1]
        mag = np.frombuffer(self.mag_buf, float,
                            count=nsum2).reshape(self.mag_shape)
        wrapped = np.frombuffer(self.wrapped_buf, float,
                                count=nsum2).reshape(self.mag_shape)
        unwrapped = np.frombuffer(self.unwrapped_buf, float,
                                  count=nsum2).reshape(self.mag_shape)
        return fstord, mag, wrapped, unwrapped


def run_worker(shared, args):
    p = Worker(shared, args)
    p.run()


class Worker:
    def __init__(self, shared, args):
        setup_logging(args)

        self.dfname = None
        self.dset = None
        self.calib_name = None
        self.calib = None
        # lastpoke
        self.run_align_state = [0]

        self.log = logging.getLogger('Worker')
        dm = open_dm(None, args)
        cam = open_cam(None, args)
        get_suitable_dmplot(args, dm)
        cam.set_exposure(cam.get_exposure_range()[0])

        shared.make_static()
        shared.fxcfyc[0] = 0.
        shared.fxcfyc[1] = 0.

        self.log.info(f'cam.shape() {cam.shape()}')
        self.log.info(f'cam.get_pixel_size() {cam.get_pixel_size()}')
        fringe = FringeAnalysis(cam.shape(), cam.get_pixel_size())
        for i in range(4):
            shared.cam_ext[i] = fringe.cam_grid[2][i] / 1000
            shared.ft_ext[i] = fringe.ft_grid[2][i] * 1000

        self.cam = cam
        self.dm = dm
        self.shared = shared
        self.fringe = fringe

    def run(self):
        cam = self.cam
        dm = self.dm
        shared = self.shared

        for cmd in iter(shared.iq.get, 'STOP'):
            self.log.info(f'cmd {cmd:}')
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
            elif cmd[0] == 'aperture':
                self.run_aperture(*cmd[1:])
            elif cmd[0] == 'calibrate':
                self.run_calibrate(*cmd[1:])
            elif cmd[0] == 'query_calib':
                self.run_query_calib(*cmd[1:])
            elif cmd[0] == 'loop':
                self.run_loop(*cmd[1:])
            else:
                raise NotImplementedError(cmd)

        self.log.info('STOP CMD')
        while not shared.iq.empty():
            self.log.info('flushing ' + str(shared.iq.get()))

        self.log.info('dies')

    def fill(self, dst, src):
        dst[:src.nbytes] = src.tobytes()

    def run_align(self, auto, repeat, poke, sleep, unwrap):
        cam = self.cam
        dm = self.dm
        shared = self.shared
        fringe = self.fringe

        while True:
            state = ('OK', )
            ts1 = time.time()

            if poke:
                shared.u[:] = 0.
                shared.u[self.run_align_state[0]] = .7
                self.run_align_state[0] += 1
                self.run_align_state[0] %= shared.u.size
                dm.write(shared.u)
                time.sleep(sleep)

            try:
                img = cam.grab_image()  # copy immediately
                shared.cam[:] = img[:]
                if img.max() == cam.get_image_max():
                    shared.cam_sat.value = 1
                else:
                    shared.cam_sat.value = 0
            except Exception as ex:
                state = ('ERR1', 'STOP', 'Camera read error')
                self.log.error(ex, exc_info=True)

            if state[0] == 'OK' and unwrap:
                try:
                    fringe.analyse(img,
                                   auto_find_orders=auto,
                                   store_logf2=True,
                                   store_logf3=True,
                                   store_mag=True,
                                   store_wrapped=True,
                                   do_unwrap=unwrap,
                                   use_mask=False)
                    elapsed = time.time() - ts1
                except Exception:
                    state = ('ERR2', 'RETRY', 'Failed to detect first orders')

            if state[0] == 'OK' and unwrap:
                try:
                    shared.ft[:] = fringe.logf2[:]
                    shared.fxcfyc[:] = fringe.fxcfyc[:]

                    self.fill(shared.fstord_buf, fringe.logf3)
                    for i in range(4):
                        shared.fstord_ext[i] = fringe.ext3[i] * 1000
                    shared.fstord_shape[:] = fringe.logf3.shape[:]
                    self.fill(shared.mag_buf, fringe.mag)
                    self.fill(shared.wrapped_buf, fringe.wrapped)
                    for i in range(4):
                        shared.mag_ext[i] = fringe.ext4[i] / 1000
                    shared.mag_shape[:] = fringe.mag.shape[:]
                    self.fill(shared.unwrapped_buf, fringe.unwrapped)
                except Exception:
                    state = ('ERR3', 'RETRY', 'Failed to unwrap phase')

            shared.oq.put(state)
            stopcmd = shared.iq.get()[1]
            shared.oq.put('')

            if not repeat or stopcmd:
                self.log.debug(
                    f'run_align stop, repeat {repeat:}, stopcmd {stopcmd:}')
                break
            else:
                elapsed = time.time() - ts1
                if elapsed < sleep:
                    self.log.debug(f'run_align repeat sleep={sleep - elapsed}')
                    time.sleep(sleep - elapsed)
                else:
                    self.log.debug('run_align repeat')

    def open_dset(self, dname):
        estr = None
        if self.dfname is None or self.dfname != dname:
            if self.dset is not None:
                self.dset.close()
            self.dset = h5py.File(dname, 'r')
            self.dfname = dname

            if 'data/images' not in self.dset:
                self.shared.oq.put(
                    (path.basename(dname) + ' does not look like a dataset', ))
                self.dset.close()
                self.dfname = None
                self.dset = None
                return -1

            try:
                img = self.dset['data/images'][0, ...]
                shape1 = self.cam.shape()
                shape2 = img.shape
                pxsize1 = self.cam.get_pixel_size()
                pxsize2 = self.dset['cam/pixel_size'][()]
                dm1 = self.dm.size()
                dm2 = self.dset['/data/U'].shape[0]
                self.log.info(f'open_dset() shape1 {shape1}')
                self.log.info(f'open_dset() shape2 {shape2}')
                self.log.info(f'open_dset() pxsize1 {pxsize1}')
                self.log.info(f'open_dset() pxsize2 {pxsize2}')
                self.log.info(f'open_dset() dm1 {dm1}')
                self.log.info(f'open_dset() dm2 {dm2}')
                if (shape1[0] != shape2[0] or shape1[1] != shape2[1]
                        or pxsize1[0] != pxsize2[0] or pxsize1[1] != pxsize2[1]
                        or dm1 != dm2):
                    self.shared.oq.put(
                        ('Configuration mismatch; Spawn new instance...',
                         dname))
                    return -1
                self.fringe.analyse(img,
                                    auto_find_orders=True,
                                    do_unwrap=True,
                                    use_mask=False)
                try:
                    self.dmplot_txs = self.dset['dmplot/DMPlot/txs'][(
                    )].tolist()
                    self.log.info(f'open_dset txs: {str(self.dmplot_txs)}')
                except KeyError:
                    self.dmplot_txs = [0, 0, 0]
                    self.log.info('open_dset txs: zero')
                return 0
            except Exception:
                self.log.info('open_dset failed fringe.analyse', exc_info=True)
                if estr is None:
                    self.shared.oq.put(('Failed to detect first orders', ))
                else:
                    self.shared.oq.put((estr, shape2, pxsize2, dm2))
                self.dfname = None
                return -1
            return 0
        else:
            return 0

    def run_query(self, dname):
        if self.open_dset(dname):
            return

        self.shared.oq.put(
            ('OK', self.dset['data/U'].shape[1], self.dmplot_txs))

    def run_calibrate(self, dname, radius, dmplot):
        if self.open_dset(dname):
            return

        try:
            wavelength = self.dset['wavelength'][()]
            dm_serial = h5_read_str(self.dset, 'dm/serial')
            dm_transform = h5_read_str(self.dset, 'dm/transform')
            cam_pixel_size = self.dset['cam/pixel_size'][()]
            cam_serial = h5_read_str(self.dset, 'cam/serial')
            hash1 = hash_file(dname)

            def make_notify():
                def f(m, cmd='UP', m2=None):
                    self.shared.oq.put((cmd, m, m2))

                return f

            notify_fun = make_notify()

            calib = RegLSCalib()
            calib.calibrate(U=self.dset['data/U'][()],
                            images=self.dset['data/images'],
                            fringe=self.fringe,
                            wavelength=wavelength,
                            dm_serial=dm_serial,
                            dm_transform=dm_transform,
                            cam_pixel_size=cam_pixel_size,
                            cam_serial=cam_serial,
                            dmplot=dmplot,
                            dname=dname,
                            hash1=hash1,
                            status_cb=notify_fun)

            now = datetime.now(timezone.utc)
            libver = 'latest'
            h5fn = path.join(
                path.dirname(dname),
                path.basename(dname).rstrip('.h5') +
                f'-{radius / 1000:.3f}mm' + '.h5')

            notify_fun(f'Saving {path.basename(h5fn)} ...')
            with h5py.File(h5fn, 'w', libver=libver) as h5f:
                write_h5_header(h5f, libver, now)
                calib.save_h5py(h5f)

            notify_fun(f'Saved {path.basename(h5fn)}; ' +
                       f'Quality {calib.mvaf.mean():.2f}%',
                       cmd='OK',
                       m2=h5fn)
        except Exception as e:
            self.log.error('run_calibrate', exc_info=True)
            self.shared.oq.put(('ERR', 'Error: ' + str(e)))

    def open_calib(self, dname):
        logging.debug(
            f'open_calib() INIT calib_name={self.calib_name} dname={dname}')
        if self.calib_name is None or self.calib_name != dname:
            with h5py.File(dname, 'r') as f:
                if 'RegLSCalib' not in f:
                    self.shared.oq.put((path.basename(dname) +
                                        ' does not look like a calibration', ))

                    self.calib_name = None
                    logging.debug(
                        'open_calib() NOT-A-CALIB ' +
                        f'calib_name={self.calib_name} dname={dname}')
                    return -1
                else:
                    shape1 = self.cam.shape()
                    shape2 = f['RegLSCalib/fringe/FringeAnalysis/shape'][()]
                    pxsize1 = self.cam.get_pixel_size()
                    pxsize2 = f['RegLSCalib/cam_pixel_size'][()]
                    dm1 = self.dm.size()
                    dm2 = f['RegLSCalib/H'].shape[1]
                    self.log.info(f'open_calib() shape1 {shape1}')
                    self.log.info(f'open_calib() shape2 {shape2}')
                    self.log.info(f'open_calib() pxsize1 {pxsize1}')
                    self.log.info(f'open_calib() pxsize2 {pxsize2}')
                    self.log.info(f'open_calib() dm1 {dm1}')
                    self.log.info(f'open_calib() dm2 {dm2}')

                    if (shape1[0] != shape2[0] or shape1[1] != shape2[1]
                            or pxsize1[0] != pxsize2[0]
                            or pxsize1[1] != pxsize2[1] or dm1 != dm2):
                        self.shared.oq.put(
                            ('Configuration mismatch; Spawn new instance...',
                             dname))

                        self.calib_name = None
                        logging.debug(
                            'open_calib() MISMATCH ' +
                            f'calib_name={self.calib_name} dname={dname}')
                        return -1

                    self.calib = RegLSCalib.load_h5py(f, lazy_cart_grid=True)
                    try:
                        self.dmplot_txs = f['RegLSCalib/dmplot/DMPlot/txs'][(
                        )].tolist()
                    except KeyError:
                        self.dmplot_txs = [0, 0, 0]

                    self.calib_name = dname
                    logging.debug(
                        'open_calib() LOADED ' +
                        f'calib_name={self.calib_name} dname={dname}')
                    return 0
        else:
            logging.debug('open_calib() SAME ' +
                          f'calib_name={self.calib_name} dname={dname}')
            return 0

    def run_query_calib(self, dname):
        if self.open_calib(dname):
            return

        self.shared.oq.put(
            ('OK', self.calib.wavelength, self.calib.get_rzern().n,
             self.calib.get_radius(), self.dmplot_txs))

    def run_aperture(self, dname, radius):
        if self.open_dset(dname):
            return

        if radius <= 0.:
            self.fringe.clear_aperture()
            self.shared.oq.put(('OK', None))
        else:
            names = h5_read_str(self.dset, 'align/names')
            if 'centre' not in names.split(','):
                self.shared.oq.put(('centre measurement is missing', ))
                return

            try:
                img_centre = self.dset['align/images'][names.index('centre'),
                                                       ...]
                img_zero = self.dset['data/images'][0, ...]
                self.fringe.estimate_aperture(img_zero, img_centre, radius)
                self.shared.oq.put(('OK', self.fringe.centre))
            except Exception:
                self.log.debug('run_aperture', exc_info=True)
                self.shared.oq.put(
                    ('Failed to estimate the aperture. Try improving the ' +
                     'fringe contrast or the illumination profile', ))

    def run_plot(self, dname, ind, radius):
        if self.open_dset(dname):
            return
        fringe = self.fringe

        t1 = self.dset['align/U'].shape[1]
        t2 = self.dset['data/U'].shape[1]
        if ind < 0 or ind > t1 + t2:
            self.shared.oq.put((f'index must be within 0 and {t1 + t2 - 1}', ))
        if ind < t1:
            addr = 'align'
        else:
            addr = 'data'
            ind -= t1
        try:
            img = self.dset[addr + '/images'][ind, ...]
            fringe.analyse(img,
                           auto_find_orders=False,
                           store_mag=True,
                           store_wrapped=True,
                           do_unwrap=True,
                           use_mask=radius > 0.)

            if img.max() == self.cam.get_image_max():
                self.shared.cam_sat.value = 1
            else:
                self.shared.cam_sat.value = 0
            self.shared.cam[:] = img[:]
            self.shared.u[:] = self.dset[addr + '/U'][:, ind]
            self.fill(self.shared.wrapped_buf, fringe.wrapped)
            for i in range(4):
                self.shared.mag_ext[i] = fringe.ext4[i] / 1000
            self.shared.mag_shape[:] = fringe.mag.shape[:]
            self.fill(self.shared.unwrapped_buf, fringe.unwrapped)
            self.shared.oq.put(('OK', ))
        except Exception as e:
            self.log.error('run_plot', exc_info=True)
            self.shared.oq.put((str(e), ))

    def run_dataacq(self, wavelength, dmplot, pokemag, sleep):
        cam = self.cam
        dm = self.dm
        shared = self.shared

        Ualign = []
        align_names = []
        for name in ('centre', 'cross', 'x', 'rim', 'checker', 'arrows'):
            try:
                Ualign.append(dm.preset(name, pokemag).reshape(-1, 1))
                align_names.append(name)
            except Exception:
                pass
        if len(Ualign) > 0:
            Ualign = np.hstack(Ualign)
        else:
            Ualign = np.zeros((dm.size(), 0))

        libver = 'latest'
        now = datetime.now(timezone.utc)
        h5fn = now.astimezone().strftime('%Y%m%d_%H%M%S.h5').replace('#', 'p')
        dmsn = dm.get_serial_number()
        if dmsn:
            h5fn = dmsn + '_' + h5fn

        U = make_normalised_input_matrix(dm.size(), 5, pokemag)

        with h5py.File(h5fn, 'w', libver=libver) as h5f:
            write_h5_header(h5f, libver, now)

            dmplot.save_h5py(h5f, 'dmplot/')

            h5_store_str(h5f, 'cam/serial', cam.get_serial_number())
            h5_store_str(h5f, 'cam/settings', cam.get_settings())
            h5f['cam/pixel_size'] = cam.get_pixel_size()
            h5f['cam/pixel_size'].attrs['units'] = 'um'
            h5f['cam/exposure'] = cam.get_exposure()
            h5f['cam/exposure'].attrs['units'] = 'ms'
            h5_store_str(h5f, 'cam/dtype', cam.get_image_dtype())
            h5f['cam/max'] = cam.get_image_max()

            h5f['wavelength'] = wavelength
            h5f['wavelength'].attrs['units'] = 'nm'
            h5f['sleep'] = sleep
            h5f['sleep'].attrs['units'] = 's'

            h5_store_str(h5f, 'dm/serial', dm.get_serial_number())
            h5_store_str(h5f, 'dm/transform', str(dm.get_transform()))

            h5f['data/U'] = U
            h5f['data/U'].dims[0].label = 'actuators'
            h5f['data/U'].dims[1].label = 'step'
            h5f.create_dataset('data/images', (U.shape[1], ) + cam.shape(),
                               dtype=cam.get_image_dtype())
            h5f['data/images'].dims[0].label = 'step'
            h5f['data/images'].dims[1].label = 'height'
            h5f['data/images'].dims[1].label = 'width'

            h5f['align/U'] = Ualign
            h5f['align/U'].dims[0].label = 'actuators'
            h5f['align/U'].dims[1].label = 'step'
            h5f.create_dataset('align/images',
                               (Ualign.shape[1], ) + cam.shape(),
                               dtype=cam.get_image_dtype())
            h5f['align/images'].dims[0].label = 'step'
            h5f['align/images'].dims[1].label = 'height'
            h5f['align/images'].dims[1].label = 'width'
            h5_store_str(h5f, 'align/names', ','.join(align_names))

            tot = U.shape[1] + Ualign.shape[1]
            count = [0]

            todo = ((Ualign, 'align/images'), (U, 'data/images'))
            for U1, imaddr in todo:
                for i in range(U1.shape[1]):
                    try:
                        dm.write(U1[:, i])
                        time.sleep(sleep)
                        img = cam.grab_image()  # copy immediately
                    except Exception as e:
                        self.log.error('run_dataacq', exc_info=True)
                        shared.oq.put((str(e), ))
                        return
                    if img.max() == cam.get_image_max():
                        shared.cam_sat.value = 1
                    else:
                        shared.cam_sat.value = 0
                    h5f[imaddr][i, ...] = img
                    shared.u[:] = U1[:, i]
                    shared.cam[:] = img
                    shared.oq.put(('OK', count[0], tot))
                    self.log.debug('run_dataacq iteration')

                    stopcmd = shared.iq.get()[1]
                    shared.oq.put(('', ))

                    if stopcmd:
                        self.log.debug('run_dataacq stop_cmd')
                        h5f.close()
                        try:
                            os.remove(h5fn)
                        except OSError:
                            pass
                        return
                    else:
                        self.log.debug('run_dataacq continue')

                    count[0] += 1

        self.log.debug('run_dataacq finished')
        shared.oq.put(('finished', h5fn))

    def run_loop(self, dname, flat, noflat_index, closed_loop, sleep):
        if self.open_calib(dname):
            return

        calib = self.calib
        fringe = self.calib.fringe
        cam = self.cam
        t1 = time.time()
        dm = ZernikeControl(self.dm, calib)
        t2 = time.time()
        self.log.debug(f'run_loop() ZernikeControl {t2 - t1:.3f}')
        shared = self.shared
        shared.z_size.value = dm.ndof

        calib.reflatten(noflat_index)
        self.log.debug(f'run_loop() reflatten {noflat_index}')
        dm.flat_on = flat

        for i in range(4):
            self.shared.mag_ext[i] = fringe.ext4[i] / 1000
        shared.mag_shape[:] = fringe.unwrapped.shape[:]
        while True:
            try:
                t1 = time.time()
                dm.write(self.shared.z_sp[:dm.ndof])
                self.shared.u[:] = dm.u[:]
                if dm.saturation:
                    self.shared.dm_sat.value = 1
                else:
                    self.shared.dm_sat.value = 0
                t2 = time.time()

                time.sleep(sleep)
                img = cam.grab_image()

                t3 = time.time()
                fringe.analyse(img, use_mask=True)
                unwrapped = fringe.unwrapped
                calib.apply_aperture_mask(unwrapped)
                self.fill(shared.unwrapped_buf, unwrapped)
                t4 = time.time()

                t7 = time.time()
                shared.z_ms[:dm.ndof] = calib.zernike_fit(unwrapped)
                shared.z_ms[0] = 0
                t8 = time.time()

                self.log.debug(f'run_loop() s:{sleep:.3f} h:{t2 - t1:.3f} ' +
                               f'u:{t4 - t3:.3f} p2:{t8 - t7:.3f}')

            except Exception as e:
                self.log.info('run_loop()', exc_info=True)
                shared.oq.put((str(e), ))
                return

            shared.oq.put(('OK', ))

            stopcmd = shared.iq.get()[1]
            shared.oq.put('')

            if stopcmd:
                self.log.debug('run_loop() stopcmd')
                return
            else:
                self.log.debug('run_loop() continue')

        self.log.debug('run_loop() finished')
        shared.oq.put(('finished', ))


def config_like(args, h5):
    log = logging.getLogger('config_like')
    with h5py.File(h5, 'r') as h5:
        if 'dmplot/DMPlot' in h5:
            args.sim_cam_shape = h5['data/images'][0, ...].shape
            args.sim_cam_pix_size = h5['cam/pixel_size'][()]
            log.info(
                f'dmplot/DMPlot {args.sim_cam_shape} {args.sim_cam_pix_size}')
            return DMPlot.load_h5py(h5, 'dmplot/')
        elif 'RegLSCalib/dmplot/DMPlot' in h5:
            args.sim_cam_shape = h5['RegLSCalib/fringe/FringeAnalysis/shape'][(
            )]
            args.sim_cam_pix_size = h5['/RegLSCalib/cam_pixel_size'][()]
            log.info(
                f'dmplot/DMPlot {args.sim_cam_shape} {args.sim_cam_pix_size}')
            return DMPlot.load_h5py(h5, 'RegLSCalib/dmplot/')
        else:
            log.info('None')
            return None


def main():
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn')

    app = QApplication(sys.argv)
    if platform.system() == 'Windows':
        try:
            app.setStyle(QStyleFactory.create('Fusion'))
        except Exception:
            pass

    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='DM calibration GUI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_log_parameters(parser)
    add_dm_parameters(parser)
    add_cam_parameters(parser)
    parser.add_argument('--config-like',
                        type=argparse.FileType('rb'),
                        default=None,
                        metavar='HDF5')
    parser.add_argument('--min-delay', type=float, default=.2, metavar='SEC')
    args = parser.parse_args(args[1:])
    setup_logging(args)

    if args.config_like is not None:
        args.config_like.close()
        args.config_like = args.config_like.name
        dmplot = config_like(args, args.config_like)
    else:
        dmplot = None

    dm = open_dm(app, args)
    cam = open_cam(app, args)
    cam_name = cam.get_serial_number()
    dm_name = dm.get_serial_number()

    dmplot = get_suitable_dmplot(args, dm, dmplot=dmplot)

    shared = Shared(cam, dm)
    dm.close()
    cam.close()

    p = Process(name='worker', target=run_worker, args=(shared, args))
    p.start()

    control = Control(p, shared, cam_name, dm_name, dmplot, args.min_delay)
    control.show()

    exit = app.exec_()

    shared.iq.put('STOP')
    p.join()
    sys.exit(exit)


if __name__ == '__main__':
    main()
