#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import numpy as np
import json
import argparse

from os import path
from h5py import File
from numpy.linalg import norm
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from datetime import datetime
from copy import deepcopy

from PyQt5.QtCore import Qt, QMutex, pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QKeySequence
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QGroupBox, QGridLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QScrollArea, QSlider, QDoubleSpinBox, QFrame,
    QErrorMessage, QApplication, QMainWindow, QSplitter, QShortcut,
    QMessageBox, QSizePolicy, QInputDialog, QTabWidget, QComboBox, QDialog,
    )


from zernike.czernike import RZern

from dmlib.version import __version__
from dmlib.dmplot import DMPlot
from dmlib.core import (
    add_dm_parameters, open_dm, add_log_parameters, setup_logging)
from dmlib.calibration import WeightedLSCalib
from dmlib import control
from dmlib.control import ZernikeControl


class MyQDoubleValidator(QDoubleValidator):

    def setFixup(self, val):
        self.fixupval = val

    def fixup(self, txt):
        return str(self.fixupval)


class MyQIntValidator(QIntValidator):

    def setFixup(self, val):
        self.fixupval = val

    def fixup(self, txt):
        return str(self.fixupval)


class OptionsPanel(QFrame):

    def setup(self, pars, name, defaultd, infod):
        self.lines = []
        self.pars = pars
        self.name = name
        self.defaultd = defaultd
        self.infod = infod

        layout = QGridLayout()
        self.setLayout(layout)

        combo = QComboBox()
        for k in defaultd.keys():
            combo.addItem(k)
        layout.addWidget(combo, 0, 0)
        combo.setCurrentIndex(0)
        self.combo = combo

        scroll = QScrollArea()
        scroll.setWidget(QWidget())
        scroll.setWidgetResizable(True)
        lay = QGridLayout(scroll.widget())
        self.scroll = scroll
        self.lay = lay

        layout.addWidget(scroll, 1, 0)
        addr_options = name + '_options'
        addr_selection = name + '_name'
        self.addr_options = addr_options
        self.addr_selection = addr_selection

        self.selection = combo.currentText()
        if addr_options not in self.pars:
            self.pars[addr_options] = defaultd
        if addr_selection not in self.pars:
            self.pars[addr_selection] = self.selection

        self.from_dict(
            self.selection,
            self.infod[self.selection],
            self.pars[self.addr_options][self.selection])

        def f():
            def f(selection):
                self.clear_all()
                self.from_dict(
                    selection,
                    self.infod[selection],
                    self.pars[self.addr_options][selection])
                self.selection = selection
            return f

        combo.currentTextChanged.connect(f())

    def update(self, selection, k, v):
        self.pars[self.addr_options][selection][k] = v
        if selection == self.combo.currentText():
            found = 0
            for l in self.lines:
                if l[0][1].text() == k:
                    type1 = self.infod[selection][k][0]
                    l[0][0].setText(str(type1(v)))
                    found = 1
                    break
            if found == 0:
                raise RuntimeError()

    def get_options(self):
        return (
            self.selection,
            deepcopy(self.pars[self.addr_options][self.selection])
            )

    def from_dict(self, selection, infod, valuesd):
        count = 0
        for k, v in infod.items():
            lab = QLabel(k)

            type1 = v[0]
            bounds = v[1]
            desc = v[2]

            lab.setToolTip(desc)
            self.lay.addWidget(lab, count, 0)

            def fle(k, le, val, type1):
                def f():
                    newval = type1(le.text())
                    self.pars[
                        self.addr_options][selection][k] = type1(le.text())
                    val.setFixup(newval)
                return f

            def ledisc(w, hand):
                def f():
                    w.editingFinished.disconnect(hand)
                return f

            curval = valuesd[k]
            if type1 in (int, float):
                le = QLineEdit(str(curval))
                le.setToolTip(desc)
                if type1 == int:
                    vv = MyQIntValidator()
                else:
                    vv = MyQDoubleValidator()
                vv.setFixup(curval)
                if bounds[0] is not None:
                    vv.setBottom(bounds[0])
                if bounds[1] is not None:
                    vv.setTop(bounds[1])
                le.setValidator(vv)
                hand = fle(k, le, vv, type1)
                le.editingFinished.connect(hand)
                disc = ledisc(le, hand)
            elif type1 == list:
                le = QLineEdit(', '.join([str(c) for c in curval]))
                le.setToolTip(desc)

                def make_validator(k, le, type1, bounds):
                    def f():
                        old = self.pars[self.addr_options][selection][k]
                        try:
                            tmp = [
                                bounds(s) for s in le.text().split(',')
                                if s != '']
                        except Exception:
                            tmp = old
                        self.pars[self.addr_options][selection][k] = tmp
                        le.blockSignals(True)
                        le.setText(', '.join([str(c) for c in tmp]))
                        le.blockSignals(False)
                    return f

                hand = make_validator(k, le, type1, bounds)
                le.editingFinished.connect(hand)
                disc = ledisc(le, hand)
            else:
                raise RuntimeError()

            self.lay.addWidget(le, count, 1)
            self.lines.append(((le, lab), disc))
            count += 1

    def clear_all(self):
        for l in self.lines:
            for w in l[0]:
                self.lay.removeWidget(w)
                w.setParent(None)
            l[1]()
        self.lines.clear()


class RelSlider:

    def __init__(self, val, cb):
        self.old_val = None
        self.fto100mul = 100
        self.cb = cb

        self.sba = QDoubleSpinBox()
        self.sba.setMinimum(-1000)
        self.sba.setMaximum(1000)
        self.sba.setDecimals(6)
        self.sba.setToolTip('Effective value')
        self.sba.setValue(val)
        self.sba_color(val)
        self.sba.setSingleStep(1.25e-3)

        self.qsr = QSlider(Qt.Horizontal)
        self.qsr.setMinimum(-100)
        self.qsr.setMaximum(100)
        self.qsr.setValue(0)
        self.qsr.setToolTip('Drag to apply relative delta')

        self.sbm = QDoubleSpinBox()
        self.sbm.setMinimum(0.01)
        self.sbm.setMaximum(1000)
        self.sbm.setSingleStep(1.25e-3)
        self.sbm.setToolTip('Maximum relative delta')
        self.sbm.setDecimals(2)
        self.sbm.setValue(4.0)

        def sba_cb():
            def f():
                self.block()
                val = self.sba.value()
                self.sba_color(val)
                self.cb(val)
                self.unblock()
            return f

        def qs1_cb():
            def f(t):
                self.block()

                if self.old_val is None:
                    self.qsr.setValue(0)
                    self.unblock()
                    return

                val = self.old_val + self.qsr.value()/100*self.sbm.value()
                self.sba.setValue(val)
                self.sba_color(val)
                self.cb(val)

                self.unblock()
            return f

        def qs1_end():
            def f():
                self.block()
                self.qsr.setValue(0)
                self.old_val = None
                self.unblock()
            return f

        def qs1_start():
            def f():
                self.block()
                self.old_val = self.get_value()
                self.unblock()
            return f

        self.sba_cb = sba_cb()
        self.qs1_cb = qs1_cb()
        self.qs1_start = qs1_start()
        self.qs1_end = qs1_end()

        self.sba.valueChanged.connect(self.sba_cb)
        self.qsr.valueChanged.connect(self.qs1_cb)
        self.qsr.sliderPressed.connect(self.qs1_start)
        self.qsr.sliderReleased.connect(self.qs1_end)

    def sba_color(self, val):
        if abs(val) > 1e-4:
            self.sba.setStyleSheet("font-weight: bold;")
        else:
            self.sba.setStyleSheet("font-weight: normal;")
        # self.sba.update()

    def block(self):
        self.sba.blockSignals(True)
        self.qsr.blockSignals(True)
        self.sbm.blockSignals(True)

    def unblock(self):
        self.sba.blockSignals(False)
        self.qsr.blockSignals(False)
        self.sbm.blockSignals(False)

    def enable(self):
        self.sba.setEnabled(True)
        self.qsr.setEnabled(True)
        self.sbm.setEnabled(True)

    def disable(self):
        self.sba.setEnabled(False)
        self.qsr.setEnabled(False)
        self.sbm.setEnabled(False)

    def fto100(self, f):
        return int((f + self.m2)/(2*self.m2)*self.fto100mul)

    def get_value(self):
        return self.sba.value()

    def set_value(self, v):
        self.sba_color(v)
        return self.sba.setValue(v)

    def add_to_layout(self, l1, ind1, ind2):
        l1.addWidget(self.sba, ind1, ind2)
        l1.addWidget(self.qsr, ind1, ind2 + 1)
        l1.addWidget(self.sbm, ind1, ind2 + 2)

    def remove_from_layout(self, l1):
        l1.removeWidget(self.sba)
        l1.removeWidget(self.qsr)
        l1.removeWidget(self.sbm)

        self.sba.setParent(None)
        self.qsr.setParent(None)
        self.sbm.setParent(None)

        self.sba.valueChanged.disconnect(self.sba_cb)
        self.qsr.valueChanged.disconnect(self.qs1_cb)
        self.qsr.sliderPressed.disconnect(self.qs1_start)
        self.qsr.sliderReleased.disconnect(self.qs1_end)

        self.sba_cb = None
        self.qs1_cb = None
        self.qs1_start = None
        self.qs1_end = None

        self.sb = None
        self.qsr = None


class ZernikePanel(QWidget):

    def_pars = {'zernike_labels': {}, 'shown_modes': 21}

    def __init__(
            self, wavelength, n_radial, z0=None, callback=None, pars={},
            parent=None):
        super().__init__(parent=parent)
        self.log = logging.getLogger(self.__class__.__name__)

        self.pars = {**deepcopy(self.def_pars), **deepcopy(pars)}
        self.units = 'rad'
        self.status = None
        self.mul = 1.0
        self.fig = None
        self.ax = None
        self.im = None
        self.cb = None
        self.shape = (128, 128)
        self.P = 1

        self.rzern = RZern(n_radial)
        dd = np.linspace(-1, 1, self.shape[0])
        xv, yv = np.meshgrid(dd, dd)
        self.rzern.make_cart_grid(xv, yv)
        self.rad_to_nm = wavelength/(2*np.pi)
        self.callback = callback
        self.zernike_rows = []

        if z0 is None:
            self.z = np.zeros(self.rzern.nk)
        else:
            self.z = z0.copy()
        assert(self.rzern.nk == self.z.size)

        top1 = QGroupBox('phase')
        toplay1 = QGridLayout()
        top1.setLayout(toplay1)
        self.fig = FigureCanvas(Figure(figsize=(2, 2)))
        self.ax = self.fig.figure.add_subplot(1, 1, 1)
        phi = self.rzern.matrix(self.rzern.eval_grid(np.dot(self.P, self.z)))
        self.im = self.ax.imshow(phi, origin='lower')
        self.cb = self.fig.figure.colorbar(self.im)
        self.cb.locator = ticker.MaxNLocator(nbins=5)
        self.cb.update_ticks()
        self.ax.axis('off')
        self.status = QLabel('')
        toplay1.addWidget(self.fig, 0, 0)
        toplay1.addWidget(self.status, 1, 0)

        def nmodes():
            return min(self.pars['shown_modes'], self.rzern.nk)

        top = QGroupBox('Zernike')
        toplay = QGridLayout()
        top.setLayout(toplay)
        labzm = QLabel('shown modes')
        lezm = QLineEdit(str(nmodes()))
        lezm.setMaximumWidth(50)
        lezmval = MyQIntValidator(1, self.rzern.nk)
        lezmval.setFixup(nmodes())
        lezm.setValidator(lezmval)

        brad = QCheckBox('rad')
        brad.setChecked(True)
        breset = QPushButton('reset')
        toplay.addWidget(labzm, 0, 0)
        toplay.addWidget(lezm, 0, 1)
        toplay.addWidget(brad, 0, 2)
        toplay.addWidget(breset, 0, 3)

        scroll = QScrollArea()
        toplay.addWidget(scroll, 1, 0, 1, 5)
        scroll.setWidget(QWidget())
        scrollLayout = QGridLayout(scroll.widget())
        scroll.setWidgetResizable(True)

        def make_hand_slider(ind):
            def f(r):
                self.z[ind] = r
                self.update_phi_plot()
            return f

        def make_hand_lab(le, i):
            def f():
                self.pars['zernike_labels'][str(i)] = le.text()
            return f

        def default_zernike_name(i, n, m):
            if i == 1:
                return 'piston'
            elif i == 2:
                return 'tip'
            elif i == 3:
                return 'tilt'
            elif i == 4:
                return 'defocus'
            elif m == 0:
                return 'spherical'
            elif abs(m) == 1:
                return 'coma'
            elif abs(m) == 2:
                return 'astigmatism'
            elif abs(m) == 3:
                return 'trefoil'
            elif abs(m) == 4:
                return 'quadrafoil'
            elif abs(m) == 5:
                return 'pentafoil'
            else:
                return ''

        def make_update_zernike_rows():
            def f(mynk=None):
                if mynk is None:
                    mynk = len(self.zernike_rows)
                ntab = self.rzern.ntab
                mtab = self.rzern.mtab
                if len(self.zernike_rows) < mynk:
                    for i in range(len(self.zernike_rows), mynk):
                        lab = QLabel(
                            f'Z<sub>{i + 1}</sub> ' +
                            f'Z<sub>{ntab[i]}</sub><sup>{mtab[i]}</sup>')
                        slider = RelSlider(self.z[i], make_hand_slider(i))

                        if str(i) in self.pars['zernike_labels'].keys():
                            zname = self.pars['zernike_labels'][str(i)]
                        else:
                            zname = default_zernike_name(
                                i + 1, ntab[i], mtab[i])
                            self.pars['zernike_labels'][str(i)] = zname
                        lbn = QLineEdit(zname)
                        lbn.setMaximumWidth(120)
                        hand_lab = make_hand_lab(lbn, i)
                        lbn.editingFinished.connect(hand_lab)

                        scrollLayout.addWidget(lab, i, 0)
                        scrollLayout.addWidget(lbn, i, 1)
                        slider.add_to_layout(scrollLayout, i, 2)

                        self.zernike_rows.append((lab, slider, lbn, hand_lab))

                    assert(len(self.zernike_rows) == mynk)

                elif len(self.zernike_rows) > mynk:
                    for i in range(len(self.zernike_rows) - 1, mynk - 1, -1):
                        lab, slider, lbn, hand_lab = self.zernike_rows.pop()

                        scrollLayout.removeWidget(lab)
                        slider.remove_from_layout(scrollLayout)
                        scrollLayout.removeWidget(lbn)

                        lbn.editingFinished.disconnect(hand_lab)
                        lab.setParent(None)
                        lbn.setParent(None)

                    assert(len(self.zernike_rows) == mynk)
            return f

        self.update_zernike_rows = make_update_zernike_rows()

        def reset_fun():
            self.z *= 0.
            self.update_gui_controls()
            self.update_phi_plot()

        def change_nmodes():
            try:
                ival = int(lezm.text())
                assert(ival > 0)
                assert(ival <= self.rzern.nk)
            except Exception:
                lezm.setText(str(len(self.zernike_rows)))
                return

            if ival != len(self.zernike_rows):
                self.update_zernike_rows(ival)
                self.update_phi_plot()
                lezm.setText(str(len(self.zernike_rows)))

        def f2():
            def f(b):
                if b:
                    self.units = 'rad'
                    self.mul = 1.0
                else:
                    self.units = 'nm'
                    self.mul = self.rad_to_nm
                self.update_phi_plot()
            return f

        self.update_zernike_rows(nmodes())

        brad.stateChanged.connect(f2())
        breset.clicked.connect(reset_fun)
        lezm.editingFinished.connect(change_nmodes)

        split = QSplitter(Qt.Vertical)
        split.addWidget(top1)
        split.addWidget(top)
        l1 = QGridLayout()
        l1.addWidget(split)
        self.setLayout(l1)
        self.lezm = lezm

    def save_parameters(self, merge={}):
        d = {**merge, **self.pars}
        d['shown_modes'] = len(self.zernike_rows)
        return d

    def load_parameters(self, d):
        self.pars = {**deepcopy(self.def_pars), **deepcopy(d)}
        nmodes = min(self.pars['shown_modes'], self.rzern.nk)
        self.pars['shown_modes'] = nmodes
        self.lezm.blockSignals(True)
        self.lezm.setText(str(nmodes))
        self.lezm.blockSignals(False)
        self.update_zernike_rows(0)
        self.update_zernike_rows(nmodes)

    def update_gui_controls(self):
        for i, t in enumerate(self.zernike_rows):
            slider = t[1]
            slider.block()
            slider.set_value(self.z[i])
            slider.unblock()

    def update_phi_plot(self, run_callback=True):
        phi = self.mul*self.rzern.matrix(
            self.rzern.eval_grid(np.dot(self.P, self.z)))
        inner = phi[np.isfinite(phi)]
        min1 = inner.min()
        max1 = inner.max()
        rms = self.mul*norm(self.z)
        self.status.setText(
            '{} [{: 03.2f} {: 03.2f}] {: 03.2f} PV {: 03.2f} RMS'.format(
                self.units, min1, max1, max1 - min1, rms))
        self.im.set_data(phi)
        self.im.set_clim(inner.min(), inner.max())
        self.fig.figure.canvas.draw()

        if self.callback and run_callback:
            self.callback(self.z)


class PlotCoeffs(QDialog):

    def set_data(self, u, z):
        self.setWindowTitle('Zernike coefficients')
        frame = QFrame()
        fig = FigureCanvas(Figure(figsize=(7, 5)))
        layout = QGridLayout()
        frame.setLayout(layout)
        nav = NavigationToolbar2QT(fig, frame)
        layout.addWidget(nav, 0, 0)
        layout.addWidget(fig, 1, 0)
        fig.figure.subplots_adjust(
            left=.125, right=.9,
            bottom=.1, top=.9,
            wspace=0.45, hspace=0.45)
        self.fig = fig

        gs = GridSpec(2, 1)
        ax0 = fig.figure.add_subplot(gs[0, 0])
        ax0.plot(u, marker='.')
        ax0.grid()
        ax0.set_xlabel('actuators')
        ax0.set_ylim((-1, 1))
        ax1 = fig.figure.add_subplot(gs[1, 0])
        ax1.plot(range(1, z.size + 1), z, marker='.')
        ax1.grid()
        ax1.set_xlabel('Noll')
        ax1.set_ylabel('[rad]')

        self.ax0 = ax0
        self.ax1 = ax1

        l1 = QGridLayout()
        self.setLayout(l1)
        l1.addWidget(frame)


class ZernikeWindow(QMainWindow):

    sig_acquire = pyqtSignal(tuple)
    sig_release = pyqtSignal(tuple)
    sig_lock = pyqtSignal()
    sig_unlock = pyqtSignal()
    sig_draw = pyqtSignal(tuple)

    def __init__(self, app, dm, calib, pars={}, parent=None):
        super().__init__(parent)
        self.log = logging.getLogger(self.__class__.__name__)
        self.can_close = True
        self.pars = pars
        self.control_enabled = True

        self.dm = dm
        self.calib = calib
        try:
            self.zcontrol = ZernikeControl(
                self.dm, self.calib, self.pars['ZernikeControl'])
            self.log.info('loaded control parameters')
        except Exception as ex:
            self.log.info(f'failed to load control parameters {str(ex)}')
            self.zcontrol = ZernikeControl(self.dm, self.calib)

        self.app = app
        self.mutex = QMutex()

        self.setWindowTitle('ZernikeWindow ' + __version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        self.dmplot = DMPlot()
        self.dmplot.update_txs(self.zcontrol.calib.dmplot_txs)
        dmstatus = QLabel()

        def make_figs():
            fig = FigureCanvas(Figure(figsize=(2, 2)))
            ax = fig.figure.subplots(2, 1)
            ima = self.dmplot.draw(ax[0], self.zcontrol.u)
            img = ax[1].imshow(self.dmplot.compute_gauss(self.zcontrol.u))
            ax[0].axis('off')
            ax[1].axis('off')

            return ax, ima, img, fig

        ax, ima, img, fig = make_figs()

        def make_write_dm():
            def f(z, do_write=True):
                # callback for zpanel
                if do_write:
                    self.zcontrol.write(z)

                if self.zcontrol.saturation:
                    satind = 'SAT'
                else:
                    satind = 'OK'
                dmstatus.setText(
                    f'u [{self.zcontrol.u.min():+0.3f} ' +
                    f'{self.zcontrol.u.max():+0.3f}] {satind}')

                ima.set_data(self.dmplot.compute_pattern(self.zcontrol.u))
                g = self.dmplot.compute_gauss(self.zcontrol.u)
                img.set_data(g)
                img.set_clim(g.min(), g.max())
                ax[0].figure.canvas.draw()
            return f

        write_dm = make_write_dm()
        self.write_dm = write_dm

        if 'ZernikePanel' not in pars:
            pars['ZernikePanel'] = {}
        self.zpanel = ZernikePanel(
            self.zcontrol.calib.wavelength, self.zcontrol.calib.get_rzern().n,
            self.zcontrol.z, callback=write_dm, pars=pars['ZernikePanel'])
        self.zpanel.z[:] = self.zcontrol.u2z()
        self.zpanel.update_gui_controls()
        self.zpanel.update_phi_plot()
        write_dm(self.zpanel.z)

        def make_select_cb():
            def f(e):
                self.mutex.lock()
                if e.inaxes is not None:
                    ind = self.dmplot.index_actuator(e.xdata, e.ydata)
                    if ind != -1:
                        val, ok = QInputDialog.getDouble(
                            self, f'Actuator {ind} ' + str(ind),
                            'range [-1, 1]', self.zcontrol.u[ind],
                            -1., 1., 4)
                        if ok:
                            self.zcontrol.u[ind] = val
                            self.zpanel.z[:] = self.zcontrol.u2z()
                            self.zpanel.update_gui_controls()
                            self.zpanel.update_phi_plot()
                self.mutex.unlock()
            return f

        ax[0].figure.canvas.callbacks.connect(
            'button_press_event', make_select_cb())

        dmstatus.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.zpanel)
        split.addWidget(fig)

        self.tabs = QTabWidget()
        front = QFrame()
        layout = QGridLayout()
        front.setLayout(layout)
        layout.addWidget(split, 0, 0, 1, 4)
        layout.addWidget(dmstatus, 1, 0, 1, 4)

        self.add_lower(layout)
        self.tabs.addTab(front, self.zcontrol.dm.get_serial_number())
        self.make_control_tab()

        self.setCentralWidget(self.tabs)

        def lock():
            self.mutex.lock()
            self.can_close = False
            for i in range(self.tabs.count()):
                self.tabs.widget(i).setEnabled(False)

        def unlock():
            for i in range(self.tabs.count()):
                self.tabs.widget(i).setEnabled(True)
            self.can_close = True
            self.mutex.unlock()

        def make_release_hand():
            def f(t):
                if t[0] is not None:
                    if norm(t[0].u) != 0:
                        self.zcontrol.u[:] = t[0].u
                    self.zpanel.z[:] = self.zcontrol.u2z()
                    self.zpanel.update_gui_controls()
                    self.zpanel.update_phi_plot()
                unlock()
            return f

        def make_acquire_hand():
            def f(t):
                lock()
            return f

        self.sig_release.connect(make_release_hand())
        self.sig_acquire.connect(make_acquire_hand())

        def make_lock_hand():
            def f():
                lock()
            return f

        def make_unlock_hand():
            def f():
                unlock()
            return f

        self.sig_lock.connect(make_lock_hand())
        self.sig_unlock.connect(make_unlock_hand())

        def f():
            def f(t):
                u = t[0]
                self.zcontrol.u[:] = u
                z = self.zcontrol.u2z()
                self.zpanel.z[:] = z
                self.zpanel.update_phi_plot(run_callback=False)
                self.write_dm(z, do_write=False)
            return f

        self.sig_draw.connect(f())

    def __str__(self):
        return f'<dmlib.zpanel.{self.__class__.__name__}>'

    def make_control_tab(self):
        control_options = OptionsPanel()
        control_options.setup(
            self.pars, 'control',
            control.get_default_parameters(),
            control.get_parameters_info())
        self.control_options = control_options
        self.tabs.addTab(control_options, 'control')

    def instance_control(self):
        try:
            self.zcontrol = ZernikeControl(
                self.dm, self.calib, self.pars['ZernikeControl'])
        except Exception as ex:
            self.log.error(f'failed instance_control {str(ex)}')
            self.zcontrol = ZernikeControl(self.dm, self.calib)
        self.bflat.blockSignals(True)
        if self.zcontrol.flat_on:
            self.bflat.setChecked(True)
        else:
            self.bflat.setChecked(False)
        self.bflat.blockSignals(False)
        self.zpanel.z[:] = self.zcontrol.u2z()
        self.zpanel.update_gui_controls()
        self.zpanel.update_phi_plot()

    def load_parameters(self, d):
        self.pars = {**deepcopy(self.pars), **deepcopy(d)}
        with File(self.pars['calibration'], 'r') as f:
            self.calib = WeightedLSCalib.load_h5py(f, lazy_cart_grid=True)
        self.instance_control()
        if 'ZernikePanel' in self.pars:
            self.zpanel.load_parameters(self.pars['ZernikePanel'])

    def save_parameters(self, asflat=False):
        self.pars['ZernikeControl'] = self.zcontrol.save_parameters(
            asflat=asflat)
        self.pars['ZernikePanel'] = self.zpanel.save_parameters()
        return deepcopy(self.pars)

    def add_lower(self, layout):
        def hold():
            self.mutex.lock()
            self.setDisabled(True)

        def release():
            self.setDisabled(False)
            self.mutex.unlock()

        def hand_load():
            def f():
                hold()
                fileName, _ = QFileDialog.getOpenFileName(
                    self, 'Select a parameters file',
                    filter='JSON (*.json);;All Files (*)')
                if fileName:
                    try:
                        calibration = self.pars['calibration']
                        with open(fileName, 'r') as f:
                            self.pars = json.load(f)
                        self.pars['calibration'] = calibration
                        self.instance_control()
                        if 'ZernikePanel' in self.pars:
                            self.zpanel.load_parameters(
                                self.pars['ZernikePanel'])
                    except Exception as ex:
                        self.log.error(f'error loading parameters {str(ex)}')
                        QMessageBox.information(
                            self, 'error loading parameters', str(ex))
                release()
            return f

        def hand_calib():
            def f():
                hold()
                fileName, _ = QFileDialog.getOpenFileName(
                    self, 'Select a calibration file',
                    filter='H5 (*.h5);;All Files (*)')
                if fileName:
                    try:
                        with File(fileName, 'r') as f:
                            self.calib = WeightedLSCalib.load_h5py(
                                f, lazy_cart_grid=True)
                        self.instance_control()
                        if 'ZernikePanel' in self.pars:
                            self.zpanel.load_parameters(
                                self.pars['ZernikePanel'])
                    except Exception as ex:
                        self.log.error(f'error loading calibration {str(ex)}')
                        QMessageBox.information(
                            self, 'error loading calibration', str(ex))
                release()
            return f

        def hand_save(flat):
            def f():
                fdiag, _ = QFileDialog.getSaveFileName(
                    self, 'Save parameters', directory=(
                        datetime.now().strftime('%Y%m%d_%H%M%S_') +
                        self.zcontrol.calib.dm_serial + '.json'),
                    filter='JSON (*.json);;All Files (*)')
                if fdiag:
                    try:
                        with open(fdiag, 'w') as f:
                            json.dump(
                                self.save_parameters(flat), f,
                                sort_keys=True, indent=4)
                    except Exception as ex:
                        self.log.error(f'error saving parameters {str(ex)}')
                        QMessageBox.information(
                            self, 'error saving parameters', str(ex))
            return f

        def hand_flat():
            def f(b):
                self.zcontrol.flat_on = b
                self.write_dm(self.zpanel.z)
            return f

        calibname = QLabel(self.pars['calibration'])
        calibname.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(calibname, 2, 0, 1, 4)

        bplot = QPushButton('plot')

        bcalib = QPushButton('load calib')
        bsave = QPushButton('save params')
        bsaveflat = QPushButton('save flat')
        bload = QPushButton('load params')
        bflat = QCheckBox('flat')
        bflat.setChecked(self.zcontrol.flat_on)
        bflipx = QCheckBox('flipx')
        bflipx.setChecked(self.zcontrol.pars['flipx'])
        bflipy = QCheckBox('flipy')
        bflipy.setChecked(self.zcontrol.pars['flipy'])
        lerotate = QLineEdit(str(self.zcontrol.pars['rotate']))

        g1 = QGroupBox()
        l1 = QGridLayout()
        l1.addWidget(bflat, 0, 0)
        l1.addWidget(bflipx, 0, 1)
        l1.addWidget(bflipy, 0, 2)
        l1.addWidget(QLabel('rotate'), 0, 3)
        l1.addWidget(lerotate, 0, 4)
        l1.addWidget(bplot, 0, 5)
        g1.setLayout(l1)
        layout.addWidget(g1, 3, 0, 1, 4)

        g2 = QGroupBox()
        l2 = QGridLayout()
        l2.addWidget(bcalib, 0, 0)
        l2.addWidget(bload, 0, 1)
        l2.addWidget(bsave, 0, 2)
        l2.addWidget(bsaveflat, 0, 3)
        g2.setLayout(l2)
        layout.addWidget(g2, 4, 0, 1, 4)

        bcalib.clicked.connect(hand_calib())
        bsave.clicked.connect(hand_save(False))
        bsaveflat.clicked.connect(hand_save(True))
        bload.clicked.connect(hand_load())
        bflat.stateChanged.connect(hand_flat())

        def plotf():
            def f():
                self.mutex.lock()
                p = PlotCoeffs()
                p.set_data(self.zcontrol.u, self.zcontrol.z)
                p.exec_()
                self.mutex.unlock()
            return f

        bplot.clicked.connect(plotf())

        def handle1(name, cb):
            def f(b):
                self.control_options.update('ZernikeControl', name, b)
                self.zcontrol.pars[name] = b
                self.zcontrol.transform_pupil()
                if self.zcontrol.P is not None:
                    self.zpanel.P = self.zcontrol.P
                else:
                    self.zpanel.P = 1
                self.zpanel.update_phi_plot()
                self.write_dm(self.zpanel.z)
            return f

        def handle2():
            def f():
                try:
                    f = float(lerotate.text())
                    self.control_options.update('ZernikeControl', 'rotate', f)
                    self.zcontrol.pars['rotate'] = f
                    self.zcontrol.transform_pupil()
                    if self.zcontrol.P is not None:
                        self.zpanel.P = self.zcontrol.P
                    else:
                        self.zpanel.P = 1
                    self.zpanel.update_phi_plot()
                    self.write_dm(self.zpanel.z)
                except ValueError:
                    lerotate.setText(str(self.zcontrol.pars['rotate']))
            return f

        bflipx.clicked.connect(handle1('flipx', bflipx))
        bflipy.clicked.connect(handle1('flipy', bflipy))
        lerotate.editingFinished.connect(handle2())

        self.bflat = bflat

    def acquire_control(self, h5f):
        self.sig_acquire.emit((h5f,))

        try:
            cname, pars = self.control_options.get_options()
            pars['enabled'] = self.control_enabled
            pars['flat_on'] = 1
            pars['uflat'] = self.zcontrol.u.tolist()
            pars['u'] = np.zeros_like(self.zcontrol.u).tolist()
            pars['all'] = 0
            c = control.new_control(
                self.zcontrol.dm,
                self.zcontrol.calib,
                cname, pars, h5f)

            def make_gui_callback():
                def f():
                    self.sig_draw.emit((c.u,))
                return f

            c.gui_callback = make_gui_callback()
        except Exception as ex:
            self.sig_release.emit((None, h5f))
            raise ex

        return c

    def release_control(self, control, h5f):
        self.sig_release.emit((control, h5f))

    def enable_control(self, b):
        self.control_enabled = b

    def lock_gui(self):
        self.sig_lock.emit()

    def unlock_gui(self):
        self.sig_unlock.emit()

    def closeEvent(self, event):
        if self.can_close:
            if self.app:
                self.app.quit()
            else:
                event.accept()
        else:
            event.ignore()


def add_arguments(parser):
    add_dm_parameters(parser)
    parser.add_argument(
        '--dm-calibration', type=argparse.FileType('rb'), default=None,
        metavar='HDF5')
    parser.add_argument(
        '--dm-parameters', type=argparse.FileType('r'), default=None,
        metavar='JSON', help='Load a previous configuration file')


def load_parameters(app, args):
    def quit(str1):
        e = QErrorMessage()
        e.showMessage(str1)
        sys.exit(e.exec_())

    if args.no_params:
        # blank parameters
        pars = {}
    elif args.params is not None:
        # command-line pars
        try:
            pars = json.load(args.params)
        except Exception:
            quit('cannot load ' + args.params.name)
    else:
        pars = {}

    if args.dm_calibration:
        args.dm_calibration.close()
        pars['calibration'] = path.abspath(args.dm_calibration.name)
        args.dm_calibration = pars['calibration']

    def choose_calib_file():
        fileName, _ = QFileDialog.getOpenFileName(
            None, 'Select a calibration', filter='H5 (*.h5);;All Files (*)')
        if not fileName:
            sys.exit()
        else:
            pars['calibration'] = path.abspath(fileName)

    if 'calibration' not in pars:
        choose_calib_file()

    try:
        dminfo = WeightedLSCalib.query_calibration(pars['calibration'])
    except Exception as e:
        quit(str(e))

    return dminfo, pars


def new_zernike_window(app, args, pars={}):
    def quit(str1):
        e = QErrorMessage()
        e.showMessage(str1)
        sys.exit(e.exec_())

    calib_file = None

    # argparse specified parameters can override pars
    if args.dm_parameters is not None:
        d = json.loads(args.dm_parameters.read())
        pars = {**pars, **d}
        args.dm_parameters = args.dm_parameters.name

    # calibration from pars
    if 'calibration' in pars:
        calib_file = pars['calibration']
    # calibration from argparse
    if args.dm_calibration is not None:
        calib_file = args.dm_calibration.name
        args.dm_calibration = calib_file
    # no calibration found, ask user for calibration
    if calib_file is None:
        fileName, _ = QFileDialog.getOpenFileName(
            None, 'Select a DM calibration', filter='H5 (*.h5);;All Files (*)')
        if not fileName:
            sys.exit()
        else:
            calib_file = path.abspath(fileName)
    pars['calibration'] = calib_file

    try:
        dminfo = WeightedLSCalib.query_calibration(calib_file)
    except Exception as e:
        quit(str(e))

    calib_dm_name = dminfo[0]
    calib_dm_transform = dminfo[1]

    if args.dm_name is None:
        args.dm_name = calib_dm_name
    dm = open_dm(app, args, calib_dm_transform)

    try:
        with File(calib_file, 'r') as f:
            calib = WeightedLSCalib.load_h5py(f, lazy_cart_grid=True)
    except Exception as e:
        quit(f'error loading calibration {pars["calibration"]}: {str(e)}')

    zwindow = ZernikeWindow(None, dm, calib, pars)
    zwindow.show()

    return zwindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='Zernike DM control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_log_parameters(parser)
    add_arguments(parser)
    parser.add_argument(
        '--params', type=argparse.FileType('rb'), default=None, metavar='JSON')
    parser.add_argument('--no-params', action='store_true')
    args = parser.parse_args(args[1:])
    setup_logging(args)

    dminfo, pars = load_parameters(app, args)
    calib_dm_name = dminfo[0]
    calib_dm_transform = dminfo[1]

    if args.dm_name is None:
        args.dm_name = calib_dm_name
    dm = open_dm(app, args, calib_dm_transform)

    try:
        with File(pars['calibration'], 'r') as f:
            calib = WeightedLSCalib.load_h5py(f, lazy_cart_grid=True)
    except Exception as e:
        quit(
            f'error loading calibration {pars["calibration"]}: {str(e)}')

    zwindow = ZernikeWindow(app, dm, calib, pars)
    zwindow.show()

    sys.exit(app.exec_())
