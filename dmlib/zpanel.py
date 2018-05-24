#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from numpy.linalg import norm
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QKeySequence
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QGroupBox, QGridLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QScrollArea, QSlider, QDoubleSpinBox, QFrame,
    QErrorMessage, QApplication, QMainWindow, QSplitter, QShortcut,
    )

from sensorless.czernike import RZern

from dmplot import DMPlot

import version


class ZernikePanel(QWidget):

    callback = None
    nmodes = 1
    units = 'rad'
    status = None
    mul = 1.0
    fig = None
    ax = None
    im = None
    cb = None
    shape = (128, 128)
    settings = {'zernike_labels': {}}

    def update_gui(self):
        phi = self.mul*self.rzern.eval_grid(self.z).reshape(
            self.shape, order='F')
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

        if self.callback:
            self.callback(self.z)

    def __init__(
            self, wavelength, n_radial, callback=None, settings=None,
            parent=None):
        super().__init__(parent=parent)

        if settings:
            self.settings = {**self.settings, **settings}

        self.rzern = RZern(n_radial)
        dd = np.linspace(-1, 1, self.shape[0])
        xv, yv = np.meshgrid(dd, dd)
        self.rzern.make_cart_grid(xv, yv)
        self.rad_to_nm = wavelength/(2*np.pi)
        self.nmodes = min((21, self.rzern.nk))
        self.callback = callback

        if 'z' not in self.settings:
            self.z = np.zeros((self.rzern.nk,))
        else:
            self.z = np.array(self.settings['z'])

        zernike_rows = list()
        fto100mul = 100

        top1 = QGroupBox('phase')
        toplay1 = QGridLayout()
        top1.setLayout(toplay1)
        self.fig = FigureCanvas(Figure(figsize=(2, 2)))
        self.ax = self.fig.figure.add_subplot(1, 1, 1)
        phi = self.rzern.eval_grid(self.z).reshape(self.shape, order='F')
        self.im = self.ax.imshow(phi, origin='lower')
        self.cb = self.fig.figure.colorbar(self.im)
        self.cb.locator = ticker.MaxNLocator(nbins=5)
        self.cb.update_ticks()
        self.ax.axis('off')
        self.status = QLabel('')
        toplay1.addWidget(self.fig, 0, 0)
        toplay1.addWidget(self.status, 1, 0)

        top = QGroupBox('Zernike')
        toplay = QGridLayout()
        top.setLayout(toplay)
        labzm = QLabel('shown modes')
        lezm = QLineEdit(str(self.nmodes))
        lezm.setMaximumWidth(50)
        lezm.setValidator(QIntValidator(1, self.rzern.nk))

        brad = QCheckBox('rad')
        brad.setChecked(True)
        reset = QPushButton('reset')
        toplay.addWidget(labzm, 0, 0)
        toplay.addWidget(lezm, 0, 1)
        toplay.addWidget(brad, 0, 2)
        toplay.addWidget(reset, 0, 3)

        scroll = QScrollArea()
        toplay.addWidget(scroll, 1, 0, 1, 5)
        scroll.setWidget(QWidget())
        scrollLayout = QGridLayout(scroll.widget())
        scroll.setWidgetResizable(True)

        def fto100(f, amp):
            maxrad = float(amp.text())
            return int((f + maxrad)/(2*maxrad)*fto100mul)

        def make_hand_spinbox(slider, ind, amp):
            def f(r):
                slider.blockSignals(True)
                slider.setValue(fto100(r, amp))
                slider.blockSignals(False)

                self.z[ind] = r
                self.update_gui()
            return f

        def make_hand_amp(spinbox, slider, le, i):
            def f():
                amp = float(le.text())
                spinbox.setRange(-amp, amp)
                spinbox.setValue(spinbox.value())
                slider.setValue(fto100(self.z[i], le))
            return f

        def make_hand_lab(le, settings, i):
            def f():
                settings['zernike_labels'][str(i)] = le.text()
            return f

        def make_hand_slider(s, amp):
            def f(t):
                maxrad = float(amp.text())
                s.setValue(t/fto100mul*(2*maxrad) - maxrad)
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

        def update_zernike_rows():
            tick_interval = 20
            single_step = 0.01

            mynk = self.nmodes
            ntab = self.rzern.ntab
            mtab = self.rzern.mtab
            if len(zernike_rows) < mynk:
                for i in range(len(zernike_rows), mynk):
                    lab = QLabel(
                        'Z<sub>{}</sub> Z<sub>{}</sub><sup>{}</sup>'.format(
                            i + 1, ntab[i], mtab[i]))
                    slider = QSlider(Qt.Horizontal)
                    spinbox = QDoubleSpinBox()
                    maxamp = max((8, self.z[i]))
                    if str(i) in self.settings['zernike_labels'].keys():
                        zname = self.settings['zernike_labels'][str(i)]
                    else:
                        zname = default_zernike_name(i + 1, ntab[i], mtab[i])
                        self.settings['zernike_labels'][str(i)] = zname
                    lbn = QLineEdit(zname)
                    lbn.setMaximumWidth(120)
                    amp = QLineEdit(str(maxamp))
                    amp.setMaximumWidth(50)
                    val = QDoubleValidator()
                    val.setBottom(0.0)
                    amp.setValidator(val)

                    slider.setMinimum(0)
                    slider.setMaximum(fto100mul)
                    slider.setFocusPolicy(Qt.StrongFocus)
                    slider.setTickPosition(QSlider.TicksBothSides)
                    slider.setTickInterval(tick_interval)
                    slider.setSingleStep(single_step)
                    slider.setValue(fto100(self.z[i], amp))
                    spinbox.setRange(-maxamp, maxamp)
                    spinbox.setSingleStep(single_step)
                    spinbox.setValue(self.z[i])

                    hand_slider = make_hand_slider(spinbox, amp)
                    hand_spinbox = make_hand_spinbox(slider, i, amp)
                    hand_amp = make_hand_amp(spinbox, slider, amp, i)
                    hand_lab = make_hand_lab(lbn, self.settings, i)

                    slider.valueChanged.connect(hand_slider)
                    spinbox.valueChanged.connect(hand_spinbox)
                    amp.editingFinished.connect(hand_amp)
                    lbn.editingFinished.connect(hand_lab)

                    scrollLayout.addWidget(lab, i, 0)
                    scrollLayout.addWidget(lbn, i, 1)
                    scrollLayout.addWidget(spinbox, i, 2)
                    scrollLayout.addWidget(slider, i, 3)
                    scrollLayout.addWidget(amp, i, 4)

                    zernike_rows.append((
                        lab, slider, spinbox, hand_slider, hand_spinbox, amp,
                        hand_amp, lbn, hand_lab))

                assert(len(zernike_rows) == mynk)

            elif len(zernike_rows) > mynk:
                for i in range(len(zernike_rows) - 1, mynk - 1, -1):
                    tup = zernike_rows.pop()
                    lab, slider, spinbox, h1, h2, amp, h3, lbn, h4 = tup

                    scrollLayout.removeWidget(lab)
                    scrollLayout.removeWidget(lbn)
                    scrollLayout.removeWidget(spinbox)
                    scrollLayout.removeWidget(slider)
                    scrollLayout.removeWidget(amp)

                    slider.valueChanged.disconnect(h1)
                    spinbox.valueChanged.disconnect(h2)
                    amp.editingFinished.disconnect(h3)
                    lbn.editingFinished.disconnect(h4)

                    lab.setParent(None)
                    slider.setParent(None)
                    spinbox.setParent(None)
                    amp.setParent(None)
                    lbn.setParent(None)

                assert(len(zernike_rows) == mynk)

        def reset_fun():
            self.z *= 0.
            for t in zernike_rows:
                t[2].setValue(0.0)
            self.update_gui()

        def change_nmodes():
            try:
                ival = int(lezm.text())
                assert(ival > 1)
                assert(ival <= self.rzern.nk)
            except Exception as exp:
                lezm.setText(str(self.nmodes))
                return

            self.nmodes = ival
            update_zernike_rows()
            self.update_gui()
            lezm.setText(str(self.nmodes))

        def f2():
            def f(b):
                if b:
                    self.units = 'rad'
                    self.mul = 1.0
                else:
                    self.units = 'nm'
                    self.mul = self.rad_to_nm
                self.update_gui()
            return f

        update_zernike_rows()

        brad.stateChanged.connect(f2())
        reset.clicked.connect(reset_fun)
        lezm.editingFinished.connect(change_nmodes)

        l1 = QGridLayout()
        l1.addWidget(top1, 0, 0)
        l1.addWidget(top, 1, 0)
        self.setLayout(l1)


class ZernikeWindow(QMainWindow):

    settings = {}

    def make_figs(self):
        fig = FigureCanvas(Figure(figsize=(2, 2)))
        ax = fig.figure.subplots(2, 1)
        ima = self.dmplot.draw(ax[0], control.u)
        img = ax[1].imshow(self.dmplot.compute_gauss(control.u))
        ax[0].axis('off')
        ax[1].axis('off')

        return ax, ima, img, fig

    def __init__(self, control, settings={}, parent=None):
        super().__init__()

        self.setWindowTitle('ZernikeWindow ' + version.__version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        if settings:
            self.settings = {**self.settings, **settings}

        if 'ZernikePanel' not in settings:
            settings['ZernikePanel'] = {}

        self.dmplot = DMPlot()
        self.dmplot.update_txs(calib.dmplot_txs)

        ax, ima, img, fig = self.make_figs()
        lab = QLabel()
        lab2 = QLabel(self.settings['calibration'])

        def f1():
            def f2(z):
                control.write(z)

                if control.saturation:
                    tmp = 'SAT'
                else:
                    tmp = 'OK'
                lab.setText('DM {} u [{:+0.3f} {:+0.3f}] {}'.format(
                    control.dm.get_serial_number(),
                    control.u.min(), control.u.max(), tmp))

                ima.set_data(self.dmplot.compute_pattern(control.u))
                g = self.dmplot.compute_gauss(control.u)
                img.set_data(g)
                img.set_clim(g.min(), g.max())
                ax[0].figure.canvas.draw()
            return f2

        zpanel = ZernikePanel(
            control.calib.wavelength, control.calib.get_rzern().n,
            callback=f1(), settings=settings['ZernikePanel'])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(zpanel)
        split.addWidget(fig)

        bopen = QPushButton('open')

        central = QFrame()
        layout = QGridLayout()
        central.setLayout(layout)
        layout.addWidget(split, 0, 0, 1, 4)
        layout.addWidget(lab, 1, 0, 1, 4)
        layout.addWidget(lab2, 2, 0, 1, 4)
        layout.addWidget(bopen, 3, 0)

        self.setCentralWidget(central)


if __name__ == '__main__':
    import argparse
    import json
    from os import path
    from pathlib import Path
    from h5py import File

    from core import add_dm_parameters, open_dm
    from calibration import WeightedLSCalib
    from control import ZernikeControl

    app = QApplication(sys.argv)
    args = app.arguments()
    parser = argparse.ArgumentParser(
        description='Zernike DM control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_dm_parameters(parser)
    args = parser.parse_args(args[1:])

    savepath = path.join(Path.home(), '.zpanel.json')
    try:
        with open(savepath, 'r') as f:
            settings = json.load(f)
    except Exception:
        settings = {}

    def quit(str1):
        e = QErrorMessage()
        e.showMessage(str1)
        sys.exit(app.exec_())

    if 'calibration' not in settings:
        fileName, _ = QFileDialog.getOpenFileName(
            None, 'Select a calibration', '', 'H5 (*.h5);;All Files (*)')
        if not fileName:
            sys.exit()
        else:
            settings['calibration'] = fileName

    try:
        with File(fileName, 'r') as f:
            if 'WeightedLSCalib' not in f:
                quit(
                    fileName +
                    ' does not seem like a calibration file')
            else:
                calib = WeightedLSCalib.load_h5py(f)
    except Exception as e:
        quit(str(e) + ' loading ' + fileName)

    if args.dm_name is None:
        args.dm_name = calib.dm_serial

    dm = open_dm(app, args, calib.dm_transform)
    control = ZernikeControl(dm, calib)

    zwindow = ZernikeWindow(control, settings)
    zwindow.show()

    sys.exit(app.exec_())
