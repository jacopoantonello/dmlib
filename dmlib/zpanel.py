#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import subprocess
import json
import argparse

from os import path
from pathlib import Path
from h5py import File
from numpy.linalg import norm
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime

from PyQt5.QtCore import Qt, QMutex, pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QKeySequence
from PyQt5.QtWidgets import (
    QWidget, QFileDialog, QGroupBox, QGridLayout, QLabel, QPushButton,
    QLineEdit, QCheckBox, QScrollArea, QSlider, QDoubleSpinBox, QFrame,
    QErrorMessage, QApplication, QMainWindow, QSplitter, QShortcut,
    QMessageBox, QSizePolicy, QInputDialog,
    )

from zernike.czernike import RZern

from dmlib.version import __version__
from dmlib.dmplot import DMPlot
from dmlib.core import (
    add_dm_parameters, open_dm, add_log_parameters, setup_logging)
from dmlib.calibration import WeightedLSCalib
from dmlib.control import ZernikeControl


fto100mul = 100


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


class RelSlider:

    def __init__(self, val, cb):
        self.old_val = None
        self.fto100mul = 100
        self.cb = cb

        self.sba = QDoubleSpinBox()
        self.sba.setValue(val)
        self.sba.setSingleStep(1.25e-3)
        self.sba.setToolTip('Effective value')
        self.sba.setMinimum(-1000)
        self.sba.setMaximum(1000)
        self.sba.setDecimals(3)

        self.qsr = QSlider(Qt.Horizontal)
        self.qsr.setMinimum(-100)
        self.qsr.setMaximum(100)
        self.qsr.setValue(0)
        self.qsr.setToolTip('Drag to apply relative delta')

        self.sbm = QDoubleSpinBox()
        self.sbm.setValue(4.0)
        self.sbm.setSingleStep(1.25e-3)
        self.sbm.setToolTip('Maximum relative delta')
        self.sbm.setDecimals(3)
        self.sbm.setMinimum(0.001)

        def sba_cb():
            def f():
                self.block()
                self.cb(self.sba.value())
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

        self.sba.editingFinished.connect(self.sba_cb)
        self.qsr.valueChanged.connect(self.qs1_cb)
        self.qsr.sliderPressed.connect(self.qs1_start)
        self.qsr.sliderReleased.connect(self.qs1_end)

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

        self.sba.editingFinished.disconnect(self.sba_cb)
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
    pars = {'zernike_labels': {}, 'shown_modes': 21}

    def save_parameters(self, merge={}):
        d = {**merge, **self.pars}
        d['z'] = self.z.tolist()
        d['shown_modes'] = len(self.zernike_rows)
        return d

    def update_controls(self):
        for i, t in enumerate(self.zernike_rows):
            slider = t[1]

            slider.disable()
            slider.set_value(self.z[i])
            slider.enable()

    def update_gui(self, run_callback=True):
        phi = self.mul*self.rzern.matrix(self.rzern.eval_grid(self.z))
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

    def __init__(
            self, wavelength, n_radial, callback=None, pars=None,
            parent=None):
        super().__init__(parent=parent)

        if pars:
            self.pars = {**self.pars, **pars}

        self.rzern = RZern(n_radial)
        dd = np.linspace(-1, 1, self.shape[0])
        xv, yv = np.meshgrid(dd, dd)
        self.rzern.make_cart_grid(xv, yv)
        self.rad_to_nm = wavelength/(2*np.pi)
        self.nmodes = min((self.pars['shown_modes'], self.rzern.nk))
        self.callback = callback

        self.z = np.zeros((self.rzern.nk,))
        if 'z' in self.pars:
            try:
                min1 = min(self.z.size, len(self.pars['z']))
                self.z[:min1] = np.array(self.pars['z'])[:min1]
            except Exception:
                pass

        zernike_rows = list()
        self.zernike_rows = zernike_rows

        top1 = QGroupBox('phase')
        toplay1 = QGridLayout()
        top1.setLayout(toplay1)
        self.fig = FigureCanvas(Figure(figsize=(2, 2)))
        self.ax = self.fig.figure.add_subplot(1, 1, 1)
        phi = self.rzern.matrix(self.rzern.eval_grid(self.z))
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
        lezmval = MyQIntValidator(1, self.rzern.nk)
        lezmval.setFixup(self.nmodes)
        lezm.setValidator(lezmval)

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

        def make_hand_slider(ind):
            def f(r):
                self.z[ind] = r
                self.update_gui()
            return f

        def make_hand_lab(le, pars, i):
            def f():
                pars['zernike_labels'][str(i)] = le.text()
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
            mynk = self.nmodes
            ntab = self.rzern.ntab
            mtab = self.rzern.mtab
            if len(zernike_rows) < mynk:
                for i in range(len(zernike_rows), mynk):
                    lab = QLabel(
                        f'Z<sub>{i + 1}</sub> ' +
                        f'Z<sub>{ntab[i]}</sub><sup>{mtab[i]}</sup>')
                    slider = RelSlider(self.z[i], make_hand_slider(i))

                    if str(i) in self.pars['zernike_labels'].keys():
                        zname = self.pars['zernike_labels'][str(i)]
                    else:
                        zname = default_zernike_name(i + 1, ntab[i], mtab[i])
                        self.pars['zernike_labels'][str(i)] = zname
                    lbn = QLineEdit(zname)
                    lbn.setMaximumWidth(120)
                    hand_lab = make_hand_lab(lbn, self.pars, i)
                    lbn.editingFinished.connect(hand_lab)

                    scrollLayout.addWidget(lab, i, 0)
                    scrollLayout.addWidget(lbn, i, 1)
                    slider.add_to_layout(scrollLayout, i, 2)

                    zernike_rows.append((lab, slider, lbn, hand_lab))

                assert(len(zernike_rows) == mynk)

            elif len(zernike_rows) > mynk:
                for i in range(len(zernike_rows) - 1, mynk - 1, -1):
                    lab, slider, lbn, hand_lab = zernike_rows.pop()

                    scrollLayout.removeWidget(lab)
                    slider.remove_from_layout(scrollLayout)
                    scrollLayout.removeWidget(lbn)

                    lbn.editingFinished.disconnect(hand_lab)
                    lab.setParent(None)
                    lbn.setParent(None)

                assert(len(zernike_rows) == mynk)

        def reset_fun():
            self.z *= 0.
            self.update_controls()
            self.update_gui()

        def change_nmodes():
            try:
                ival = int(lezm.text())
                assert(ival > 0)
                assert(ival <= self.rzern.nk)
            except Exception:
                lezm.setText(str(self.nmodes))
                return

            if ival != self.nmodes:
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

        split = QSplitter(Qt.Vertical)
        split.addWidget(top1)
        split.addWidget(top)
        l1 = QGridLayout()
        l1.addWidget(split)
        self.setLayout(l1)


class ZernikeWindow(QMainWindow):

    sig_acquire = pyqtSignal(tuple)
    sig_release = pyqtSignal(tuple)

    can_close = True
    pars = {}

    def __init__(self, app, control, pars={}, parent=None):
        super().__init__()
        self.control = control
        self.app = app
        self.mutex = QMutex()

        self.setWindowTitle('ZernikeWindow ' + __version__)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)

        if pars:
            self.pars = {**self.pars, **pars}

        if 'ZernikePanel' not in pars:
            pars['ZernikePanel'] = {}
        if 'flat_on' in pars:
            control.flat_on = int(pars['flat_on'])

        self.dmplot = DMPlot()
        self.dmplot.update_txs(control.calib.dmplot_txs)
        dmstatus = QLabel()

        def make_figs():
            fig = FigureCanvas(Figure(figsize=(2, 2)))
            ax = fig.figure.subplots(2, 1)
            ima = self.dmplot.draw(ax[0], self.control.u)
            img = ax[1].imshow(self.dmplot.compute_gauss(self.control.u))
            ax[0].axis('off')
            ax[1].axis('off')

            return ax, ima, img, fig

        ax, ima, img, fig = make_figs()

        def make_write_fun():
            def f(z):
                control.write(z)

                if control.saturation:
                    tmp = 'SAT'
                else:
                    tmp = 'OK'
                dmstatus.setText('DM {} u [{:+0.3f} {:+0.3f}] {}'.format(
                    control.dm.get_serial_number(),
                    control.u.min(), control.u.max(), tmp))

                ima.set_data(self.dmplot.compute_pattern(control.u))
                g = self.dmplot.compute_gauss(control.u)
                img.set_data(g)
                img.set_clim(g.min(), g.max())
                ax[0].figure.canvas.draw()
            return f

        write_fun = make_write_fun()

        self.zpanel = ZernikePanel(
            control.calib.wavelength, control.calib.get_rzern().n,
            callback=write_fun, pars=pars['ZernikePanel'])

        def make_select_cb():
            def f(e):
                self.mutex.lock()
                if e.inaxes is not None:
                    ind = self.dmplot.index_actuator(e.xdata, e.ydata)
                    if ind != -1:
                        val, ok = QInputDialog.getDouble(
                            self, f'Actuator {ind} ' + str(ind),
                            'range [-1, 1]', control.u[ind],
                            -1., 1., 4)
                        if ok:
                            control.u[ind] = val
                            self.zpanel.z[:] = control.u2z()
                            self.zpanel.update_controls()
                            self.zpanel.update_gui()
                self.mutex.unlock()
            return f

        ax[0].figure.canvas.callbacks.connect(
            'button_press_event', make_select_cb())

        write_fun(self.zpanel.z)

        dmstatus.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.zpanel)
        split.addWidget(fig)

        central = QFrame()
        layout = QGridLayout()
        central.setLayout(layout)
        layout.addWidget(split, 0, 0, 1, 4)
        layout.addWidget(dmstatus, 1, 0, 1, 4)

        self.add_lower(layout, write_fun)
        self.setCentralWidget(central)
        self.write_fun = write_fun

        def make_release_hand():
            def f(t):
                self.control.u[:] = t[0].u
                self.zpanel.z[:] = self.control.u2z()
                self.zpanel.update_controls()
                self.zpanel.update_gui()
                self.setEnabled(True)
                self.can_close = True
                self.mutex.unlock()
            return f

        def make_acquire_hand():
            def f(t):
                self.mutex.lock()
                self.can_close = False
                self.setEnabled(False)
            return f

        self.sig_release.connect(make_release_hand())
        self.sig_acquire.connect(make_acquire_hand())

    def add_lower(self, layout, write_fun):
        def f3(name, glob, param):
            def f():
                self.setDisabled(True)
                fileName, _ = QFileDialog.getOpenFileName(
                    None, name, '', glob)
                if not fileName:
                    self.setDisabled(False)
                    return
                else:
                    self.close()
                    self.control.dm.close()
                    myargs = list(sys.argv)
                    myargs.append(param)
                    myargs.append(fileName)
                    subprocess.Popen([sys.executable, *myargs])
            return f

        def f4():
            def f():
                fdiag, _ = QFileDialog.getSaveFileName(directory=(
                    self.control.calib.dm_serial +
                    datetime.now().strftime('_%Y%m%d_%H%M%S.json')))
                if fdiag:
                    try:
                        with open(fdiag, 'w') as f:
                            json.dump(self.save_parameters(), f)
                    except Exception as e:
                        QMessageBox.information(self, 'error', str(e))
            return f

        def f5():
            def f(b):
                self.control.flat_on = b
                write_fun(self.zpanel.z)
            return f

        calibname = QLabel(self.pars['calibration'])
        calibname.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(calibname, 2, 0, 1, 4)

        bcalib = QPushButton('load calibration')
        bsave = QPushButton('save parameters')
        bload = QPushButton('load parameters')
        bflat = QCheckBox('flat')
        bflat.setChecked(self.control.flat_on)

        bcalib.clicked.connect(f3(
            'Select a calibration file', 'H5 (*.h5);;All Files (*)',
            '--calibration'))
        bsave.clicked.connect(f4())
        bload.clicked.connect(f3(
            'Select a parameters file', 'JSON (*.json);;All Files (*)',
            '--parameters'))
        bflat.stateChanged.connect(f5())

        layout.addWidget(bcalib, 3, 0)
        layout.addWidget(bsave, 3, 1)
        layout.addWidget(bload, 3, 2)
        layout.addWidget(bflat, 3, 3)

    def save_parameters(self, merge={}):
        self.pars['ZernikePanel'] = self.zpanel.save_parameters()
        self.pars['flat_on'] = self.control.flat_on
        d = {**merge, **self.pars}
        return d

    def acquire_control(self, h5f):
        self.sig_acquire.emit((h5f,))

        def make_gui_update():
            def f(u):
                self.control.u[:] = u
                self.zpanel.z[:] = self.control.u2z()
                self.zpanel.update_gui()
            return f

        class DummyControl(self.control.__class__):

            def set_gui_update(self, gui_update):
                self.gui_update = gui_update

            def write(self, x):
                super().write(x)
                self.gui_update(self.u)

        pars = {
            'control': {
                'Zernike': {
                    'include': [],
                    'exclude': [1, 2, 3, 4],
                    'min': 5,
                    'max': 6,
                    'all': 0,
                    }
                }
            }

        control = DummyControl(
            self.control.dm, self.control.calib, pars=pars, h5f=h5f)
        control.set_gui_update(make_gui_update())
        return control

    def release_control(self, control, h5f):
        self.sig_release.emit((control, h5f))

    def closeEvent(self, event):
        if self.can_close:
            with open(path.join(Path.home(), '.zpanel.json'), 'w') as f:
                json.dump(self.save_parameters(), f)
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


def load_parameters(app, args, last_pars='.zpanel.json'):
    def quit(str1):
        e = QErrorMessage()
        e.showMessage(str1)
        sys.exit(e.exec_())

    if args.no_params:
        # blank parameters
        pars = {}
    elif args.params is None:
        # last run parameters
        savepath = path.join(Path.home(), last_pars)
        try:
            with open(savepath, 'r') as f:
                pars = json.load(f)
        except Exception:
            pars = {}
    else:
        # command-line pars
        try:
            pars = json.load(args.params)
        except Exception:
            quit('cannot load ' + args.params.name)

    if args.dm_calibration:
        args.dm_calibration.close()
        pars['calibration'] = args.dm_calibration.name
        args.dm_calibration = pars['calibration']

    def choose_calib_file():
        fileName, _ = QFileDialog.getOpenFileName(
            None, 'Select a calibration', '', 'H5 (*.h5);;All Files (*)')
        if not fileName:
            sys.exit()
        else:
            pars['calibration'] = fileName

    if 'calibration' not in pars:
        choose_calib_file()

    try:
        dminfo = WeightedLSCalib.query_calibration(pars['calibration'])
    except Exception as e:
        quit(str(e))

    return dminfo, pars


def new_zernike_window(app, args, params={}):
    # args can override params

    def quit(str1):
        e = QErrorMessage()
        e.showMessage(str1)
        sys.exit(e.exec_())

    calib_file = None

    if 'calibration' in params:
        calib_file = params['calibration']
    if args.dm_calibration is not None:
        calib_file = args.dm_calibration.name
        args.dm_calibration = calib_file

    if calib_file is None:
        fileName, _ = QFileDialog.getOpenFileName(
            None, 'Select a DM calibration', '', 'H5 (*.h5);;All Files (*)')
        if not fileName:
            sys.exit()
        else:
            calib_file = fileName

    params['calibration'] = calib_file

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
        quit('error loading calibration {}: {}'.format(
            params['calibration'], str(e)))

    control = ZernikeControl(dm, calib)
    zwindow = ZernikeWindow(None, control, params)
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
        '--params', type=argparse.FileType('rb'), default=None,
        metavar='JSON')
    parser.add_argument('--no-params', action='store_true')
    args = parser.parse_args(args[1:])
    setup_logging(args)

    dminfo, params = load_parameters(app, args)
    calib_dm_name = dminfo[0]
    calib_dm_transform = dminfo[1]

    if args.dm_name is None:
        args.dm_name = calib_dm_name
    dm = open_dm(app, args, calib_dm_transform)

    try:
        with File(params['calibration'], 'r') as f:
            calib = WeightedLSCalib.load_h5py(f, lazy_cart_grid=True)
    except Exception as e:
        quit('error loading calibration {}: {}'.format(
            params['calibration'], str(e)))

    control = ZernikeControl(dm, calib)

    zwindow = ZernikeWindow(app, control, params)
    zwindow.show()

    sys.exit(app.exec_())
