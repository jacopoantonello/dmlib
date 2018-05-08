#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt  # noqa:

from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvas  # noqa:
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT  # noqa:
from matplotlib.figure import Figure  # noqa:

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # noqa:
from PyQt5.QtGui import (  # noqa:
    QImage, QPainter, QDoubleValidator, QIntValidator, QKeySequence,
    )
from PyQt5.QtWidgets import (  # noqa:
    QMainWindow, QDialog, QTabWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QGroupBox, QGridLayout, QCheckBox, QVBoxLayout, QFrame,
    QApplication, QShortcut, QSlider, QDoubleSpinBox, QToolBox,
    QWidget, QFileDialog, QScrollArea, QMessageBox, QSplitter,
    QInputDialog, QStyleFactory, QSizePolicy, QErrorMessage,
    )

from sensorless.czernike import RZern


class ZernikePanel(QWidget):

    fig = None
    ax = None
    im = None
    cb = None
    shape = (128, 128)
    settings = {'zernike_labels': {}}

    def update_gui(self):
        phi = self.rzern.eval_grid(self.z).reshape(self.shape, order='F')
        inner = phi[np.isfinite(phi)]
        self.im.set_data(phi)
        self.im.set_clim(inner.min(), inner.max())
        self.fig.figure.canvas.draw()

    def __init__(self, n_radial=5, settings=None):
        super().__init__()

        self.rzern = RZern(n_radial)
        dd = np.linspace(-1, 1, self.shape[0])
        xv, yv = np.meshgrid(dd, dd)
        self.rzern.make_cart_grid(xv, yv)
        self.z = np.zeros((self.rzern.nk,))

        if settings:
            self.settings = {**self.settings, **settings}

        zernike_rows = list()
        multiplier = 100

        top1 = QGroupBox('phase')
        toplay1 = QGridLayout()
        top1.setLayout(toplay1)
        self.fig = FigureCanvas(Figure(figsize=(2, 2)))
        self.ax = self.fig.figure.add_subplot(1, 1, 1)
        phi = self.rzern.eval_grid(self.z).reshape(self.shape, order='F')
        self.im = self.ax.imshow(phi)
        self.cb = self.fig.figure.colorbar(self.im)
        self.cb.locator = ticker.MaxNLocator(nbins=5)
        self.cb.update_ticks()
        self.ax.axis('off')
        toplay1.addWidget(self.fig)

        top = QGroupBox('Zernike')
        toplay = QGridLayout()
        top.setLayout(toplay)
        labzm = QLabel('max radial order')
        lezm = QLineEdit(str(self.rzern.n))
        lezm.setMaximumWidth(50)
        lezm.setValidator(QIntValidator(1, 255))

        reset = QPushButton('reset')
        toplay.addWidget(labzm, 0, 0)
        toplay.addWidget(lezm, 0, 1)
        toplay.addWidget(reset, 0, 3)

        scroll = QScrollArea()
        toplay.addWidget(scroll, 1, 0, 1, 5)
        scroll.setWidget(QWidget())
        scrollLayout = QGridLayout(scroll.widget())
        scroll.setWidgetResizable(True)

        def fto100(f, amp):
            maxrad = float(amp.text())
            return int((f + maxrad)/(2*maxrad)*multiplier)

        def update_coeff(slider, ind, amp):
            def f(r):
                slider.blockSignals(True)
                slider.setValue(fto100(r, amp))
                slider.blockSignals(False)

                self.z[ind] = r
                # TODO UPDATE HERE
                self.update_gui()
                # slm.set_aberration(slm.aberration)
                # slm.update()
                # phase_display.update_phase(slm.rzern.n, slm.aberration)
                # phase_display.update()
            return f

        def update_amp(spinbox, slider, le, i):
            def f():
                amp = float(le.text())
                spinbox.setRange(-amp, amp)
                spinbox.setValue(spinbox.value())
                slider.setValue(fto100(self.z[i], le))
            return f

        def update_zlabel(le, settings, i):
            def f():
                settings['zernike_labels'][str(i)] = le.text()
            return f

        def update_spinbox(s, amp):
            def f(t):
                maxrad = float(amp.text())
                s.setValue(t/multiplier*(2*maxrad) - maxrad)
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
            mynk = self.rzern.nk
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
                    slider.setMaximum(multiplier)
                    slider.setFocusPolicy(Qt.StrongFocus)
                    slider.setTickPosition(QSlider.TicksBothSides)
                    slider.setTickInterval(20)
                    slider.setSingleStep(0.01)
                    slider.setValue(fto100(self.z[i], amp))
                    spinbox.setRange(-maxamp, maxamp)
                    spinbox.setSingleStep(0.01)
                    spinbox.setValue(self.z[i])

                    hand1 = update_spinbox(spinbox, amp)
                    hand2 = update_coeff(slider, i, amp)
                    hand3 = update_amp(spinbox, slider, amp, i)
                    hand4 = update_zlabel(lbn, self.settings, i)
                    slider.valueChanged.connect(hand1)
                    spinbox.valueChanged.connect(hand2)
                    amp.editingFinished.connect(hand3)
                    lbn.editingFinished.connect(hand4)

                    scrollLayout.addWidget(lab, i, 0)
                    scrollLayout.addWidget(lbn, i, 1)
                    scrollLayout.addWidget(spinbox, i, 2)
                    scrollLayout.addWidget(slider, i, 3)
                    scrollLayout.addWidget(amp, i, 4)

                    zernike_rows.append((
                        lab, slider, spinbox, hand1, hand2, amp,
                        hand3, lbn, hand4))

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
            for t in zernike_rows:
                t[2].setValue(0.0)

        def change_radial():
            try:
                ival = int(lezm.text())
            except Exception as exp:
                lezm.setText(str(self.rzern.n))
                return
            n = (ival + 1)*(ival + 2)//2
            newab = np.zeros((n,))
            minn = min((n, self.rzern.n))
            newab[:minn] = self.z[:minn]
            self.z = newab
            update_zernike_rows()
            # TODO update here
            # self.set_aberration(newab)
            # slm.update()
            # phase_display.update_phase(slm.rzern.n, slm.aberration)
            # phase_display.update()
            lezm.setText(str(self.rzern.n))

        # TODO update here
        # phase_display.update_phase(slm.rzern.n, slm.aberration)
        update_zernike_rows()

        reset.clicked.connect(reset_fun)
        lezm.editingFinished.connect(change_radial)

        l1 = QGridLayout()
        l1.addWidget(top1, 0, 0)
        l1.addWidget(top, 1, 0)
        self.setLayout(l1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    zp = ZernikePanel()
    zp.show()
    sys.exit(app.exec_())
