#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.random import uniform

from PyQt5.QtWidgets import QErrorMessage, QInputDialog


class SquareRoot:

    name = 'v = 2.0*np.sqrt((u + 1.0)/2.0) - 1.0'

    def __call__(self, u):
        assert(np.all(np.isfinite(u)))

        if norm(u, np.inf) > 1.:
            print('SquareRoot saturation')
            u[u > 1.] = 1.
            u[u < -1.] = -1.
        assert(norm(u, np.inf) <= 1.)

        v = 2*np.sqrt((u + 1.0)/2.0) - 1.0
        assert(np.all(np.isfinite(v)))
        assert(norm(v, np.inf) <= 1.)
        del u

        return v

    def __str__(self):
        return self.name


class FakeCam():
    exp = 0.06675 + 5*0.06675
    fps = 4

    name = None

    def open(self, name):
        self.name = name
        print('FakeCam open ' + name)

    def close(self):
        print('FakeCam close ' + self.name)

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

    def get_devices(self):
        return ['cam0']


class FakeDM():

    name = None
    transform = None

    def open(self, name):
        self.name = name
        print('FakeDM open ' + name)

    def close(self):
        print('FakeDM close ' + self.name)

    def get_devices(self):
        return ['dm0', 'C17W005#050']

    def size(self):
        return 140

    def write(self, v):
        if self.transform:
            v = self.transform(v)
        print('FakeDM', v)

    def get_transform(self):
        return self.transform

    def set_transform(self, tx):
        self.transform = tx

    def get_serial_number(self):
        return 'simdm0'

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


def choose_device(app, args, dev, name, def1, set1):
    devs = dev.get_devices()
    if len(devs) == 0:
        if app:
            e = QErrorMessage()
            e.showMessage('no {} found'.format(name))
            sys.exit(app.exec_())
        else:
            raise ValueError('no {} found'.format(name))
    elif def1 is not None and def1 not in devs:
        if app:
            e = QErrorMessage()
            e.showMessage(
                '{d} {n} not detected, available {d} are {ll}'.format(
                    d=name, n=def1, ll=str(devs)))
            sys.exit(app.exec_())
        else:
            raise ValueError(
                '{d} {n} not detected, available {d} are {ll}'.format(
                    d=name, n=def1, ll=str(devs)))
    elif def1 is None:
        if len(devs) == 1:
            set1(devs[0])
        else:
            if app:
                item, ok = QInputDialog.getItem(
                    None, '', 'select ' + name + ':', devs, 0, False)
                if ok and item:
                    set1(item)
                else:
                    sys.exit(0)
            else:
                raise ValueError('must specify a device name')


def open_cam(app, args):

    # choose driver
    if args.cam == 'sim':
        cam = FakeCam()
    elif args.cam == 'thorcam':
        from devwraps.thorcam import ThorCam
        cam = ThorCam()
    else:
        raise NotImplementedError(args.cam)

    # choose device
    def set_cam(t):
        args.cam_name = t

    choose_device(app, args, cam, 'cam', args.cam_name, set_cam)

    # open device
    cam.open(args.cam_name)

    return cam


def open_dm(app, args, dm_transform=None):

    # choose driver
    if args.dm == 'sim':
        dm = FakeDM()
    elif args.dm == 'bmc':
        from devwraps.bmc import BMC
        dm = BMC()
    elif args.dm == 'ciusb':
        from devwraps.ciusb import CIUsb
        dm = CIUsb()
    else:
        raise NotImplementedError(args.dm)

    # choose device
    def set_dm(t):
        args.dm_name = t

    choose_device(app, args, dm, 'dm', args.dm_name, set_dm)

    # open device
    dm.open(args.dm_name)

    if dm_transform is None:
        dm.set_transform(SquareRoot())
    elif dm_transform == SquareRoot.name:
        dm.set_transform(SquareRoot())
    elif dm_transform == 'v = u':
        pass
    else:
        raise NotImplementedError('unknown transform ' + dm_transform)

    return dm


def add_dm_parameters(parser):
    parser.add_argument(
        '--dm', choices=['sim', 'bmc', 'ciusb'], default='sim')
    parser.add_argument('--dm-name', type=str, default=None, metavar='SERIAL')


def add_cam_parameters(parser):
    parser.add_argument(
        '--cam', choices=['sim', 'thorcam'], default='sim')
    parser.add_argument('--cam-name', type=str, default=None, metavar='SERIAL')
