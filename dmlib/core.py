#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import h5py
import hashlib

from datetime import datetime
from numpy.linalg import norm
from PyQt5.QtWidgets import QErrorMessage, QInputDialog

import dmlib.test

from dmlib.version import __date__, __version__, __commit__


LOG = logging.getLogger('core')


# https://stackoverflow.com/questions/22058048

def hash_file(fname):
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(fname, 'rb') as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
    hexdig = hasher.hexdigest()
    return hexdig


def write_h5_header(h5f, libver, now):
    h5f['datetime'] = now.isoformat()

    # save HDF5 library info
    h5f['h5py/libver'] = libver
    h5f['h5py/api_version'] = h5py.version.api_version
    h5f['h5py/version'] = h5py.version.version
    h5f['h5py/hdf5_version'] = h5py.version.hdf5_version
    h5f['h5py/info'] = h5py.version.info

    # save dmlib info
    h5f['dmlib/__date__'] = __date__
    h5f['dmlib/__version__'] = __version__
    h5f['dmlib/__commit__'] = __commit__


class SquareRoot:

    name = 'v = 2.0*np.sqrt((u + 1.0)/2.0) - 1.0'

    def __init__(self):
        self.log = logging.getLogger('SquareRoot')

    def __call__(self, u):
        assert(np.all(np.isfinite(u)))

        if norm(u, np.inf) > 1.:
            self.log.info('saturation')
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

    def __init__(self, shape, pxsize):
        self.log = logging.getLogger(self.__class__.__name__)
        self._shape = shape
        self._pxsize = pxsize

    def open(self, name):
        self.name = name
        self.log.info(f'open {name:}')

    def close(self):
        self.log.info(f'close {self.name:}')

    def grab_image(self):
        return dmlib.test.load_int3(self._shape)

    def shape(self):
        return self._shape

    def get_pixel_size(self):
        return self._pxsize

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
        return self.name

    def get_settings(self):
        return 'camera settings'

    def get_image_dtype(self):
        return 'uint8'

    def get_image_max(self):
        return 0xff

    def get_devices(self):
        return ['simcam0', 'simcam1']


class FakeDM():

    name = None
    transform = None

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def open(self, name):
        self.name = name
        self.log.info(f'open {name:}')

    def close(self):
        self.log.info(f'close {self.name:}')

    def get_devices(self):
        return ['simdm0', 'simdm1']

    def size(self):
        return 140

    def write(self, v):
        if self.transform:
            v = self.transform(v)

    def get_transform(self):
        return self.transform

    def set_transform(self, tx):
        self.transform = tx

    def get_serial_number(self):
        return self.name

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
        exit_error(app, 'no {} found'.format(name), ValueError)
    elif def1 is not None and def1 not in devs:
        exit_error(
            app, '{d} {n} not detected, available {d} are {ll}'.format(
                d=name, n=def1, ll=str(devs)), ValueError)
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


def attempt_open(app, what, devname, devtype):
    try:
        what.open(devname)
    except Exception:
        exit_error(
            app, f'unable to open {devtype} {devname}',
            ValueError)


def exit_exception(app, txt, exc):
    msg = txt + ' ' + str(exc)
    if app:
        e = QErrorMessage()
        e.showMessage(msg)
        sys.exit(e.exec_())
    else:
        raise exc(msg)


def exit_error(app, text, exc):
    if app:
        e = QErrorMessage()
        e.showMessage(text)
        sys.exit(e.exec_())
    else:
        raise exc(text)


def open_cam(app, args):

    try:
        # choose driver
        if args.cam_driver == 'sim':
            cam = FakeCam(args.sim_cam_shape, args.sim_cam_pix_size)
        elif args.cam_driver == 'thorcam':
            from devwraps.thorcam import ThorCam
            cam = ThorCam()
        elif args.cam_driver == 'ueye':
            from devwraps.ueye import uEye
            cam = uEye()
        elif args.cam_driver == 'ximea':
            from devwraps.ximea import Ximea
            cam = Ximea()
        else:
            raise NotImplementedError(args.cam_driver)
    except Exception as e:
        exit_exception(
            app, f'error loading camera {args.cam_driver} drivers', e)

    if args.cam_list:
        devs = cam.get_devices()
        LOG.info('cam devices: ' + str(cam.get_devices()))
        if app:
            e = QErrorMessage()
            e.showMessage('detected cameras are: ' + str(devs))
            sys.exit(e.exec_())
        else:
            sys.exit()

    # choose device
    def set_cam(t):
        args.cam_name = t

    choose_device(app, args, cam, 'cam', args.cam_name, set_cam)

    # open device
    attempt_open(app, cam, args.cam_name, 'cam')

    return cam


def open_dm(app, args, dm_transform=None):

    # choose driver
    try:
        if args.dm_driver == 'sim':
            dm = FakeDM()
        elif args.dm_driver == 'bmc':
            from devwraps.bmc import BMC
            dm = BMC()
        elif args.dm_driver == 'ciusb':
            from devwraps.ciusb import CIUsb
            dm = CIUsb()
        else:
            raise NotImplementedError(args.dm_driver)
    except Exception as e:
        exit_exception(
            app, f'error loading dm {args.cam_driver} drivers', e)

    if args.dm_list:
        devs = dm.get_devices()
        LOG.info('dm devices: ' + str(dm.get_devices()))
        if app:
            e = QErrorMessage()
            e.showMessage('detected dms are: ' + str(devs))
            sys.exit(e.exec_())
        else:
            sys.exit()

    # choose device
    def set_dm(t):
        args.dm_name = t

    choose_device(app, args, dm, 'dm', args.dm_name, set_dm)

    # open device
    attempt_open(app, dm, args.dm_name, 'dm')

    if dm_transform is None:
        dm.set_transform(SquareRoot())
    elif dm_transform == SquareRoot.name:
        dm.set_transform(SquareRoot())
    elif dm_transform == 'v = u':
        pass
    else:
        raise NotImplementedError('unknown transform ' + dm_transform)

    return dm


def add_log_parameters(parser):
    parser.add_argument(
        '--no-file-log', action='store_true',
        help='Disable logging to a file')
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='DEBUG')


def setup_logging(args):
    if args.log_level == 'DEBUG':
        level = logging.DEBUG
    elif args.log_level == 'INFO':
        level = logging.INFO
    elif args.log_level == 'WARNING':
        level = logging.WARNING
    elif args.log_level == 'ERROR':
        level = logging.ERROR
    elif args.log_level == 'CRITICAL':
        level = logging.CRITICAL
    else:
        raise NotImplementedError('Unknown logging level {args.log_level}')

    if not args.no_file_log:
        fn = datetime.now().strftime(
            '%Y%m%d-%H%M%S-' + str(os.getpid()) + '.log')
        logging.basicConfig(filename=fn, level=level)
    else:
        logging.basicConfig(level=level)


def add_dm_parameters(parser):
    parser.add_argument(
        '--dm-driver', choices=['sim', 'bmc', 'ciusb'], default='sim')
    parser.add_argument('--dm-name', type=str, default=None, metavar='SERIAL')
    parser.add_argument(
        '--dm-list', action='store_true', help='List detected devices')


def add_cam_parameters(parser):
    parser.add_argument(
        '--cam-driver', choices=['sim', 'thorcam', 'ximea', 'ueye'],
        default='sim')
    parser.add_argument('--cam-name', type=str, default=None, metavar='SERIAL')
    parser.add_argument(
        '--cam-list', action='store_true', help='List detected devices')
    parser.add_argument(
        '--sim-cam-shape', metavar=('H', 'W'), type=int, nargs=2,
        default=(1024, 1280))
    parser.add_argument(
        '--sim-cam-pix-size', metavar=('H', 'W'), type=float, nargs=2,
        default=(5.20, 5.20))
