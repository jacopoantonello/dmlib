#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from os import path

import h5py
import numpy as np
from numpy.linalg import norm
from PyQt5.QtWidgets import QErrorMessage, QInputDialog

import dmlib.test
from dmlib.dmplot import dmplot_from_layout, get_layouts
from dmlib.version import __commit__, __date__, __version__

LOG = logging.getLogger('core')


def h5_store_str(f, a, s):
    f.create_dataset(a,
                     data=np.array(s.encode('utf-8'),
                                   dtype=h5py.string_dtype('utf-8', len(s))))


def h5_read_str(f, a):
    tmp = f[a][()]
    if isinstance(tmp, bytes):
        tmp = tmp.decode('utf-8')
    return tmp


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
    h5_store_str(h5f, 'datetime', now.isoformat())

    # save HDF5 library info
    h5_store_str(h5f, 'h5py/libver', libver)
    h5_store_str(h5f, 'h5py/api_version', h5py.version.api_version)
    h5_store_str(h5f, 'h5py/version', h5py.version.version)
    h5_store_str(h5f, 'h5py/hdf5_version', h5py.version.hdf5_version)
    h5_store_str(h5f, 'h5py/info', h5py.version.info)

    # save dmlib info
    h5_store_str(h5f, 'dmlib/__date__', __date__)
    h5_store_str(h5f, 'dmlib/__version__', __version__)
    h5_store_str(h5f, 'dmlib/__commit__', __commit__)


class SquareRoot:

    name = 'v = 2.0*np.sqrt((u + 1.0)/2.0) - 1.0'

    def __init__(self):
        self.log = logging.getLogger('SquareRoot')

    def __call__(self, u):
        assert (np.all(np.isfinite(u)))

        if norm(u, np.inf) > 1.:
            self.log.info('saturation')
            u[u > 1.] = 1.
            u[u < -1.] = -1.
        assert (norm(u, np.inf) <= 1.)

        v = 2 * np.sqrt((u + 1.0) / 2.0) - 1.0
        assert (np.all(np.isfinite(v)))
        assert (norm(v, np.inf) <= 1.)
        del u

        return v

    def __str__(self):
        return self.name


class FakeCam():
    def __init__(self, shape=(1024, 1280), pxsize=(7.4, 7.4)):
        self.log = logging.getLogger(self.__class__.__name__)
        self.exp = 0.06675 + 5 * 0.06675
        self.fps = 4
        self.name = None
        assert(len(shape) == 2)
        assert(len(pxsize) == 2)
        self._shape = shape
        self._pxsize = pxsize

    def open(self, name=''):
        self.name = name
        self.log.info(f'open {name:}')

    def close(self):
        self.log.info(f'close {self.name:}')

    def grab_image(self):
        img = dmlib.test.load_int3(self._shape)
        img = img.astype(self.get_image_dtype())
        assert (img.dtype == self.get_image_dtype())
        assert (img.shape == self.shape())
        return img

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
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.name = None
        self.transform = None
        self._size = 140
        self.presets = {}

    def open(self, name=''):
        self.name = name
        self.log.info(f'open {name:}')

    def close(self):
        self.log.info(f'close {self.name:}')

    def get_devices(self):
        return ['simdm0', 'simdm1']

    def size(self):
        return self._size

    def write(self, v):
        if self.transform:
            v = self.transform(v)

    def get_transform(self):
        return self.transform

    def set_transform(self, tx):
        self.transform = tx

    def get_serial_number(self):
        return self.name

    def from_dmplot(self, dmplot):
        self._size = dmplot.size()
        self.presets = dmplot.presets

    def preset(self, name, mag=0.7):
        return mag * self.presets[name]


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
                item, ok = QInputDialog.getItem(None, '',
                                                'select ' + name + ':', devs,
                                                0, False)
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
        exit_error(app, f'unable to open {devtype} {devname}', ValueError)


def exit_exception(app, txt, exc=None):
    if exc is None:
        exc = ValueError(txt)
        msg = txt
    else:
        msg = txt + '; ' + str(exc)
    if app:
        e = QErrorMessage()
        e.showMessage(msg)
        sys.exit(e.exec_())
    else:
        raise exc


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
        elif args.cam_driver == 'custom':
            from customcam import CustomCam
            cam = CustomCam()
        else:
            raise NotImplementedError(args.cam_driver)
    except Exception as e:
        exit_exception(app, f'error loading camera {args.cam_driver} drivers',
                       e)

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


def get_suitable_dmplot(args, dm, calib=None, dmplot=None):
    if dmplot is not None:
        return dmplot
    if args.dm_driver == 'sim':
        if calib is not None:
            if calib.dmplot is not None:
                dmplot = calib.dmplot
            elif args.dm_layout is not None:
                dmplot = dmplot_from_layout(args.dm_layout)
            elif calib.nactuators() == 140:
                dmplot = dmplot_from_layout('multidm140')
            elif calib.nactuators() == 69:
                dmplot = dmplot_from_layout('alpao69')
            elif calib.nactuators() == 52:
                dmplot = dmplot_from_layout('mirao52e')
            else:
                raise ValueError(
                    f'Unknown DMPlot for {calib.size()} actuators')
            dm.from_dmplot(dmplot)
            return dmplot
        else:
            if args.dm_layout is None:
                args.dm_layout = 'multidm140'
            dmplot = dmplot_from_layout(args.dm_layout)
            dm.from_dmplot(dmplot)
            return dmplot
    else:
        if calib is not None:
            if calib.nactuators() != dm.size():
                raise ValueError(f'Calibration has {calib.size()} actuators ' +
                                 f'but DM has {dm.size()}')
            elif calib.dmplot is not None:
                return calib.dmplot
            elif args.dm_layout is not None:
                dmplot = dmplot_from_layout(args.dm_layout)
                if dmplot.size() != dm.size():
                    raise ValueError(
                        f'Calibration has {calib.size()} actuators ' +
                        f'but {args.dm_layout} has {dmplot.size()}')
                else:
                    return dmplot
            else:
                raise ValueError('Cannot find suitable DMPlot')
        else:
            if args.dm_layout is None:
                raise ValueError('Cannot find suitable DMPlot')
            else:
                dmplot = dmplot_from_layout(args.dm_layout)
                if dmplot.size() != dm.size():
                    raise ValueError(
                        f'DM layout has {dmplot.size()} actuators ' +
                        f'but DM has {dm.size()}')
                return dmplot

    raise RuntimeError()


def open_dm(app, args, dm_transform=None):

    # choose a driver & adjust DM layout and transform if not set
    try:
        if args.dm_driver == 'sim':
            dm = FakeDM()
            if dm_transform is None:
                dm_transform = SquareRoot.name
        elif args.dm_driver == 'bmc':
            from devwraps.bmc import BMC
            dm = BMC()
            if args.dm_layout is None:
                args.dm_layout = 'multidm140'
            if dm_transform is None:
                dm_transform = SquareRoot.name
        elif args.dm_driver == 'asdk':
            from devwraps.asdk import ASDK
            dm = ASDK()
            if args.dm_layout is None:
                args.dm_layout = 'alpao69'
            if dm_transform is None:
                dm_transform == 'v = u'
        elif args.dm_driver == 'mirao52e':
            from devwraps.mirao52e import Mirao52e
            dm = Mirao52e()
            if args.dm_layout is None:
                args.dm_layout = 'mirao52e'
            if dm_transform is None:
                dm_transform == 'v = u'
        elif args.dm_driver == 'custom':
            from customdm import CustomDM
            dm = CustomDM()
            if dm_transform is None:
                dm_transform == 'v = u'
        else:
            exit_exception(app, NotImplementedError(args.dm_driver))
    except Exception as e:
        exit_exception(app, f'Error loading DM driver: {args.cam_driver}', e)

    if args.dm_list:
        devs = dm.get_devices()
        LOG.info('dm devices: ' + str(dm.get_devices()))
        if app:
            e = QErrorMessage()
            e.showMessage('Detected dms are: ' + str(devs))
            sys.exit(e.exec_())
        else:
            sys.exit()

    if args.dm_list_layouts:
        lays = get_layouts()
        str1 = ', '.join(lays)
        LOG.info(f'layouts: {str1}')
        if app:
            e = QErrorMessage()
            e.showMessage(f'Detected dms are: {str1}')
            sys.exit(e.exec_())
        else:
            sys.exit()

    # choose device
    def set_dm(t):
        args.dm_name = t

    choose_device(app, args, dm, 'dm', args.dm_name, set_dm)

    # open device
    attempt_open(app, dm, args.dm_name, 'dm')

    # choose a driver
    if dm_transform == 'v = u':
        pass
    elif dm_transform == SquareRoot.name:
        dm.set_transform(SquareRoot())
    else:
        exit_exception(app, 'DM transform not set')

    return dm


def add_log_parameters(parser):
    parser.add_argument('--file-log', dest='file_log', action='store_true')
    parser.add_argument('--no-file-log', dest='file_log', action='store_false')
    parser.set_defaults(file_log=False)
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='ERROR')


# https://stackoverflow.com/questions/6631299/
def spawn_file(p):
    if platform.system() == 'Windows':
        subprocess.Popen(['explorer', '/select,', p])
    elif platform.system() == 'Darwin':
        subprocess.Popen(['open', '-a', 'Finder', p])
    else:
        subprocess.Popen(['xdg-open', path.dirname(p)])


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
        raise NotImplementedError(f'Unknown logging level {args.log_level}')

    if args.file_log:
        fn = datetime.now().strftime('%Y%m%d-%H%M%S-' + str(os.getpid()) +
                                     '.log')
        logging.basicConfig(filename=fn, level=level)
    else:
        logging.basicConfig(level=level)


def add_dm_parameters(parser):
    parser.add_argument('--dm-driver',
                        choices=[
                            'sim',
                            'bmc',
                            'asdk',
                            'mirao52e',
                            'custom',
                        ],
                        default='sim')
    parser.add_argument('--dm-name', type=str, default=None, metavar='SERIAL')
    parser.add_argument('--dm-list-layouts',
                        action='store_true',
                        help='List included layouts')
    parser.add_argument('--dm-layout',
                        type=str,
                        default=None,
                        metavar='Layout name or external JSON file')
    parser.add_argument('--dm-list',
                        action='store_true',
                        help='List detected devices')


def add_cam_parameters(parser):
    parser.add_argument('--cam-driver',
                        choices=['sim', 'thorcam', 'ximea', 'ueye', 'custom'],
                        default='sim')
    parser.add_argument('--cam-name', type=str, default=None, metavar='SERIAL')
    parser.add_argument('--cam-list',
                        action='store_true',
                        help='List detected devices')
    parser.add_argument('--sim-cam-shape',
                        metavar=('H', 'W'),
                        type=int,
                        nargs=2,
                        default=(1024, 1280))
    parser.add_argument('--sim-cam-pix-size',
                        metavar=('H', 'W'),
                        type=float,
                        nargs=2,
                        default=(5.20, 5.20))
