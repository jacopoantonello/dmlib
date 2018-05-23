#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.random import uniform


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
