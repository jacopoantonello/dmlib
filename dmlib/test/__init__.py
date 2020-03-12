#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from imageio import imread
from numpy.random import uniform
from skimage.transform import resize


def load_int3(shape):
    img = imread(os.path.join(
        os.path.dirname(__file__), 'int3.tif'))
    img = np.roll(img, int(np.round(uniform(-200, 200))), axis=0)
    img = np.roll(img, int(np.round(uniform(-200, 200))), axis=1)
    assert(img.dtype == np.uint8)
    img = resize(img, shape, mode='constant', anti_aliasing=False)
    img *= 255/img.max()
    img = img.astype(np.uint8)
    assert(img.dtype == np.uint8)
    return img
