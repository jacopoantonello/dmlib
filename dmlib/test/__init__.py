#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform
from scipy.misc import imresize


def load_int3(shape):
    img = plt.imread(os.path.join(
        os.path.dirname(__file__), 'int3.tif'))
    img = np.roll(img, int(np.round(uniform(-200, 200))), axis=0)
    img = np.roll(img, int(np.round(uniform(-200, 200))), axis=1)
    return imresize(img, shape)
