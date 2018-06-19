#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform


def load_int3():
    img = plt.imread(os.path.join(
        os.path.dirname(__file__), 'int3.tif'))
    img = np.roll(img, int(np.round(uniform(-200, 200))), axis=0)
    img = np.roll(img, int(np.round(uniform(-200, 200))), axis=1)
    return img
