#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from h5py import File

from dmlib.core import FakeDM, FakeCam
from dmlib.interf import FringeAnalysis
from dmlib.calibration import make_normalised_input_matrix, u2v

"""Example script to collect DM calibration data.

This script shows how you should collect the DM calibration data. It generates
an HDF5 file that can be later supplied to the calibration.py script.

You can adapt this script to work with your own hardware that is not supported
by dmlib. In order to do so, adjust the camera and DM parameters accordingly.
Then replace the calls to dmlib's dummy camera (FakeCam) and DM objects
(FakeDM) with actual calls to your manufacturer libraries.

You can still run this script from beginning to end using FakeCam and FakeDM,
without using any real hardware. Note however that these objects are for GUI
testing purposes only, and produce random data. As a result the output
calibration is meaningless.

"""

wavelength_nm = 775

# DM parameters
N = 8                     # number of actuators in your DM
vmin = 0                  # min actuator value that can be set
vmax = 100                # max actuator value that can be set
sqrt = 0                  # set to 1 if expecting quadratic response
aperture_radius_mm = 2.1  # aperture radius over the DM surface

# camera parameters
cam_shape = (1024, 1280)
cam_pixel_size_um = 7.4*np.ones(2)

# calibration parameters
sleep_time = .1  # settling time for the DM
calib_steps = 5  # must be odd and >= 3
calib_mag = .7   # must be in (0, 1]

# replace the following lines with calls to your own hardware libraries
cam = FakeCam()
cam.open()
dm = FakeDM()
dm.open()

# input values for the DM
U = make_normalised_input_matrix(N, calib_steps, calib_mag)
u_zero = np.zeros(N)    # a vector of actuator values all set to zero
u_centre = np.zeros(N)  # a vector with only the central actuator[s]
u_centre[N//2] = .7     # set to non-zero


def write_dm(u):
    "Convenience function to write the DM"
    v = u2v(u, vmin, vmax)
    dm.write(v)
    print(f'DM u:{u}')


# this object handles all the interferometry
fringe = FringeAnalysis(cam_shape, cam_pixel_size_um)

# first take an initial measurement and adjust the interferometer tilt and
# camera exposure until satisfied
while True:
    write_dm(u_zero)
    sleep(sleep_time)
    img_zero = cam.grab_image()
    fringe.analyse(
        img_zero, auto_find_orders=True, do_unwrap=True,
        use_mask=False)

    plt.close('all')
    plt.figure(1)
    plt.title('interferogram')
    plt.imshow(
        img_zero, origin='lower',
        extent=[fringe.cam_grid[2][i]/1000 for i in range(4)])
    plt.xlabel('mm')
    plt.figure(2)
    plt.imshow(
        fringe.unwrapped, origin='lower',
        extent=[fringe.ext4[i]/1000 for i in range(4)])
    plt.xlabel('mm')
    plt.title('phase')
    plt.show(block=False)

    user = input('set a new camera exposure or enter to continue> ')
    try:
        newexp = cam.set_exposure(float(user))  # replace with own hardware
        print(f'exposure changed to {newexp}')
    except Exception:
        break

# take a second measurement by poking the centre of your DM
# dmlib will automatically center the calibration using this reference
write_dm(u_centre)
sleep(sleep_time)
img_centre = cam.grab_image()

# set the aperture radius of your calibration
while True:
    fringe.estimate_aperture(img_zero, img_centre, aperture_radius_mm*1000)
    fringe.analyse(img_centre, auto_find_orders=False, do_unwrap=True)

    plt.close('all')
    plt.imshow(
        fringe.unwrapped, origin='lower',
        extent=[fringe.ext4[i]/1000 for i in range(4)])
    plt.xlabel('mm')
    plt.title('phase')
    plt.show(block=False)

    user = input('set a new aperture radius [mm] or enter to continue> ')
    try:
        aperture_radius_mm = float(user)
    except Exception:
        break

# acquire the calibration data
Ucheck = []
images = []
for i in range(U.shape[1]):
    u = U[:, i]
    write_dm(u)
    sleep(sleep_time)
    img = cam.grab_image()  # take a snapshot with the camera
    Ucheck.append(u.reshape(-1, 1))
    images.append(img)

# you MUST take the DM pokes in the right order!
assert(np.allclose(U, np.hstack(Ucheck)))

# save the collected interferometric data
fout = 'calib-data2.h5'
with File(fout, 'w') as f:
    f['align/images'] = img_centre.reshape((1,) + cam_shape)
    f['align/names'] = 'centre'
    f['data/images'] = images
    f['data/U'] = U
    f['wavelength'] = wavelength_nm
    f['cam/pixel_size'] = cam_pixel_size_um
print(f'Saved {fout}')
