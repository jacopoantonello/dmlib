#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from h5py import File

from dmlib.interf import FringeAnalysis
from dmlib.calibration import WeightedLSCalib

"""Example about using dmlib to compute a calibration.

First, download calib-data.h5 from

https://drive.google.com/file/d/1koAdxAXhjp-ZGuQk-mp8leNyF95HJKCr/view?usp=sharing

which contains interferometric data recorded with a Boston Multi-DM. The rest
of the script loads this data and computes a calibration, given the radius of
the aperture. You can check out the calibration afterwards by running
dmlib.zpanel.

"""

fname = 'calib-data.h5'

# load the interferograms from the HDF5 file
with File(fname, 'r') as f:
    align = f['align/images'][()]
    names = f['align/names'][()].split(',')
    images = f['data/images'][()]
    cam_pixel_size_um = f['cam/pixel_size'][()]
    U = f['data/U'][()]
    wavelength_nm = f['wavelength'][()]

# define the pupil radius
radius_um = 900

# pull an interferogram with all actuators at rest
img_zero = images[0, ...]
# pull an interferogram where the central actuators have been poked
img_centre = align[names.index('centre'), ...]

# make a fringe analysis object
fringe = FringeAnalysis(images[0, ...].shape, cam_pixel_size_um)

# use img_zero to lay out the FFT masks
fringe.analyse(
    img_zero, auto_find_orders=True, do_unwrap=True,
    use_mask=False)
# find the aperture position automatically
fringe.estimate_aperture(img_zero, img_centre, radius_um)
# or set the position manually with fringe.set_aperture(centre, radius)

# compute the calibration
calib = WeightedLSCalib()
calib.calibrate(
    U, images, fringe, wavelength_nm, cam_pixel_size_um, status_cb=print)

# save the calibration to a file
fout = 'calib.h5'
with File(fout, 'w', libver='latest') as h5f:
    calib.save_h5py(h5f)
print(f'Saved {fout}')
print(f'To test the {fout}, run:')
print(f'$ python -m dmlib.zpanel --dm-name simdm0 --dm-calibration ./calib.h5')
