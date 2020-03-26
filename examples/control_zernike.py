#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from traceback import print_exc
from numpy.random import uniform

from h5py import File

from dmlib.core import SquareRoot, FakeDM
from dmlib.calibration import RegLSCalib
from dmlib.control import ZernikeControl

"""Example about using dmlib to control the DM using Zernike modes.
"""

# set this flag to use the real hardware
use_real_hardware = 0

try:
    if use_real_hardware:
        # load the Boston BMC driver
        from devwraps.bmc import BMC
        dm = BMC()

        # get a list of connected devices
        # NB you need to install the correct DM profile files for this to work!
        devs = dm.get_devices()

        # open first device
        dm.open(devs[0])
    else:
        # use a dummy DM object instead of real hardware

        dm = FakeDM()

        devs = dm.get_devices()
        dm.open(devs[0])

    # Membrane DMs, like the Boston Multi-DM, exhibit a quadratic response of
    # the displacement `d` of one actuator with respect to the applied voltage
    # `v`, i.e., `d=v**2`. To have a better linear response you can define a
    # new control variable `u =`v**2`, so that control of the DM is established
    # as `d = u`.  From a given `u` you can then compute the correct voltage to
    # apply as `v = np.sqrt(u)`. This transformation is handled internally by
    # using the the following line. For other DM technologies, such as piezo,
    # you need not use this transformation.
    dm.set_transform(SquareRoot())

    # load a DM calibration file (e.g. generated using calibration.py)
    calib_file = './data/calib-example.h5'
    with File(calib_file, 'r') as f:
        calib = RegLSCalib.load_h5py(f, lazy_cart_grid=True)

    # The control matrix can be retrieved as `calib.C`. You can also read the
    # control matrix using any HDF5 viewer or library from the address
    # "/RegLSCalib/C"

    # instance an object to control using Zernike modes
    zcontrol = ZernikeControl(dm, calib)

    # random Zernikes like in the bar of the GUI
    z = uniform(-.5, .5, size=zcontrol.ndof)

    # The Zernike coefficients are in rad at the wavelength used for
    # calibration.  Use `calib.get_rad_to_nm()*z` to convert to nm.

    # write random Zernike vector
    zcontrol.write(z)

    print(z)

except Exception:
    print_exc()
finally:
    # avoid opening the DM multiple times if launching this from ipython
    try:
        dm.close()
    except Exception:
        pass
