#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from traceback import print_exc
from numpy.random import uniform

from dmlib.core import SquareRoot, FakeDM


"""Example about using dmlib to control the DM using raw voltages (no DM
   calibration).

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

    # choose a random voltage to apply to the DM (by convention `u` is limited
    # to [-1, 1], which is automatically mapped to the minimum and maximum
    # voltage supported by the DM.
    u = uniform(-.8, .8, size=dm.size())

    # write vector u
    dm.write(u)
    print(u)

except Exception:
    print_exc()
finally:
    # avoid opening the DM multiple times if launching this from ipython
    try:
        dm.close()
    except Exception:
        pass
