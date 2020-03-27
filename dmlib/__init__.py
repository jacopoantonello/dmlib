#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dmlib import version

__author__ = 'J Antonello'
__copyright__ = 'Copyright 2019, J. Antonello'
__license__ = 'to be defined'
__email__ = 'jacopo@antonello.org'
__status__ = 'Prototype'
__all__ = [
    'calibration', 'control', 'core', 'dmplot', 'gui', 'interf',
    'version', 'zpanel']
__version__ = version.__version__
__date__ = version.__date__
__commit__ = version.__commit__
__doc__ = """
To be defined.

author:  {}
date:    {}
version: {}
commit:  {}
""".format(
    __author__,
    __date__,
    __version__,
    __commit__)
