#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
from os import path
from subprocess import check_output

from setuptools import setup

# Get the long description from the relevant file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def update_version():
    try:
        toks = check_output('git describe --tags --long --dirty',
                            universal_newlines=True,
                            shell=True).strip().split('-')
        version = toks[0].strip('v') + '.' + toks[1]
        last = check_output('git log -n 1',
                            universal_newlines=True,
                            shell=True)
        date = re.search(r'^Date:\s+([^\s].*)$', last, re.MULTILINE).group(1)
        commit = re.search(r'^commit\s+([^\s]{40})', last,
                           re.MULTILINE).group(1)

        with open(path.join('dmlib', 'version.py'), 'w', newline='\n') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('# -*- coding: utf-8 -*-\n\n')
            f.write(f"__version__ = '{version}'\n")
            f.write(f"__date__ = '{date}'\n")
            f.write(f"__commit__ = '{commit}'")
    except Exception as e:
        print('Cannot update version {}'.format(str(e)), file=sys.stderr)


update_version()


def lookup_version():
    with open(os.path.join('dmlib', 'version.py'), 'r') as f:
        m = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    return m.group(1)


setup(name='dmlib',
      version=lookup_version(),
      description='Python tools for deformable mirror calibration',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jacopoantonello/dmlib',
      author='The DMLib Project Contributors',
      author_email='jacopo@antonello.org',
      license='GPLv3+',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Physics',
          ('License :: OSI Approved :: ' +
           'GNU General Public License v3 or later (GPLv3+)'),
          'Programming Language :: Python :: 3',
      ],
      packages=['dmlib', 'dmlib.test', 'dmlib.dmlayouts'],
      package_data={
          'dmlib.test': ['*.tif'],
          'dmlib.dmlayouts': ['*.json']
      },
      setup_requires=['numpy'],
      install_requires=['numpy', 'h5py', 'scikit-image', 'zernike'],
      extras_require={
          'user interface': ['pyqt5'],
          'plot': ['matplotlib'],
      },
      entry_points={
          'console_scripts': [
              'dmlib.calibrate = dmlib.gui:main',
              'dmlib.control = dmlib.zpanel:main'
          ]
      })
