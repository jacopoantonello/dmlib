#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from os import path


# Get the long description from the relevant file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dmlib',
    version='',
    description='Python tools for deformable mirror calibration',
    long_description=long_description,
    url='',
    author='fixme',
    author_email='fixme',
    license='tobedefined',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(
        exclude=['tests*', 'examples*']),
    setup_requires=['numpy'],
    install_requires=['numpy', 'h5py', 'skimage'],
    extras_require={
        'user interface': ['pyqt5'],
        'plot': ['matplotlib'],
        },
)
