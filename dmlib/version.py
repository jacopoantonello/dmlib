#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = '0.0.5+10.g71762dd.dirty'
__date__ = 'Thu Aug 23 15:20:37 2018 +0100'
__commit__ = '71762dd7b03d3ea9b4c4a89f348335986b5e528c'


import h5py
import hashlib


# https://stackoverflow.com/questions/22058048

def hash_file(fname):
    BLOCKSIZE = 65536
    hasher = hashlib.md5()
    with open(fname, 'rb') as f:
        buf = f.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(BLOCKSIZE)
    hexdig = hasher.hexdigest()
    return hexdig


def write_h5_header(h5f, libver, now):
    h5f['datetime'] = now.isoformat()

    # save HDF5 library info
    h5f['h5py/libver'] = libver
    h5f['h5py/api_version'] = h5py.version.api_version
    h5f['h5py/version'] = h5py.version.version
    h5f['h5py/hdf5_version'] = h5py.version.hdf5_version
    h5f['h5py/info'] = h5py.version.info

    # save dmlib info
    h5f['dmlib/__date__'] = __date__
    h5f['dmlib/__version__'] = __version__
    h5f['dmlib/__commit__'] = __commit__
