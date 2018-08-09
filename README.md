DMlib
=====

Python tools for deformable mirror calibration.


Install
-------

Make sure you have setup your `~/.ssh/config` as in the [wiki](https://uni.eng.ox.ac.uk/wiki/index.php/Phabricator). Then clone recursively.

    $ git clone --recurse-submodules dop-git.eng.ox.ac.uk:diffusion/8/dmlib.git

Install the `devwraps` and `zernike` dependencies, then install this package

    $ python setup.py bdist_wheel
    $ pip install dist\dmlib-*.whl


Run the calibration GUI
-----------------------

    $ python -m dmlib.gui --help


Run the Zernike control GUI
---------------------------

    $ python -m dmlib.zpanel --help
