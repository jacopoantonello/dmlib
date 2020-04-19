# Index

This folder contains a number of examples about using `dmlib`.

## calibration

This example shows how to compute the DM calibration from a set of input-output
data that has been previously recorded with an interferometer. First you need
to download the calibration dataset.

## data_collection
This example shows how to collect the DM calibration data using an
interferometer and your own hardware, in case this is not directly supported by
`dmlib`. If your hardware is supported by `dmlib`, you can use the GUI
`dmlib.gui` for this purpose. Otherwise using this script might be the simplest
option for you. You need to replace the function calls to the dummy DM and
Camera objects with the correct function calls for your hardware. Once the data
has been collected, it is saved into a file `calib-data2.h5`. You can then
compute the DM calibration using this file and the script above.

## control_voltages
This example shows how to drive the DM using raw actuator voltages. It uses the
built-in wrappers from the `devwraps` package.

## control_zernike
This example shows how to drive the DM using Zernike modes. It loads a
calibration computed using `calibration.py`.

## export_calibration
This example exports the calibration into a text file (JSON), so that it can be
more easily loaded by an external application like LabVIEW. It also shows how
to scale correctly the control variable to compute the raw voltage to apply to
the DM. You can open, edit and run this example with Spyder.

## run_calibration_gui
Example BAT file to launch the calibration GUI ` dmlib.gui` from Windows.

## run_control_gui
Example BAT file to launch the Zernike control GUI `dmlib.zpanel` from Windows.
