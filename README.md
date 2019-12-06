# dmlib
Python library for calibration and control of deformable mirrors (DM).

`dmlib` is meant to be used as an external library in a Python application that
needs to control deformable mirrors. `dmlib` is not bound to a particular
hardware, and can be used with any kind of DM. To control actual hardware you
can either use your own manufacturer-provided library or one of the existing
wrappers found in [devwraps](https://github.com/jacopoantonello/devwraps).

There are two possible use cases

- Full calibration and control using `dmlib`; You can use `dmlib`'s GUIs and
  embed them into your Python application. In this case your DM and calibration
  camera must be directly supported via
  [devwraps](https://github.com/jacopoantonello/devwraps).
- Calibration with `dmlib` and control using your own code. If your DM is
  directly supported by `dmlib` then you can use `dmlib`'s calibration GUI to
  generate a calibration file. Otherwise you can adjust the data collection
  script `examples/data_collection.py` for your hardware to collect the
  calibration data and subsequently run `examples/calibration.py` to generate a
  calibration file. You can then extract the control matrix from the calibration
  file and load it into your own application to control the DM.

## Installation
### Windows

* install the hardware drivers and external requirements listed in
	[devwraps](https://github.com/jacopoantonello/devwraps) if you need to run
	real hardware, otherwise simulation only will be available
* clone recursively this repository with `git clone --recurse-submodules
  https://github.com/jacopoantonello/dmlib`
* double-click on `install.bat`

## Running dmlib GUIs
### DM calibration (dmlib.gui)

The calibration GUI can be used to interferometrically calibrate and test a DM.
To run the calibration GUI, open an Anaconda Prompt and query the available
drivers with `python -m dmlib.gui --help`. After selecting the appropriate
drivers flags you can run the GUI without the `--help` flag.

For example
```bash
python -m dmlib.gui --dm-driver bmc --dm-name DM_SERIAL_NUMBER --cam-driver
thorcam --cam-name CAM_SERIAL_NUMBER
```
would run the calibration GUI using the Boston BMC Multi-DM with serial number
`DM_SERIAL_NUMBER` and the Thorlabs camera with serial number
`CAM_SERIAL_NUMBER`.

To run the calibration GUI in simulation mode without any hardware, use
```bash
python -m dmlib.gui --dm-driver sim --cam-driver sim
```

You can create a `BAT` file with the correct parameters so that you can open
the GUI by double-clicking on it (see `examples/run_calibration_gui.bat`).

### DM control (dmlib.zpanel)
The control GUI can load a DM calibration and control the DM in open-loop. To
run the control GUI, use the `-m dmlib.gui` flag and select the parameters as
explained above.

You can create a `BAT` file with the correct parameters so that you can open
the GUI by double-clicking on it (see `examples/run_control_gui.bat`).
