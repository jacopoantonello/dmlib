# DMlib
Python library for calibration and control of deformable mirrors (DM).

`dmlib` is meant to be used as an external library in a Python application that needs to control deformable mirrors. `dmlib` has no hardware requirements and can be used with any kind of DM, provided an appropriate Python wrapper is available (see `examples/calibration_script.py`). A collection of hardware wrappers is readily available in the external package [devwraps](https://github.com/jacopoantonello/devwraps).

`dmlib` is a flexible and compact library that can be used in multiple scenarios. You can directly run the GUIs in `dmlib` to calibrate and control a DM. Alternatively, you can perform the DM calibration using a script and saving it to a file (see `examples/calibration_script.py`). You can then load and use the calibration on your own Python application or GUI. Another possibility is to embed the GUIs in `dmlib` into your own Python GUI.

## Installation
### Windows
* install the hardware drivers and external requirements listed in [devwraps](https://github.com/jacopoantonello/devwraps) if you need to run real hardware, otherwise simulation only will be available
* clone recursively this repository with `git clone --recurse-submodules https://github.com/jacopoantonello/dmlib`
* double-click on `install.bat`

## Running DMlib GUIs
### DM calibration (dmlib.gui)
The calibration GUI can be used to interferometrically calibrate and test a DM. To run the calibration GUI, open an Anaconda Prompt and query the available drivers with `python -m dmlib.gui --help`. After selecting the appropriate drivers flags you can run the GUI without the `--help` flag.

For example
```bash
python -m dmlib.gui --dm-driver bmc --dm-name DM_SERIAL_NUMBER --cam-driver thorcam --cam-name CAM_SERIAL_NUMBER
```
would run the calibration GUI using the Boston BMC Multi-DM with serial number `DM_SERIAL_NUMBER` and the Thorlabs camera with serial number `CAM_SERIAL_NUMBER`.

To run the calibration GUI in simulation mode without any hardware, use
```bash
python -m dmlib.gui --dm-driver sim --cam-driver sim
```

You can create a `BAT` file with the correct parameters so that you can open the GUI by double-clicking on it (see `examples/run_calibration_gui.bat`).

### DM control (dmlib.zpanel)
The control GUI can load a DM calibration and control the DM in open-loop. To run the control GUI, use the `-m dmlib.gui` flag and select the parameters as explained above.

You can create a `BAT` file with the correct parameters so that you can open the GUI by double-clicking on it (see `examples/run_control_gui.bat`).
