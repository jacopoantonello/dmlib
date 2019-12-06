@echo off

REM lines starting with REM are commented out and not executed
REM only ONE line must start with SET PARAMETERS, the remaining lines
REM MUST be commented out with REM

SET CURDIR=%~dp0

REM choose a GUI by modifying the following line
SET MODULE=dmlib.zpanel
REM choose your parameters by modifying the following line
SET PARAMETERS=--dm-driver sim --dm-name simdm0 --dm-calibration '%CURDIR%\data\calib-example.h5'
REM use "SET PARAMETERS=--help" to get a list of the possible parameters

REM do not edit below
SET BASE_RELATIVE_PATH=..\devwraps\scripts\base.ps1
SET CMD=. %BASE_RELATIVE_PATH%; Activate-Anaconda; cd ~; python -m %MODULE% %PARAMETERS%
echo "%CMD%"
Powershell.exe -executionpolicy bypass -NoExit -Command "%CMD%"
