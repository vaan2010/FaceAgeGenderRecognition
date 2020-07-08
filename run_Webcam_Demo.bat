@echo off
set current_path=%cd%
for /f "tokens=1* delims=\" %%a in ("%current_path%") do (
	set execution_file=%%a
)
@SET /P env=Please input your conda environment: 
call conda.bat activate %env%
echo Start activating environment...
C:
cd /d C:/Program Files (x86)/IntelSWTools/openvino/bin
echo Start setting OpenVINO environment variables...
call setupvars.bat
%execution_file%
@SET /P res=Please input resolution(1920, 1280, 960): 
@SET /P xml=Please input the path of .xml file(If you wanna use default, please enter d): 
@SET /P bin=Please input the path of .bin file(If you wanna use default, please enter d): 
if "%xml%" == "d" set xml=./Training/Results/Openvino_IR/MFN.xml
if "%bin%" == "d" set bin=./Training/Results/Openvino_IR/MFN.bin
python ./Demo/RealTime/Webcam.py --x %xml% --b %bin% --r %res%
echo.
echo You has already shut down the real-time face recognition!
Pause