@echo off
set current_path=%cd%
for /f "tokens=1* delims=\" %%a in ("%current_path%") do (
	set execution_file=%%a
)
@SET /P env=Please input your conda environment: 
call conda.bat activate %env%
echo Start activating environment...
@SET /P model=Please input the model type you wanna use(Keras or OpenVINO):
IF "%model%" == "OpenVINO" goto S1
IF "%model%" == "Keras" goto S2
goto:error
:S1
echo The Model you use is %model% now...
C:
cd /d C:/Program Files (x86)/IntelSWTools/openvino/bin
echo Start setting OpenVINO environment variables...
call setupvars.bat
%execution_file%
@SET /P xml=Please input the path of .xml file(If you wanna use default, please enter d): 
@SET /P bin=Please input the path of .bin file(If you wanna use default, please enter d): 
if "%xml%" == "d" set xml=./Training/Results/Openvino_IR/MFN.xml
if "%bin%" == "d" set bin=./Training/Results/Openvino_IR/MFN.bin
python ./Demo/Image/Image_Test.py --x %xml% --b %bin% --i ./Demo/Image/Demo_Images --o ./Demo/Image/Results/ --m %model%
PAUSE
exit/b
:S2
echo The Model you use is %model% now...
@SET /P hh=Please input the path of .h5 file(If you wanna use default, please enter d): 
if "%hh%" == "d" set hh=./Training/Results/Keras_h5/MFN.h5
python ./Demo/Image/Image_Test.py --h %hh% --i ./Demo/Image/Demo_Images --o ./Demo/Image/Results/ --m %model%
PAUSE
