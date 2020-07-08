@echo off
set current_path=%cd%
for /f "tokens=1* delims=\" %%a in ("%current_path%") do (
	set execution_file=%%a
)
@SET /P env=Please input your conda environment: 
call conda.bat activate %env%
echo Start activating environment...
cd %execution_file%\FaceAgeGenderRecognition\Training
python MFN_Train.py
