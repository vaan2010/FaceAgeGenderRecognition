# High Speed Real-Time Face-Age-Gender-Recognition
+ A real-time age and gender recognition of face model with only 1mb, 0.531×10^6 parameters and 0.074 GFLOPs.
+ Using OpenVINO to accelerate the speed of real-time recognition on intel D415.
+ The backbone of model structure is MobileFaceNet.
+ CelebA, AFAD, MegaAge-Asian and UTKFace dataset is used for model training.

**Code Author: Vaan Lin**

**Last update: 2020/07/08**
### Face image inference demo

​<img src="Demo/Image/Results/1.jpg" height="300"/>
​<img src="Demo/Image/Results/3.jpg" height="300"/>
​<img src="Demo/Image/Results/2.jpg" height="267"/>

### Real-time webcam(intel D415) demo
​<img src="Demo/RealTime/realtime_test.gif" height="300"/>
​<img src="Demo/RealTime/realtime_test.PNG" height="303.5"/>


## Requirements
Please install Anaconda(Python 3.7 64-Bit) first.
#### NVIDIA
+ CUDA            10.1
+ cudnn           7.6.4
#### Python
+ Tensorflow-GPU  2.2.0
+ Keras           2.3.1
+ cvlib           0.2.2
+ numpy           1.17.4

You can install the package of above at once by using command: 
```
pip install -r requirements.txt
```
#### OpenVINO     
+ OpenVINO        2020.3

## Dependencies
+ Anaconda
+ MobileFaceNet
(https://arxiv.org/ftp/arxiv/papers/1804/1804.07573.pdf)
+ cvlib
(https://github.com/arunponnusamy/cvlib)

## Simply Use
Insure you have already installed OpenVINO.
+ If you wanna use the model to inference your images

1. Put your images in .\FaceAgeGenderRecognition\Demo\Image\Demo_Image
2. Double click run_Image_Demo.bat
3. The results will be saved in .\FaceAgeGenderRecognition\Demo\Image\Results

+ If you wanna use the model to achieve the real-time recognition on intel D415

1. Double click run_Webcam_Demo.bat

+ If you want to re-train the model of using your dataset of TFRecords

1. Double click run_MFN_Train.bat

