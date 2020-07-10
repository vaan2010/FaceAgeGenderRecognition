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
+ CUDA            10.1
+ cudnn           7.6.4
+ Tensorflow-GPU  2.2.0
+ OpenVINO        2020.3
+ Keras           2.3.1
+ cvlib           0.2.2
+ numpy           1.17.4

## Dependencies
+ 
