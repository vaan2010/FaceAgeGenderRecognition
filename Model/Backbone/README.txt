[2020/07/08] Face Recognition of Age and Gender

------------------------------------------------------
1. This model can be used on Face Recognition of Age and Gender with European, American and Asian. 
2. This model's backbone is based on MobileFaceNet, and you can check FaceAgeGenderRecognition.doc to get more details like model structure, training step or parameter tuning.
------------------------------------------------------

======================================================

MobileFaceNet.py:
The original structure of model. In this code, model's width will be adjustmented by multiplying 0.75, then generating a file named MFN_62_075.h5. 62 means input image size is 62*62.
