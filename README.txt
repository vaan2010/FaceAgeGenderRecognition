[2020/6/29] Face Recognition of Age and Gender

------------------------------------------------------
1. This model can be used on Face Recognition of Age and Gender with European, American and Asian. 
2. This model's backbone is based on MobileFaceNet, and you can check FaceAgeGenderRecognition.doc or .pdf to get more details like model structure, training step or parameter tuning.
------------------------------------------------------

======================================================

###### Requirement
CUDA            10.1
cudnn           7.6.4
Anaconda        Python 3.7 64-Bit 
Tensorflow-GPU  2.2.0
OpenVINO        2020.3
Keras           2.3.1
cvlib           0.2.2
numpy           1.17.4

###### Command of Code Execution
--Insure you have already installed OpenVINO
--If you wanna use the model to inference your images
1. Activate your Anaconda environment
2. Put your images in .\FaceAgeGenderRecognition\Demo\Image\Demo_Image
3. python ./Demo/Image/Image_Test.py --x ./Training/Results/Openvino_IR/MFN.xml --b ./Training/Results/Openvino_IR/MFN.bin --i ./Demo/Image/Demo_Image --o ./Demo/Image/Results/ 
4. --x means path of OpenVINO .xml file
   --b means path of OpenVINO .bin file
   --i means path of input images
   --o means path of output images

--If you wanna use the model to achieve the real-time recognition on intel D415
1. Activate your Anaconda environment
2. python ./Demo/RealTime/Webcam.py --x ./Training/Results/Openvino_IR/MFN.xml --b ./Training/Results/Openvino_IR/MFN.bin --r 1280
3. --x means path of OpenVINO .xml file
   --b means path of OpenVINO .bin file
   --r means the resolution of D415(include 1920(1920x1080), 1280(1280x720) and 960(960x540))

--If you want to re-train the model of using default dataset
1. Activate your Anaconda environment
2. cd ./FaceAgeGenderRecognition/Training
3. python MFN_Train.py --tn P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Outputs/TFRecords/train/ --ts P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Outputs/TFRecords/test/ --m .././Model/Backbone/MFN_62_075_gender_pre-trained.h5 --p Y
4. --tn means path of training tfrecords
   --ts means path of testing tfrecords
   --m  means the path of model
   --p  means If use pre-trained model or not

--If you want to create the tfrecords by yourself
1. Activate your Anaconda environment
2. (Asian)python ./TFRecords_Create/gen_TFRecords.py --i P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Datasets/Asian_FaceData/ --c ./TFRecords_Create/Asian_FaceData.csv --t ./TFRecords_Create/TFRecords --n Asian
3. (UTK)  python ./TFRecords_Create/gen_TFRecords.py --i P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Datasets/UTK_FaceData/ --c ./TFRecords_Create/UTK_FaceData.csv --t ./TFRecords_Create/TFRecords --n UTK
4. --i means the path of dataset
   --c means the path of csv file
   --t means the path of output tfrecords
   --n means the name of output tfrecords

###### Simple Steps of Execution 
--Insure you have already installed OpenVINO
--If you wanna use the model to inference your images
1. Put your images in .\FaceAgeGenderRecognition\Demo\Image\Demo_Image
2. Double click run_Image_Demo.bat
3. The results will be saved in .\FaceAgeGenderRecognition\Demo\Image\Results

--If you wanna use the model to achieve the real-time recognition on intel D415
1. Double click run_Webcam_Demo.bat

--If you want to re-train the model of using default dataset
1. Double click run_MFN_Train.bat


###### Directory structure description
./FaceAgeGenderRecognition
│  README.txt
│  run_Image_Demo.bat
│  run_MFN_Train.bat
│  run_Webcam_Demo.bat
│  
├─Datasets
├─Demo
│  ├─Image
│  │  │  Image_Test.py
│  │  │  
│  │  ├─Demo_Image
│  │  │      158528155ea3f80b1d.jpg
│  │  │      20170614-095856_U7324_M290366_aa8b.jpg
│  │  │      48437584274ffc3e_k.jpg
│  │  │      
│  │  └─Results
│  └─RealTime
│          Webcam.py
│          
├─Model
│  └─Backbone
│      │  MFN_62_075_gender_pre-trained.h5
│      │  MobileFaceNet.py
│      │  README.txt
│      │  
│      ├─Tools
│      │  │  Keras_custom_layers.py
│              
├─TFRecords_Create
│  │  Asian_FaceData.csv
│  │  gen_TFRecords.py
│  │  UTK_FaceData.csv
│  │  
│  └─TFRecords
│      ├─test
│      └─train
└─Training
    │  MFN_Train.ipynb
    │  MFN_Train.py
    │  solve_cudnn_error.py
    │  training_step.jpg
    │      
    ├─Results
    │  ├─Keras_h5
    │  │      MFN.h5
    │  │      
    │  ├─Openvino_IR
    │  │      MFN.bin
    │  │      MFN.mapping
    │  │      MFN.xml
    │  │      
    │  ├─Tensorflow_pb
    │  │      MFN.pb
    │  │      
    │  └─Training_Checkpoints
    │          MFN_Recognition_0.h5

