[2020/07/16] Face Recognition of Age and Gender

------------------------------------------------------
1. This model can be used on Face Recognition of Age and Gender with European, American and Asian. 
2. This model's backbone is based on MobileFaceNet, and you can check FaceAgeGenderRecognition.doc or .pdf to get more details like model structure, training step or parameter tuning.
------------------------------------------------------

======================================================
###### Convert h5 to pb then converting to IR

The following steps will tell you how to convert your .h5 of Keras to IR file of OpenVINO.

1. Prepare your .h5 file of trained model and convert it to .pb file in tensorflow 1.15.0 environment by using Keras2pb.py
2. python Keras2pb.py --s [Save pb path] --m [The h5 file you wanna convert]
3. Copy your .pb file into C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer.
4. In terminal, you need to type the following command to convert your .pb file to IR file:
	python mo_tf.py --input_model [Your .pb file] --output_dir [Your output dir] --batch 1 

5. You will see *.bin, *.mapping and *.xml in your output dir.

