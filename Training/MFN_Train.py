import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from solve_cudnn_error import *
import math
import argparse

from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dropout 

solve_cudnn_error()

parser = argparse.ArgumentParser()

parser.add_argument('--tn', help="Input training tfrecords Path", dest="training_tfrecords", default='P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Outputs/TFRecords/train/')
parser.add_argument('--ts', help="Input testing tfrecords Path", dest="testing_tfrecords", default='P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Outputs/TFRecords/test/')
parser.add_argument('--m', help="Input Model Path", dest="Model", default='.././Model/Backbone/MFN_62_075_gender_pre-trained.h5')
parser.add_argument('--p', help="If use pre-trained model or not", dest="pre_train", default='Y')

args = parser.parse_args()

######### Load Model and add another branch for age training ####################
if args.pre_train == 'Y':
    model = tf.keras.models.load_model(args.Model)
    
    prelu_output = model.get_layer('p_re_lu_34').output
    model_output = model.get_layer('dense_2').output

    M2 = Dropout(rate = 0.2, name = 'dropout_2')(prelu_output)
    M2 = Flatten(name = 'flatten_2')(M2)

    M2 = Dense(128, activation = None, use_bias = False, kernel_initializer = 'glorot_normal', name = 'dense_3')(M2)

    Y2 = Dense(units = 48, activation = 'softmax', name='age_output')(M2)

    model = Model(inputs = model.input, outputs = [Y2 ,model_output], name = 'customed_model')
else:
    sys.path.append('.././Model/Backbone/')
    from MobileFaceNet import *
    model = mobile_face_net_train(2, 0.75, 'softmax')
   
    prelu_output = model.get_layer('p_re_lu_33').output
    model_output = model.get_layer('dense_1').output

    M2 = Dropout(rate = 0.2, name = 'dropout_2')(prelu_output)
    M2 = Flatten(name = 'flatten_2')(M2)

    M2 = Dense(128, activation = None, use_bias = False, kernel_initializer = 'glorot_normal', name = 'dense_3')(M2)

    Y2 = Dense(units = 48, activation = 'softmax', name='age_output')(M2)

    model = Model(inputs = model.input, outputs = [Y2 ,model_output], name = 'customed_model')

    
#################################################################################

tfrecord_train_dir = args.training_tfrecords
train_tfrecord = [tfrecord_train_dir + tfrec for tfrec in os.listdir(tfrecord_train_dir)]

tfrecord_test_dir = args.testing_tfrecords
test_tfrecord = [tfrecord_test_dir + tfrec for tfrec in os.listdir(tfrecord_test_dir)]
# print(train_tfrecord)

def _parse_example_train(example_string): # 將 TFRecord 文件中的每一個序列化的 tf.train.Example 解碼，並且進行預處理
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解碼JPEG圖片  
    # Datatype is int8
    x = tf.random.uniform(shape=[], minval=0.1, maxval=45.0)*0.0174532925
    feature_dict['image'] = tfa.image.rotate(feature_dict['image'], x)
    feature_dict['image'] = tf.image.random_brightness(feature_dict['image'], 0.2)
    feature_dict['image'] = tf.image.random_hue(feature_dict['image'], max_delta=0.03)
    feature_dict['image'] = tf.image.random_contrast(feature_dict['image'], 0.5, 1.5)
    feature_dict['image'] = tf.image.random_saturation(feature_dict['image'], 0.5, 1.3)
    feature_dict['image'] = tf.image.random_flip_left_right(feature_dict['image'])
    
    
    # Datatype is Float32
    feature_dict['image'] = tf.image.convert_image_dtype(feature_dict['image'], tf.float32)
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [62, 62])  # Resize 圖片
    
    feature_dict['Age'] = feature_dict['Age'] - 18  # 將 Age 扣掉 18，達成 48 分類
#     feature_dict['Age'] = tf.one_hot(feature_dict['Age'], 48)
    
    feature_dict['Gender'] = tf.one_hot(feature_dict['Gender'], 2)
    
    return feature_dict['image'], feature_dict['Age'], feature_dict['Gender']

def _parse_example_test(example_string): # 將 TFRecord 文件中的每一個序列化的 tf.train.Example 解碼，並且進行預處理
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解碼JPEG圖片  
    
    # Datatype is Float32
    feature_dict['image'] = tf.image.convert_image_dtype(feature_dict['image'], tf.float32)
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [62, 62])  # Resize 圖片
    
    feature_dict['Age'] = feature_dict['Age'] - 18  # 將 Age 扣掉 18，達成 48 分類
#     feature_dict['Age'] = tf.one_hot(feature_dict['Age'], 48)
    
    feature_dict['Gender'] = tf.one_hot(feature_dict['Gender'], 2)
    
    return feature_dict['image'], feature_dict['Age'], feature_dict['Gender']
    
feature_description = { # 定義Feature結構，告訴解碼器每個Feature的類型是什麼
    'image': tf.io.FixedLenFeature([], tf.string),
    'Age': tf.io.FixedLenFeature([], tf.int64),
    'Gender': tf.io.FixedLenFeature([], tf.int64)
}

raw_dataset = []
for i in range(0, 4):
    raw_dataset.append(tf.data.TFRecordDataset([train_tfrecord[i]]).map(_parse_example_train).repeat().batch(8).prefetch(4))

for i in range(4, 8):
    raw_dataset.append(tf.data.TFRecordDataset([train_tfrecord[i]]).map(_parse_example_train).repeat().batch(6).prefetch(4))
    
for i in range(8, 16):
    raw_dataset.append(tf.data.TFRecordDataset([train_tfrecord[i]]).map(_parse_example_train).repeat().batch(1).prefetch(2))
  
for i in range(16, 20):
    raw_dataset.append(tf.data.TFRecordDataset([train_tfrecord[i]]).map(_parse_example_train).repeat().batch(5).prefetch(4))
    
for i in range(20, 24):
    raw_dataset.append(tf.data.TFRecordDataset([train_tfrecord[i]]).map(_parse_example_train).repeat().batch(5).prefetch(4))
    
for i in range(24, 32):
    raw_dataset.append(tf.data.TFRecordDataset([train_tfrecord[i]]).map(_parse_example_train).repeat().batch(3).prefetch(2))
    
raw_test_dataset = []
for i in range(0, 2):
    raw_test_dataset.append(tf.data.TFRecordDataset([test_tfrecord[i]]).map(_parse_example_test).batch(2))

loss_obj_age = tf.keras.losses.SparseCategoricalCrossentropy()
loss_obj_gender = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

loss_age = tf.keras.metrics.Mean(name='age loss')
loss_gender = tf.keras.metrics.Mean(name='gender loss')

error_age = tf.keras.metrics.SparseCategoricalAccuracy()
error_gender = tf.keras.metrics.CategoricalAccuracy()

def plot_training_process(train_age_loss_results, train_age_accuracy_results, train_gender_loss_results, train_gender_accuracy_results):
    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(20, 10))
    fig.suptitle('Training Metrics', fontsize=35)

    axes[0][0].set_ylabel("Age Loss", fontsize=14)
    axes[0][0].set_xlabel("Step", fontsize=14)
    axes[0][0].plot(train_age_loss_results, color = 'r')

    axes[1][0].set_ylabel("Age Accuracy", fontsize=14)
    axes[1][0].set_xlabel("Step", fontsize=14)
    axes[1][0].plot(train_age_accuracy_results)
    
    axes[0][1].set_ylabel("Gender Loss", fontsize=14)
    axes[0][1].set_xlabel("Step", fontsize=14)
    axes[0][1].plot(train_gender_loss_results, color = 'r')

    axes[1][1].set_ylabel("Gender Accuracy", fontsize=14)
    axes[1][1].set_xlabel("Step", fontsize=14)
    axes[1][1].plot(train_gender_accuracy_results)
    
    fig.savefig('training_step.jpg')
    fig.clf()
    plt.close(fig)
    
@tf.function
def train_step(inputs, y_age, y_gender):
    with tf.GradientTape() as tape:
        pred_age, pred_gender = model(inputs)
        age_loss = loss_obj_age(y_age, pred_age)
        gender_loss = loss_obj_gender(y_gender, pred_gender)

    gradients = tape.gradient([age_loss, gender_loss], model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_age(age_loss)
    loss_gender(gender_loss)

    error_age(y_age, pred_age)
    error_gender(y_gender, pred_gender)

t = 1
train_age_loss_results = []
train_age_accuracy_results = []
train_gender_loss_results = []
train_gender_accuracy_results = []

print('\n--------------------------------------------------------------------')
print(" Start training...")
print(" Notice: Every 10000 steps will save a training step plot, training model and test the testing set.")
print('--------------------------------------------------------------------\n')

for (image1, age1, gender1), (image2, age2, gender2), (image3, age3, gender3), (image4, age4, gender4), (image5, age5, gender5),(image6, age6, gender6),(image7, age7, gender7),(image8, age8, gender8),(image9, age9, gender9),(image10, age10, gender10),(image11, age11, gender11),(image12, age12, gender12),(image13, age13, gender13),(image14, age14, gender14),(image15, age15, gender15),(image16, age16, gender16), (image17, age17, gender17), (image18, age18, gender18), (image19, age19, gender19), (image20, age20, gender20), (image21, age21, gender21),(image22, age22, gender22),(image23, age23, gender23),(image24, age24, gender24),(image25, age25, gender25),(image26, age26, gender26),(image27, age27, gender27),(image28, age28, gender28),(image29, age29, gender29),(image30, age30, gender30),(image31, age31, gender31),(image32, age32, gender32) in zip(*raw_dataset):

    combine_image = tf.concat([image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11, image12, image13, image14, image15, image16, image17, image18, image19, image20, image21, image22, image23, image24, image25, image26, image27, image28, image29, image30, image31, image32], 0)
    combine_age = tf.concat([age1, age2, age3, age4, age5, age6, age7, age8, age9, age10, age11, age12, age13, age14, age15, age16, age17, age18, age19, age20, age21, age22, age23, age24, age25, age26, age27, age28, age29, age30, age31, age32], 0)
    combine_gender = tf.concat([gender1, gender2, gender3, gender4, gender5, gender6, gender7, gender8, gender9, gender10, gender11, gender12, gender13, gender14, gender15, gender16, gender17, gender18, gender19, gender20, gender21, gender22, gender23, gender24, gender25, gender26, gender27, gender28, gender29, gender30, gender31, gender32], 0)

    indices = tf.range(start=0, limit=tf.shape(combine_age)[0], dtype=tf.int32)
    idx = tf.random.shuffle(indices)

    combine_image = tf.gather(combine_image, idx)
    combine_age = tf.gather(combine_age, idx)
    combine_gender = tf.gather(combine_gender, idx)

    train_step(combine_image, combine_age, combine_gender)

    train_gender_accuracy_results.append(error_gender.result())
    train_gender_loss_results.append(loss_gender.result())
    
    train_age_accuracy_results.append(error_age.result())
    train_age_loss_results.append(loss_age.result())
    
    template = 'Step {:>2}, ------ Total Loss: {:>4.5f}, ------ Age_Loss: {:>4.5f}, ------ Gender_Loss: {:>4.5f}, ------ Age_Acc: {:>4.5f}, ------ Gender_Acc: {:>4.5f}'
    print('\r', template.format(t+1, loss_age.result() + loss_gender.result(), loss_age.result(), loss_gender.result(), error_age.result(), error_gender.result()), end = '')

    if t % 10000 == 0:
        print('\n Saving model...')
        model.save('./Results/Training_Checkpoints/MFN_Recognition_' + str(t) + '.h5')
        
        plot_training_process(train_age_loss_results, train_age_accuracy_results, train_gender_loss_results, train_gender_accuracy_results)
        
        print('--------------------------------------------------------------------')
        print(' Using current model to test the testing set...')
        
        itera = 0
        number = 0
        bar = 50
        
        for (image_Asian_test, age_Asian_test, gender_Asian_test), (image_UTK_test, age_UTK_test, gender_UTK_test) in zip(*raw_test_dataset):
        
            combine_image_test = tf.concat([image_Asian_test, image_UTK_test], 0)
            combine_age_test = tf.concat([age_Asian_test, age_UTK_test], 0)
            combine_gender_test = tf.concat([gender_Asian_test, gender_UTK_test], 0)
            
            pred_age_test, pred_gender_test = model(combine_image_test)
            
            age_test_loss = loss_obj_age(combine_age_test, pred_age_test)
            gender_test_loss = loss_obj_gender(combine_gender_test, pred_gender_test)
            
            loss_age(age_test_loss)
            loss_gender(gender_test_loss)
            
            error_age(combine_age_test, pred_age_test)
            error_gender(combine_gender_test, pred_gender_test)
            
            print('\r Processing[' + '■' *number + '  '*(bar-1-number) + ']', end='')
            
            itera+=1
            
            if itera == int(800/bar):
                number+=1
                itera = 0
            
        template_test = 'Test Age Loss: {:>4.5f}, ------ Test Gender Loss: {:>4.5f}, ------ Test Age Accuracy: {:>4.5f}, ------ Test Gender Accuracy: {:>4.5f}'
        print('\n',template_test.format(loss_age.result(), loss_gender.result(), error_age.result(), error_gender.result()))
        print('--------------------------------------------------------------------')
        
    loss_age.reset_states()
    loss_gender.reset_states()

    error_age.reset_states()
    error_gender.reset_states()
    t+=1    
#     for image, agee, genderr in zip(combine_image, combine_age, combine_gender):
#         print(label)
#         print(agee.numpy())
#         print(genderr.numpy())
#         print("第", str(t), "張圖片")
#         plt.imshow(image.numpy())
#         plt.show()
#         t+=1
#     print('------------------------------------------------------------------------')


    
