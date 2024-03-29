# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:55:57 2019

@author: TMaysGGS
"""

'''Last updated on 12/24/2019 10:21'''
'''Importing the libraries''' 
import tensorflow as tf
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dense, Dropout 
from tensorflow.keras.models import Model 

from Tools.Keras_custom_layers import ArcFaceLossLayer 

'''Building Block Functions'''
def conv_block(inputs, filters, kernel_size, strides, padding, alpha, last = False):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if last == False:
        Z = Conv2D(int(filters*alpha), kernel_size, strides = strides, padding = padding, use_bias = False)(inputs)
        Z = BatchNormalization(axis = channel_axis)(Z)
        A = PReLU(shared_axes = [1, 2])(Z)
    else:
        Z = Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = False)(inputs)
        Z = BatchNormalization(axis = channel_axis)(Z)
        A = PReLU(shared_axes = [1, 2])(Z)
    return A

def separable_conv_block(inputs, filters, kernel_size, strides, alpha):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = SeparableConv2D(int(filters*alpha), kernel_size, strides = strides, padding = "same", use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = PReLU(shared_axes = [1, 2])(Z)
    
    return A

def bottleneck(inputs, filters, kernel, t, s, alpha, r = False):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    Z1 = conv_block(inputs, tchannel, 1, 1, 'same', alpha)
    
    Z1 = DepthwiseConv2D(kernel, strides = s, padding = 'same', depth_multiplier = 1, use_bias = False)(Z1)
    Z1 = BatchNormalization(axis = channel_axis)(Z1)
    A1 = PReLU(shared_axes = [1, 2])(Z1)
    
    Z2 = Conv2D(int(filters*alpha), 1, strides = 1, padding = 'same', use_bias = False)(A1)
    Z2 = BatchNormalization(axis = channel_axis)(Z2)
    
    if r:
        Z2 = add([Z2, inputs])
    
    return Z2

def inverted_residual_block(inputs, filters, kernel, t, strides, n, alpha):
    
    Z = bottleneck(inputs, filters, kernel, t, strides, alpha)
    
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel, t, 1, alpha, True)
    
    return Z

def linear_GD_conv_block(inputs, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = DepthwiseConv2D(kernel_size, strides = strides, padding = 'valid', depth_multiplier = 1, use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    
    return Z

'''Building the MobileFaceNet Model'''
def mobile_face_net_train(num_labels, alpha, loss = 'arcface'):
    
    X = Input(shape = (62, 62, 3)) ##################
    label = Input((num_labels, ))

    M = conv_block(X, 64, 3, 2, 'same', alpha = alpha) # Output Shape: (56, 56, 64) 

    M = separable_conv_block(M, 64, 3, 1, alpha) # (56, 56, 64) 
    
    M = inverted_residual_block(M, 64, 3, t = 2, strides = 2, n = 5, alpha = alpha) # (28, 28, 64) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1, alpha = alpha) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 6, alpha = alpha) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1, alpha = alpha) # (7, 7, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 2, alpha = alpha) # (7, 7, 128) 
    
    M = conv_block(M, 512, 1, 1, 'valid', alpha = alpha, last = True) # (7, 7, 512) 
    
    M = linear_GD_conv_block(M, 4, 1) # (1, 1, 512) ##################
    # kernel_size = 7 for 112 x 112; 4 for 62 x 62
    
    M = conv_block(M, 128, 1, 1, 'valid', alpha = alpha, last = True)
    M = Dropout(rate = 0.2)(M)
    M = Flatten()(M)
    
    M = Dense(128, activation = None, use_bias = False, kernel_initializer = 'glorot_normal')(M) 
    
    if loss == 'arcface': 
        Y = ArcFaceLossLayer(class_num = num_labels)([M, label]) 
        model = Model(inputs = [X, label], outputs = Y, name = 'mobile_face_net') 
    else: 
        Y = Dense(units = num_labels, activation = 'softmax')(M) 
        model = Model(inputs = X, outputs = Y, name = 'mobile_face_net') 
    
    return model 

# model = mobile_face_net_train(2, 0.75, 'softmax')
# model.save('./MFN_62_075.h5')
# model.summary()