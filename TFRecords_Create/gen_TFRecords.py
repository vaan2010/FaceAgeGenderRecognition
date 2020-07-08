import tensorflow as tf
import random
import argparse
import csv
import time
from random import sample

parser = argparse.ArgumentParser()

parser.add_argument('--i', help="Input Image Path", dest="image", default="P:/3.軟體開發/1.演算法/Project/FaceAgeGenderRecognition/Datasets/UTK_FaceData/")
parser.add_argument('--c', help="Input CSV Path", dest="csv", default="./UTK_FaceData.csv")
parser.add_argument('--t', help="Output TFRecords Path", dest="tfrecord", default="./TFRecords")
parser.add_argument('--n', help="Output TFRecords Name", dest="tfrecord_name", default="UTK")

args = parser.parse_args()

train_filename = [] #18-20, 21-23, ......
group_file = [] #(18-20), (21-23), ......
name_gender_age_file = []
test_file = []

i = 18
h = 0

with open(args.csv,"r+") as f:
    csv_read = csv.reader(f)
    for line in csv_read:
        if line[2] == 'Age':
            pass
            
        else:    
            if int(line[2]) == i:
                name_gender_age_file = []
                train_filename = []
                name_gender_age_file.append(line[0])
                name_gender_age_file.append(line[1])
                name_gender_age_file.append(line[2])
                train_filename.append(name_gender_age_file)
                group_file.append(train_filename)
                i+=3
            else:
                name_gender_age_file = []
                name_gender_age_file.append(line[0])
                name_gender_age_file.append(line[1])
                name_gender_age_file.append(line[2])
                train_filename.append(name_gender_age_file)
    

i = 18  
bar = 20

for j in range(0, 16):
    random.shuffle(group_file[j])
    idx = 0
    number = 0
    
    # print(group_file[j])
    for test_set in sample(group_file[j], 100):
        test_file.append(test_set)
        group_file[j].remove(test_set)
    
    with tf.io.TFRecordWriter(args.tfrecord + '\\train\\' + args.tfrecord_name + str(i) + '.tfrecords') as writer:
        for groups in group_file[j]:
            filename = args.image + '\\' + groups[0]
            age = groups[2]
            gender = groups[1]

            
            image = open(filename, 'rb').read()     # 讀取數據集圖片到內存，image 為一個 Byte 類型的字符串
            feature = {                             # 建立 tf.train.Feature 字典
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 圖片是一個 Bytes 對象
                'Age': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(age)])),   # 標籤是一個 Int 對象
                'Gender': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(gender)]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通過字典建立 Example
            writer.write(example.SerializeToString())   # 將Example序列化並寫入 TFRecord 文件

            print('\r Training TFRecords is Processing[' + '■' *number + '  '*(bar-number) + ']', end='')

            idx += 1
            
            if idx == int(len(group_file[j])/bar):
                number+=1
                idx = 0

    print('\n ------> Sucessfully Create ' + args.tfrecord + '/' + args.tfrecord_name + str(i) + '.tfrecords\n')
    i+=3
    
with tf.io.TFRecordWriter(args.tfrecord + '\\test\\' + args.tfrecord_name + '_test' + '.tfrecords') as writer:
    for testset in test_file:
        filename = args.image + '\\' + testset[0]
        age = testset[2]
        gender = testset[1]

        
        image = open(filename, 'rb').read()     # 讀取數據集圖片到內存，image 為一個 Byte 類型的字符串
        feature = {                             # 建立 tf.train.Feature 字典
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 圖片是一個 Bytes 對象
            'Age': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(age)])),   # 標籤是一個 Int 對象
            'Gender': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(gender)]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通過字典建立 Example
        writer.write(example.SerializeToString())   # 將Example序列化並寫入 TFRecord 文件

        print('\r Testing TFRecords is Processing[' + '■' *number + '  '*(bar-number) + ']', end='')

        idx += 1
        
        if idx == int(len(test_file)/bar):
            number+=1
            idx = 0

    print('\n ------> Sucessfully Create ' + args.tfrecord + '/' + args.tfrecord_name + '_test' + '.tfrecords\n')
        

