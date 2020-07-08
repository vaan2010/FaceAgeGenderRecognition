import cv2
import os
import numpy as np
import cvlib as cv
import matplotlib.image as mpimg
import time
import pyrealsense2 as rs
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--x', help="Input xml file", dest="xml", default='../../Training/Results/Openvino_IR/MFN.xml')
parser.add_argument('--b', help="Input bin file", dest="bin", default='../../Training/Results/Openvino_IR/MFN.bin')
parser.add_argument('--r', help="Webcam Resolution", dest="Res", default='1280')

args = parser.parse_args()

image_size = 62
    
def pre_process_image(image, img_height=image_size):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    
    image = image[...,::-1] #BGR2RGB
    
    processedImg_o = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
    
    # processedImg = processedImg[...,::-1] #
    
    # Normalize to keep data between 0 - 1
    processedImg = (np.array(processedImg_o)) / 255.0

    # Change data layout from HWC to CHW
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg_f = processedImg.reshape((n, c, h, w))

    return processedImg_f, image 

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:/n[ {} ] {}".format(exception_type, e))
    sys.exit(1)
    
plugin_dir = None
model_xml = args.xml
model_bin = args.bin

# Devices: GPU (intel), CPU, MYRIAD
plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)

# Read IR
net = IENetwork.from_ir(model=model_xml, weights=model_bin)
assert len(net.inputs.keys()) == 1
assert len(net.outputs) == 2 ##############################

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

# Load network to the plugin
exec_net = plugin.load(network=net)
del net

pipeline = rs.pipeline()
config = rs.config()

if args.Res == '1280':
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
elif args.Res == '1920':
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
elif args.Res == '960':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
else:
    print('The input resolution is not supported, we adjust it to 960x540.')
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)

# Start streaming
pipeline.start(config)

i = u = c = agggg = face_n = 0
m_ages = moving_a = [0]
ages = []

for k in range(18, 66):
    ages.append(k)
ages = np.reshape(np.array(ages), (48, 1))

try:
    while True:

        #用來判斷性別與年紀並加以標示的函式
        frames = pipeline.wait_for_frames()     
        color_frame = frames.get_color_frame() #BGR
        frame = np.asanyarray(color_frame.get_data())

        # 找出圖片中的臉孔
        tc = time.time()
        face, confidence = cv.detect_face(frame)
        
        w = len(face)
        facedist = []
        
        if w == 0:
            agggg = c = 0
            m_ages = moving_a = [0]

        elif w > 1:

            if m_ages == [0] or face_n != w:
                m_ages = [0]*w
                moving_a = [0]*w

            else:
                m_ages = m_ages
                moving_a = moving_a
       
            face_n = w
            c = 0
            agggg = 0
            

            for idx, f in enumerate(face):
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]
                face = frame[startY:endY, startX:endX]

                try:
                    processedImg, processedImg_o = pre_process_image(face)
              
                    res = exec_net.infer(inputs={input_blob: processedImg})          
                    
                    output_node_age = list(res.keys())[0] ##############################
                    output_node_gender = list(res.keys())[1] ##############################
                    
                    res1 = res[output_node_age] ##############################
                    res2 = res[output_node_gender] ##############################

                    idx2 = np.argsort(res2[0])[-1]
                
                    pro = res2[0][idx2] * 100
                    
                    label = ['Female', 'Male']

                    if m_ages[idx] == 0:
                        m_ages[idx] = ((res1.dot(ages)[0][0]))
                        moving_a[idx] =1

                    elif abs((m_ages[idx]/moving_a[idx])-((res1.dot(ages)[0][0]))) <= 10:
                        m_ages[idx] += ((res1.dot(ages)[0][0]))
                        moving_a[idx] +=1

                    elif abs((m_ages[idx]/moving_a[idx])-((res1.dot(ages)[0][0]))) > 10:
                        m_ages[idx] = 0
                        moving_a[idx] = 0
                        m_ages[idx] += ((res1.dot(ages)[0][0]))
                        moving_a[idx] += 1
                    
                    else:
                        m_ages[idx] += ((res1.dot(ages)[0][0]))
                        moving_a[idx] += 1
                    
                    text = "{}: {:.2f}%, , age = {:.0f}".format(label[idx2], pro, m_ages[idx]/moving_a[idx])

                    # 根據不同性別採用不同顏色的顯示文字
                    if label[idx2] == 'Male':
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.putText(frame, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                except Exception as e:
                    pass
     
        elif w == 1:
            
            m_ages = [0]
            moving_a = [0]
            
            (startX, startY) = face[0][0], face[0][1]
            (endX, endY) = face[0][2], face[0][3]
            face = frame[startY:endY, startX:endX]

            try:
                processedImg, processedImg_o = pre_process_image(face)
                
                res = exec_net.infer(inputs={input_blob: processedImg})  
                
                output_node_age = list(res.keys())[0] ##############################
                output_node_gender = list(res.keys())[1] ##############################
                
                res1 = res[output_node_age] ##############################
                res2 = res[output_node_gender] ##############################

                idx2 = np.argsort(res2[0])[-1]

                pro = res2[0][idx2] * 100

                label = ['Female', 'Male']

                # 準備顯示用的文字，包含性別與年紀
                if agggg == 0:
                    agggg = ((res1.dot(ages)[0][0]))
                    c = 1
                
                elif abs(((res1.dot(ages)[0][0]))-(agggg/c)) > 10: 

                    c = 0
                    agggg = 0
                    agggg += ((res1.dot(ages)[0][0]))
                    c += 1
                
                elif abs(((res1.dot(ages)[0][0]))-(agggg/c)) <= 10: 
                    agggg += ((res1.dot(ages)[0][0]))
                    c +=1
     
                else:
                    agggg += ((res1.dot(ages)[0][0]))
                    c += 1

                text = "{}: {:.2f}%, , age = {:.0f}".format(label[idx2], pro, (agggg/c))

                if label[idx2] == 'Male':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                
                cv2.putText(frame, text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                # print('X: ',endX-startX,' Y: ', endY-startY)
            except Exception as e:
                pass
                
        tf = time.time()
        u+=(tf - tc)
        i+=1
        
        if i % 20 == 0:
            print('\r Average Processing Time: {:>.3f}'.format(u/20),end = '')
            u = 0
        
        
        cv2.imshow('RealSense', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    # Stop streaming
    pipeline.stop()
   