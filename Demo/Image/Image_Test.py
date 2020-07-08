import cv2
import os
import numpy as np
import cvlib as cv
import matplotlib.image as mpimg
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--x', help="Input xml file", dest="xml", default='../../Training/Results/Openvino_IR/MFN.xml')
parser.add_argument('--b', help="Input bin file", dest="bin", default='../../Training/Results/Openvino_IR/MFN.bin')
parser.add_argument('--i', help="Input Image Path", dest="Image", default='./Demo_Image')
parser.add_argument('--o', help="Output Image Path", dest="Output_dir", default='./Results/')

args = parser.parse_args()

try:
    from openvino import inference_engine as ie
    from openvino.inference_engine import IENetwork, IEPlugin
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:/n[ {} ] {}".format(exception_type, e))
    sys.exit(1)

image_size = 62
    
def pre_process_image(image, img_height=image_size):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    
    # image = image[...,::-1] #RGB2BGR
    
    processedImg = cv2.resize(image, (h, w), interpolation=cv2.INTER_CUBIC)
    
    # processedImg = processedImg[...,::-1] #BGR2RGB
    
    # Normalize to keep data between 0 - 1
    processedImg = (np.array(processedImg)) / 255.0

    # Change data layout from HWC to CHW
    processedImg = processedImg.transpose((2, 0, 1))
    processedImg = processedImg.reshape((n, c, h, w))

    return processedImg

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

ages = []
for k in range(18, 66):
    ages.append(k)
ages = np.reshape(np.array(ages), (48, 1))

image_path = args.Image

u = 1

for imaggg in os.listdir(image_path):
    
    frame = mpimg.imread(os.path.join(image_path, imaggg))

    face, confidence = cv.detect_face(frame)
    
    w = len(face)
                
    facedist = []

    if w > 0:
        for idx, f in enumerate(face):

            # get corner points of face rectangle       
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
        

            face = frame[startY:endY, startX:endX]
            try:
                processedImg = pre_process_image(face)

                res = exec_net.infer(inputs={input_blob: processedImg})  

                output_node_age = list(res.keys())[0] ##############################

                output_node_gender = list(res.keys())[1] ##############################

                res1 = res[output_node_age] ##############################
                res2 = res[output_node_gender] ##############################

                idx2 = np.argsort(res2[0])[-1]

                pro = res2[0][idx2] * 100
                
                label = ['Female', 'Male']
                
                # 準備顯示用的文字，包含性別與年紀
                text = "{}: {:.2f}%".format(label[idx2], pro) ##############################
                text2 = "age = {:.0f}".format((res1.dot(ages)[0][0]))

                # 根據不同性別採用不同顏色的顯示文字
                if label[idx2] == 'Male':
                    color = (0, 255, 0)
                    
                else:
                    color = (255, 0, 0)

                cv2.putText(frame, text, (startX, endY+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, text2, (startX, endY+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            except Exception as e:
                pass
      
        frame = frame[...,::-1]
        
        print('\r Processing[' + '■' *u + ']', end='')
        
        cv2.imwrite(args.Output_dir + str(u) + '.jpg', frame)
        u+=1
print('\n Successful!')


    
    




    

