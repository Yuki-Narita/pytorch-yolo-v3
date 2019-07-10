from __future__ import division
import ros_path_deleter
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

class RosYolo:
    def __init__(self):
        rospy.get_param("~parameter")

        # cfgfile = "cfg/yolov3.cfg"
        # weightsfile = "yolov3.weights"
        cfgfile = rospy.get_param("~cfg_path")
        weightsfile = rospy.get_param("~weights_path")
        self.class_names = rospy.get_param("~class_names_path")
        self.pallete = rospy.get_param("~pallete_path")

        self.num_classes = 80
        args = arg_parse()
        self.confidence = float(args.confidence)
        self.nms_thesh = float(args.nms_thresh)
        self.start = 0
        self.CUDA = torch.cuda.is_available()

        bbox_attrs = 5 + self.num_classes

        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)

        self.model.net_info["height"] = args.reso
        self.inp_dim = int(self.model.net_info["height"])
        
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32

        if self.CUDA:
            self.model.cuda()
                
        self.model.eval()

        self.frames = 0
        self.start = time.time()

        self.pub = rospy.Publisher("yolo_image", Image, queue_size=1)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber("camera_image", numpy_msgs(Image), self.callback)

    def callback(self,data):
        # rosmsg を ndarray に変換したい
        
        frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

        img, orig_im, dim = prep_image(frame, self.inp_dim)
            
        im_dim = torch.FloatTensor(dim).repeat(1,2)                        
        
        
        if self.CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        output = self.model(Variable(img), self.CUDA)
        output = write_results(output, self.confidence, self.num_classes, nms = True, nms_conf = self.nms_thesh)

        if type(output) == int:
            self.frames += 1
            print("FPS of the video is {:5.2f}".format( self.frames / (time.time() - self.start)))
            # cv2.imshow("frame", orig_im)
            # key = cv2.waitKey(1)
            self.pub.publish(self.bridge.cv2_to_imgmsg(orig_im, "bgr8"))
            # if key & 0xFF == ord('q'):
            #     break
            # continue
            return

        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(self.inp_dim))/self.inp_dim
            
        im_dim = im_dim.repeat(output.size(0), 1)
        output[:,[1,3]] *= frame.shape[1]
        output[:,[2,4]] *= frame.shape[0]

        # classes = load_classes('data/coco.names')
        # colors = pkl.load(open("pallete", "rb"))
        classes = load_classes(self.class_names)
        colors = pkl.load(open(self.pallete, "rb"))


        list(map(lambda x: write(x, orig_im), output))

        # cv2.imshow("frame", orig_im) #orig_imをmsgに変換
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break
        self.pub.publish(self.bridge.cv2_to_imgmsg(orig_im, "bgr8"))
        self.frames += 1
        print("FPS of the video is {:5.2f}".format( self.frames / (time.time() - self.start)))

# ROSで実行する場合のファイルパス
# pythonのopencv関連のパスとか

def main():
    rospy.init_node('ros_yolo', anonymous=True)

    ros_yolo = RosYolo()

    rospy.spin()

if __name__ == '__main__':
    main()