import torch
from torch.autograd import Variable
import cv2 
from data import BaseTransform, VOC_CLASSES as labelmap 
from ssd import build_ssd
import imageio

#creating the function 

def detectThings(frame, net, transform): #neural net here is our SSD, transform to make it comptaible with the NN
    height, width = frame.shape[:2] #height and width of the frame. shape is an in-built func 
    frame_t = transform(frame)[0] #getting the transformed frame. op would be a numpy arr 
    x = torch.from_numpy(frame_t).permute(2,0,1)#converting numpy arr to torch tensor. reversing the colors with permute
    x = Variable(x.unsqueeze(0)) #getting one dimensional variable
    y = net(x) #feeding to the nn
    detections = y.data #all the boxes that come up with an object detection
    scale = torch.Tensor([width,height,width,height]) #normalise scale of detected objects bw 0 and 1
    
