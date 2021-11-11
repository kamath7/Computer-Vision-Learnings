import torch
from torch.autograd import Variable
import cv2 
from data import BaseTransform, VOC_CLASSES as labelmap 
from ssd import build_ssd
import imageio

#creating the function 

def detectThings(frame, net, transform): #neural net here is our SSD, transform to make it comptaible with the NN
    height, width = frame.shape[:2] #height and width of the frame. shape is an in-built func 
    