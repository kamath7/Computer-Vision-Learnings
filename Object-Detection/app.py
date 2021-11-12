from os import read
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# creating the function


# neural net here is our SSD, transform to make it comptaible with the NN
def detectThings(frame, net, transform):
    # height and width of the frame. shape is an in-built func
    height, width = frame.shape[:2]
    # getting the transformed frame. op would be a numpy arr
    frame_t = transform(frame)[0]
    # converting numpy arr to torch tensor. reversing the colors with permute
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))  # getting one dimensional variable
    y = net(x)  # feeding to the nn
    detections = y.data  # all the boxes that come up with an object detection
    # normalise scale of detected objects bw 0 and 1
    scale = torch.Tensor([width, height, width, height])
    # detections tensor has 4 values. 1st value -> batch(batches of ops), 2nd value s-> no of classes(objects that are detected - beach, car, plane), 3rd value of no of occurences of the object (no of cars detected etc.) 4th value is  a tuple (score, x0, y0, x1,y1). if score is < 0.6 object not found else object is present and this is for each class
    for i in range(detections.size(1)):  # detection.size(1) -> no of classes
        j = 0  # number of times class has occurred
        while detections[0, i, j, 0] >= 0.6:  # basically gets the score
            # drawing the rectangles
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(
                pt[1])), (int(pt[2]), int(pt[3])), (0, 255, 0), 2)  # 2 - thickness of the text
            cv2.putText(frame, labelmap[i-1], (int(pt[2]), int(pt[3])),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)  # mentioning what the object is
            j += 1
    return frame


network_neural = build_ssd('test')  # already trained hence test
network_neural.load_state_dict(torch.load(
    'ssd300_mAP_77.43_v2.pth', map_location=lambda storage, loc: storage))  # feeding it weights as tensors

#transformation process
transform = BaseTransform(network_neural.size, (104/256.0, 117/256.0, 123/256.0)) #target size of images. scale values for colors 
reader = imageio.get_reader('funny_dog.mp4')
fps = reader.get_meta_data()['fps'] #getting the frames per second. to be used in op 
writer = imageio.get_writer('op.mp4',fps=fps)

for i, frame in enumerate(reader): #processing frame by frame
    frame = detectThings(frame, network_neural.eval(), transform)
    writer.append_data(frame)
    print(i) #which frame is going on 

writer.close()