import os
import cv2
import torch
import numpy as np
from detector import Detector
import pandas as pd
from custom_utils import *



model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)
# img = cv2.imread(r"D:\WIN_20221004_18_16_31_Pro.jpg")
# areas, centers = detect_yolo(model, img)

# print("centers: ", centers)

vid = cv2.VideoCapture(0)
first_frame = True
area_dist = {'area': []}

# 
ret, img = vid.read()
rect = cv2.selectROI("select the area", img)
first_frame = False
print(rect)
x1=rect[0]
y1=rect[1]
x2=rect[2]
y2=rect[3]
img_crop=img[y1:y1+y2,x1:x1+x2]
show(img_crop)
detector = Detector(img_crop)

while(True):
    ret, img = vid.read()
    img = cv2.resize(img, (360,360))
    areas, centers = detect_yolo(model, img)
    centers, _, argmax, _ = detector(img, centers)
    # print("argmax: ", argmax)
    if argmax!=None:
        centers = [centers[argmax]]
    print("centers: ", centers)
    if len(centers) == 2:
        import pdb; pdb.set_trace()
    img = draw_figures(img, centers)
    cv2.imshow('frame', img)
    area_dist['area'].append(areas)
    area_df = pd.DataFrame(area_dist)
    area_df.to_csv("data.csv")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

