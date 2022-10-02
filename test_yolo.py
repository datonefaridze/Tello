import os
import cv2
import torch
import numpy as np
from detector import Detector
import pandas as pd
from custom_utils import *



model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)
vid = cv2.VideoCapture(0)
area_dist = {'area': []}


while(True):
    ret, img = vid.read()
    img = cv2.resize(img, (360,360))
    areas, centers = detect_yolo(model, img)
    img = draw_figures(img, centers)
    cv2.imshow('frame', img)
    area_dist['area'].append(areas)
    area_df = pd.DataFrame(area_dist)
    area_df.to_csv("data.csv")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

