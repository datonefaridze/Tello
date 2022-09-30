import os
import cv2
import torch
import numpy as np
from detector import Detector
import pandas as pd


model = torch.hub.load('yolov5', 'yolov5s', source='local', pretrained=True)

def show(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

vid = cv2.VideoCapture(0)


def filter_detections(results):
    confidence_flt = results['confidence'] >= 0.6
    class_flt = results['class'] == 0
    return results[confidence_flt&class_flt]


while(True):
    ret, img = vid.read()
    results = model(img)
    # import pdb; pdb.set_trace()
    df = results.pandas().xyxy[0]
    df_filtered = filter_detections(df)
    print("img: ", img.shape)
    for id in range(len(df_filtered)):
        print("df_filtered[id]: ", df_filtered.iloc[id])
        xmin, ymin, xmax, ymax, _, _, _ = df_filtered.iloc[id]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        area = (xmax-xmin) * (ymax-ymin)
        center = ((xmin+xmax)//2, (ymin+ymax)//2)
        print(center)
        print("center[0]: ", center[0])
        # cv2.circle(img, int(center[0]), int(center[1]), 5, (0,255,255), cv2.FILLED)
        cv2.circle(img, center, 5, (0,255,255), cv2.FILLED)
        # cropped_image = img[ymin:ymax, xmin:xmax]

        print(img.shape)
        cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

vid.release()
cv2.destroyAllWindows()

