import torch
import cv2
import torch.nn as nn
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch
import torch.nn as nn
import json
import os
import cv2
import numpy as np
import os
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class Detector():
    def __init__(self, init_img):
        self.threshold = 0.6
        self.weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=self.weights)
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.preprocess = self.weights.transforms()
        self.model.eval()
        self.saved_vec = None
        self.overwrite_saved_vec(init_img)
        
    def overwrite_saved_vec(self, img):
        if type(img) == np.ndarray:
            img = torch.tensor(img, device=self.device).permute(2, 0, 1)
            processed_img = self.preprocess(img)
        else:
            processed_img = img
        self.saved_vec = self.model(processed_img.unsqueeze(0))
        print("saved_vec: ", self.saved_vec)

    def predict(self, batch):
        batch_pred = self.model(batch)
        cattened = torch.cat((batch_pred, self.saved_vec), 0).cpu().detach().numpy()
        cattened = cattened / np.linalg.norm(cattened)
        similarities = cosine_similarity(cattened)
        print(similarities)

        return similarities

    def __call__(self, img, centers):
        print("len(centers): ", len(centers))
        if len(centers) == 0 : return None, None, None
        processed_imgs = []
        for center in centers:
            center_, bbox = center
            (xmin, ymin),(xmax, ymax) = bbox
            print(ymin, ymax, xmin, xmax)
            tensor = torch.tensor(img[ymin:ymax, xmin:xmax], device=self.device)
            print(tensor.shape)
            processed = self.preprocess(tensor.permute(2, 0, 1))
            processed_imgs.append(processed.unsqueeze(0))

        batch = torch.cat(processed_imgs, 0)
        selected_argmax, argmax, predicted_score  = self.process(batch)
        print("predicted_score: ", predicted_score)
        print("argmax: ", argmax)
        
        self.overwrite_saved_vec(selected_argmax)
        
        return selected_argmax, argmax, predicted_score
    
    def process(self, batch):
        predicted_score = self.predict(batch)
        argmax = np.argmax(predicted_score[-1, 0:-1], axis=0)
    
        return batch[argmax], argmax, predicted_score
    