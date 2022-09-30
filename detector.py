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
    def __init__(self):
        self.threshold = 0.6
        self.weights = EfficientNet_B0_Weights.DEFAULT
        self.model = efficientnet_b0(weights=self.weights)
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2))

        self.preprocess = self.weights.transforms()
        self.model.eval()
        self.saved_vec = torch.ones((1,1280))

   
    def predict(self, batch):
        batch_pred = self.model(batch)
        cattened = torch.cat((batch_pred, self.vector), 0).detach().numpy()
        cattened = cattened / np.linalg.norm(cattened)
        similarities = cosine_similarity(cattened)
        print(similarities)

        return similarities

    def __call__(self, batch):
        predicted_score = self.predict(batch)
        argmax = np.argmax(predicted_score[-1, 0:-1], axis=0)
    
        return batch[argmax], argmax, predicted_score 