import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def overlayCAM(heatmap, origin_img_path ,plt = plt):
    img = cv2.imread(origin_img_path)
    heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    plt.imshow(img)
    plt.imshow(heatmap,cmap='seismic',alpha=0.4)

def input_link(name):
    img = Image.open(name).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)


class Flatten(torch.nn.Module):
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # x = torch.relu(x)  # Applying ReLU activation function
        return x
    
def generate_model(model):
    new_model = torch.nn.Sequential(*(list(model.children())[0] + torch.nn.Sequential(model.avgpool) + torch.nn.Sequential(Flatten()) + list(model.children())[2]))
    layers = torch.nn.Sequential(*(list(new_model.children())))
    return copy.deepcopy(layers)

def predict_decode(yhat):
    probabilities = torch.softmax(yhat, dim=1)
    top5_prob_indices = torch.topk(probabilities, 5)[1].squeeze(0).tolist()
    with open("../labels.txt") as f:
        labels = [line.strip() for line in f.readlines()]
    
    top5_labels = [(labels[i], probabilities[0, i].item(),i) for i in top5_prob_indices]
    for i in top5_labels:
        print(i)
    return (top5_labels[0], int(yhat.argmax()))
