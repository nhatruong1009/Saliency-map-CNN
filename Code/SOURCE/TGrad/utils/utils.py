import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def input_link(name):
    img = Image.open(name).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)

def predict_decode(yhat):
    probabilities = torch.softmax(yhat, dim=1)
    top5_prob_indices = torch.topk(probabilities, 5)[1].squeeze(0).tolist()

    with open("../labels.txt") as f:
        labels = [line.strip() for line in f.readlines()]

    top5_labels = [(labels[i], probabilities[0, i].item()) for i in top5_prob_indices]
    for i in top5_labels:
        print(i)
    return (top5_labels[0], int(yhat.argmax()))
class Flatten(torch.nn.Module):
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return x
    
def generate_flat_model(model):
    new_model = torch.nn.Sequential(*(list(model.children())[0] + torch.nn.Sequential(model.avgpool) + torch.nn.Sequential(Flatten()) + list(model.children())[2]))
    layers = torch.nn.Sequential(*(list(new_model.children())))
    return layers
def show_cam_on_image(
    img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET, image_weight: float = 0.005) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint16(255 * cam)
def cam_func(model, image_tensor, image, target_layers, cam_model):
    resized_img = cv2.resize(image, (224, 224))
    # Convert BGR to RGB (OpenCV loads images as BGR by default)
    image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    cam = cam_model(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
    # model_outputs = cam.outputs
    # plt.imshow(visualization)
    # plt.show()
    return visualization

def visualization(img):
    plt.imshow(img)
    plt.show()