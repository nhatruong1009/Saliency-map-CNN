# Khoá luận tốt nghiệp 2024

## Tên đề tài: Diễn giải đặc trưng mô hình học sâu cho tác vụ phân lớp ảnh

## Tên tiếng anh: Deep learning feature interpretation for image classification tasks

Thông tin sinh viên:

- Mai Quý Trung - MSSV 20127370
- Trần Nhật Trường - MSSV 20127376

Giáo viên hướng dẫn: GS. TS. Lê Hoài Bắc

## Thông tin bài báo đã nộp:

### T-Grad: A Trustworthy Gradient Explanation for Convolutional Neural Networks

### Authors: [Truong Tran](https://orcid.org/0009−0005−2467−3175), [Trung Mai](https://orcid.org/0009−0007−9947−145X), [Duc Le](https://orcid.org/0009-0006-1574-1743), [Bac Le](https://orcid.org/0000−0002−4306−6945)

> **Abstract:** In this paper, we introduce T-Grad, a trustworthy AI interpretation method designed to enhance the understanding of features that explain model behavior in image classification tasks. By combining gradients and biases with advanced upgrades to generate saliency maps, T-Grad offers a more accurate interpretation of a model's true behavior. This method has demonstrated its effectiveness through various evaluations on the ImageNet1K dataset, including performance experiments, visual assessments, and execution speed analyses. We believe that T-Grad can significantly contribute to the overall concept of Explainable AI (XAI) in general and improve the reliability and robustness of image classification models specifically.

## Summary

Our proposed method, T-Grad, is provided in `TGrad/MyCam.py` file. This method works well with any pytorch model but is only complete with models that can be decomposed into matrix multiplication or Hadamard multiplication.

## Prequesites:

- Install Python ($\ge 3.9$)
- Install `pip` (Python Install Packager)
- Install required packages:
```shell
pip install -r requirements.txt
```

## Download dataset

```python
python DownloadDataset/download.py -dataset "imagenet" -version "2012" -split "train"

python DownloadDataset/download.py -dataset "imagenet" -version "2012" -split "valid"

tar -xvf ILSVRC2012_img_val.tar
```

## How to use:
```python
from MyCam import MyCam2,MyCam
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


# create a model
model = models.vgg16(pretrained=True)

# method creation
method =  MyCam(model,device='gpu')

# load image
img = Image.open('./test.jpeg').convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_tensor = preprocess(img).unsqueeze(0)


#create saliency map

sal_map = method(img_tensor) # for the highest class
# sal_map = method(img_tensor,class_idx = 10) # for the specified class


# visualize saliency map
plt.imshow(sal_map)

# after create sal_map, the method still save inner maps at self.maps and its correspond weight at self.w
plt.imshow(method.maps[7].detach().numpy())
```

## Generate saliency maps for the dataset

```python
python CAM/generate_explanations.py --model vgg16 --saliency scorecam --cuda --batch_size 32 --image_folder ILSVRC2012_val_folders
```

## Evaluate with Insertion, Deletion and Insertion-Deletion metrics

```
python CAM/evaluation.py --model vgg16 --saliency_npz vgg16_scorecam.npz --cuda --image_folder ILSVRC2012_val_folders --batch_size 64
```

## Evaluate with % Increase of Confidence and Average Drop % metrics

```
python CAM/evaluation_2.py --model vgg16 --saliency_npz vgg16_scorecam.npz --cuda --image_folder ILSVRC2012_val_folders --batch_size 64
```

**Note:**

- Results for evaluation metrics are located in `CAM/metrics.ipynb`

- Results for creating saliency maps using T-Grad and Sanity check are located in `CAM/testcam.ipynb`
