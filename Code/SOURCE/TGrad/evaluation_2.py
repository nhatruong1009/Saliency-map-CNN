import argparse
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from benchmarks.imagesDataset import ImagesDataset
from benchmarks.metrics.average_drop_inscrease import AverageDropIncrease
from benchmarks.metrics.average_drop_confidence import AverageDropConfidence
from benchmarks.metrics.loc import *
import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Saliency map faithfullness metrics evaluation from npz")
parser.add_argument("--image_folder", type=str, default=None, help="path to images repository")
parser.add_argument("--image_list", type=str, default='/Users/trungmai-eh/Developer/CAMProjectXAI/datasets/ILSVRC2012_val_folders/image_list.txt',
                    help="path to images list file")
parser.add_argument("--labels_list", type=str, default='/Users/trungmai-eh/Developer/CAMProjectXAI/datasets/ILSVRC2012_val_folders/labels.txt',
                    help="path to labels list file in txt")
parser.add_argument("--model", type=str, default='vgg16', help="model type: vgg16 (default) or resnet50")
parser.add_argument("--dataset_name", type=str, default='imagenet', help="model type: vgg16 (default) or resnet50")
parser.add_argument("--saliency_npz", type=str, default='', help="saliency file")
parser.add_argument("--cuda", dest="gpu", action='store_true', help="use cuda")
parser.add_argument("--cpu", dest="gpu", action='store_false', help="use cpu instead of cuda (default)")
parser.set_defaults(gpu=False)
parser.add_argument("--npz_folder", type=str, default="./npz", help="Path to the folder where npz are stored")
parser.add_argument("--csv_folder", type=str, default="./csv", help="Path to the folder to store the csv outputs")
parser.add_argument("--batch_size", type=int, default=64, help="max batch size, default to 1")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="max batch size, default to 1")
parser.add_argument("--loc_threshold", type=float, default=0.2, help="max batch size, default to 1")
parser.add_argument("--percentile", type=int, default=100, help="max batch size, default to 1")

def main():
    global args
    args = parser.parse_args()
    if args.image_folder == None and args.dataset_name == 'imagenet':
        args.image_folder = f"../../ILSVRC2012_val_folders"
    if args.image_list == None and args.dataset_name == 'imagenet':
        args.image_list = f"../ILSVRC2012_val_folders/image_list.txt"
    if args.labels_list == None and args.dataset_name == 'imagenet':
        args.labels_list = f"../ILSVRC2012_val_folders/labels.txt"

    if args.model == 'resnet50':
        model = models.resnet50(True)
    elif args.model == 'vgg16':
        model = models.vgg16(True)
    else:
        print("model: " + args.model + " unknown, set to resnet18 by default")
        model = models.resnet18(True)

    model.eval()
    model_softmax = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))
    if args.gpu:
        model = model.to("mps")
        model_softmax.to("mps")

    input_size = 224

    # get saliencies from file
    saliencies = np.load(args.npz_folder + "/" + args.saliency_npz, allow_pickle=True)

    # Set metrics, use input size as step size
    # average_drop_increase = AverageDropIncrease(model_softmax)
    average_drop_confidence = AverageDropConfidence(model_softmax, 224, args.batch_size)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((input_size, input_size)),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    dataset = ImagesDataset(args.image_list, args.labels_list, args.image_folder, transform=transform)

    drop_ADI_dict = dict()
    increase_ADI_dict = dict()

    for i in tqdm(range(len(dataset))):
        image, _, image_name = dataset[i]
        image = image.unsqueeze(0)
        saliency = torch.tensor(saliencies[image_name])

        # upscale saliency
        sh, sw = saliency.shape[-2:]
        saliency = saliency.view(1,1,sh,sw)
        saliency = F.interpolate(saliency, image.shape[-2:], mode='bilinear')

        if args.gpu:
            image = image.to("mps")
            saliency = saliency.to("mps")

        kq = model(image)
        class_idx = kq.max(1)[1].item()
        original_score = torch.softmax(kq,1)[0,class_idx]

        # _, increase_ADI = average_drop_increase(image, saliency, class_idx=class_idx)
        _, scores = average_drop_confidence(image, saliency, class_idx=class_idx)
        # scores = scores.cpu().numpy()
        # original_score = original_score.cpu().detach().numpy()
        drop_scores = [(original_score - score) / original_score for score in scores]
        avg_drop = 100 * sum(drop_scores) / len(drop_scores)

        drop_ADI_dict[image_name] = avg_drop.cpu().detach().numpy()

    csv_suffix = '.'.join(args.saliency_npz.split('.')[:-1]) + ".csv"
    pd.DataFrame.from_dict(drop_ADI_dict, orient='index').to_csv(args.csv_folder + "/" + 'drop_adi_' + csv_suffix)
    # pd.DataFrame.from_dict(increase_ADI_dict, orient='index').to_csv(args.csv_folder + "/" + 'increase_adi_' + csv_suffix)

if __name__ == "__main__":
    main()
