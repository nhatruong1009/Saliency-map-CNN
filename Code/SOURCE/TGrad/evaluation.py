import argparse
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from benchmarks.imagesDataset import ImagesDataset
from benchmarks.metrics.insertion import Insertion
from benchmarks.metrics.deletion import Deletion
from benchmarks.metrics.average_drop_inscrease import AverageDropIncrease
from benchmarks.metrics.road import ROADCombined
from benchmarks.metrics.loc import *
from benchmarks.meters import AverageEpochMeter
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
    
    if args.image_folder == None:
        if args.dataset_name == 'imagenet':
            args.image_folder = f"../../ILSVRC2012_val_folders"
    if args.image_list == None:
        if args.dataset_name == 'imagenet':
            args.image_list = f"../ILSVRC2012_val_folders/image_list.txt"
        
    if args.labels_list == None:
        if args.dataset_name == 'imagenet':
            args.labels_list = f"../ILSVRC2012_val_folders/labels.txt"
    
    # Initiate meters - Top1-Cls, GT-known Loc, Top1-Loc
    top1_cls_meter = AverageEpochMeter('Top-1 Cls')
    top5_cls_meter = AverageEpochMeter('Top-5 Cls')
    gt_loc_meter = AverageEpochMeter('GT-Known Loc with {}'.format(args.loc_threshold))
    top1_loc_meter = AverageEpochMeter('Top-1 Loc {}'.format(args.loc_threshold))

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
    insertion = Insertion(model_softmax, input_size, args.batch_size)
    deletion = Deletion(model_softmax, input_size, args.batch_size)
    average_drop_increase = AverageDropIncrease(model_softmax)
    road_combined = ROADCombined()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((input_size, input_size)),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    dataset = ImagesDataset(args.image_list, args.labels_list, args.image_folder, transform=transform)

    ins_auc_dict = dict()
    del_auc_dict = dict()
    ins_details_dict = dict()
    del_details_dict = dict()
    drop_ADI_dict = dict()
    increase_ADI_dict = dict()
    # road_combined_dict = dict()
    # energy_point_game_dict = dict()
    # gt_loc_dict = dict()
    # top1_loc_dict = dict()
    # top1_cls_dict = dict()
    # top5_cls_dict = dict()

    for i in tqdm(range(len(dataset))):
        # image, label, image_name, gt_boxes = dataset[i]
        image, label, image_name = dataset[i]
        # gt_boxes = dataset[i][3] if len(dataset[i]) > 3 else []
        image = image.unsqueeze(0)
        saliency = torch.tensor(saliencies[image_name])

        # upscale saliency
        sh, sw = saliency.shape[-2:]
        saliency = saliency.view(1,1,sh,sw)
        saliency = F.interpolate(saliency, image.shape[-2:], mode='bilinear')

        # set image and saliency to gpu if required
        if args.gpu:
            image = image.to("mps")
            saliency = saliency.to("mps")

        # get class predicted by the model for the full image, it's the class used to generate saliency map
        class_idx = model(image).max(1)[1].item()
        
        
        # Compute Top-1 and Top-5 Cls
        predictions = class_idx
        # top1_cls, top5_cls = topk_accuracy(predictions, label, topk=(1,5))
        # top1_cls_meter.update(top1_cls, args.batch_size)
        # top5_cls_meter.update(top5_cls, args.batch_size)

        # Class activation map
        # gt_cams, _ = cam(model, labels=label, truncate=False)
        # unnormalized_images = unnormalize_images(image, args.dataset_name)
        # bboxes, blended_bboxes = extract_bbox(unnormalized_images, gt_cams, gt_boxes, args.loc_threshold, percentile=args.percentile)

        # compute insertion and deletion for each step + auc on the image
        ins_auc, ins_details = insertion(image, saliency, class_idx=class_idx)
        del_auc, del_details = deletion(image, saliency, class_idx=class_idx)
        drop_ADI, increase_ADI = average_drop_increase(image, saliency, class_idx=class_idx)
        # epg = energy_point_game(bboxes, saliency)
        # road_details = road_combined(input_tensor=image, cams=, targets=, model=model)

        # gt_loc, top1_loc = loc_accuracy(predictions, label, gt_boxes, bboxes, args.iou_threshold)
        # gt_loc_meter.update(gt_loc, args.batch_size)
        # top1_loc_meter.update(top1_loc, args.batch_size)
        # top1_cls = top1_cls_meter.compute()
        # top5_cls = top5_cls_meter.compute()
        # gt_loc = gt_loc_meter.compute()
        # top1_loc = top1_loc_meter.compute()
        
        # store every values for the image in dictionary
        ins_auc_dict[image_name] = ins_auc.cpu().numpy()
        ins_details_dict[image_name] = ins_details.cpu().numpy()
        del_auc_dict[image_name] = del_auc.cpu().numpy()
        del_details_dict[image_name] = del_details.cpu().numpy()
        drop_ADI_dict[image_name] = drop_ADI
        increase_ADI_dict[image_name] = increase_ADI
        # energy_point_game_dict[image_name] = epg
        # top1_cls_dict[image_name] = top1_cls
        # top5_cls_dict[image_name] = top5_cls
        # gt_loc_dict[image_name] = gt_loc
        # top1_loc_dict[image_name] = top1_loc

    csv_suffix = '.'.join(args.saliency_npz.split('.')[:-1]) + ".csv"
    pd.DataFrame.from_dict(ins_auc_dict, orient='index').to_csv(args.csv_folder + "/" + 'ins_auc_' + csv_suffix)
    pd.DataFrame.from_dict(del_auc_dict, orient='index').to_csv(args.csv_folder + "/" + 'del_auc_' + csv_suffix)
    pd.DataFrame.from_dict(ins_details_dict, orient='index').to_csv(args.csv_folder + "/" + 'ins_details_' + csv_suffix)
    pd.DataFrame.from_dict(del_details_dict, orient='index').to_csv(args.csv_folder + "/" + 'del_details_' + csv_suffix)

if __name__ == "__main__":
    main()
