import argparse
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from benchmarks.imagesDataset import ImagesDataset
from MyCam import MyCam, MyCam2
from utils import *
from BaseGradient import *

try:
    from pytorch_grad_cam import GradCAMPlusPlus, XGradCAM, EigenCAM, LayerCAM, HiResCAM, FullGrad, RandomCAM
    from methods.grad_cam import GradCAM
    from methods.score_cam import ScoreCAM
    torchcam = True
except:
    print("torchcam not installed: GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM not availables")
    torchcam = False

try:
    from RISE.explanations import RISE
    rise = True
except:
    print("RISE not installed, not available")
    rise = False

try:
    from captum.attr import IntegratedGradients, InputXGradient, Lime, Occlusion, Saliency, NoiseTunnel
    captum = True
except:
    print("captum not installed, IntegratedGradients, InputXGradient, SmoothGrad, Occlusion not availables")
    captum = False

try:
    from zoomcam import object_class
    from zoomcam import function
    zoomcam = True
except:
    print("zoomcam not found")
    zoomcam = False

import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Saliency map methods evaluation")
parser.add_argument("--image_folder", type=str, default='images', help="path to images repository")
parser.add_argument("--image_list", type=str, default='/Users/trungmai-eh/Developer/CAMProjectXAI/datasets/ILSVRC2012_val_folders/image_list.txt',
                    help="path to images list file")
parser.add_argument("--labels_list", type=str, default='/Users/trungmai-eh/Developer/CAMProjectXAI/datasets/ILSVRC2012_val_folders/labels.txt',
                    help="path to labels list file")
parser.add_argument("--model", type=str, default='vgg16', help="model type: vgg16 (default) or resnet50")
parser.add_argument("--saliency", type=str, default='pcampm',
                    help="saliency type: pcamp, pcamm, pcampm (default), gradcam, gradcampp, smoothgradcampp, ig (=IntegratedGrad), ixg (=InputxGrad), sg (=SmoothGrad), occlusion, rise")
parser.add_argument("--cuda", dest="gpu", action='store_true', help="use cuda")
parser.add_argument("--cpu", dest="gpu", action='store_false', help="use cpu instead of cuda (default)")
parser.set_defaults(gpu=False)
parser.add_argument("--batch_size", type=int, default=1, help="max batch size (when saliency method use it), default to 1")
parser.add_argument("--npz_folder", type=str, default="./npz", help="Path to the folder to store the output file")
parser.add_argument("--suffix", type=str, default="", help="Add SUFFIX string to the checkpoint name")

def main():
    global args
    args = parser.parse_args()

    target_layer = None
    if args.model == 'resnet50':
        model = models.resnet50(True)
    elif args.model == 'vgg16':
        model = models.vgg16(True)
        target_layer = [model.features[-1]]
        target_layers = ['features.3', 'features.8', 'features.15', 'features.22', 'features.29']
    else:
        print("model: " + args.model + " unknown, set to vgg16 by default")
        model = models.vgg16(True)

    if args.saliency.lower() == 'rise':
        if not rise:
            print("Cannot use rise, import not available")
            return
        model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
        for p in model.parameters():
            p.requires_grad = False

    if args.saliency.lower() in ["ig", "ixg", "occlusion", "lime", "sg"] and not captum:
        print("cannot use captum methods, import not available")
        return

    if args.saliency.lower() in ["gradcam", "scorecam", "gradcampp", "smoothgradcampp", "sscam", "iscam"] and not torchcam:
        print("cannot use torchcam methods, import not available")
        return

    model.eval()
    if args.gpu:
        model = model.to("mps")

    input_size = 224

    n_maps = 1
    library = None
    if args.saliency.lower() == 'gradcam':
        saliency = GradCAM(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam1':
        saliency = GradCAM(model, target_layers=target_layers[-2])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam2':
        saliency = GradCAM(model, target_layers=target_layers[-3])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam3':
        saliency = GradCAM(model, target_layers=target_layers[-4])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcam4':
        saliency = GradCAM(model, target_layers=target_layers[-5])
        library = "torchcam"
    elif args.saliency.lower() == 'gradcampp':
        saliency = GradCAMPlusPlus(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'xgradcam':
        saliency = XGradCAM(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'eigencam':
        saliency = EigenCAM(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'hirescam':
        saliency = HiResCAM(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'fullgrad':
        saliency = FullGrad(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'mycam':
        saliency = MyCam(model)
    elif args.saliency.lower() == 'mycam2':
        m = generate_model(model)
        saliency = MyCam2(m)
    elif args.saliency.lower() == 'vanilla':
        saliency = VallinaGradient(model)
    elif args.saliency.lower() == 'ig':
        saliency = IntegratedGradient(model)
    elif args.saliency.lower() == 'input':
        saliency = InputGradient(model)
    elif args.saliency.lower() == 'inputx':
        saliency = InputXGradient(model)
    elif args.saliency.lower() == 'smooth':
        saliency = SmoothGradient(model)
    elif args.saliency.lower() == 'randomcam':
        saliency = RandomCAM(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam':
        saliency = ScoreCAM(model, target_layers=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam1':
        saliency = ScoreCAM(model, target_layers=target_layers[-2])
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam2':
        saliency = ScoreCAM(model, target_layers=target_layers[-3])
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam3':
        saliency = ScoreCAM(model, target_layers=target_layers[-4])
        library = "torchcam"
    elif args.saliency.lower() == 'scorecam4':
        saliency = ScoreCAM(model, target_layers=target_layers[-5])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam0':
        saliency = LayerCAM(model, target_layer=target_layer)
        library = "torchcam"
    elif args.saliency.lower() == 'layercam1':
        saliency = LayerCAM(model, target_layer=target_layers[-2])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam2':
        saliency = LayerCAM(model, target_layer=target_layers[-3])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam3':
        saliency = LayerCAM(model, target_layer=target_layers[-4])
        library = "torchcam"
    elif args.saliency.lower() == 'layercam4':
        saliency = LayerCAM(model, target_layer=target_layers[-5])
        library = "torchcam"
    elif args.saliency.lower()[:-1] == 'layercamfusion':
        n_layers = int(args.saliency.lower()[-1])
        first_layer_idx = len(target_layers) - n_layers - 1
        layercam_dict = {}
        for layer in target_layers[first_layer_idx:]:
            layercam_dict[layer] = LayerCAM(model, target_layer=layer)
        def scale_fn(cam, factor):
            return torch.tanh((factor * cam) / cam.max())
        def layer_fusion(inputs, class_idx=0):
            out = model(inputs)
            cam_fusion = torch.zeros((1,1,1,1))
            if args.gpu:
                cam_fusion = cam_fusion.to("mps")
            for layer in reversed(target_layers[first_layer_idx:]):
                saliency_map = layercam_dict[layer](class_idx, out)
                saliency_map = saliency_map.view((1,1,saliency_map.shape[-2],saliency_map.shape[-1]))
                if layer in (target_layers[:3]):
                    saliency_map = scale_fn(saliency_map, 2)
                cam_fusion = F.interpolate(cam_fusion, saliency_map.shape[-2:], mode="bilinear")
                cam_fusion = torch.maximum(cam_fusion, saliency_map)
            return cam_fusion   
        saliency = layer_fusion
        library = "overlay"
    elif args.saliency.lower() == 'rise':
        saliency = RISE(model, (input_size, input_size), args.batch_size)
        saliency.generate_masks(N=6000, s=8, p1=0.1)
        library = "rise"
    elif args.saliency.lower() == 'ig':
        saliency = IntegratedGradients(model)
        library = "captum"
    elif args.saliency.lower() == 'ixg':
        saliency = InputXGradient(model)
        library = "captum"
    elif args.saliency.lower() == 'lime':
        saliency = Lime(model)
        library = "captum"
    elif args.saliency.lower() == 'sg':
        gradient = Saliency(model)
        sg = NoiseTunnel(gradient)
        def sg_fn(inputs, class_idx=0):
            return sg.attribute(inputs, nt_samples=50, nt_samples_batch_size=args.batch_size, target=class_idx).sum(1)
        saliency = sg_fn
        library = "overlay"
    elif args.saliency.lower() == 'occlusion':
        occlusion = Occlusion(model)
        def occ_fn(inputs, class_idx=0):
            return occlusion.attribute(inputs, target=class_idx, sliding_window_shapes=(3,64,64), strides=(3,8,8)).sum(1)
        saliency = occ_fn
        library = "overlay"

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((input_size, input_size)),
                                    transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

    dataset = ImagesDataset(args.image_list, args.labels_list, args.image_folder, transform=transform)

    # Handle multiple saliency maps if needed
    saliencies = [dict() for n in range(n_maps)] if n_maps > 1 else dict()

    for i in tqdm(range(len(dataset)), desc='generating saliency maps'):
        sample, labels, sample_name = dataset[i]
        sample = sample.unsqueeze(0)
        if args.gpu:
            sample = sample.to("mps")
        out = model(sample)
        class_idx = out.squeeze(0).argmax().item()
        # generate saliency map depending on the choosen method
        if library == "torchcam":
            saliency_map = saliency(sample, None)
        elif library == "rise":
            saliency_map = saliency(sample)[class_idx]
        elif library == "captum":
            saliency_map = saliency.attribute(sample, target=class_idx).sum(1)
        else:
            saliency_map = saliency(sample, class_idx=class_idx)

        if n_maps > 1:
            for n in range(n_maps):
                saliencies[n][sample_name] = saliency_map[n].cpu().detach().numpy()
        else:
            saliencies[sample_name] = saliency_map

    # PolyCAM methods output multiples maps for intermediate layers, export in separate files
    if n_maps > 1:
        for n in range(n_maps):
            np.savez(args.npz_folder + "/" + args.model + "_" + args.saliency + args.suffix + str(n), **saliencies[n])
    else:
        np.savez(args.npz_folder + "/" + args.model + "_" + args.saliency + args.suffix, **saliencies)

if __name__ == "__main__":
    main()