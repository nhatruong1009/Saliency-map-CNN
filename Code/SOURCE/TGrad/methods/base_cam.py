import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple, Optional
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from methods.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 tta_transforms: Optional[tta.Compose] = None) -> None:
        # print("init basecan")
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        # print(self.activations_and_grads.activations)
        # print(self.activations_and_grads.gradients)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """
    def get_activations_and_grads(self):
        return self.activations_and_grads
    def get_target_layers(self):
        return self.target_layers
    def get_grads_list(self):
        return self.grads_list
    def get_activations_list(self):
        return self.activations_list
    def get_weights(self):
        return self.weights
    
    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        # print(f"get_cam_weights")
        raise Exception("Not Implemented")

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:
        # print(f"get_cam_image")

        self.weights = weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads)
        
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:
        # print(f"forward")
        # print(f"input tensor: {input_tensor}")
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)
        # print(f"input_tensor : {input_tensor.shape}")

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        # print(f"self.outputs : {self.outputs.shape}")

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        # print(f"cam_per_layer : {cam_per_layer[0].shape}")
        
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        # print(f"get_target_width_height")
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        # print(f"compute_cam_per_layer")
        self.activations_list = activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        # print(f"len activations_list: {len(activations_list)}")
        self.grads_list = grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # print(f"len grads_list: {len(grads_list)}")
        target_size = self.get_target_width_height(input_tensor)
        # print(f"tar : {target_size}")
        # print(f"grads_list : {grads_list[0].shape}")
        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                # layer_activations = activations_list[i]
                # print(f"loop : {activations_list[i].shape}")
                layer_activations = scale_cam_image(activations_list[i], target_size)
            if i < len(grads_list):
                # layer_grads = grads_list[i]
                layer_grads = scale_cam_image(grads_list[i], target_size)

            # print(f"layer_activations: {layer_activations.shape}")
            # print(f"layer_grads: {layer_grads.shape}")
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            # print(f"cam: {cam.shape}")
            cam = np.maximum(cam, 0)
            # print(f"cam: {cam.shape}")
            scaled = scale_cam_image(cam, target_size)
            # print(f"scaled: {scaled.shape}")
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        # print(f"aggregate_multi_layers")
        # print(f"cam_per_target_layer: {cam_per_target_layer[0].shape}")
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        # print(f"cam_per_target_layer: {cam_per_target_layer[0].shape}")
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        # print(f"cam_per_target_layer: {cam_per_target_layer[0].shape}")
        result = np.mean(cam_per_target_layer, axis=1)
        # print(f"result: {result.shape}")
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        # print(f"forward_augmentation_smoothing")
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:
        # print(f"__call__")

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True