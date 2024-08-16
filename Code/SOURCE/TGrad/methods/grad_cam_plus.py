import numpy as np
from methods.base_cam import BaseCAM


class GradCAMPlus(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAMPlus,
            self).__init__(
            model,
            target_layers,
            reshape_transform)
        print("GradCAM")
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        print(f"get_cam_weights GradCAM")
        return np.mean(grads, axis=(2, 3))