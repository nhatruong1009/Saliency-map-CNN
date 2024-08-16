import torch
import tqdm
from methods.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None):
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
        # print(f"input_tensor: {input_tensor.shape}")
        # print(f"targets: {targets}")
        # print(f"activations: {activations.shape}")
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            # print(f"upsample: {upsample}")
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to('mps')
            # print(f"activation_tensor: {activation_tensor.shape}")

            upsampled = upsample(activation_tensor)
            print(f"upsampled: {upsampled.shape}")
            self.upsample = upsampled

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            # print(f"maxs: {maxs.shape}")
            # print(f"mins: {mins.shape}")
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]
            # print(f"input_tensors: {input_tensors.shape}")

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 128
            # print(f"BATCH_SIZE: {BATCH_SIZE}")

            scores = []
            for target, tensor in zip(targets, input_tensors):
                # print(f"target: {target}")
                # print(f"tensor: {tensor.shape}")
                # for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :].to('mps')
                    outputs = [target(o).cpu().item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
                    # print(outputs)
            scores = torch.Tensor(scores)
            # print(f"scores: {scores.shape}")
            scores = scores.view(activations.shape[0], activations.shape[1])
            # print(f"scores: {scores.shape}")
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            # print(f"weights: {weights.shape}")
            return weights