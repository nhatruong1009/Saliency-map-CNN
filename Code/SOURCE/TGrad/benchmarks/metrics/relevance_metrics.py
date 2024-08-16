import torch
import torchvision
import math


def blur_image(input_image):
    """
    blur the input image tensor
    """
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def grey_image(input_image):
    """
    generate a grey image tensor with same shape as input
    """
    return 0.5 * torch.ones_like(input_image)

class RelevanceMetric:
    """
    base class for Insertion and Deletion
    """
    def __init__(self, model, n_steps, batch_size, baseline="blur",device = "mps"):
        self.device = device
        self.model = model.to(self.device)
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.baseline_fn = grey_image if baseline == "grey" else blur_image

    def __call__(self, image, saliency_map, class_idx=None, *args, **kwargs):
        assert image.shape[-2:] == saliency_map.shape[-2:], "Image and saliency map should have the same resolution"

        with torch.no_grad():
            class_idx = class_idx or self.model(image).max(1)[1].item()
            h, w = image.shape[-2:]

            # generate baseline
            baseline = self.baseline_fn(image).to(self.device)
            # index of pixels in the saliency map in descending order
            sorted_index = torch.flip(saliency_map.view(-1, h * w).argsort(), dims=[-1]).to(self.device)
            samples = self.generate_samples(sorted_index, image, baseline)
            # running sum of the scores
            scores = torch.zeros(self.n_steps).to(self.device)

            for idx in range(math.ceil(samples.shape[0] / self.batch_size)):
                selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, samples.shape[0]))
                with torch.no_grad():
                    res = self.model(samples[selection_slice])
                    scores[selection_slice] = res[:, class_idx]

            auc = torch.sum(scores) / self.n_steps
            return auc, scores

    def generate_samples(self, *args, **kwargs):
        raise NotImplementedError