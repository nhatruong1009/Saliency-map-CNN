import torch
from benchmarks.metrics.relevance_metrics import RelevanceMetric

class Insertion(RelevanceMetric):
    def __init__(self, model, step, batch_size, baseline="blur"):
        super(Insertion, self).__init__(model, step, batch_size, baseline=baseline)

    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * baseline
        for step in range(self.n_steps):
            pixels = index[:, :pixels_per_steps*(step+1)]
            samples[step].view(-1, h * w)[..., pixels] = image.view(-1, h * w)[..., pixels]
        return samples

class CustomRelevance(RelevanceMetric):

    def __call__(self, image, saliency_map, class_idx=None, *args, **kwargs):
        assert image.shape[-2:] == saliency_map.shape[-2:], "Image and saliency map should have the same resolution"

        with torch.no_grad():
            class_idx = class_idx or self.model(image).max(1)[1].item()
            h, w = image.shape[-2:]

            baseline = self.baseline_fn(image)
            sorted_index = torch.flip(saliency_map.view(-1, h * w).argsort(), dims=[-1])
            return self.generate_samples(sorted_index, image, baseline)

class CustomInsertion(CustomRelevance):
    """CustomInsertion."""

class CustomDeletion(CustomRelevance):
    """CustomDeletion."""
