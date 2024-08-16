import torch
from benchmarks.metrics.relevance_metrics import RelevanceMetric

class Deletion(RelevanceMetric):
    def __init__(self, model, step, batch_size, baseline="blur"):
        super(Deletion, self).__init__(model, step, batch_size, baseline=baseline)

    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * image
        for step in range(self.n_steps):
            pixels = index[:, :pixels_per_steps*(step+1)]
            samples[step].view(-1, h * w)[..., pixels] = baseline.view(-1, h * w)[..., pixels]
        return samples