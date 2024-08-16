import torch
from benchmarks.metrics.relevance_metrics import RelevanceMetric

class AverageDropConfidence(RelevanceMetric):
    def __init__(self, model, step, batch_size, baseline="blur"):
        super(AverageDropConfidence, self).__init__(model, step, batch_size, baseline=baseline)

    def generate_samples(self, index, image, baseline):
        h, w = image.shape[-2:]
        pixels_per_steps = (h * w + self.n_steps - 1) // self.n_steps
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * image  # Start with the original image
        custom_index = torch.flip(index, dims=(1,))
        for step in range(self.n_steps):
            pixels = custom_index[:, :pixels_per_steps*(step+1)]
            samples[step].view(-1, h * w)[..., pixels] = baseline.view(-1, h * w)[..., pixels]
        return samples

    def compute_metric(self, image, baseline, label, index):
        original_confidence = self.model(image).softmax(dim=1)[0, label].item()
        samples = self.generate_samples(index, image, baseline)
        confidences = [self.model(sample.unsqueeze(0)).softmax(dim=1)[0, label].item() for sample in samples]
        drop_confidences = [(original_confidence - confidence) / original_confidence for confidence in confidences]
        average_drop_confidence = 100 * sum(drop_confidences) / len(drop_confidences)
        return average_drop_confidence
    