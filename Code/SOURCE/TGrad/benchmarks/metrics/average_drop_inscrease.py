import torch

class AverageDropIncrease:
    """
    Return a tuple of [AverageDrop, IncreaseOfConfidence]
    """
    def __init__(self, model):
        self.model = model

    def __call__(self, image, saliency_map, class_idx=None, *args, **kwargs):
        assert image.shape[-2:] == saliency_map.shape[-2:], "Image and saliency map should have the same resolution"

        with torch.no_grad():
            class_idx = class_idx or self.model(image).max(1)[1].item()

        mask = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        masked_image = mask * image

        base_score = torch.softmax(self.model(image), dim=1)[:, class_idx]
        score = torch.softmax(self.model(masked_image), dim=1)[:, class_idx]
        drop = torch.maximum(torch.zeros(1).to(base_score.device), base_score - score) / base_score
        increase = int(score > base_score)
        return drop.item(), increase