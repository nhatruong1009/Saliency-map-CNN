import numpy as np
import torch
from torch.nn.modules import Module
import torchvision

def blur_image(input_image):
    """
    blur the input image tensor
    """
    return torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])

def grey_image(input_image):
    """
    generate a grey image tensor with same shape as input
    """
    return 0.5 * torch.ones_like(input_image).to(input_image.device)

def black_image(input_image):
    """
    generate a black image tensor with same shape as input
    """
    return 0.0 * torch.ones_like(input_image).to(input_image.device)

class BaseGradient:
    def __init__(self,
                 model: torch.nn.Module,
                 sofmax = True,
                 device="mps") -> None:
        
        self.device = device
        self.model = model.eval().to(self.device)
        self.sofmax = sofmax

    def __call__(self,
                input_tensor: torch.Tensor,
                class_idx) -> np.ndarray:
        raise Exception("Not Implemented") 

class VallinaGradient(BaseGradient):
    def __init__(self, model: Module,sofmax = True, device="mps") -> None:
        super().__init__(model,sofmax ,device)

    def __call__(self, input_tensor: torch.Tensor, class_idx=None) -> np.ndarray:
        input = input_tensor.to(self.device).detach().requires_grad_(True)
        out = self.model(input)
        if class_idx == None:
            class_idx = out.argmax().item()
        if self.sofmax:
            out = torch.softmax(out,dim=1)[:,class_idx]
        else:
            out = out[:,class_idx]
        out.backward()
        return input.grad.abs().max(dim=1)[0][0].cpu().detach().numpy()
        
class InputGradient(BaseGradient):
    def __init__(self, model: Module,sofmax = True, device="mps") -> None:
        super().__init__(model,sofmax ,device)

    def __call__(self, input_tensor: torch.Tensor, class_idx=None) -> np.ndarray:
        input = input_tensor.to(self.device).detach().requires_grad_(True)
        out = self.model(input)
        if class_idx == None:
            class_idx = out.argmax().item()
        if self.sofmax:
            out = torch.softmax(out,dim=1)[:,class_idx]
        else:
            out = out[:,class_idx]
        out.backward()
        sal = (input.grad * input).mean((0,1)).relu().cpu().detach().numpy()
        return sal
    
class IntegratedGradient(BaseGradient):
    def __init__(self, model: Module,sofmax = True, device="mps") -> None:
        super().__init__(model,sofmax ,device)

    def __call__(self, input_tensor: torch.Tensor, baseline = None,step= 50, class_idx = None) -> np.ndarray:
        if class_idx == None:
            class_idx = self.model(input_tensor).argmax().item()
        if baseline is None:
            baseline = black_image(input_tensor)
        inputs = self.make_samples(input_tensor.to(self.device),baseline.to(self.device),step)
        out = self.model(inputs)
        if self.sofmax:
            out = torch.softmax(out,dim=1)[:,class_idx]
        else:
            out = out[:,class_idx]
        return torch.autograd.grad(out,inputs,grad_outputs=torch.ones(step,device=self.device))[0].mean((0,1)).abs().cpu().detach().numpy()

    def make_samples(self,input,baseline,step):
        step_img = (input-baseline) / step
        return torch.stack([(baseline + step_img*i)[0] for i in range(step)]).requires_grad_(True)
    
class SmoothGradient(BaseGradient):
    def __init__(self, model: Module,sofmax = True, mean=0,std=0.1,device="mps") -> None:
        super().__init__(model,sofmax ,device)
        self.mean = mean
        self.std = std

    def __call__(self, input_tensor: torch.Tensor,num_sample= 50, class_idx = None) -> np.ndarray:
        if class_idx == None:
            class_idx = self.model(input_tensor).argmax().item()

        inputs = self.make_samples(input_tensor.to(self.device),self.mean,self.std,num_sample)
        out = self.model(inputs)
        if self.sofmax:
            out = torch.softmax(out,dim=1)[:,class_idx]
        else:
            out = out[:,class_idx]
        return torch.autograd.grad(out,inputs,grad_outputs=torch.ones(num_sample,device=self.device))[0].mean((0,1)).abs().cpu().detach().numpy()
    
    def make_samples(self,input,mean,std,num_sample):
        ssize = list(input.shape)
        ssize[0] = num_sample
        noise = torch.normal(mean,std, size=ssize,device = self.device)
        return (input.repeat(num_sample,1,1,1) + noise).requires_grad_(True)
