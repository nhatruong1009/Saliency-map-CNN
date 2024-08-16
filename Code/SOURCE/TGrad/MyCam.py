import torch

class MyCam2:
    def __init__(self,model):
        model = model.to("mps")
        self.model = model.eval()
        self.activations = []
        self.gradients = []
        self.layers = []

    def __call__(self, input, class_idx=None):
        input = input.to("mps")
        self.get_activations_grads(input,class_idx)
        self.get_data()
        return self.aggregate_smap(self.maps, self.w).cpu().detach().numpy()

    def get_data(self):
        maps = [ (self.activations[0] * self.gradients[0]).clamp(min=0).sum((0,1))]
        w = [  (self.activations[0] * self.gradients[0]).sum()/self.score]
        for i in range(13):
            t = (self.activations_bias[i] * self.gradients[i+1]).clamp(min=0)
            maps.append(t.sum((0,1)))
            sc = (self.activations_bias[i] * self.gradients[i+1]).sum()/self.score
            w.append(sc)
        self.maps = maps
        self.w = w

    def aggregate_smap(self, maps, w):
        mw = []
        for i in maps:
            mw.append(i.clone())
        size = mw[0].shape
        kq = torch.zeros(size)
        kq = kq.to("mps")
        for i in range(len(mw)):
            mw[i] = torch.nn.functional.interpolate(mw[i][None,None],size,mode='bilinear')[0,0]
            mw[i] = mw[i]/mw[i].sum()
            kq += mw[i] *  torch.relu(w[i])
        return kq

    def get_activations_grads(self,input,class_idx = None):
        if class_idx == None:
            class_idx = self.model(input).argmax().item()
        activations = [input.clone()]
        try:
            activations[-1].requires_grad = True
        except:
            try:
                activations[-1].require_grad = True
            except:
                pass
        out = self.model(activations[-1])[0,class_idx]
        grads = [torch.autograd.grad(out,activations[-1])[0]]
        self.score = out.item()
        l = []
        for i in range(len(self.model)):
            if isinstance(self.model[i], torch.nn.Conv2d):
                activations.append( self.model[:i+1](input).clamp(min=0))
                try:
                    activations[-1].requires_grad = True
                except:
                    try:
                        activations[-1].require_grad = True
                    except:
                        pass
                out = self.model[i+1:](activations[-1])[0,class_idx]
                grads.append(torch.autograd.grad(out,activations[-1])[0])
                l.append(i)
        self.activations = activations
        self.gradients = grads
        self.layers = l

        self.activations_bias = []
        k = 1
        for i in self.model:
            if isinstance(i,torch.nn.Conv2d):
                self.activations_bias.append((activations[k] > 0).float() * i.bias[None,:,None,None])
                k+=1

class MyCam:
    def __init__(self, model,device="mps"):
        self.device = device
        self.model = model.to(self.device)

    def assign_hook(self):
        self.hooks = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                forward_handle = layer.register_forward_hook(self.forward_hook)
                backward_handle = layer.register_backward_hook(self.backward_hook)

                self.hooks.append(forward_handle)
                self.hooks.append(backward_handle)

    def remove_hooks(self):
        for i in self.hooks:
            i.remove()
        self.hooks=[]

    def __call__(self, input, class_idx=None):
        self.activations_bias = []
        self.gradients = []
        self.input = input.to(self.device).detach().requires_grad_(True)
        self.get_activations_grads(self.input,class_idx)
        self.get_data()
        return self.aggregate_smap(self.maps, self.w).cpu().detach().numpy()

    def get_data(self):
        maps = [ (self.input* self.gradients[0]).clamp(min=0).sum((0,1))]
        w = [  (self.input * self.gradients[0]).sum()/(self.score+1e-16)]
        for i in range(13):
            t = (self.activations_bias[i] * self.gradients[i+1]).clamp(min=0)
            maps.append(t.sum((0,1)))
            sc = (self.activations_bias[i] * self.gradients[i+1]).sum()/(self.score+1e-16)
            w.append(sc)
        self.maps = maps
        self.w = w

    def aggregate_smap(self, maps, w):
        mw = []
        for i in maps:
            mw.append(i.clone())
        size = mw[0].shape
        kq = torch.zeros(size).to(self.device)
        for i in range(len(mw)):
            mw[i] = torch.nn.functional.interpolate(mw[i][None,None],size,mode='bilinear')[0,0]
            mw[i] = mw[i]/(mw[i].sum()+1e-16)
            kq += mw[i] *  torch.relu(w[i])
        return kq


    def forward_hook(self, module, input, output):
        output = (output >0).float() * module.bias[None,:,None,None]  # ReLU
        self.activations_bias.append(output)

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients.insert(0,grad_out[0])

    def get_activations_grads(self, input, class_idx=None):
        self.assign_hook()
        out = self.model(input)
        if class_idx == None:
            class_idx = out.argmax().item()
        out = out[0,class_idx]
        self.score = out
        out.backward()
        self.gradients.insert(0,input.grad)
        self.remove_hooks()