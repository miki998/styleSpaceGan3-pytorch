import torch
from torch.autograd.functional import jacobian
from torchvision.transforms import Resize
import torch.nn.functional as F

# we make a function so to compute the jacobian of such a function
def func_generator(SG, YS, RGBYS, layer=4, fs=32):
    """
    desc: generate a function to compute the jacobian of
    args: 
        - YS   ::[list] (list of layers containing style channels)
        - RGBYS::[list] (list of layers for toRGB)
        - layer::[int]  (layer #th to give as input to generated func)
        - fs   ::[int]  (filter size applied to gradient map to avg resize)
    ret : 
        - f    ::[func] (function to generate resized image)
        - inp  ::[array](supposed input for function generated)

    """
    def f(x):
        inputs   = YS[:layer] + [x] + YS[layer+1:]
        toResize = SG.generate_image_from_ys(inputs, RGBYS, raw=True)[0]

        # pooling 
        pooling  = torch.nn.AvgPool2d((fs,fs), stride=(fs,fs))
        
        return pooling(toResize)

    inp = YS[layer]

    return f, inp

# we make a function so to compute the jacobian of such a function
def func_generator2(SG, YS, layer=4, fs=32):
    """
    desc: generate a function to compute the jacobian of
    args: 
        - YS   ::[list] (list of layers containing style channels)
        - layer::[int]  (layer #th to give as input to generated func)
        - fs   ::[int]  (filter size applied to gradient map to avg resize)
    ret : 
        - f    ::[func] (function to generate resized image)
        - inp  ::[array](supposed input for function generated)

    """
    def f(x):
        inputs   = YS[:layer] + [x] + YS[layer+1:]
        toResize = SG.generate_image_from_ys(inputs, raw=True)[0]

        # pooling 
        pooling  = torch.nn.AvgPool2d((fs,fs), stride=(fs,fs))
        
        return pooling(toResize)

    inp = YS[layer]

    return f, inp