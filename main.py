import torch
import numpy
print(torch.__version__)

x = torch.rand(2,2)
y = torch.rand(2,2)

z = torch.mul(x,y)

