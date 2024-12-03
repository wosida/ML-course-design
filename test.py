import torch
import  torch.nn.functional as F
a=torch.randn(2,3)
print(a)
b=F.softmax(a,dim=1)
print(b)
c=F.softmax(a,dim=0)
print(c)