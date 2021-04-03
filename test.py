import torch

a = torch.ones((1,16))
a.reshape(-1)
print(a.shape)