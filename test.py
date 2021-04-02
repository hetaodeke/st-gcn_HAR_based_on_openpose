import os
# # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import json
import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
# import pickle

# from net.utils.graph import Graph
# from utils import load_data

# import matplotlib.pyplot as plt

# import torch
# print(torch.cuda.is_available())

train_data_path = 'data/ucf101_skeleton/skeleton'
label = os.listdir(train_data_path)
print(len(label))