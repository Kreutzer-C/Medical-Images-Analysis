import torch
import torch.nn as nn
from model.Mymodel import MLP
from torchsummary import summary

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

model = MLP(64,10).to(device)
summary(model, (64,))