import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP,self).__init__()

        self.input_dim = input_dim
        self.output_dim = num_classes

        self.linear1 = nn.Linear(self.input_dim, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64,self.output_dim)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
            self.linear1,
            self.act,
            self.linear2,
            self.act,
            self.linear3
        )

    def forward(self, input):
        output = self.mlp(input)
        return output