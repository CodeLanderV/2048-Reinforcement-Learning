import torch
import torch.nn as nn


class ManagerModelConfig:
    def __init__(self, input_dim=16, hidden_dims=(512, 256), output_dim=4):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim


class ManagerDQN(nn.Module):
    """A small DQN used as the high-level manager.

    Inputs: flattened board (16)
    Outputs: a discrete set of high-level goals (we reuse the 4 move actions for now,
    but this can represent directions or goal-type choices).
    """

    def __init__(self, config: ManagerModelConfig):
        super().__init__()
        layers = []
        last = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
