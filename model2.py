# create transformer model with two heads (value and policy)
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import wandb


class SmallTransformerEncoder(nn.Module):
    def __init__(self):
        super(SmallTransformerEncoder, self).__init__()
        self.linear1 = nn.Linear(49, 256)
        self.linear2 = nn.Linear(256, 256)
        self.P_head = nn.Linear(256, 7)
        # self.V_head = nn.Linear(d_model * 49, 1)

        self.l1_loss = torch.nn.CrossEntropyLoss()

    def forward(self, src, targets=None):
        if targets is None:
            pass

        else:
            src = src.view(-1, 7, 7)
            src = src.view(-1, 49)
            src = src.view(-1, 49)
            src = self.linear1(src)
            src = torch.nn.functional.relu(src)
            src = self.linear2(src)
            src = torch.nn.functional.relu(src)
            P = self.P_head(src).view(-1, 7)
            # calculate entropy
            entropy = (
                Categorical(probs=torch.nn.functional.softmax(P, dim=-1))
                .entropy()
                .mean()
            )

            # P = torch.nn.functional.softmax(P, dim=-1)
            P_t, V_t = targets
            l1 = self.l1_loss(P, P_t)
            return entropy, l1


def connect_model():

    model = SmallTransformerEncoder()
    return model
