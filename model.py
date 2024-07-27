# create transformer model with two heads (value and policy)
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import wandb


class SmallTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(SmallTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embed = nn.Linear(1, d_model)
        self.encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layers, num_layers=num_layers
        )
        self.P_head = nn.Linear(d_model * 49, 7)
        self.V_head = nn.Linear(d_model * 49, 1)

        self.l1_loss = torch.nn.CrossEntropyLoss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, src, targets=None):
        if targets is None:
            src = torch.from_numpy(src).to(torch.float32)
            src = src.view(-1, 7, 7)
            src = src.view(-1, 49)
            src = src.view(-1, 49, 1)
            src = self.embed(src)
            src = self.transformer_encoder(src)
            src = src.view(-1, self.d_model * 49)
            P = self.P_head(src)
            V = self.V_head(src)
            P = torch.nn.functional.softmax(P, dim=-1)
            # print(P.shape)
            return {"P": P.view(7).numpy(), "V": V.view(-1).numpy()}

        else:
            src = src.view(-1, 7, 7)
            src = src.view(-1, 49)
            src = src.view(-1, 49, 1)
            src = self.embed(src)
            src = self.transformer_encoder(src)
            src = src.view(-1, self.d_model * 49)
            P = self.P_head(src).view(-1, 7)
            V = self.V_head(src).view(-1, 1)
            # calculate entropy
            entropy = (
                Categorical(probs=torch.nn.functional.softmax(P, dim=-1))
                .entropy()
                .mean()
            )

            # P = torch.nn.functional.softmax(P, dim=-1)
            P_t, V_t = targets
            l1 = self.l1_loss(P, P_t)
            l2 = self.l2_loss(V.view(-1, 1), V_t.view(-1, 1))
            return entropy, l1 + l2


def connect_model():

    d_model = 512  # Dimensionality of input
    nhead = 8  # Number of heads in the transformer model
    num_layers = 6  # Number of transformer layers
    model = SmallTransformerEncoder(d_model, nhead, num_layers)
    return model
