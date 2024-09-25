import pickle

import numpy as np
import torch
from tqdm import tqdm

import wandb
from connect4 import Connect4
from model import connect_model,Parentmodel,MLP,Perfect
from self_play import train, calculate_policy,hashable


model = connect_model()
#model.load_state_dict(torch.load("model.pt"))
parentmodel = Parentmodel()
#parentmodel = MLP()
"""
for param in parentmodel.model.parameters():
    param.requires_grad = False"""
wandb.init(project="probe")

#load ds

Xs = []
S = []

#generate 10000 random positions
for i in tqdm(range(100000)):
    board = Connect4()
    board.initialize_random(num_moves=20)

    Xs.append(board.board)
    S.append(board.to_move)


Xs = np.array(Xs)
S = np.array(S)

# shuffle (X,P,V)
permutation = np.random.permutation(len(Xs))

Xs = Xs[permutation]

S = S[permutation]

print(Xs.shape,S.shape)
opt = torch.optim.Adam(parentmodel.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
batch_size = 100
steps = Xs.shape[0] // batch_size
epochs = 1
parentmodel = parentmodel.to("cuda")
for epoch in range(epochs):
    for step in range(steps):
        X = (
            torch.from_numpy(Xs[step * (batch_size) : (step + 1) * batch_size])
            .to(torch.float32)
            .to("cuda")
        )
        S_i = (
            torch.from_numpy(S[step * (batch_size) : (step + 1) * batch_size])
            .to(torch.float32)
            .to("cuda")
        )
        opt.zero_grad()
        #print(X)
        pred = parentmodel(X)

        #print(pred)
        #print(S_i)

        #print(pred.shape,S_i.shape)
        l = loss_fn(pred,S_i.view(-1,1))

        print(step, l.item())
        pred_sign = torch.sign(pred)
        
        # Compare pred_sign with target
        correct = (pred_sign == S_i.view(-1,1)).float()
        
        # Calculate accuracy
        accuracy = correct.mean().detach().item()
        wandb.log({"loss":l.item(),"accuracy": accuracy})

        l.backward()
        opt.step()