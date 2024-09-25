import pickle
import os
import numpy as np
import torch
from tqdm import tqdm

import wandb
from connect4 import Connect4
from model3 import connect_model
from self_play import train, calculate_policy,hashable


model = connect_model()

#parentmodel = Parentmodel(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)
'''
wandb.init(project="SupervisedConnect4")
batch_size = 200
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
files = []
for file in os.listdir():
    if file.endswith('.pkl'):
        files.append(file)
#print(files)

Xs = []
Pis = []
Vs = []
for file  in tqdm(files):
    with open(file, "rb") as inp:
        result_step = pickle.load(inp)

    for res in result_step:
        Xs.append(res[0])
        Pis.append(res[1])
        Vs.append(res[2])

Xs = np.array(Xs)
Pis = np.array(Pis)
Vs = np.array(Vs)

# shuffle (X,P,V)
permutation = np.random.permutation(len(Xs))    

Xs = Xs[permutation]
Pis = Pis[permutation]
Vs = Vs[permutation]

print(Xs.shape, Pis.shape, Vs.shape)
train(model, Xs, Pis, Vs, opt, batch_size,epochs=1)
'''