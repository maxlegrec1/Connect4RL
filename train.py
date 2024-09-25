import pickle

import numpy as np
import torch
from tqdm import tqdm

import wandb
from connect4 import Connect4
from model3 import connect_model
from self_play_par import train

wandb.init(project="reinforcement")
model = connect_model()
#model.load_state_dict(torch.load("model.pt"))
games_per_step = 10000*16
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100,200,300], gamma=1.2,verbose = True)
Xs = []
Ps = []
Vs = []
for j in tqdm(range(games_per_step)):
    with open(f"ds_{j}.pkl", "rb") as inp:
        result_step = pickle.load(inp)

    for i, res in enumerate(result_step):
        Xs.append(res[0])
        #print(i, res[0])
        Ps.append(res[1])
        Vs.append(res[2])

Xs = np.array(Xs, dtype=np.float32)
Ps = np.array(Ps, dtype=np.float32)
Vs = np.array(Vs, dtype=np.float32)

permutation = np.random.permutation(len(Xs))
Xs = Xs[permutation]
Ps = Ps[permutation]
Vs = Vs[permutation]

batch_size = 200
print(Xs.shape[0])

train(model, Xs, Ps, Vs, opt, batch_size, epochs=3)
