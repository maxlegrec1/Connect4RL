import pickle

import numpy as np
import torch
from tqdm import tqdm

import wandb
from connect4 import Connect4
from model import connect_model
from self_play import train

# wandb.init(project="reinforcement")
model = connect_model()
model.load_state_dict(torch.load("model.pt"))
games_per_step = 100
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

Xs = []
Ps = []
Vs = []
for j in tqdm(range(games_per_step)):
    with open(f"ds_{j}.pkl", "rb") as inp:
        result_step = pickle.load(inp)

    for i, res in enumerate(result_step):
        Xs.append(res[0])
        print(i, res[0])
        Ps.append(res[1])
        Vs.append(res[2])
    break
Xs = np.array(Xs, dtype=np.float32)
Ps = np.array(Ps, dtype=np.float32)
Vs = np.array(Vs, dtype=np.float32)
b = Connect4(pos=Xs[-1])
b.check_for_win()
print(b.winner)
print(b)
with torch.no_grad():
    print(model(b.board))
print(Vs[-1], Xs[-1])
exit()
permutation = np.random.permutation(len(Xs))
Xs = Xs[permutation]
Ps = Ps[permutation]
Vs = Vs[permutation]

batch_size = 100


train(model, Xs, Ps, Vs, opt, batch_size, epochs=10)
