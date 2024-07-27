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

b = Connect4()
with torch.no_grad():
    print(model(b.board))
b.move(6)

b.move(1)
with torch.no_grad():
    print(model(b.board))

b.move(6)
b.move(0)
with torch.no_grad():
    print(model(b.board))

b.move(6)
b.move(6)

print(b)
with torch.no_grad():
    print(model(b.board))

b.move(2)
b.move(3)
print(b)
with torch.no_grad():
    print(model(b.board))

b.move(2)
b.move(2)
print(b)
with torch.no_grad():
    print(model(b.board))

b.move(2)
b.move(3)
print(b)
with torch.no_grad():
    print(model(b.board))
b.move(3)
b.move(4)
print(b)
with torch.no_grad():
    print(model(b.board))

b.move(3)
b.move(4)
print(b)
with torch.no_grad():
    print(model(b.board))

b.move(3)
b.move(3)
print(b)
with torch.no_grad():
    print(model(b.board))

b.move(5)
b.move(4)
print(b)
with torch.no_grad():
    print(model(b.board))
b.move(5)

print(b)
with torch.no_grad():
    print(model(b.board))
