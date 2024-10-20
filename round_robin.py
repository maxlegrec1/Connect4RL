from battle import battle
import torch
from inference_model import connect_model
import numpy as np
models_path = ["random_trained.pt","model_0.pt","model_1.pt","model_2.pt","model_3.pt","model_4.pt","model_5.pt","model_6.pt","model_7.pt","model_8.pt","model_9.pt"]
l = len(models_path)
models = []

for model_path in models_path:
    model = connect_model()
    model.load_state_dict(torch.load(model_path))
    models.append(model)

results = np.zeros((l,l))
#start the round robin

with torch.no_grad():
    for i,modelA in enumerate(models):
        for j,modelB in enumerate(models):
            w,d,l = battle(modelA,modelB)
            results[i,j] = w/(w+l)

print(results)
