import pickle
import os
import numpy as np
import torch
from tqdm import tqdm

import wandb
from connect4 import Connect4
from model3 import connect_model
from torch_model_clean import connect_model as connect_model_onnx
from battle import battle
import ray
import time
import onnxruntime
import random

def hashable(array):
    arr = array.flatten()
    return tuple(arr)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


@ray.remote
class Worker():
    def __init__(self,worker_id,step):
        self.session = onnxruntime.InferenceSession("onnx_model.onnx", providers=['CPUExecutionProvider'])
        self.worker_id = worker_id
        self.step = step
    def calculate_policy(self,board, num_rollouts,worker_id = None, return_dicts=False):
        Q_dict = {}
        N_dict = {}
        P_dict = {}
        V_dict = {}
        for i in range(num_rollouts):
            self.update(Q_dict, N_dict, P_dict, V_dict, board, main_path=True)
        policy = N_dict[hashable(board.board)] / np.sum(N_dict[hashable(board.board)])
        if worker_id == 0:
            print(Q_dict[hashable(board.board)])
            print(N_dict[hashable(board.board)])
        #policy = softmax(5 * Q_dict[hashable(board.board)] +  board.legal_moves_mask()  )
        if not return_dicts:
            return policy
        else:
            return policy, (Q_dict, N_dict, P_dict, V_dict)


    def update(self,Q, N, P, V, board, main_path=False):
        if hashable(board.board) not in Q:
            ort_inputs = {self.session.get_inputs()[0].name: board.board}
            pred = self.session.run(None, ort_inputs)
            P[hashable(board.board)] = pred[0]
            if board.winner is None:
                V[hashable(board.board)] = pred[1]
            else:
                V[hashable(board.board)] = -1
            Q[hashable(board.board)] = np.zeros(7)
            N[hashable(board.board)] = np.zeros(7)
            return -V[hashable(board.board)]
        U = Q[hashable(board.board)] + 3 * P[hashable(board.board)] * max(
            np.sqrt(np.sum(N[hashable(board.board)])), 1
        ) / (1 + N[hashable(board.board)])
        """if main_path:
            print(Q[hashable(board.board)])
            print(U)
            print(N[hashable(board.board)])
            print(P[hashable(board.board)])
            print()"""
        U = U + board.legal_moves_mask()

        a = np.argmax(U)
        if board.winner is None:
            s_prime = board.copy()
            try:
                s_prime.move(a)
            except:
                print("failed")
                print(s_prime)
                print(board.legal_moves_mask())
                print(U)
                exit()

            v = self.update(Q, N, P, V, s_prime)
        else:
            v = -1

        N[hashable(board.board)][a] += 1
        Q[hashable(board.board)][a] += (v - Q[hashable(board.board)][a]) / N[
            hashable(board.board)
        ][a]

        return -v
    @torch.no_grad()
    def perform_games(self,num_games,num_workers):
        j = 0
        t_0 = time.time()
        num_moves = 0
        num_rollouts_per_move = 50
        while j<= num_games:
            result_step = []
            board = Connect4()
            if random.random()<=0.9:
                board.initialize_random()
            if board.winner != None:
                continue
            print(self.worker_id,j)
            j+=1
            while True: 
                improved_policy = self.calculate_policy(
                    board, num_rollouts_per_move,self.worker_id
                )
                #improved_policy = temp_softmax(improved_policy)
                if(self.worker_id==0):
                    print(improved_policy)
                sampled_move = sample(improved_policy)
                previous_board = board.copy()
                board.move(sampled_move)
                num_moves+=1
                if(self.worker_id==0):
                    print(round((time.time()-t_0)/(num_workers*num_moves),2))
                    print(previous_board)
                result_step.append([previous_board.board, improved_policy, None])
                if board.winner != None:
                    break
            if board.winner == 0:
                result_step[-1][2] = 0
            else:
                result_step[-1][2] = 1
            # change the value of the truple to add the winner
            for i in range(len(result_step) - 2, -1, -1):
                result_step[i][2] = -result_step[i + 1][2]

            # concatenate result_step to result_dict                                        
            # save result_step as "ds_j.pkl"
            with open(f"games/ds_{self.step}_{self.worker_id*num_games+(j-1)}.pkl", "wb") as outp:
                pickle.dump(result_step, outp, pickle.HIGHEST_PROTOCOL)



def sample(policy):
    """Samples an action from the given policy.

    Args:
        policy: A list of probabilities for each action.

    Returns:
        The index of the sampled action.
    """
    return np.random.choice(len(policy), p=(policy))


def temp_softmax(policy):
    # Apply temperature
    new_pol = policy * policy
    return new_pol / np.sum(new_pol)


def test():
    # model is initialized with random weights
    model = connect_model()
    #model.load_state_dict(torch.load("model.pt"))
    # load_weights here

    num_steps = 1_000_000

    print(torch.cuda.is_available())


    batch_size = 50

    num_workers = 16
    games_per_step = 10000*num_workers

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    wandb.init(project="reinforcement")

    ray.init()
    '''
    torch.save(model.state_dict(), "model.pt")
    onnx_model = connect_model_onnx()
    onnx_model.load_state_dict(torch.load("model.pt"))
    torch_input = torch.randn((7,7),dtype=torch.float32)
    torch.onnx.export(onnx_model, torch_input,"onnx_model.onnx")'''
    for i in tqdm(range(num_steps)):

        
        workers = [Worker.remote(k,i) for k in range(num_workers) ]

        work_ref = [worker.perform_games.remote(games_per_step//num_workers,num_workers) for worker in workers]
        #compute
        _ = ray.get(work_ref)
        #kill actors
        _ = [ray.kill(worker) for worker in workers]
        # load dataset
        Xs = []
        Pis = []
        Vs = []
        Is = [i,i-1] if i > 0 else [i]
        for k in Is:
            for j in tqdm(range(games_per_step)):
                with open(f"games/ds_{k}_{j}.pkl", "rb") as inp:
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
        train(model, Xs, Pis, Vs, opt, batch_size,epochs=3)

        old_model = connect_model()
        old_model.load_state_dict(torch.load("old_model.pt"))
        with torch.no_grad():
            w,d,l = battle(model,old_model)
        wandb.log(
            {
                "win": w,
                "draw": d,
                "lose": l,
            }
        )




def train(model, Xs, Ps, Vs, opt, batch_size, epochs=1, scheduler = None):
    steps = Ps.shape[0] // batch_size

    model = model.to("cuda")
    for epoch in range(epochs):
        for step in range(steps):
            X = (
                torch.from_numpy(Xs[step * (batch_size) : (step + 1) * batch_size])
                .to(torch.float32)
                .to("cuda")
            )
            P = (
                torch.from_numpy(Ps[step * (batch_size) : (step + 1) * batch_size])
                .to(torch.float32)
                .to("cuda")
            )
            V = (
                torch.from_numpy(Vs[step * (batch_size) : (step + 1) * batch_size])
                .to(torch.float32)
                .to("cuda")
            )
            opt.zero_grad()
            entropy, l1, l2, l3, acc1, acc2, acc3, lm = model(X, targets=(P, V))
            #loss = l1 + l2 + l3
            loss = l2 + l3 + l1
            print(step, loss.item())
            wandb.log(
                {
                    "loss": loss.item(),
                    "l1": l1.item(),
                    "l2": l2.item(),
                    "l3": 0.1*l3.item(),
                    "acc1" : acc1.item(),
                    "acc2" : acc2.item(),
                    "acc3" : acc3.item(),
                    "entropy": entropy.item(),
                    "lm": lm.item(),
                    "target_entropy": torch.distributions.Categorical(probs=P)
                    .entropy()
                    .mean(),
                }
            )

            loss.backward()

            opt.step()
            if scheduler != None:
                scheduler.step()
    #rename last model.pt to old_model.pt
    os.rename("model.pt","old_model.pt")
    torch.save(model.state_dict(), "model.pt")
    model = model.to("cpu")
    onnx_model = connect_model_onnx()
    onnx_model.load_state_dict(torch.load("model.pt"))
    torch_input = torch.randn((7,7),dtype=torch.float32)
    torch.onnx.export(onnx_model, torch_input,"onnx_model.onnx")


if __name__ == "__main__":
    test()
