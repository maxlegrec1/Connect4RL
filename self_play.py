import pickle

import numpy as np
import torch
from tqdm import tqdm
import time
import wandb
from connect4 import Connect4
from model import connect_model
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def hashable(array):
    arr = array.flatten()
    return tuple(arr)


def calculate_policy(board, num_rollouts, model, return_dicts=False):
    Q_dict = {}
    N_dict = {}
    P_dict = {}
    V_dict = {}
    for i in range(num_rollouts):
        update(Q_dict, N_dict, P_dict, V_dict, board, model, main_path=True)
    policy = N_dict[hashable(board.board)] / np.sum(N_dict[hashable(board.board)])
    if not return_dicts:
        return policy
    else:
        return policy, (Q_dict, N_dict, P_dict, V_dict)


def update(Q, N, P, V, board, model, main_path=False):
    if hashable(board.board) not in Q:
        #pred = model(board.board)
        #P[hashable(board.board)] = pred["P"]
        P[hashable(board.board)] = softmax(np.random.randn(7))
        # print(type(pred["P"]))
        if board.winner is None:
            #V[hashable(board.board)] = pred["V"]
            V[hashable(board.board)] = np.random.randn()
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

        v = update(Q, N, P, V, s_prime, model)
    else:
        v = -1

    N[hashable(board.board)][a] += 1
    Q[hashable(board.board)][a] += (v - Q[hashable(board.board)][a]) / N[
        hashable(board.board)
    ][a]

    return -v


def sample(policy):
    """Samples an action from the given policy.

    Args:
        policy: A list of probabilities for each action.

    Returns:
        The index of the sampled action.
    """
    return np.random.choice(len(policy), p=(policy))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def temp_softmax(policy):
    # Apply temperature
    new_pol = policy * policy
    return new_pol / np.sum(new_pol)


def test():
    # model is initialized with random weights
    model = connect_model()
    #model.load_state_dict(torch.load("model.pt"))
    # load_weights here

    num_steps = 10_000

    games_per_step = 100

    num_rollouts_per_move = 1000

    batch_size = 50

    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    wandb.init(project="reinforcement")


    Xs = []
    Pis = []
    Vs = []
    for j in tqdm(range(games_per_step)):
        with open(f"ds_{j}.pkl", "rb") as inp:
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
    train(model, Xs, Pis, Vs, opt, batch_size,epochs=2)



    for i in tqdm(range(num_steps)):
        j = 0

        while j<= games_per_step:
            result_step = []
            board = Connect4()
            board.initialize_random()
            if board.winner != None:
                continue
            j+=1
            while True: 
                with torch.no_grad():
                    improved_policy = calculate_policy(
                        board, num_rollouts_per_move, model
                    )
                    improved_policy = temp_softmax(improved_policy)
                    print(improved_policy)
                    sampled_move = sample(improved_policy)
                    previous_board = board.copy()
                    board.move(sampled_move)
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
            with open(f"ds_{j}.pkl", "wb") as outp:
                pickle.dump(result_step, outp, pickle.HIGHEST_PROTOCOL)

        # load dataset
        Xs = []
        Pis = []
        Vs = []
        for j in tqdm(range(games_per_step)):
            with open(f"ds_{j}.pkl", "rb") as inp:
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
        train(model, Xs, Pis, Vs, opt, batch_size,epochs=2)
        # for (X,pi,v) in ds:
        # perform gradient descent on X, pi v

        # perform backpropagation on the result_dict dataset


def train(model, Xs, Ps, Vs, opt, batch_size, epochs=1):
    steps = Ps.shape[0] // batch_size

    model = model.to("cuda:0")
    for epoch in range(epochs):
        for step in range(steps):
            X = (
                torch.from_numpy(Xs[step * (batch_size) : (step + 1) * batch_size])
                .to(torch.float32)
                .to("cuda:0")
            )
            P = (
                torch.from_numpy(Ps[step * (batch_size) : (step + 1) * batch_size])
                .to(torch.float32)
                .to("cuda:0")
            )
            V = (
                torch.from_numpy(Vs[step * (batch_size) : (step + 1) * batch_size])
                .to(torch.float32)
                .to("cuda:0")
            )
            opt.zero_grad()
            entropy, l1, l2, l3, acc1, acc2, acc3 = model(X, targets=(P, V))
            loss = l1 + l2 + l3
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
                    "target_entropy": torch.distributions.Categorical(probs=P)
                    .entropy()
                    .mean(),
                }
            )
            writer.add_scalar("Accuracy", acc1.item(), epoch*steps + step)
            #time.sleep(0.2)

            loss.backward()

            opt.step()
    torch.save(model.state_dict(), "model.pt")
    model.to("cpu")


if __name__ == "__main__":
    test()
