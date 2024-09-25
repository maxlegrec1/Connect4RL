import torch
from tqdm import tqdm
from model3 import connect_model
from self_play import calculate_policy,sample,softmax
from connect4 import Connect4
def battle(model1,model2,num_games = 1_000):
    model1_w = 0
    model1_d = 0
    model1_l = 0

    #model1 starts (model1 is blue)
    for game in tqdm(range(num_games)//2):
        board = Connect4()

        while board.winner == None:
            if board.to_move == 1:
                pred = model1(board.board)
            else:
                pred = model2(board.board)

            policy = pred["P"]
            policy_masked = softmax(policy + board.legal_moves_mask())
            move = sample(policy_masked)
            #print(board)
            board.move(move)
        if board.winner == 1:
            model1_w +=1
        elif board.winner == 0:
            model1_d += 1
        else:
            model1_l +=1
    #model2 starts
    for game in tqdm(range(num_games)//2):
        board = Connect4()

        while board.winner == None:
            if board.to_move == 1:
                pred = model2(board.board)
            else:
                pred = model1(board.board)

            policy = pred["P"]
            policy_masked = softmax(policy + board.legal_moves_mask())
            move = sample(policy_masked)
            #print(board)
            board.move(move)
        if board.winner == -1:
            model1_w +=1
        elif board.winner == 0:
            model1_d += 1
        else:
            model1_l +=1

    print(model1_w,model1_d,model1_l)
    return 
    #model2 starts (model2 is blue)

'''
model1 = connect_model()
model2 = connect_model()
model2.load_state_dict(torch.load("model.pt"))
with torch.no_grad():
    battle(model1,model2)
'''