# create transformer model with two heads (value and policy)
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import wandb
import math

def create_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.view(-1,max_len,d_model)

class SmallTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(SmallTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embed = nn.Linear(3, d_model//2-1)
        self.pos_encoding = create_positional_encoding(49,d_model//2)
        self.encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,dropout=0
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layers, num_layers=num_layers
        )
        self.P_head = nn.Linear((self.d_model) * 49, 7)
        self.V_head = nn.Linear((self.d_model) * 49, 3)
        self.mlp_head = nn.Linear((self.d_model)* 49,1)
        self.l1_loss = torch.nn.CrossEntropyLoss()
        self.l2_loss = torch.nn.CrossEntropyLoss()
        self.l3_loss = torch.nn.MSELoss()



    def create_planes(self,src):
        src_pos = torch.gt(src,0.5).float()
        src_neg = torch.gt(-src,0.5).float()
        src_zero = torch.eq(src,0).float()
        return torch.cat((src_pos,src_neg,src_zero),dim=-1)


    def forward(self, src, targets=None, compute_mid = False):

        if compute_mid:
            src = src.view(-1, 49, 1)
            src = self.create_planes(src)
            #print(src)
            src = self.embed(src)
            self.pos_encoding = self.pos_encoding.to(src.device).expand(src.shape[0],-1,-1)
            src = torch.cat((src,self.pos_encoding),dim = -1)
            src = self.transformer_encoder(src)
            src = src.view(-1, self.d_model * 49)
            return src

        if targets is None:
            src = torch.from_numpy(src).to(torch.float32)
            side_tgt = src.view(-1,49)
            side_tgt = torch.sum(side_tgt,dim = 1).view(-1,1)
            side_tgt = -2*side_tgt -1
            tgt_embed = side_tgt.view(-1,1,1).expand(1,49,1)
            src = src.view(-1, 49, 1)
            src = self.create_planes(src)
            src = self.embed(src)
            self.pos_encoding = self.pos_encoding.to(src.device)[0:1,:,:]
            src = torch.cat((src,self.pos_encoding,tgt_embed),dim = -1)
            src = self.transformer_encoder(src)
            src = src.view(-1, (self.d_model) * 49)
            P = self.P_head(src)
            V = self.V_head(src)
            P = torch.nn.functional.softmax(P, dim=-1)
            V = torch.nn.functional.softmax(V, dim=-1)
            #print(V)
            #V = 2*V[:,2]+ V[:,1] -1
            V = V[:,2] - V[:,0] 
            return {"P": P.view(7).numpy(), "V": V.view(-1).numpy()}

        else:
            last_row_is_zero = (src[:,:,-1] == 0).float()
            side_tgt = src.view(-1,49)
            side_tgt = torch.sum(side_tgt,dim = 1).view(-1,1)
            side_tgt = -2*side_tgt -1
            tgt_embed = side_tgt.view(-1,1,1).expand(src.shape[0],49,1)
            src = src.view(-1, 49, 1)
            src = self.create_planes(src)
            src = self.embed(src)
            #move pos_encoding to the right device
            self.pos_encoding = self.pos_encoding.to(src.device).expand(src.shape[0],-1,-1)
            src = torch.cat((src,self.pos_encoding,tgt_embed),dim = -1)
            src = self.transformer_encoder(src)
            src = src.view(-1, (self.d_model) * 49)
            #print(src.shape)
            side = self.mlp_head(src)
            side = nn.functional.tanh(side)
            P = self.P_head(src).view(-1, 7)
            V = self.V_head(src).view(-1, 3)
            # calculate entropy
            entropy = (
                Categorical(probs=torch.nn.functional.softmax(P, dim=-1))
                .entropy()
                .mean()
            )

            # P = torch.nn.functional.softmax(P, dim=-1)
            P_t, V_t = targets
            ldw = (V_t + 1).to(torch.long)
            l1 = self.l1_loss(P, P_t)
            l2 = self.l2_loss(V.view(-1, 3), ldw.view(-1))
            #print(side.shape,side_tgt.shape)
            l3 = self.l3_loss(side,side_tgt)

            acc1 = (torch.argmax(P,dim = -1) == torch.argmax(P_t,dim=-1)).float().mean()
            acc2 = (torch.argmax(V.view(-1, 3),dim = -1) == ldw.view(-1)).float().mean()
            acc3 = (torch.sign(side) == side_tgt).float().mean()
            #lm = (torch.gather(last_rows,dim = 1,index = torch.argmax(P,dim = -1).view(-1,1)) == 0).float().mean()
            lm = torch.sum((torch.nn.functional.softmax(P, dim=-1) * last_row_is_zero),dim = -1).mean()
            return entropy, l1 , l2, l3 , acc1 , acc2, acc3, lm


def connect_model():

    d_model = 128  # Dimensionality of input
    nhead = 4  # Number of heads in the transformer model
    num_layers = 6  # Number of transformer layers
    model = SmallTransformerEncoder(d_model, nhead, num_layers)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    return model
#connect_model()

class Parentmodel(torch.nn.Module):
    
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super(Parentmodel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embed = nn.Linear(3, d_model//2)
        self.pos_encoding = create_positional_encoding(49,d_model//2)
        self.encoder_layers = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,dropout=0
        )
        self.transformer_encoder = TransformerEncoder(
            self.encoder_layers, num_layers=num_layers
        )

        self.mlp_head = nn.Linear(self.d_model * 49,1)


    def create_planes(self,src):

        src_pos = torch.gt(src,0.5).float()
        src_neg = torch.gt(-src,0.5).float()
        src_zero = torch.eq(src,0).float()
        return torch.cat((src_pos,src_neg,src_zero),dim=-1)


    def forward(self, src):

        src = src.view(-1, 49, 1)
        src = self.create_planes(src)
        #print(src)
        src = self.embed(src)
        self.pos_encoding = self.pos_encoding.to(src.device).expand(src.shape[0],-1,-1)

        src = torch.cat((src,self.pos_encoding),dim = -1)
        src = self.transformer_encoder(src)
        src = src.view(-1, self.d_model * 49)
        src = self.mlp_head(src)
        src = nn.functional.tanh(src)
        return src

    
    
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.mlp = nn.Linear(49*3,1)

    def create_planes(self,src):
        src_pos = torch.gt(src,0.5).float()
        src_neg = torch.gt(-src,0.5).float()
        src_zero = torch.eq(src,0).float()
        return torch.cat((src_pos,src_neg,src_zero),dim=-1)
    
    def forward(self,src):
        src = src.view(-1, 49,1)
        src = self.create_planes(src).view(-1,49*3)
        src = self.mlp(src) 
        src = nn.functional.tanh(src)
        #print(src)
        #src = self.bn(src)
        return src
    

    
class Perfect(torch.nn.Module):
    def __init__(self):
        super(Perfect,self).__init__()


    def forward(self,src):
        src = src.view(-1,49)
        src = torch.sum(src,dim = 1)
        src = -2*src -1
        #print(src.shape)
        return src
