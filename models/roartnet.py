from typing import List
from itertools import combinations
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ResLayer(nn.Module):
    def __init__(self, dim_in:int, dim_out:int, bn:bool=False, ln:bool=False, dropout:float=0.):
        super().__init__()
        self.is_bn = bn
        self.is_ln = ln
        self.fc1 = nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        if ln:
            self.ln1 = nn.LayerNorm(dim_out)
        else:
            self.ln1 = lambda x: x
        self.fc2 = nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if ln:
            self.ln2 = nn.LayerNorm(dim_out)
        else:
            self.ln2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = self.fc1(x)
        if len(x.shape) > 3 or len(x.shape) < 2:
            raise ValueError("x.shape should be (B, N, D) or (N, D)")
        elif len(x.shape) == 3 and self.is_bn:
            x = x.permute(0, 2, 1)      # from (B, N, D) to (B, D, N)
            x = self.bn1(x)
            x = x.permute(0, 2, 1)      # from (B, D, N) to (B, N, D)
        elif len(x.shape) == 2 and self.is_bn:
            x = self.bn1(x)
        elif self.is_ln:
            x = self.ln1(x)
        else:
            x = self.bn1(x)             # actually self.bn1 is identity function
        x = F.relu(x)

        x = self.fc2(x)
        if len(x.shape) > 3 or len(x.shape) < 2:
            raise ValueError("x.shape should be (B, N, D) or (N, D)")
        elif len(x.shape) == 3 and self.is_bn:
            x = x.permute(0, 2, 1)      # from (B, N, D) to (B, D, N)
            x = self.bn2(x)
            x = x.permute(0, 2, 1)      # from (B, D, N) to (B, N, D)
        elif len(x.shape) == 2 and self.is_bn:
            x = self.bn2(x)
        elif self.is_ln:
            x = self.ln2(x)
        else:
            x = self.bn2(x)             # actually self.bn2 is identity function
        x = self.dropout(x + x_res)
        return x


def create_MLP(input_dim:int, hidden_dims:List[int], output_dim:int, 
               bn:bool, ln:bool, dropout:float) -> nn.Module:
    fcs = hidden_dims
    fcs.insert(0, input_dim)
    fcs.append(output_dim)
    MLP = nn.Sequential(
        *[ResLayer(fcs[i], fcs[i+1], bn=bn, ln=ln, dropout=dropout) 
          for i in range(len(fcs) - 1)]
    )
    return MLP


def create_shot_encoder(shot_hidden_dims:List[int], shot_feature_dim:int, 
                        shot_bn:bool, shot_ln:bool, shot_dropout:float) -> nn.Module:
    return create_MLP(input_dim=352, hidden_dims=shot_hidden_dims, output_dim=shot_feature_dim, 
                      bn=shot_bn, ln=shot_ln, dropout=shot_dropout)

def create_encoder(num_more:int, shot_feature_dim:int, has_rgb:bool, 
                   overall_hidden_dims:List[int], rot_bin_num:int, joint_num:int, 
                   overall_bn:bool, overall_ln:bool, overall_dropout:float) -> nn.Module:
    # input order: (coords, normals, shots(, rgb))
    overall_input_dim = len(list(combinations(np.arange(num_more + 2), 2))) * 4 + (num_more + 2) * shot_feature_dim + (3 * (num_more + 2) if has_rgb else 0)
    # output order: (J*tr, J*rot, J*afford, J*conf)
    overall_output_dim = (2 + rot_bin_num + 2 + 1) * joint_num
    
    return create_MLP(input_dim=overall_input_dim, hidden_dims=overall_hidden_dims, output_dim=overall_output_dim, 
                      bn=overall_bn, ln=overall_ln, dropout=overall_dropout)
