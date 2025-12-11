import os
import numpy as np
import torch
import torch.nn.functional as F
from Params import args
import random
import numpy as np
import scipy.sparse as sp
import torch 
import torch.nn as nn

def cal_loss_r(preds, labels, mask):
    loss = torch.sum(torch.square(preds - labels) * mask) / torch.sum(mask)
    return loss

def cal_metrics_r(preds, labels, mask):
    loss = np.sum(np.square(preds - labels) * mask) / np.sum(mask)
    sqLoss = np.sum(np.sum(np.square(preds - labels) * mask, axis=0), axis=0)
    absLoss = np.sum(np.sum(np.abs(preds - labels) * mask, axis=0), axis=0)
    tstNums = np.sum(np.sum(mask, axis=0), axis=0)
    posMask = mask * np.greater(labels, 0.5)
    apeLoss = np.sum(np.sum(np.abs(preds - labels) / (labels + 1e-8) * posMask, axis=0), axis=0)
    posNums = np.sum(np.sum(posMask, axis=0), axis=0)
    return loss, sqLoss, absLoss, tstNums, apeLoss, posNums

def cal_metrics_r_mask(preds, labels, mask, mask_sparsity):
    loss = np.sum(np.square(preds - labels) * mask) / np.sum(mask)
    sqLoss = np.sum(np.sum(np.square(preds - labels) * mask * mask_sparsity, axis=0), axis=0)
    absLoss = np.sum(np.sum(np.abs(preds - labels) * mask * mask_sparsity, axis=0), axis=0)
    tstNums = np.sum(np.sum(mask * mask_sparsity, axis=0), axis=0)
    posMask = mask * mask_sparsity * np.greater(labels, 0.5)
    apeLoss = np.sum(np.sum(np.abs(preds - labels) / (labels + 1e-8) * posMask, axis=0), axis=0)
    posNums = np.sum(np.sum(posMask, axis=0), axis=0)
    return loss, sqLoss, absLoss, tstNums, apeLoss, posNums

def Informax_loss(logits):
    """
    logits: (B, 2)
        logits[:, 0] = positive score
        logits[:, 1] = negative score
    """
    pos = logits[:, 0]
    neg = logits[:, 1]

    # maximize pos, minimize neg
    loss = - torch.log(torch.sigmoid(pos) + 1e-9).mean() \
           - torch.log(1 - torch.sigmoid(neg) + 1e-9).mean()

    return loss

def infoNCEloss(q, k):
    T = args.t
    q = q.expand_as(k)
    q = q.permute(0, 3, 4, 2, 1)
    k = k.permute(0, 3, 4, 2, 1)
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    pos_sim = torch.sum(torch.mul(q, k), dim=-1)
    neg_sim = torch.matmul(q, k.transpose(-1, -2))
    pos = torch.exp(torch.div(pos_sim, T))
    neg = torch.sum(torch.exp(torch.div(neg_sim, T)), dim=-1)
    denominator = neg + pos
    return torch.mean(-torch.log(torch.div(pos, denominator)))

def seed_torch(seed=523):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def makePrint(name, ep, reses):
    ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
    for metric in reses:
        val = reses[metric]
        ret += '%s = %.4f, ' % (metric, val)
    ret = ret[:-2] + '  '
    return ret

def create_net(n_inputs, n_outputs, n_layers = 0, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def graph_grad(adj_mx):
    """Fetch the graph gradient operator."""
    num_nodes = adj_mx.shape[0]

    num_edges = (adj_mx > 0.).sum()
    grad = torch.zeros(num_nodes, num_edges)
    e = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mx[i, j] == 0:
                continue

            grad[i, e] = 1.
            grad[j, e] = -1.
            e += 1
    return grad

def init_network_weights(net, std = 0.1):
    """
    Just for nn.Linear net.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def build_crime_adj():
    """
    4-neighbour grid per crime type.

    Nodes = (region, crime_type).
    Within each crime type, each region is connected to its
    4 spatial neighbours. No cross-type edges.
    """
    R, C, K = args.row, args.col, args.cateNum
    A = R * C
    N = A * K

    adj = np.zeros((N, N), dtype=np.float32)

    def idx(r, c, k):
        return k * A + r * C + c

    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    for k in range(K):
        for r in range(R):
            for c in range(C):
                u = idx(r, c, k)
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < R and 0 <= cc < C:
                        v = idx(rr, cc, k)
                        adj[u, v] = 1.0

    return adj