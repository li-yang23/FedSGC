"""
按fedsgc的逻辑，分两方，A首先计算S^K，B初始化权重W并计算XW，然后交给A
A再计算(S^K)XW，然后计算损失，再把损失传回给B
B最后计算梯度，更新权重
"""
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, sgc_precompute, set_seed, sgc_s_precompute
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

# setting random seeds
set_seed(args.seed, args.cuda)

adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)

# Bob 先初始化模型
model = get_model(args.model, features.size(1), labels.max().item()+1, args.hidden, args.dropout, args.cuda)

# Alice 先执行S^K的计算
if args.model == "SGC":
    adj, precompute_time = sgc_s_precompute(adj ,args.degree)
    print("{:.4f}s".format(precompute_time))

def train_regression(model,
                     features, labels,
                     idx_train, idx_val,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    t = perf_counter()

    for epoch in range(epochs):
        # 先过模型
        model.train()
        optimizer.zero_grad()

        # Bob 计算XW
        output = model(features)

        # Bob 将XW和Y发送给Alice，
        # Alice 计算 Y=(S^K)XW
        # print(adj.shape, output.shape)
        output = torch.spmm(adj, output)
        # Alice 计算损失
        loss = F.cross_entropy(output[idx_train], labels[idx_train])

        # Alice将损失发回给Bob，Bob计算梯度
        loss.backward()
        # Bob 更新权重
        optimizer.step()
    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features)
        output = torch.spmm(adj, output)
        acc_val = accuracy(output[idx_val], labels[idx_val])
    return model, acc_val, train_time

def test_regression(model, features, labels, idx_test):
    model.eval()
    return accuracy(torch.spmm(adj, model(features))[idx_test], labels[idx_test])

if args.model == "SGC":
    model, acc_val, train_time = train_regression(model, 
                                                  features, labels, 
                                                  idx_train, idx_val,
                                                  args.epochs, args.weight_decay, 
                                                  args.lr, args.dropout)
    acc_test = test_regression(model, features, labels, idx_test)

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time, precompute_time+train_time))