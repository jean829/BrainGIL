from helper import get_args, get_args_details
"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import torch.optim as optim
from torch.utils.data import DataLoader
# from load_data import *
from load_data import *

## training
from model import GraphEnvAug
from utils import init_weights
from train_eval import train_model


if __name__=='__main__':
    args = get_args()
    get_args_details(args)
    setup_seed(args.seed)    
    
    acc = []
    sen = []
    spe = []
    f1 = []
    auc = []
    file = args.file
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    myDataset = FSDataset(args.data_path, args.folds)
    for k in range(args.num_exp):
        acc_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []
        auc_iter = []
        for i in range(args.folds):
            # train_dataset, val_dataset, test_dataset = myDataset.kfold_split(args.batch_size, i)
            train_dataset, val_dataset, test_dataset = myDataset.get_set()
            # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            train_loader = DataLoader(train_dataset[i], batch_size=args.batch_size, shuffle=True)
            valid_loader = DataLoader(val_dataset[i], batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset[i], batch_size=args.batch_size, shuffle=False)
            
            """ Model init """
            model = GraphEnvAug(args, gnn_type = args.gnn, num_tasks = args.num_tasks, num_layer = args.num_layer,
                         emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device) 
                    
            # define optimizer and learning rate scheduler
            init_weights(model, args.initw_name, init_gain=0.02)
            opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
            opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters()), lr=args.lr, weight_decay=args.l2reg)
            optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
            if args.use_lr_scheduler:
                schedulers = {}
                for opt_name, opt in optimizers.items():
                    schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
            else:
                schedulers = None
            
            best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc = train_model(args,i, device, model, optimizers, schedulers, train_loader, valid_loader)
            test_acc, test_sen, test_spe, test_f1, test_auc = best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc
            
            acc_iter.append(test_acc*100)
            sen_iter.append(test_sen*100)
            spe_iter.append(test_spe*100)
            f1_iter.append(test_f1*100)
            auc_iter.append(test_auc*100)
            print('Test set results, accuracy = {:.6f}, sensitivity = {:.6f}, '
                  'specificity = {:.6f}, f1 = {:.6f}, auc = {:.6f}'.format( test_acc, test_sen,
                                                                           test_spe, test_f1, test_auc))
            save_each_fold(file, test_acc, test_sen, test_spe, test_f1, test_auc)
            # save_pred(file,list(np.array(y).reshape(-1)), pred)

        acc.append(np.mean(acc_iter))
        sen.append(np.mean(sen_iter))
        spe.append(np.mean(spe_iter))
        f1.append(np.mean(f1_iter))
        auc.append(np.mean(auc_iter))
        save_std(file, acc_iter, sen_iter, spe_iter, f1_iter, auc_iter)

    # print(args)
    # print(model)
    save_std(file, acc, sen, spe, f1, auc)
    save_args(file, args)

# nohup python /home/weijiayin/OOD/GREA-main/leave_one_site_com_exp/main.py > a.txt 2>&1