from helper import get_args
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
from load_data22 import *
from torch.optim import lr_scheduler

## training
# from model import GraphEnvAug
from GCN import GCN
from train_eval_gcn import train_model, evaluate


if __name__=='__main__':
    args = get_args()
    # get_args_details(args)
    setup_seed(args.seed)
    
    acc = []
    sen = []
    spe = []
    f1 = []
    auc = []
    file = args.file
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    for k in range(args.num_exp):
        # myDataset = FSDataset(args.data_path, args.folds, args.data_seed[k])
        myDataset = FSDataset_leave_one(args.data_path, args.folds)
        acc_iter = []
        sen_iter = []
        spe_iter = []
        f1_iter = []
        auc_iter = []
        for i in range(args.folds):
            # train_dataset, val_dataset, test_dataset = myDataset.kfold_split(args.batch_size, i)
            train_dataset, val_dataset, test_dataset = myDataset.get_set(i)
            # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            valid_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
            """ Model init """
            # define optimizer and learning rate scheduler
            model = GCN(args).to(args.device)
            # model = GIN(args).to(args.device)
            optimizer = optim.Adam(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            criterion = torch.nn.NLLLoss()
            # criterion = torch.nn.MSELoss()
            scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100], gamma=0.5)
            best_model = train_model(args, model, criterion, optimizer, scheduler, train_loader, valid_loader, i)
            # model.load_state_dict(torch.load('{}/{}_fold_best_model.pth'.format(args.savepath, i)))
            model.load_state_dict(torch.load(os.path.join(args.savepath+'gcn_best_model.pth')))
            test_acc, test_loss, test_sen, test_spe, test_f1, test_auc, y, pred = evaluate(args, model, criterion, test_loader)
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

# nohup /home/weijiayin/OOD/GREA-main/main_asd22_more_for_one.py > a.txt 2>&1