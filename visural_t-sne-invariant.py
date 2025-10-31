from helper import get_args, get_args_details
"""
    IMPORTING LIBS
"""
# torch_geometric     2.5.0
import dgl

import numpy as np
import os
import socket
import time
import random
# import glob
from glob import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator
import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset

from torch_geometric.data import Data  # DataLoader
from torch_geometric.loader import DataLoader

# from load_data import *
# from load_data2 import *
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data  # DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

## training
from leave_one_site_com_exp.model import GraphEnvAug
from utils import init_weights
# from train_eval import train_model
import scipy.io as io

def get_adj(connectivity,threshold):
    num_nodes = connectivity.shape[0]
    connectivity[np.isinf(connectivity)] = 1
    connectivity[np.isnan(connectivity)] = 0
    subj_fc_adj_up = connectivity[np.triu_indices(num_nodes, k=1)]  # 对角线及其以下都是0
    subj_fc_adj_list = subj_fc_adj_up.reshape((-1))
    thindex = int(threshold * subj_fc_adj_list.shape[0])
    thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex - 1]]
    adj = np.zeros((num_nodes, num_nodes))
    adj[connectivity > thremax] = 1
    return adj

def sensitivity_specificity(args, Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)
    if tp ==0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    return sensitivity, specificity, f1, auc

def gcn_evaluate(args, model,device, criterion, val_loader):
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    h_rr=[]
    predd=[]
    for data in val_loader:
        with torch.no_grad():
            data = data.to(device)
            logits,h_pre = model(data)
            h_rr.append(h_pre.tolist())
            predd.append(logits.tolist())
            out = logits.log_softmax(dim=1)
            pred = out.argmax(1)
            correct += pred.eq(data.y).float().sum().item()
            test_loss = criterion(out, data.y).item() * data.x.size(0) / 90
            for num in range(len(out)):
                Y_pred.append(pred.cpu().numpy()[num])
                Y_test.append(data.y.cpu().numpy()[num])
    test_acc = correct / len(val_loader.dataset)
    test_loss = test_loss / len(val_loader.dataset)
    test_sen, test_spe, test_f1, test_auc = sensitivity_specificity(args, Y_test, Y_pred)
    h_rr = [subitem for item in h_rr for subitem in item]
    predd = [subitem for item in predd for subitem in item]
    return h_rr,predd,test_acc,  test_sen, test_spe,test_f1,test_auc

def evaluate(args, model, device, loader):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    # for step, batch in enumerate(loader):
    h_rr=[]
    gate_M=[]
    predd=[]
    for data in loader:
        with torch.no_grad():
            data = data.to(args.device)
            
            # h_r, pred = model.eval_forward(data)
            # output=model.eval_forward(data)
            pred, gate = model.eval_forward(data)
            gate_M=gate.tolist()
            # h_rr.append(h_r.tolist())
            # predd.append(pred.tolist())
            prob = torch.sigmoid(pred)
            prob = torch.squeeze(prob)
            prob = (prob >= 0.5).float()
            correct += prob.eq(data.y).float().sum().item()
            for num in range(len(pred)):
                Y_pred.append(prob.cpu().numpy()[num])
                Y_test.append(data.y.cpu().numpy()[num])
    
    test_acc = correct / len(data)
    
    test_sen, test_spe, test_f1, test_auc = sensitivity_specificity(args, Y_test, Y_pred)
    # h_rr = [subitem for item in h_rr for subitem in item]
    # predd = [subitem for item in predd for subitem in item]
    # return h_rr,predd,test_acc, test_sen, test_spe, test_f1, test_auc
    return gate_M[:90],predd,test_acc, test_sen, test_spe, test_f1, test_auc


def max_min_norm(sub_region_series):
    subj_fc_mat_list = sub_region_series.reshape((-1))  # 246x246->60516x1
    subj_fc_feature = (sub_region_series - min(subj_fc_mat_list)) / (max(subj_fc_mat_list) - min(subj_fc_mat_list))
    return subj_fc_feature



if __name__=='__main__':
    args = get_args()
    
    acc = []
    sen = []
    spe = []
    f1 = []
    auc = []
    file = args.file
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    root_dir='/home/weijiayin/wwh_spatial_temporal/data/MDD_wwh_667'
    myDataset = FSDataset(root_dir)
    data=myDataset.fc
    label=myDataset.site
    data = DataLoader(data, batch_size=128, shuffle=False)
    # '''
    """ Model init invariant learn"""
    model = GraphEnvAug(args, gnn_type = args.gnn, num_tasks = args.num_tasks, num_layer = args.num_layer,
                    emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, gamma=args.gamma, use_linear_predictor = args.use_linear_predictor).to(device) 

    # define optimizer and learning rate scheduler
   
    model.load_state_dict(torch.load('/home/weijiayin/OOD/GREA-main/leave_one_site_com_exp/result/MDD_wwh_667/M_region/3_fold_best_model.pth'))

    # '''
    '''
    # GCN
    from pre_exp_GCN.GCN import GCN
    args.num_tasks=2
    args.num_layer=3
    model = GCN(args).to(device)
    # model.load_state_dict(torch.load('/home/weijiayin/OOD/GREA-main/result/ASD/ckptgcn_best_model.pth'))
    model.load_state_dict(torch.load('/home/weijiayin/OOD/GREA-main/pre_exp_GCN/ASD/ckpt/9_fold_best_model.pth'))
    '''
    
    # /home/weijiayin/OOD/GREA-main/result/ASD/ckpt/0_fold_best_model.pth
    
    # init_weights(model, args.initw_name, init_gain=0.02)
    # opt_separator = optim.Adam(model.separator.parameters(), lr=args.lr, weight_decay=args.l2reg)
    # opt_predictor = optim.Adam(list(model.graph_encoder.parameters())+list(model.predictor.parameters()), lr=args.lr, weight_decay=args.l2reg)
    # optimizers = {'separator': opt_separator, 'predictor': opt_predictor}
    # if args.use_lr_scheduler:
    #     schedulers = {}
    #     for opt_name, opt in optimizers.items():
    #         schedulers[opt_name] = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-4)
    # else:
    #     schedulers = None
    criterion = torch.nn.NLLLoss()
    # h_r,pred,best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc = gcn_evaluate(args, model, device,criterion, data)
    gate,pred,best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc = evaluate(args, model, device, data)
    data_array = np.array(gate)
    top_20_indices = np.argsort(data_array.flatten())[-90:][::-1]
    aal=[
    ['PreCG.L'],
    ['PreCG.R'],
    ['SFGdor.L'],
    ['SFGdor.R'],
    ['ORBsup.L'],
    ['ORBsup.R'],
    ['MFG.L'],
    ['MFG.R'],
    ['ORBmid.L'],
    ['ORBmid.R'],
    ['IFGoperc.L'],
    ['IFGoperc.R'],
    ['IFGtriang.L'],
    ['IFGtriang.R'],
    ['ORBinf.L'],
    ['ORBinf.R'],
    ['ROL.L'],
    ['ROL.R'],
    ['SMA.L'],
    ['SMA.R'],
    ['OLF.L'],
    ['OLF.R'],
    ['SFGmed.L'],
    ['SFGmed.R'],
    ['ORBsupmed.L'],
    ['ORBsupmed.R'],
    ['REC.L'],
    ['REC.R'],
    ['INS.L'],
    ['INS.R'],
    ['ACG.L'],
    ['ACG.R'],
    ['DCG.L'],
    ['DCG.R'],
    ['PCG.L'],
    ['PCG.R'],
    ['HIP.L'],
    ['HIP.R'],
    ['PHG.L'],
    ['PHG.R'],
    ['AMYG.L'],
    ['AMYG.R'],
    ['CAL.L'],
    ['CAL.R'],
    ['CUN.L'],
    ['CUN.R'],
    ['LING.L'],
    ['LING.R'],
    ['SOG.L'],
    ['SOG.R'],
    ['MOG.L'],
    ['MOG.R'],
    ['IOG.L'],
    ['IOG.R'],
    ['FFG.L'],
    ['FFG.R'],
    ['PoCG.L'],
    ['PoCG.R'],
    ['SPG.L'],
    ['SPG.R'],
    ['IPL.L'],
    ['IPL.R'],
    ['SMG.L'],
    ['SMG.R'],
    ['ANG.L'],
    ['ANG.R'],
    ['PCUN.L'],
    ['PCUN.R'],
    ['PCL.L'],
    ['PCL.R'],
    ['CAU.L'],
    ['CAU.R'],
    ['PUT.L'],
    ['PUT.R'],
    ['PAL.L'],
    ['PAL.R'],
    ['THA.L'],
    ['THA.R'],
    ['HES.L'],
    ['HES.R'],
    ['STG.L'],
    ['STG.R'],
    ['TPOsup.L'],
    ['TPOsup.R'],
    ['MTG.L'],
    ['MTG.R'],
    ['TPOmid.L'],
    ['TPOmid.R'],
    ['ITG.L'],
    ['ITG.R']
]

    # 打印出前十个最大值的位置和对应的值
    for idx in top_20_indices:
        print(f"Index: {idx}, AAL:{aal[idx]}, Value: {data_array[idx][0]}")
    print(gate)
    # 'Yale' 'UM' 'USM' 'Leuven'
    # gates = (gate > 0.5).to(gate)
    '''
    # Yale_rep = [h_r[i] for i in range(len(label)) if label[i] == 'Yale']
    test_UM_rep = [h_r[i] for i in range(len(label)) if label[i] == 'UM']
    # USM_rep = [h_r[i] for i in range(len(label)) if label[i] == 'USM']
    # Leuven_rep = [h_r[i] for i in range(len(label)) if label[i] == 'Leuven']
    train_rep = [h_r[i] for i in range(len(label)) if label[i] != 'UM']
    # Yale_rep, UM_rep, USM_rep, Leuven_rep =np.array(Yale_rep), np.array(UM_rep), np.array(USM_rep), np.array(Leuven_rep)
    test_UM_rep, train_rep =np.array(test_UM_rep), np.array(train_rep)
    # np.save('inv_Yale.npy', Yale_rep)
    # np.save('inv_UM.npy', UM_rep)
    # np.save('inv_USM.npy', USM_rep)
    # np.save('inv_Leuven.npy', Leuven_rep)
    
    # np.save('inv_UM_test_leave_one.npy', test_UM_rep)
    # np.save('inv_train_leave_oneUM.npy', train_rep)
    
    np.save('/home/weijiayin/OOD_distribution_visual/INV/inv_UM_test_leave_one.npy', test_UM_rep)
    np.save('/home/weijiayin/OOD_distribution_visual/INV/inv_train_leave_oneUM.npy', train_rep)
    # np.save('inv_UM_test_leave_one_weightfrommixtrain.npy', test_UM_rep)
    # np.save('inv_train_leave_oneUM_weightfrommixtrain.npy', train_rep)
    
    # X = np.vstack([Yale_rep, UM_rep, USM_rep, Leuven_rep])
    # X = np.vstack([UM_rep, USM_rep, Leuven_rep])
    X = np.vstack([test_UM_rep, train_rep])
    
    # labels = np.array([0]*len(Yale_rep) + [1]*len(UM_rep) + [2]*len(USM_rep) + [3]*len(Leuven_rep))
    labels = np.array([0]*len(test_UM_rep) + [1]*len(train_rep))
    # 创建 t-SNE 实例
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    # 对数据进行降维
    X_embedded = tsne.fit_transform(X)

    # 可视化
    plt.figure(figsize=(10, 6))
    # site_names = ['Yale', 'UM', 'USM', 'Leuven']
    # site_names = ['UM', 'USM', 'Leuven']
    site_names=['test', 'train']
    for label in np.unique(labels):
        # plt.scatter(X_embedded[labels == label, 0], X_embedded[labels == label, 1], label=f'Site {label + 1}')
        plt.scatter(X_embedded[labels == label, 0], X_embedded[labels == label, 1], label=site_names[label])
    plt.legend()
    plt.show()
    # plt.savefig('Invariant_leave_one_UM_tsne_plot.png')
    # plt.savefig('inv_leave_one_UM_weightfrommixtrain_tsne_plot.png')
    # plt.savefig('general_gcn_tsne_plot.png')
    # plt.savefig('/home/weijiayin/OOD_distribution_visual/GCN/leave_one_gcn_tsne_plot.png')
    plt.savefig('/home/weijiayin/OOD_distribution_visual/INV/leave_one_inv_tsne_plot.png')
    '''