import numpy as np
import copy
import torch
import dgl
# from utils import set_seed
import time
from helper import *

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
               
# 分类损失和回归损失 
cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()             
                
def train(args, model, device, loader, optimizers, optimizer_name):
    optimizer = optimizers[optimizer_name]
    model.train()
    # if optimizer_name == 'predictor':
    #     set_requires_grad([model.graph_encoder, model.predictor], requires_grad=True)
    #     set_requires_grad([model.separator], requires_grad=False)
    # if optimizer_name == 'separator':
    #     set_requires_grad([model.separator], requires_grad=True)
    #     set_requires_grad([model.graph_encoder,model.predictor], requires_grad=False)
        
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred, h_r = model(batch)
            criterion = cls_criterion
            criterion1 = torch.nn.BCEWithLogitsLoss(reduction='none')

            target = batch.y.to(torch.float32).view(-1,1)
            
            
            # logits级别的拉近
            # pred_mdd = torch.mul(pred['pred_rem'], target)
            # pred_hc = torch.mul(pred['pred_rem'], (1-target))
            # pred_mdd_mean = pred_mdd[pred_mdd.sum(dim=1).nonzero().squeeze(),:].mean(dim=0, keepdim=True)
            # pred_hc_mean = pred_hc[pred_hc.sum(dim=1).nonzero().squeeze(),:].mean(dim=0, keepdim=True)
            # distances_mdd = torch.cdist(pred_mdd, pred_mdd_mean)
            # distances_hc = torch.cdist(pred_hc, pred_hc_mean)
            # Loss_distance1 = torch.exp(torch.log(torch.mean(distances_mdd)) - torch.log(torch.mean(torch.cdist(pred_mdd, pred_hc_mean))))
            # Loss_distance2 = torch.exp(torch.log(torch.mean(distances_hc)) - torch.log(torch.mean(torch.cdist(pred_hc, pred_mdd_mean))))
            
            is_labeled = batch.y == batch.y
            loss = args.rem * criterion(pred['pred_rem'].to(torch.float32)[is_labeled], target[is_labeled]) 
            # loss = args.rem*criterion(pred['pred_rem'].to(torch.float32)[is_labeled], target[is_labeled]) 
            target_rep = batch.y.to(torch.float32).repeat_interleave(batch.batch[-1]+1,dim=0)
            is_labeled_rep = target_rep == target_rep
            
            # 非因果扰动
            loss += args.rep * criterion(pred['pred_rep'].to(torch.float32)[is_labeled_rep], target_rep[is_labeled_rep].view(-1,1))
          
            # # V-Rex
            # 消融
            losses_1 = criterion1(pred['pred_rep'].to(torch.float32)[is_labeled_rep], target_rep[is_labeled_rep].view(-1,1))
            tmp_loss = 0.0
            tmp_sequence_lenth = target.size(0)
            for i in range(0, tmp_sequence_lenth):
                # 获取 losses_2 中的第 target[is_labeled] 个元素
                loss2_elements = losses_1[tmp_sequence_lenth*i:tmp_sequence_lenth*(i+1)]
                # 拼接
                mean_value = torch.mean(loss2_elements)
                tmp_loss += torch.mean(torch.pow(loss2_elements - mean_value, 2))
            
            # 纯gnn-消融
            if optimizer_name == 'separator': 
                loss += pred['loss_reg']
            
            # print(Loss_distance1)
            # print(Loss_distance2)
            # loss+=(Loss_distance1+Loss_distance2)*args.logits_class_loss_weight
            
            # loss+=tmp_loss*0.06
            loss += tmp_loss*args.vrex
            
            loss.backward()
            if args.use_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
"""
1. eval
"""
def evaluate(args, model, device, val_loader):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    for step, data in enumerate(val_loader):
        with torch.no_grad():
            data = data.to(args.device)
            
            pred = model.eval_forward(data)
            
            prob = torch.sigmoid(pred)
            prob = torch.squeeze(prob)
            prob = (prob >= 0.5).float()
            correct += prob.eq(data.y).float().sum().item()
            for num in range(len(pred)):
                Y_pred.append(prob.cpu().numpy()[num])
                Y_test.append(data.y.cpu().numpy()[num])
    
    test_acc = correct / len(val_loader.dataset)
    
    test_sen, test_spe, test_f1, test_auc = sensitivity_specificity(args, Y_test, Y_pred)
    return test_acc, test_sen, test_spe, test_f1, test_auc, Y_test, Y_pred

def train_model(args, device, model, optimizers, schedulers, train_loader, val_loader):
    best_epoch = 0
    cnt_wait = 0
    best_valid_perf=0.0
    best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for epoch in range(args.epochs):
        print("=====Epoch {}".format(epoch))
        path = epoch % int(args.path_list[-1])
        if path in list(range(int(args.path_list[0]))):
            optimizer_name = 'separator' 
        elif path in list(range(int(args.path_list[0]), int(args.path_list[1]))):
            optimizer_name = 'predictor'
        
            
        train(args, model, device, train_loader, optimizers, optimizer_name)

        if schedulers != None:
            schedulers[optimizer_name].step()
        train_acc, train_sen, train_spe, train_f1, train_auc, _, _ = evaluate(args, model, device, train_loader)
        val_acc, val_sen, val_spe, val_f1, val_auc, _, _ = evaluate(args, model, device, val_loader)
        
        if val_acc >  best_valid_perf or epoch == 0:
            best_valid_perf = val_acc
            cnt_wait = 0
            best_epoch = epoch
            best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc = val_acc, val_sen, val_spe, val_f1, val_auc
            print({'Metric': 'AUC', 'Train': train_acc, 'Validation': val_acc, 'Test': val_auc})
            torch.save(model.state_dict(), os.path.join(args.savepath+'mixtrain'+'.pth'))
        else:
            print({'Train': train_acc, 'Validation': val_acc})
            cnt_wait += 1
            if cnt_wait > args.patience:
                break
    print('Finished training! Results from epoch {} with best validation {}.'.format(best_epoch, best_valid_perf))
    
    return best_val_acc, best_val_sen, best_val_spe, best_val_f1, best_val_auc
