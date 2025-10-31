import numpy as np
import copy
import torch
import dgl
# from utils import set_seed
import time
from helper import *

"""
1. eval
"""
def evaluate(args, model, criterion, val_loader):
    """
    Returns:
    out: log probability of all input data
    loss & score (float): evaluated loss & score, if idx_eval is not None, only loss & score on those idx.
    """
    model.eval()
    Y_test = []
    Y_pred = []
    correct = 0.
    test_loss = 0.
    for data in val_loader:
        with torch.no_grad():
            data = data.to(args.device)
            logits,_ = model(data)
            out = logits.log_softmax(dim=1)
            pred = out.argmax(1)
            correct += pred.eq(data.y).float().sum().item()
            test_loss = criterion(out, data.y).item() * data.x.size(0) / args.num_nodes
            for num in range(len(out)):
                Y_pred.append(pred.cpu().numpy()[num])
                Y_test.append(data.y.cpu().numpy()[num])
    test_acc = correct / len(val_loader.dataset)
    test_loss = test_loss / len(val_loader.dataset)
    test_sen, test_spe, test_f1, test_auc = sensitivity_specificity(args, Y_test, Y_pred)
    return test_acc, test_loss, test_sen, test_spe,test_f1,test_auc,Y_test, Y_pred


def train_model(args, model, criterion, optimizer, scheduler, train_loader, val_loader, i_fold):
    max_acc = 0
    patience = 0
    best_epoch = 0
    best_loss = float('inf')
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            data = data.to(args.device)
            logits,_ = model(data)
            out = logits.log_softmax(dim=1)
            loss = criterion(out, data.y)
            # loss_s = criterion1(model.edge_distribution_high(data.edge_index, repre), data.edge_weight)  #param['tau']=1.5
            # loss_all = loss + loss_s
            optimizer.zero_grad()
            loss.backward()
            # loss_all.backward()
            optimizer.step()
            train_loss += loss.item() * data.x.size(0) / args.num_nodes
        scheduler.step()      
        test_acc, test_loss, _, _, _, _, _, _ = evaluate(args, model, criterion, val_loader)

        train_loss = train_loss / len(train_loader.dataset)
        print('Epoch: {:04d}'.format(epoch), 'train_loss: {:.6f}'.format(train_loss),
              'test_loss: {:.6f}'.format(test_loss), 'test_acc: {:.6f}'.format(test_acc),
              'time: {:.6f}s'.format(time.time() - t))
        
        if test_acc >= max_acc or epoch==0:
            max_acc = test_acc
            torch.save(model.state_dict(), '{}/{}_fold_best_model.pth'.format(args.savepath, i_fold))
            torch.save(model.state_dict(), os.path.join(args.savepath+'gcn_best_model.pth'))
            print("Model saved at epoch{}".format(epoch))
            best_epoch = epoch
            patience = 0
        # if train_loss <= best_loss:
        #     best_loss = train_loss
        #     torch.save(model.state_dict(), '{}/{}_fold_best_model.pth'.format(args.savepath, i_fold))
        #     print("Model saved at epoch{}".format(epoch))
        #     best_epoch = epoch
        #     patience = 0
        else:
            patience += 1
        
        if patience == args.patience:
            break

    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t0))
    return best_epoch
