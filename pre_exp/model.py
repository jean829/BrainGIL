import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset

from conv import GNN_node, GNN_node_Virtualnode

nn_act = torch.nn.ReLU()
F_act = F.relu
class GraphEnvAug(torch.nn.Module):

    def __init__(self, args, num_tasks, num_layer = 5, emb_dim = 300, gnn_type = 'gin', drop_ratio = 0.5, gamma = 0.4, use_linear_predictor=False):
        '''
            num_tasks (int): number of labels to be predicted
        '''

        super(GraphEnvAug, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.gamma  = gamma
        self.gate = None

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split('-')[0]
        emb_dim_rat = emb_dim
        if 'virtual' in gnn_type: 
            rationale_gnn_node = GNN_node_Virtualnode(args, 2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node_Virtualnode(args, num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        else:
            rationale_gnn_node = GNN_node(2, emb_dim_rat, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
            self.graph_encoder = GNN_node(num_layer, emb_dim, JK = "last", drop_ratio = drop_ratio, residual = True, gnn_name = gnn_name)
        self.separator = separator(
            rationale_gnn_node=rationale_gnn_node, 
            gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim_rat, 2*emb_dim_rat), torch.nn.BatchNorm1d(2*emb_dim_rat), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim_rat, 1)),
            nn=None
            )
        rep_dim = emb_dim
        if use_linear_predictor:
            self.predictor = torch.nn.Linear(rep_dim, self.num_tasks)
        else:
            self.predictor = torch.nn.Sequential(torch.nn.Linear(rep_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), nn_act, torch.nn.Dropout(), torch.nn.Linear(2*emb_dim, self.num_tasks))

    
    def forward(self, batched_data):
        h_node = self.graph_encoder(batched_data, self.gate)
        h_r, h_env, r_node_num, env_node_num, gate = self.separator(batched_data, h_node)
        gate = gate.view(-1,90).mean(dim=0)
        self.gate = gate.detach().clone() 
        h_rep = (h_r.unsqueeze(1) + h_env.unsqueeze(0)).view(-1, self.emb_dim)
        pred_rem = self.predictor(h_r)
        pred_rep = self.predictor(h_rep)
        loss_reg =  torch.abs(r_node_num / (r_node_num + env_node_num) - self.gamma  * torch.ones_like(r_node_num)).mean()
        output = {'pred_rep': pred_rep, 'pred_rem': pred_rem, 'loss_reg':loss_reg}
        return output, h_r
    
    def eval_forward(self, batched_data):
        h_node = self.graph_encoder(batched_data, self.gate)
        h_r, _, _, _, _ = self.separator(batched_data, h_node)
        pred_rem = self.predictor(h_r)
        return pred_rem 




class separator(torch.nn.Module):
    def __init__(self, rationale_gnn_node, gate_nn, nn=None):
        super(separator, self).__init__()
        self.rationale_gnn_node = rationale_gnn_node
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.rationale_gnn_node)
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, batched_data, h_node, size=None):
        x = self.rationale_gnn_node(batched_data, gate=None)
        batch = batched_data.batch
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        h_node = self.nn(h_node) if self.nn is not None else h_node
        assert gate.dim() == h_node.dim() and gate.size(0) == h_node.size(0)
        gate = torch.sigmoid(gate)

        h_out = scatter_add(gate * h_node, batch, dim=0, dim_size=size)
        c_out = scatter_add((1 - gate) * h_node, batch, dim=0, dim_size=size)

        r_node_num = scatter_add(gate, batch, dim=0, dim_size=size)
        env_node_num = scatter_add((1 - gate), batch, dim=0, dim_size=size)

        return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8 , gate
        # return h_out, c_out, r_node_num + 1e-8 , env_node_num + 1e-8
        
        
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn.functional import cosine_similarity


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()

        self.features = args.in_dim
        self.hidden_dim = args.emb_dim
        self.num_layers = args.num_layer
        self.num_classes = args.num_tasks
        self.dropout = args.drop_ratio
        
        self.norms = nn.BatchNorm1d(self.hidden_dim)

        self.conv1 = GCNConv(self.features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)
        
        self.fc_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norms_fc = nn.BatchNorm1d(self.hidden_dim)
        self.fc_2 = nn.Linear(self.hidden_dim, self.num_classes)

    def fc_forward(self, x):
        x = F.relu((self.fc1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.norms(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for l, conv in enumerate(self.convs):
            if l != self.num_layers-2:
                x = F.relu(conv(x, edge_index))
                x = self.norms(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, edge_index)
        # print(x.shape)
        
        x1 = global_mean_pool(x, batch)
        # print(x.shape)
        # x = self.fc_forward(x)
        x1 = self.fc_forward(x1)

        return x1, x

    def edge_distribution_high(self, edge_index, repre):

        src = edge_index[0]
        dst = edge_index[1]
        
        feats_cos = cosine_similarity(repre[src], repre[dst], dim=1)

        return feats_cos

    def __repr__(self):
        return self.__class__.__name__

