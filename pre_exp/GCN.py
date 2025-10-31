       
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
        # x=self.conv1(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        # x = self.norms(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for l, conv in enumerate(self.convs):
            if l != self.num_layers-2:
                # x = conv(x, edge_index)
                x = F.relu(conv(x, edge_index))
                # x = self.norms(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, edge_index)
        # print(x.shape)
        
        x = global_add_pool(x, batch)
        # print(x.shape)
        # x = self.fc_forward(x)
        # x1 = self.fc_forward(x1)
        x1 = self.fc(x)

        return x1, x

    def edge_distribution_high(self, edge_index, repre):

        src = edge_index[0]
        dst = edge_index[1]
        
        feats_cos = cosine_similarity(repre[src], repre[dst], dim=1)

        return feats_cos

    def __repr__(self):
        return self.__class__.__name__

