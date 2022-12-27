import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import math
import ipdb

class singerNetLayer_R(nn.Module):

    def __init__(self, n_nodetype, n_head, input_dims, output_dim, edge_types, n_selfring):
        super(singerNetLayer_R, self).__init__()
        assert float(output_dim) / n_head == output_dim // n_head
        # w: from_nodetype * to_nodetype * head
        # self.W = nn.Parameter(torch.zeros(n_nodetype, n_nodetype, n_head, input_dim, output_dim // n_head))
        if isinstance(input_dims, int):
            self.W = nn.Parameter(torch.zeros(n_nodetype, n_nodetype, n_head, input_dims, output_dim // n_head))
            torch.nn.init.xavier_uniform_(self.W)
            # stdv = 1./math.sqrt(self.W.size(4))
            # self.W.data.uniform_(-stdv, stdv)
        else:
            # node type weights
            self.W = nn.ParameterList([nn.Parameter(torch.zeros(n_nodetype, n_head, input_dims[i], output_dim // n_head)) for i in range(n_nodetype)])
            for w in self.W:
                torch.nn.init.xavier_uniform_(w)
                # stdv = 1./math.sqrt(w.size(3))
                # w.data.uniform_(-stdv, stdv)

        # edge type attention weights
        self.attn = nn.Parameter(torch.zeros(n_head, output_dim // n_head, len(edge_types)))
        torch.nn.init.xavier_uniform_(self.attn)

        # stdv = 1./math.sqrt(self.attn.size(1))
        # self.attn.data.uniform_(-stdv, stdv)
        self.n_nodetype = n_nodetype
        self.n_head = n_head
        self.n_edgetype = len(edge_types)
        self.edge_types = edge_types
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.onehead_dim = output_dim // n_head
        self.n_selfring = n_selfring

    def forward(self, embeddings, adjs):
        # embeddings: n_nodetype * n_node * input_dim
        # adjs: n_edgetype * n_from_node * n_to_node
        assert len(embeddings) == self.n_nodetype
        assert len(adjs) == self.n_edgetype

        # attentions = [[torch.zeros_like(a[0]).cuda().unsqueeze(0).repeat(self.n_head,1,1) for a in b] for b in adjs]
        n_nodes = [embeddings[i].size(0) for i in range(len(embeddings))]
        attention_dst_sum = [torch.zeros(self.n_head, n).cuda() for n in n_nodes] # destination attention
        ret = [torch.zeros(self.n_head, t, self.onehead_dim).cuda() for t in n_nodes]

        # for to_type in range(self.n_nodetype):
        for i in range(self.n_edgetype):
            from_type, to_type = self.edge_types[i] #

            from_embedding = embeddings[from_type]
            n_from_node = from_embedding.size(0)
            from_projected = from_embedding.unsqueeze(0).repeat(self.n_head,1,1).bmm(self.W[from_type][to_type])
            to_embedding = embeddings[to_type]
            n_to_node = to_embedding.size(0)
            to_self_projected = to_embedding.unsqueeze(0).repeat(self.n_head,1,1).bmm(self.W[to_type][to_type])

            # for from_type in range(self.n_nodetype):
            adj = adjs[i].unsqueeze(0).repeat(self.n_head,1,1)

            # attn_o = torch.cat([from_projected.unsqueeze(2).repeat(1,1,n_to_node,1), to_self_projected.unsqueeze(1).repeat(1,n_from_node,1,1)], dim=3).reshape(self.n_head, n_from_node*n_to_node, -1).bmm(self.attn[:,:,i:i+1]).reshape(self.n_head, n_from_node, n_to_node)
            attn_o = from_projected.bmm((to_self_projected + self.attn[:,:,i].unsqueeze(1)).permute(0,2,1))
            attn_o = attn_o.where(adj > 0, -9e15*torch.ones_like(attn_o).cuda())
            # attn_o = attn_o.where(adj > 0, -9e15*torch.ones_like(attn_o))
            attn_o = (attn_o - attn_o.max()).exp()

            attention_dst_sum[to_type] = attention_dst_sum[to_type] + attn_o.sum(dim=1)
            # print(ret[to_type].size(), attentions.size(), from_projected.size())
            ret[to_type] = ret[to_type] + attn_o.permute(0,2,1).bmm(from_projected)

        for i in range(self.n_selfring, self.n_edgetype): # range(3,5) meta-path
            to_type, from_type = self.edge_types[i] # add self loop

            from_embedding = embeddings[from_type]
            n_from_node = from_embedding.size(0)
            from_projected = from_embedding.unsqueeze(0).repeat(self.n_head,1,1).bmm(self.W[from_type][to_type])
            to_embedding = embeddings[to_type]
            n_to_node = to_embedding.size(0)
            to_self_projected = to_embedding.unsqueeze(0).repeat(self.n_head,1,1).bmm(self.W[to_type][to_type])

            # for from_type in range(self.n_nodetype):
            adj = adjs[i].unsqueeze(0).repeat(self.n_head,1,1) # repeat n_head matrix

            # attn_o = torch.cat([from_projected.unsqueeze(2).repeat(1,1,n_to_node,1), to_self_projected.unsqueeze(1).repeat(1,n_from_node,1,1)], dim=3).reshape(self.n_head, n_from_node*n_to_node, -1).bmm(self.attn[:,:,i:i+1]).reshape(self.n_head, n_from_node, n_to_node)
            attn_o = from_projected.bmm((to_self_projected - self.attn[:,:,i].unsqueeze(1)).permute(0,2,1))
            attn_o = attn_o.where(adj.permute(0,2,1) > 0, -9e15*torch.ones_like(attn_o).cuda())
            # attn_o = attn_o.where(adj > 0, -9e15*torch.ones_like(attn_o))
            attn_o = (attn_o - attn_o.max()).exp()

            attention_dst_sum[to_type] = attention_dst_sum[to_type] + attn_o.sum(dim=1)
            # print(ret[to_type].size(), attentions.size(), from_projected.size())
            ret[to_type] = ret[to_type] + attn_o.permute(0,2,1).bmm(from_projected)
        
        # return [embeddings[t] + ret[t].permute(1,2,0).reshape(-1, self.output_dim) for t in range(self.n_nodetype)]
        ret = [(ret[i]/(attention_dst_sum[i].unsqueeze(2)+1e-9)).permute(1,0,2).reshape(-1, self.output_dim) for i in range(len(ret))]
        # ipdb.set_trace()
        return ret

class Het_Netlayer_R(nn.Module):

    def __init__(self, n_layer, n_nodetype, n_head, edge_types, input_dims, mediate_size, dr, residual, n_selfring):
        super(Het_Netlayer_R, self).__init__()
        assert len(mediate_size) == n_layer
        self.layers = [singerNetLayer_R(n_nodetype, n_head, input_dims, mediate_size[0], edge_types, n_selfring)] + [singerNetLayer_R(n_nodetype, n_head, mediate_size[i], mediate_size[i+1], edge_types, n_selfring) for i in range(n_layer-1)]
        self.layers = nn.ModuleList(self.layers)
        self.n_layer = n_layer
        self.n_nodetype = n_nodetype
        self.n_head = n_head
        self.edge_types = edge_types
        self.mediate_size = mediate_size
        self.dropout = dr
        self.residual = residual

    def forward(self, embeddings, adjs):
        # embeddings: n_nodetype * n_node * input_dim
        # adjs: n_edgetype * n_from_node * n_to_node
        emb = embeddings
        for i in range(self.n_layer):
            size1 = [int(e.size(1)) for e in emb]
            emb2 = [F.dropout(e, self.dropout, training=self.training) for e in emb]
            emb2 = self.layers[i](emb2, adjs)
            size2 = [int(e.size(1)) for e in emb2]
            if size1 == size2 and self.residual:
                emb = [emb[i] + emb2[i] for i in range(len(embeddings))]
            else:
                emb = emb2
        return emb

