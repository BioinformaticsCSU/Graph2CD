from gensim.models.doc2vec import Doc2Vec
import GIP
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import random
from seq_encoding import *
import scipy.io as sio
from scipy.spatial.distance import pdist
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score,\
    accuracy_score, precision_recall_curve, matthews_corrcoef

# Seed for reproducible numbers
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.num_fea = 213
        self.hid = 16
        self.in_head = 12
        self.out_dim = 8
        self.out_head = 8
        self.conv1 = GATConv(self.num_fea, self.hid, heads=self.in_head, dropout=0.8)
        self.conv2 = GATConv(self.hid * self.in_head, self.out_dim, heads=self.out_head,
                             dropout=0.8)
        self.linear = nn.Linear(self.out_dim * self.out_head * 2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, edge_index, node1_index, node2_index):
        x = features
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        y = self.linear(torch.cat((x[node1_index],x[node2_index]), dim=1)) # get interaction
        y_pred = torch.sigmoid(y)  # predicted
        return y_pred, x

def normalSim(sim_mat, normType):
    if normType == 1:
        sim = preprocessing.scale(sim_mat, axis=0, with_mean=True, with_std=True, copy=True)
    elif normType == 2:
        sim = preprocessing.minmax_scale(sim_mat, feature_range=(0, 1), axis=0, copy=True)
    elif normType == 3:
        sim = preprocessing.maxabs_scale(sim_mat,axis=0, copy=True)
    elif normType == 4:
        sim = preprocessing.robust_scale(sim_mat, axis=0, with_centering=True, with_scaling=True, copy=True)
    elif normType == 5:
        sim = preprocessing.normalize(sim_mat,norm='l2', axis=0, copy=True)

    return sim

def loadData():
    # dis similarity
    disNet_weight = np.load("data_cdhgnn/disSim_mat.npy")

    return disNet_weight

def getCentralities():
    with open('./data_cdhgnn/centralities.pkl', 'rb') as f:
        centralities = pickle.load(f)
    # circCentra = np.array([round(i, 3) for i in centralities[0].values()])
    # mirCentra = np.array([round(i,3) for i in centralities[1].values()])
    disCentra = np.array([round(i,3) for i in centralities[2].values()])

    return disCentra

def getDis_fea():
    ### node feature
    # load doc2vec model
    model = Doc2Vec.load("data_cdhgnn/d2v_disNarrativ.model")
    dis_narrFea = model.docvecs.vectors_docs

    # dosim_wang
    doSim_data = sio.loadmat(r'data_cdhgnn/doSimWang_mndr.mat', mat_dtype=True, struct_as_record=True)
    disSimWang_MNDR = doSim_data['dosimMNDR_wang']

    # concatenate disease similarity
    dis_fea = np.concatenate((dis_narrFea, disSimWang_MNDR), axis=1)

    return dis_fea

if __name__ == "__main__":

    disNet_weight = loadData()
    num_dis = disNet_weight.shape[0]

    # load features
    disCentra = getCentralities()
    print("start generating Doc2vec")

    dis_fea = getDis_fea()
    print("end Doc2vec")
    # initial features + structural features
    dis_merge = np.concatenate((np.around(dis_fea, decimals=3), disCentra.reshape(-1, 1)), axis=1)

    # embedding = np.random.randn(20,12).astype('float32')
    index_tuple = np.where(disNet_weight >0 )
    edge_index = [index_tuple[0], index_tuple[1]]
    label_pos = [1 for i in range(len(index_tuple[0]))]
    node1_pos_index = [i for i in index_tuple[0]]
    node2_pos_index = [i for i in index_tuple[1]]

    num_pos = len(index_tuple[0])
    neg_set = []
    node1_neg_index = []
    node2_neg_index = []
    pos_set = set()
    [pos_set.add((index_tuple[0][i], index_tuple[1][i])) for i in range(num_pos)]
    print("start generating negative samples")
    while len(neg_set)<10000:
        row_rand = random.randint(0,num_dis-1)
        col_rand = random.randint(0,num_dis-1)
        if (row_rand, col_rand) not in pos_set:
            node1_neg_index.append(row_rand)
            node2_neg_index.append(col_rand)
            neg_set.append((row_rand, col_rand))

    print("end generating negative samples")
    tmp_list = []
    tmp_list.append(np.concatenate((edge_index[0], np.array(node1_neg_index))))
    tmp_list.append(np.concatenate((edge_index[1], np.array(node2_neg_index))))
    label_neg = [0 for i in range(len(neg_set))]
    edge_index = tmp_list
    node1_index = node1_pos_index + node1_neg_index
    node2_index = node2_pos_index + node2_neg_index
    label = label_pos + label_neg

    features = torch.from_numpy(dis_merge).to(torch.float32).to(device)
    edge_index = torch.tensor(np.array(edge_index)).to(device)
    node1_index = torch.tensor(node1_index).to(device)
    node2_index = torch.tensor(node2_index).to(device)
    label = torch.tensor(label).to(device)
    # Train
    model = GAT().to(device)

    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    # optimizer = torch.optim.Adam([{'params': model.conv1.parameters(), "lr": 1e-5},
    #                                       {'params': model.conv2.parameters(), "lr": 1e-5},
    #                                       {"params": model.linear.parameters(), "lr": 1e-5}
    #                                       ], lr=0.05, weight_decay=1e-4)

    print("start training")
    # Training Loop
    model.train()
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        tr_y, x = model(features, edge_index, node1_index, node2_index)

        # loss = F.nll_loss(result, label)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(tr_y, label)
        tmp_tr = tr_y.clone()

        tr_y_pred = torch.argmax(tmp_tr, dim=1).detach().cpu()
        tr_acc = accuracy_score(label.detach().cpu(), tr_y_pred)
        tr_precision = precision_score(label.detach().cpu(), tr_y_pred)
        tr_mcc = matthews_corrcoef(label.detach().cpu(), tr_y_pred)
        tr_recall = recall_score(label.detach().cpu(), tr_y_pred)
        tr_fscore = f1_score(label.detach().cpu(), tr_y_pred, labels=np.unique(tr_y_pred))

        if epoch % 20 == 0:
            print(('Epoch:  ' + str(epoch)).center(100, '='))
            print('Train - Loss: {}, acc: {}, precision: {}, recall: {}, mcc: {}, fscore: {}' \
                      .format(round(loss.item(),3), round(tr_acc,3), round(tr_precision,3), round(tr_recall,3),
                              round(tr_mcc,3), round(tr_fscore,3)))

        loss.backward()
        optimizer.step()

    print("end training")
    torch.save(x, 'data_cdhgnn/disFea.pt')


# dist = pdist(np.vstack([x,y]),'cityblock')