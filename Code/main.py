import dataLoad_circdis2
from model import *
import numpy as np
from numpy import interp
import os
import pickle
import time
import torch.nn as nn
import torch.optim.optimizer
from utils import EarlyStopping
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score,\
    accuracy_score, precision_recall_curve, matthews_corrcoef

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('entering main')
    # torch.cuda.set_device(0)

    # setting parameters
    patience = 50
    kfold = 5
    n_layer = 4
    n_nodetype = 3
    n_head = 8
    edge_types = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
    # type0: mir; tpye1: dis; type2: circ
    mediate_size = [128,64,32,16]
    epochs = 100

    # load features

    circ_fea = torch.load('data_cdhgnn/circ_feaD100.pt').cpu().cuda(0)
    mir_fea = torch.load('data_cdhgnn/mir_fea_01.pt').cpu().cuda(0)
    dis_fea = torch.load('data_cdhgnn/dis_fea.pt').cpu().cuda(0)

    # embeddings: n_nodetype * n_node * input_dim
    embeddings = [] # # type0: mir; tpye1: dis; type2: circ; [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
    embeddings.append(mir_fea)
    embeddings.append(dis_fea)
    embeddings.append(circ_fea)

    # load dict, similarity net, interMat
    with open('data_cdhgnn/allProcessedData.pkl', 'rb') as file1:
        allProcessedList = pickle.load(file1) # circDict, mirDict, disDict, circNet, mirNet, disNet, circmir, circdis, mirdis

    circNet = allProcessedList[3]
    circNet = circNet.astype(int) # 0.2---0.5876; 0.1---0.786

    mirNet = allProcessedList[4]
    mirNet = mirNet.astype(int)

    disNet = allProcessedList[5]
    disNet = disNet.astype(int)

    circmirInterMat = allProcessedList[6]
    circdisInterMat = allProcessedList[7]
    mirdisInterMat = allProcessedList[8]

    # adjs: n_edgetype * n_from_node * n_to_node
    adjs = [] # type0: mir; tpye1: dis; type2: circ; [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
    adjs.append(torch.from_numpy(mirNet).long().cuda(0))
    adjs.append(torch.from_numpy(disNet).long().cuda(0))
    adjs.append(torch.from_numpy(circNet).long().cuda(0))
    adjs.append(torch.from_numpy(mirdisInterMat).long().cuda(0))
    adjs.append(torch.from_numpy(np.transpose(circmirInterMat)).long().cuda(0))

    discircInterMat = np.transpose(circdisInterMat.astype(int))

    # data_input = dataLoad_circdis.dataLoad(discircInterMat)  #
    mircircInterMat = np.transpose(circmirInterMat)
    data_input = dataLoad_circdis2.dataLoad(mirNet, disNet, circNet, mirdisInterMat, mircircInterMat, discircInterMat)  #
   
    adjs.append(torch.from_numpy(discircInterMat).long().cuda(0))

    input_dims = [circ_fea.detach().cpu().numpy().shape[1], mir_fea.detach().cpu().numpy().shape[1], dis_fea.detach().cpu().numpy().shape[1]]

    loss_list = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aucs = []
    Acc = []
    aucpr_list = []
    Mcc = []
    pre_value_list = []
    recall_value_list = []
    fscore1 = []

    start_time = time.time()

    y_real = []
    y_proba = []

    Ws_list = []
    X_list = []

    for i_fold in range(kfold):
        print(('Fold:  ' + str(i_fold)).center(100, '='))

        # early stopping
        save_path = "./results" + "/{}".format(i_fold)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        early_stopping = EarlyStopping(savepath=save_path, patience=patience, verbose=True, delta=0)

        # Train, Valid and Test
        tr_samples, tr_labels, val_samples, val_labels = data_input.shuffleMakeTraining()
        tr_samples = torch.from_numpy(np.array(tr_samples)).type(torch.LongTensor).cuda(0)
        val_samples = torch.from_numpy(np.array(val_samples)).type(torch.LongTensor).cuda(0)
        tr_labels_ts = torch.from_numpy(np.array(tr_labels)).type(torch.LongTensor).cpu().cuda(0)
        val_labels_ts = torch.from_numpy(np.array(val_labels)).type(torch.LongTensor).cpu().cuda(0)

        hetsann = Het_Netlayer_R(n_layer, n_nodetype, n_head, edge_types, input_dims, mediate_size, 0, True, 3) # meta path self loop

        optimizer = torch.optim.Adam(hetsann.parameters(), lr=0.00001, weight_decay=1e-3)
        loss_func = nn.CrossEntropyLoss().cuda(0)
        # model.to(device)

        hetsann.cuda(0)

        # training
        print(('Training').center(60, '='))
        # epochs
        for epoch in range(epochs):
            print('Epoch:  ', epoch + 1)
            hetsann.train()
            tr_y, emb = hetsann(embeddings, adjs, tr_samples)
            tr_loss = loss_func(tr_y, tr_labels_ts)

            optimizer.zero_grad()
            tr_loss.backward()
           
            optimizer.step()

           
            hetsann.eval()
            # Validation
            with torch.no_grad():
                # val_loss, val_y = model.forward(A, node_fea, val_samples, val_labels_ts)
                val_y, _ = hetsann(embeddings, adjs, val_samples)
                val_loss = loss_func(val_y, val_labels_ts)
                val_y_pred = torch.argmax(val_y.detach(), dim=1)
               


            early_stopping(val_loss, hetsann, epoch)

        hetsann.eval()


