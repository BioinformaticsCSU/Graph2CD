import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn import preprocessing
import pandas as pd
import GIP
import pickle

circInter = np.load("../data_cdhgnn/circInter_pro.npy")
circSim_kernel = GIP.Get_GIP_profile(circInter, "t")
circSim = preprocessing.scale(circSim_kernel, axis=0, with_mean=True, with_std=True, copy=True)
G_circ = nx.Graph()
num_circ = circSim.shape[0]
[G_circ.add_node(j) for j in range(num_circ)]  # start from 0
for i in range(num_circ):
    for j in range(num_circ):
        G_circ.add_edge(i, j, weight=circSim[i, j])
circ_centrality = nx.closeness_centrality(G_circ, max_iter=30, weight='weight', wf_improved=True)

mirNet_weight = np.load("../data_cdhgnn/mirNet_weight.npy")
mirSim = preprocessing.scale(mirNet_weight, axis=0, with_mean=True, with_std=True, copy=True)
G_mir = nx.Graph()
num_mir = mirSim.shape[0]
[G_mir.add_node(j) for j in range(num_mir)]  # start from 0
for i in range(num_mir):
    for j in range(num_mir):
        G_mir.add_edge(i, j, weight=mirSim[i, j])
mir_centrality = nx.closeness_centrality(G_mir, max_iter=60, weight='weight', wf_improved=True)

disSim_mat = np.load("../data_cdhgnn/disSim_mat.npy")
disSim = preprocessing.scale(disSim_mat, axis=0, with_mean=True, with_std=True, copy=True)
G_dis = nx.Graph()
num_dis = disSim.shape[0]
[G_dis.add_node(j) for j in range(num_dis)]  # start from 0
for i in range(num_dis):
    for j in range(num_dis):
        G_dis.add_edge(i, j, weight=disSim[i, j])
dis_centrality = nx.closeness_centrality(G_dis, max_iter=20, weight='weight', wf_improved=True)

centrality_list = []
centrality_list.append(circ_centrality)
centrality_list.append(mir_centrality)
centrality_list.append(dis_centrality)

with open("../data_cdhgnn/centralities.pkl", 'wb') as fileCentrality:
    pickle.dump(centrality_list, fileCentrality)

