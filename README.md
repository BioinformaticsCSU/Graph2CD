# License

Copyright (C) 2020 Jianxin Wang(jxwang@mail.csu.edu.cn),Chengqian Lu(chengqlu@csu.edu.cn)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.

Jianxin Wang(jxwang@mail.csu.edu.cn),Chengqian Lu(chengqlu@csu.edu.cn)
School of Information Science and Engineering
Central South University
ChangSha
CHINA, 410083

Type: Package

# Title: Predicting circRNA-disease associations based on structural graph attention and heterogeneous graph neural network

Description: This package implements the Graph2CD algorithm with structural graph attention and heterogeneous graph neural network for predicting circRNA-disease associations.

Files:
1.data
1) all_gene_disease_associations.tsv stores gene-disease associations from DisGeNET
2) circmiRna_starbase.txt stores experimentally validated circRNA-miRNA associations from starBase;
3) circRNA_human_Download.xlsx stores known circRNA-miRNA associations from CircFunBase;
4) Experimental circRNA-disease information.txt stores known circRNA-disease associations from MNDR 3.0;
5) miRNA-disease.txt stores known miRNA-disease associations from HMDD 3;
6) MISIM2_similarity.txt stores known MISIM 2.0;

2.code
1) centrality_process.py # get Closeness centrality 
2) dataLoad.py # generate training samples and load data
3) GAT_circ.py # get weighted node features
4) GIP.py # calculate similarity based on asscoaition vector
5) main.py # instance for Graph2CD
6) model.py # model of Graph2CD
7) seqEncoding # get sequence features
8) utils.py # save checkpoint and early stopping

##  Requirements
    torch
	sklearn
	scipy
	argparse	
	numpy
	

##  How to train the model
You can generate training samples and load data: 
dataLoad.py

You can run the model 
main.py
