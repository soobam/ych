#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
import math
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter
from os.path import join as pjoin
from torch.autograd import Variable


# In[ ]:


# Experiment parameters
batch_size = 64
threads = 0
lr = 0.005
epochs = 2000
log_interval = 10
wdecay = 1e-4
dataset = 'COLLAB' # 'proteins' 'collab'
model_name = 'gcn'  # 'gcn', 'unet'
device = 'cuda'  # 'cuda', 'cpu'
visualize = True
shuffle_nodes = False
op_filters = [128,128,128]
op_adj_sq = False
n_folds = 1  # 10-fold cross validation
seed = 111
print('torch', torch.__version__)


# In[ ]:


# Data loader and reader
class GraphData(torch.utils.data.Dataset):
    def __init__(self,
                 datareader,
                 fold_id,
                 split):
        self.fold_id = fold_id
        self.split = split
        self.rnd_state = datareader.rnd_state
        self.set_fold(datareader.data, fold_id)

    def set_fold(self, data, fold_id):
        self.total = len(data['targets'])
        self.N_nodes_max = data['N_nodes_max']
        self.n_classes = data['n_classes']
        self.features_dim = data['features_dim']
        self.idx = data['splits'][fold_id][self.split]
         # use deepcopy to make sure we don't alter objects in folds
        self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
        self.adj_list = copy.deepcopy([data['adj_list'][i] for i in self.idx])
        self.features_onehot = copy.deepcopy([data['features_onehot'][i] for i in self.idx])
        print('%s: %d/%d' % (self.split.upper(), len(self.labels), len(data['targets'])))
        self.indices = np.arange(len(self.idx))  # sample indices for this epoch
        
    def pad(self, mtx, desired_dim1, desired_dim2=None, value=0):
        sz = mtx.shape
        assert len(sz) == 2, ('only 2d arrays are supported', sz)
        # if np.all(np.array(sz) < desired_dim1 / 3): print('matrix shape is suspiciously small', sz, desired_dim1)
        if desired_dim2 is not None:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, desired_dim2 - sz[1])), 'constant', constant_values=value)
        else:
            mtx = np.pad(mtx, ((0, desired_dim1 - sz[0]), (0, 0)), 'constant', constant_values=value)
        return mtx
    
    def nested_list_to_torch(self, data):
        if isinstance(data, dict):
            keys = list(data.keys())           
        for i in range(len(data)):
            if isinstance(data, dict):
                i = keys[i]
            if isinstance(data[i], np.ndarray):
                data[i] = torch.from_numpy(data[i]).float()
            elif isinstance(data[i], list):
                data[i] = list_to_torch(data[i])
        return data
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        index = self.indices[index]
        N_nodes_max = self.N_nodes_max
        N_nodes = self.adj_list[index].shape[0]
        graph_support = np.zeros(self.N_nodes_max)
        graph_support[:N_nodes] = 1
        return self.nested_list_to_torch([self.pad(self.features_onehot[index].copy(), self.N_nodes_max),  # node_features
                                          self.pad(self.adj_list[index], self.N_nodes_max, self.N_nodes_max),  # adjacency matrix
                                          graph_support,  # mask with values of 0 for dummy (zero padded) nodes, otherwise 1 
                                          N_nodes,
                                          int(self.labels[index])])  # convert to torch


class DataReader():
    '''
    Class to read the txt files containing all data of the dataset
    '''
    def __init__(self,
                 data_dir,  # folder with txt files
                 rnd_state=None,
                 use_cont_node_attr=False,  # use or not additional float valued node attributes available in some datasets
                 use_cont_node_labels=False, 
                 folds=10):

        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        self.use_cont_node_labels = use_cont_node_labels
        files = os.listdir(self.data_dir)
        data = {}
        nodes, graphs = self.read_graph_nodes_relations(list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])
        
        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)                      
        data['targets'] = np.array(self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0, files))[0],
                                                       line_parse_fn=lambda s: int(float(s.strip()))))
        
        data['test'] = np.array(
            self.parse_txt_file(list(filter(lambda f: f.find('test') >= 0 , files))[0],
                                line_parse_fn=lambda s: int(float(s.strip()))))
        #if self.use_cont_node_labels:
        #    data['features'] = self.read_node_features(list(filter(lambda f: f.find('node_labels') >= 0, files))[0], 
        #                                         nodes, graphs, fn=lambda s: int(s.strip()))  

        #if self.use_cont_node_attr:
        #    data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0], 
        #                                           nodes, graphs, fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        
        #features, n_edges, degrees = [], [], []
        n_edges = []
        degrees = []
        features_onehot = []
        for sample_id, adj in enumerate(data['adj_list']):
            N = adj.shape[0]#len(adj)  # number of nodes
            #if data['features'] is not None:
            #    assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            #else:
            feature_onehot=np.empty((N,0))
            n = np.sum(adj)  # total sum of edges
            #assert n % 2 == 0, n
            n_edges.append( int(n / 2) )  # undirected edges, so need to divide by 2
            if not np.allclose(adj, adj.T):
                print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            #features.append(np.array(data['features'][sample_id]))
            feature_attr = np.empty((N,0))
            onehot_degree = np.empty((N,0))
            node_features = np.concatenate((feature_onehot,feature_attr,onehot_degree),axis=1)
            
            if node_features.shape[1] == 0:
                node_features = np.ones((N,1))
            features_onehot.append(node_features)
        # Create features over graphs as one-hot vectors for each node
        
        #features_all = np.concatenate(features)
        #features_min = features_all.min()
        #features_dim = int(features_all.max() - features_min + 1)  # number of possible values
        
        #for i, x in enumerate(features):
        #    feature_onehot = np.zeros((len(x), features_dim))
        #    for node, value in enumerate(x):
        #        feature_onehot[node, value - features_min] = 1
        #    if self.use_cont_node_attr:
        #        feature_onehot = np.concatenate((feature_onehot, np.array(data['attr'][i])), axis=1)
        #    features_onehot.append(feature_onehot)
        
        #if self.use_cont_node_attr:
        #    features_dim = features_onehot[0].shape[1]
        
        features_dim = features_onehot[0].shape[1]
        

        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']        # graph class labels
        labels -= np.min(labels)        # to start from 0
        
        
        test_idx = data['test']
        test_idx -= np.min(test_idx)  # from 0 ~
        
        N_nodes_max = np.max(shapes)    

        classes = np.unique(labels)
        n_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(n_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == n_classes, np.unique(labels)

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(shapes), np.std(shapes), np.min(shapes), np.max(shapes)))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(n_edges), np.std(n_edges), np.min(n_edges), np.max(n_edges)))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % (np.mean(degrees), np.std(degrees), np.min(degrees), np.max(degrees)))
        print('Node features dim: \t\t%d' % features_dim)
        print('N classes: \t\t\t%d' % n_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        #for u in np.unique(features_all):
        #    print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))
        
        N_graphs = len(labels)  # number of samples (graphs) in data
        assert N_graphs == len(data['adj_list']) == len(features_onehot), 'invalid data'

        # Create test sets first
        train_ids, test_ids = self.split_ids(np.arange(N_graphs), test_idx , rnd_state=self.rnd_state, folds=folds)
        
        # Create train sets
        splits = []
        for fold in range(folds):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits 
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['features_dim'] = features_dim
        data['n_classes'] = n_classes
        
        self.data = data

    def split_ids(self, ids_all, te_idx, rnd_state=None, folds=10):
        n = len(ids_all)
        stride = int(np.ceil(n / float(folds)))
        train_ids = []
        test_ids = []
        test_ids.append(te_idx)
        train_ids.append([e for e in ids_all if e not in te_idx])
        return train_ids, test_ids 

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data
    
    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            node2 = int(edge[1].strip()) - 1
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
            
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        
        return adj_list
        
    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        for graph_id in graphs:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [ None ] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst


# In[ ]:


#log_theta = Variable(torch.randn(3), requires_grad=True)
#log_theta = torch.randn(3, requires_grad=True, device="cpu") 
#print(log_theta)
# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017)
    Additional tricks (power of adjacency matrix and weight self connections) as in the Graph U-Net paper
    '''
    def __init__(self,
                in_features,
                out_features,
                activation=None,
                adj_sq=False,
                scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj_sq = adj_sq
        self.activation = activation
        self.scale_identity = scale_identity
        self.log_theta = nn.Parameter(torch.randn(1))
            
    def laplacian_batch(self, A):
        theta = torch.exp(self.log_theta)
        #print(f"Laplacian_batch-A:{A}")
        batch, N = A.shape[:2]
        A_squ = torch.zeros_like(A)
        A_cub = torch.zeros_like(A)
        #print(f"Laplacian_batch-batch:{batch}")
        if self.adj_sq:
            #print(f"Laplacian_batch:here!")
            A_squ = copy.deepcopy(torch.bmm(A, A))  # use A^2 to increase graph connectivity
            A_cub = copy.deepcopy(torch.bmm(A,A_squ))
        I = torch.eye(N).unsqueeze(0).to(device)
        if self.scale_identity:
            I = 10 * I  # increase weight of self connections
        A_hat = theta[0]*A + I #theta[2]*A_cub + theta[1]*A_squ + 
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A = data[:2]
        x = self.fc(torch.bmm(self.laplacian_batch(A), x))
        if self.activation is not None:
            x = self.activation(x)
        return (x, A)
        
class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 filters,#changed
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        
        #self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1], 
        #                                        out_features=f, 
        #                                        activation=nn.ReLU(inplace=True),
        #                                        adj_sq=adj_sq,
        #                                        scale_identity=scale_identity) for layer, f in enumerate(filters)]))
    
        self.gconv1 = nn.Sequential(*([GraphConv(in_features=in_features, 
                                                out_features=filters[0], 
                                                activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity)]))
        self.gconv2 = nn.Sequential(*([GraphConv(in_features=filters[0], 
                                                out_features=filters[1], 
                                                activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity)]))
        self.gconv3 = nn.Sequential(*([GraphConv(in_features=filters[1], 
                                                out_features=filters[2], 
                                                activation=nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity)]))
        k1 = 100#30
        self.k1 = k1
        k2 = 50#10
        conv2d = nn.Conv2d(k1,k2,(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d = conv2d

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]*len(filters)*k2
        fc.append(nn.Linear(n_last,out_features))
        #fc.append(nn.Linear(n_last,filters[-1]))    
        #fc.append(nn.Linear(filters[-1],out_features))   
        self.fc = nn.Sequential(*fc)
        
    def forward(self, data):
        #x = self.gconv(data)[0]
        
        Z = []
        x,A = self.gconv1(data)
        Z.append(x)
        x,A = self.gconv2([x,A])
        Z.append(x)
        x,A = self.gconv3([x,A])
        Z.append(x)
        #Z_all = torch.stack((Z[0],Z[1],Z[2]),2)
        #print(f"GCN:Z0:{Z[0].size()}")
        #print(f"GCN:Z1:{Z[1].size()}")#32x492x64
        #print(f"GCN:Z2:{Z[2].size()}")
        #Z_new
        #for s in range(batch_size):
            #print(f"GCN:Z[-1][0]:{Z[-1][0].size()}")#492x64
        Z_new = []
        Z_temp, topk_indices = torch.topk(Z[-1],k=self.k1,dim=1)
        #print(f"torch.topk(Z[-1],k=30,dim=1)[0]:{torch.topk(Z[-1],k=30,dim=1)[0].size()}")##32x30x64
        Z_new.append(Z_temp)
        
        for j in range(1,len(Z)):
            #print(f"torch.gather(Z[j],1,topk_indices):{torch.gather(Z[j],1,topk_indices).size()}")##32x30x64
            Z_new.append(torch.gather(Z[j],1,topk_indices))

            #print(f"Z_new[0]:{len(Z_new)}")#3
            #break
        #print(f"torch.stack(Z_new,3).size():{torch.stack(Z_new,3).size()}")#32x30x64x3
        #break
        #torch.stack(Z_new,3)
        Z_new = torch.stack(Z_new,3)
        x=self.conv2d(Z_new)
        x = x.view(x.size()[0],-1)
        #x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x  


# In[ ]:


x = torch.tensor([[[1,5,3,7],
                [4,2,6,3],
                [2,7,4,5]],
                [[10,4,6,2],
                [2,6,4,4],
                [4,0,8,6]]])
print(f"x:{x.size()}")
#print(f"torch.stack((x,x),2):{torch.stack((x,x),2).size()}")
#print(f"torch.cat((x,x),2):{torch.cat((x,x),1).size()}")
print(f"torch.topk(x,k=2,dim=0):{torch.topk(x,k=2,dim=1)[0]}")
print(f"torch.topk(x,k=2,dim=1):{torch.topk(x,k=2,dim=1)[1]}")
#print(torch.index_select(x,torch.topk(x,k=2,dim=2)[1]))
idx = torch.topk(x,k=2,dim=1)[1]
print(f"result:{torch.gather(x,1,idx)}")
#x1s=torch.topk(x,k=2,dim=1)[1][:][0]
#x2s=torch.topk(x,k=2,dim=1)[1][:][1]
#print(f"x1s:{x1s.size()}")
#print(f"x2s:{x2s}")
#temp1 = []
#temp2 = []
#for i in range(np.int32(x1s.size()[1])):
#    print(x[:,x1s[i],i])
#    temp1.append(x[:,x1s[i],i])
#for i in range(np.int32(x2s.size()[1])):
#    print(x[:,x2s[i],i])
#    temp2.append(x[:,x2s[i],i])
#temptemp = [temp1,temp2]
#print(f"result:{torch.FloatTensor(temptemp)}")
#x_new=torch.zero_like(idx)
#for i in range(x_new.size()[0]):
#    for j in range(x_new.size()[1]):
#        x_new[i,j]=temptemp


# In[ ]:


m = nn.Conv2d(16, 33, (3, 3), stride=(1, 1), padding=(1, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.size())


# In[ ]:


dataset.upper()


# In[ ]:



print('Loading data')
start_load = time.time()
datareader = DataReader(data_dir='./data/%s/' % dataset.upper(),
                        rnd_state=np.random.RandomState(seed),
                        folds=n_folds,                    
                        use_cont_node_attr=False,
                        use_cont_node_labels=False)
end_load = time.time()
print(f"Total time cost for {dataset.upper()} data :{end_load-start_load} seconds.")


# In[ ]:


def train(train_loader):
    #scheduler.step()
    model.train()
    start = time.time()
    train_loss, n_samples = 0, 0
    for batch_idx, data in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data[4])
        loss.backward()
        optimizer.step()
        scheduler.step()
        time_iter = time.time() - start
        train_loss += loss.item() * len(output)
        n_samples += len(output)
        if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                epoch, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
#             break 
def test(test_loader):
    model.eval()
    start = time.time()
    test_loss, correct, n_samples = 0, 0, 0
    for batch_idx, data in enumerate(test_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)
        output = model(data)
        loss = loss_fn(output, data[4], reduction='sum')
        test_loss += loss.item()
        n_samples += len(output)
        pred = output.detach().cpu().max(1, keepdim=True)[1]

        correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

    time_iter = time.time() - start

    test_loss /= n_samples

    acc = 100. * correct / n_samples
    print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                          test_loss, 
                                                                                          correct, 
                                                                                          n_samples, acc))
    return acc


# In[ ]:


#import itertools
fold_id = 0
print('\nFOLD', fold_id)
loaders = []
for split in ['train', 'test']:
    gdata = GraphData(fold_id=fold_id,
                         datareader=datareader,
                         split=split)

    loader = torch.utils.data.DataLoader(gdata, 
                                         batch_size=batch_size,
                                         shuffle=split.find('train') >= 0,
                                         num_workers=threads)
    loaders.append(loader)

if model_name == 'gcn':
    model = GCN(in_features=loaders[0].dataset.features_dim,
                out_features=loaders[0].dataset.n_classes,
                n_hidden=0,
                filters=op_filters,#changed
                dropout=0.2,
                adj_sq=op_adj_sq,#changed
                scale_identity=False).to(device)

else:
    raise NotImplementedError(model_name)

print('\nInitialize model')
#print(model)
#print(f"model params:{model.parameters()}")
c = 0
for p in filter(lambda p: p.requires_grad, model.parameters()):
    c += p.numel()
    #print(p)
print('N trainable parameters:', c)
#temp = model.parameters()
#itertools.chain(*params)            
#print(f"What is filtered result:{temp}")
#temp_param = Variable(torch.randn_like(model.parameters()), requires_grad=True)
#filtered = filter(lambda p: p.requires_grad, model.parameters())

#temptemp = torch.nn.utils.parameters_to_vector(model.parameters())
#temptemptemp = Variable(temptemp, requires_grad=True)
#params = [log_theta,temptemptemp]
#params.append(para for para in filtered)
#print(f"params:{params}")
#for para in filtered:
#    print(f"filtered para:{para}")
#    params.extend(para)
#    print(f"params:{params}")
#print(f"length of params:{len(params)}")
"""optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=wdecay,
            betas=(0.5, 0.999))
"""
optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=wdecay,
            betas=(0.5, 0.999))

scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)

loss_fn = F.cross_entropy

epochs_test = 1
for epoch in range(epochs_test):
    train(loaders[0])
    acc = test(loaders[0])


# In[ ]:



#temptemp = torch.nn.utils.parameters_to_vector(model.parameters())
#temptemptemp = Variable(temptemp, requires_grad=True)
#print(temptemptemp)
#print(torch.nn.utils.vector_to_parameters(temptemptemp,model.parameters()))
#print(torch.nn.Parameter(temptemptemp))


# In[ ]:


#model.parameters() = torch.nn.Parameter(temptemptemp)


# In[ ]:





# In[ ]:


testacc = []
for _ in range(1):
    start_train = time.time()
    acc_folds = []
    for fold_id in range(n_folds):
        if fold_id == 1:
            break
        print('\nFOLD', fold_id)
        loaders = []
        for split in ['train', 'test']:
            gdata = GraphData(fold_id=fold_id,
                                 datareader=datareader,
                                 split=split)

            loader = torch.utils.data.DataLoader(gdata, 
                                                 batch_size=batch_size,
                                                 shuffle=split.find('train') >= 0,
                                                 num_workers=threads)
            loaders.append(loader)

        if model_name == 'gcn':
            model = GCN(in_features=loaders[0].dataset.features_dim,
                        out_features=loaders[0].dataset.n_classes,
                        n_hidden=0,
                        filters=[64,64,64],
                        dropout=0.2,
                        adj_sq=False,
                        scale_identity=False).to(device)
        
        else:
            raise NotImplementedError(model_name)

        print('\nInitialize model')
        print(model)
        c = 0
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            c += p.numel()
        print('N trainable parameters:', c)

        optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=wdecay,
                    betas=(0.5, 0.999))

        scheduler = lr_scheduler.MultiStepLR(optimizer, [20, 30], gamma=0.1)

        def train(train_loader):
            scheduler.step()
            model.train()
            start = time.time()
            train_loss, n_samples = 0, 0
            for batch_idx, data in enumerate(train_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, data[4])
                loss.backward()
                optimizer.step()
                time_iter = time.time() - start
                train_loss += loss.item() * len(output)
                n_samples += len(output)
#                if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                if batch_idx == len(train_loader) - 1: # DELETE LOG_INTERVAL
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                        epoch, n_samples, len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1) ))
        #             break 
            return train_loss / n_samples
        def test(test_loader):
            model.eval()
            start = time.time()
            test_loss, correct, n_samples = 0, 0, 0
            for batch_idx, data in enumerate(test_loader):
                for i in range(len(data)):
                    data[i] = data[i].to(device)
                output = model(data)
                loss = loss_fn(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                pred = output.detach().cpu().max(1, keepdim=True)[1]

                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()

            time_iter = time.time() - start

            test_loss /= n_samples

            acc = 100. * correct / n_samples
            print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                                  test_loss, 
                                                                                                  correct, 
                                                                                                  n_samples, acc))
            return acc
                
        test_history = []
        train_history = []
        sum = 0.
        t_sum = 0.
        loss_fn = F.cross_entropy
        print(f"epochs:{epochs}")
        for epoch in range(epochs):
            t_loss=train(loaders[0])
            acc = test(loaders[1])
            sum += acc
            t_sum += t_loss
            # update graph per epochs
            
            # update graph per epochs
            if epoch % 10 == 9:
                test_history.append(sum/10)
                train_history.append(t_sum/10)
                sum = 0
                t_sum = 0
                plt.figure(figsize=(15, 7))
                plt.plot(test_history)
                plt.savefig('figure/fig_testAcc%d.png' % epoch)
                plt.figure(figsize=(15, 7))
                plt.plot(train_history)
                plt.savefig('figure/fig_trainLoss%d.png' % epoch)
    end_train = time.time()

    #print(f"result:{acc_folds[-1]}")
    #plt.plot(acc_folds)
    #plt.show()
    #testacc.append(acc_folds)
    #print('{}-fold cross validation avg acc (+- std): {} ({})'.format(n_folds, np.mean(acc_folds), np.std(acc_folds)))
    
    print(f"train cost {end_train-start_train} seconds")


# In[ ]:


#torch.mean(torch.tensor(testacc))


# In[ ]:


#torch.tensor(acc_folds)


# In[ ]:




