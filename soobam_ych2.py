#!/usr/bin/env python
# coding: utf-8

# # Datamodule

# In[8]:

import torch_optimizer as optim
import torch
import pytorch_lightning as pl
from timm.data.loader import MultiEpochsDataLoader
import dataloader.postech_loader as postech_loader
import torch.nn.functional as F

class POSTECHDataModule(pl.LightningDataModule):
    def __init__(self, root, edge, graph_ind, train=None, test=None, save='save.pt',
                batch_size=32, num_workers=32):
        super().__init__()
        self.root, self.edge, self.graph_ind = root, edge, graph_ind
        self.train, self.test, self.save = train, test, save
        self.batch_size, self.num_workers = batch_size, num_workers

    def setup(self, stage=None):
        train_valid_data = postech_loader.POSTECHDataset(self.root, self.edge, self.graph_ind,
                                                        train=self.train, save=self.save)
        self.train_data, self.valid_data = torch.utils.data.random_split(train_valid_data, [3800, 600])
        self.test_data = postech_loader.POSTECHDataset(self.root, self.edge, self.graph_ind,
                                                        test=self.test, save=self.save)

    def train_dataloader(self):
        return MultiEpochsDataLoader(self.train_data, batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return MultiEpochsDataLoader(self.valid_data, batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return MultiEpochsDataLoader(self.test_data, batch_size=self.batch_size,
                                    num_workers=self.num_workers, pin_memory=True)


# # GCN Module

# In[9]:


# NN layers and models
class GraphConv(pl.LightningModule):
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
        self.fc = torch.nn.Linear(in_features=in_features, out_features=out_features)
        self.adj_sq = adj_sq
        self.activation = activation
        self.scale_identity = scale_identity
        self.log_theta = torch.nn.Parameter(torch.randn(1))
            
    def laplacian_batch(self, A):
        theta = torch.exp(self.log_theta)
        #print(f"Laplacian_batch-A:{A}")
        batch, N = A.shape[:2]
        #print(f"Laplacian_batch-batch:{batch}")
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        I = torch.eye(N).unsqueeze(0).to(self.device)
        if self.scale_identity:
            I = 2 * I  # increase weight of self connections
        A_hat =  theta[0]*A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A = data
        x = self.fc(torch.bmm(self.laplacian_batch(A), x))
        if self.activation is not None:
            x = self.activation(x)
        #x = F.dropout(x,0.1)
        return (x, A)
        
class GCN(pl.LightningModule):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64,64],
                 n_hidden=0,
                 dropout=0.1,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        #self.gconv = torch.nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1], 
        #                                        out_features=f, 
        #                                        activation=torch.nn.ReLU(inplace=True),
        #                                        adj_sq=adj_sq,
        #                                        scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        self.gconv_list = torch.nn.ModuleList()
        self.gconv_list.append(torch.nn.Sequential(*([GraphConv(in_features=in_features, 
                                                out_features=filters[0], 
                                                activation=torch.nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity)])))
        
        for i in range(0,len(filters)-1):
            self.gconv_list.append(torch.nn.Sequential(*([GraphConv(in_features=filters[i], 
                                                out_features=filters[i+1], 
                                                activation=torch.nn.ReLU(inplace=True),
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity)])))


        #self.gconv1 = torch.nn.Sequential(*([GraphConv(in_features=in_features, 
        #                                        out_features=filters[0], 
        #                                        activation=torch.nn.ReLU(inplace=True),
        #                                        adj_sq=adj_sq,
        #                                        scale_identity=scale_identity)]))
        #self.gconv2 = torch.nn.Sequential(*([GraphConv(in_features=filters[0], 
        #                                        out_features=filters[1], 
        #                                        activation=torch.nn.ReLU(inplace=True),
        #                                        adj_sq=adj_sq,
        #                                        scale_identity=scale_identity)]))
        #self.gconv3 = torch.nn.Sequential(*([GraphConv(in_features=filters[1], 
        #                                        out_features=filters[2], 
        #                                        activation=torch.nn.ReLU(inplace=True),
        #                                        adj_sq=adj_sq,
        #                                        scale_identity=scale_identity)]))
        #self.gconv4 = torch.nn.Sequential(*([GraphConv(in_features=filters[2], 
        #                                        out_features=filters[3], 
        #                                        activation=torch.nn.ReLU(inplace=True),
        #                                        adj_sq=adj_sq,
        #                                        scale_identity=scale_identity)]))
                                                 
        k1 = 100
        self.k1 = k1
        k2 = 50
        self.k2 = k2
        k3 = 10
        self.k3 = k3
        #conv2d = torch.nn.Sequential(torch.nn.Conv2d(k1,k2,3,padding=1),torch.nn.Conv2d(k2,k3,3,padding=1))
        conv2d = torch.nn.Conv2d(k1,k3,3,padding=1)
        self.conv2d = conv2d




        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(torch.nn.Dropout(p=dropout))
        #if n_hidden > 0:
        #    fc.append(torch.nn.Linear(filters[-1], n_hidden))
        #    if dropout > 0:
        #        fc.append(torch.nn.Dropout(p=dropout))
        #    n_last = n_hidden
        #else:
        n_last = filters[-1]*len(filters)*k3
        fc.append(torch.nn.Linear(n_last, filters[-1]))
        fc.append(torch.nn.Linear(filters[-1], filters[-1]))
        fc.append(torch.nn.Linear(filters[-1], out_features))           
        self.fc = torch.nn.Sequential(*fc)
        
    def forward(self, data):
        #x = self.gconv(data)[0]
        Z = []
        x,A = self.gconv_list[0](data)
        Z.append(x)
        for i in range(1,len(self.gconv_list)):
            x,A = self.gconv_list[i]([x,A])
            Z.append(x)
            
        
        #x,A = self.gconv2([x,A])
        #Z.append(x)
        #x,A = self.gconv3([x,A])
        #Z.append(x)
        #x,A = self.gconv4([x,A])
        #Z.append(x)

        Z_new = []
        Z_temp, topk_indices = torch.topk(Z[-1],k=self.k1,dim=1)
        Z_new.append(Z_temp)
        for j in range(1,len(Z)):
            Z_new.append(torch.gather(Z[j],1,topk_indices))
        
        #Z_new = torch.stack(Z_new,3)
        Z_new = torch.stack(Z_new,3) #batch_size x k1 x filters[-1] x len(filters)

        x=self.conv2d(Z_new)
        x = x.view(x.size()[0],-1)
        #print(f"GCN:x.size{x.size()}")
        #x = torch.max(x, dim=1)[0].squeeze() # Max pooling over layers
        x = self.fc(x)
        return x  


# # Lightning GCN

# In[10]:


import pytorch_lightning as pl
import torch
import pprint

class LightningGCN(pl.LightningModule):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.model = GCN(1, 3)

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        mask_2d, adjacency_matrix, original_index, degree, y = batch
        y_hat = self((mask_2d, adjacency_matrix))
        weight = torch.Tensor([4400,4400,4400]) / torch.Tensor([2400,575,1425])
        weight = weight / weight.sum()
        loss = torch.nn.functional.cross_entropy(y_hat, y,weight=weight.to(self.device))
        correct = y_hat.argmax(dim=1).eq(y).sum().detach()

        self.log('train_acc', correct / len(batch[-1]), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mask_2d, adjacency_matrix, original_index, degree, y = batch
        y_hat = self((mask_2d, adjacency_matrix))
        weight = torch.Tensor([4400,4400,4400]) / torch.Tensor([2400,575,1425])
        weight = weight / weight.sum()
        loss = torch.nn.functional.cross_entropy(y_hat, y,weight=weight.to(self.device))
        correct = y_hat.argmax(dim=1).eq(y).sum().detach()

        self.log('val_loss', loss)
        batch_dictionary = {
            'val_loss': loss,
            'val_correct': correct,
            'val_total': len(batch[-1])
        }
        return batch_dictionary

    def validation_epoch_end(self, results):
        validation_avg_loss = torch.stack([result.get('val_loss', None) for result in results]).mean()
        validation_correct = sum([result.get('val_correct', 0) for result in results])
        validation_total = sum([result.get('val_total', 0) for result in results])

        self.log('val_accuracy', validation_correct / validation_total, prog_bar=True, on_epoch=True)
        epoch_dictionary = {
            'val_avg_loss': validation_avg_loss,
            'val_accuracy': validation_correct / validation_total
        }
        return epoch_dictionary

    def test_step(self, batch, batch_idx):
        mask_2d, adjacency_matrix, original_index, degree = batch
        y_hat = self((mask_2d, adjacency_matrix))

        batch_dictionary = {
            'original_index': original_index,
            'y_hat': y_hat
        }
        return batch_dictionary

    def test_epoch_end(self, results):
        test_original_index = torch.cat([result.get('original_index', None) for result in results])
        test_y_hat = torch.cat([result.get('y_hat', None) for result in results])

        self.save_to_csv(test_original_index, test_y_hat)

    def save_to_csv(self, original_indices, y_hats):
        preds = y_hats.argmax(dim=1)
        original_indices, preds = original_indices.cpu(), preds.cpu()

        with open(f'answer_version{self.logger.version}.csv', 'w') as f:
            f.write('Id,Category\n')
            for original_index, pred in zip(original_indices, preds):
                f.write(f'{int(original_index)},{int(pred.item()) + 1}\n')

    def configure_optimizers(self):
        return torch.optim.Adam (model.parameters(), lr=0.0008)
       # return optim.RAdam(model.parameters(), lr=0.005)
        #torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False) #


# In[11]:


def wrong_answer_finder(filename):
    with open(filename, 'r') as f:
        correct = 0
        wrong = 0
        wrong_idx = []
        wrong_pred = []
        wrong_ans = []
        columns = f.readline()
        for line in f:
            index, pred = tuple(int(i) for i in line.split(','))
            if index <= 2600:
                if pred == 1:
                    correct += 1
                else:
                    wrong_idx.append(index)
                    wrong_pred.append(pred)
                    wrong_ans.append(1)
                    wrong += 1
            elif index <= 3375:
                if pred == 2:
                    correct += 1
                else:
                    wrong_idx.append(index)
                    wrong_pred.append(pred)
                    wrong_ans.append(2)
                    wrong += 1
            else:
                if pred == 3:
                    correct += 1
                else:
                    wrong_idx.append(index)
                    wrong_pred.append(pred)
                    wrong_ans.append(3)
                    wrong += 1
        wrong_idx = torch.tensor(wrong_idx)
        wrong_pred = torch.tensor(wrong_pred)
        wrong_ans = torch.tensor(wrong_ans)
        wrong_ans,temp_idx=torch.sort(wrong_ans)
        wrong_idx = torch.gather(wrong_idx,0,temp_idx)
        wrong_pred = torch.gather(wrong_pred,0,temp_idx)
        what_is_wrong = torch.stack((wrong_idx,wrong_ans,wrong_pred),1)
        
    return what_is_wrong
        #total = correct + wrong
        #print(f'{100 * correct / total:.2f}% ({correct}/{total})')

def grader(filename):
    with open(filename, 'r') as f:
        correct = 0
        wrong = 0

        columns = f.readline()
        for line in f:
            index, pred = tuple(int(i) for i in line.split(','))
            if index <= 2600:
                if pred == 1:
                    correct += 1
                else:
                    wrong += 1
            elif index <= 3375:
                if pred == 2:
                    correct += 1
                else:
                    wrong += 1
            else:
                if pred == 3:
                    correct += 1
                else:
                    wrong += 1

        total = correct + wrong
        print(f'{100 * correct / total:.2f}% ({correct}/{total})')


# In[12]:



pl.seed_everything(2021)

POSTECH_data_module = POSTECHDataModule('data', 'graph.txt', 'graph_ind.txt',
                                        train='train.txt', test='test.txt',
                                        batch_size=32, num_workers=0)
model = LightningGCN(0.0008)

early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=15,
    verbose=True,
    mode='min'
)
trainer = pl.Trainer(callbacks=[early_stop_callback], gpus=1)# gpus=1, accelerator='ddp_spawn')
trainer.fit(model, datamodule=POSTECH_data_module)


# In[13]:


result = trainer.test(model, datamodule=POSTECH_data_module)
print(f'answer_version{model.logger.version}.csv')
grader(f'answer_version{model.logger.version}.csv')  


# In[ ]:


wrong_answer_finder(f'answer_version{model.logger.version}.csv')


# In[ ]:




