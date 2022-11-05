from utils import *

import os 
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
import numpy as np
import torch



dataset = 'R8'
# data = Planetoid(root='./', name='Cora')[0]
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
if not os.path.exists('./edge.npy'):
    adj_np = adj.toarray()
    edge_index = np.array([[i, j] for i, j in zip(adj.nonzero()[0], adj.nonzero()[1])])
    edge_attr = np.array([adj_np[i, j] for [i, j] in edge_index])
    edge_index = edge_index.reshape(2, -1)
    # print("ADJ", edge_index.shape)
    # print("weight", edge_weight.shape)

    with open('edge.npy', 'wb') as f:
        np.save(f, np.array(edge_index))
        np.save(f, np.array(edge_attr))
else:
    with open('edge.npy', 'rb') as f:
        edge_index = np.load(f)
        edge_attr = np.load(f)
features, edge_index, edge_attr = torch.tensor(features.toarray()), torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr)

print("EDGE: ", edge_index.shape, edge_attr.shape)
print("all features: ", features.shape)
print("adj :", adj.shape)
print("MASK: ", train_mask.sum(), val_mask.sum(), test_mask.sum(), train_size, test_size)  
data = Data(x=features, y=y_train + y_val + y_test, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, edge_index=edge_index)
loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30, 30],
    # Use a batch size of 128 for sampling training nodes
    batch_size=16,
    edge_label_index=data.edge_index
)
print("Data: ", data.x.shape, data.y.shape)


sampled_data = next(iter(loader))
print(sampled_data.x.shape)