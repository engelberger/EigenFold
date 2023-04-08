# TensorProductConvLayer

This code defines a custom neural network layer called `TensorProductConvLayer`. This layer performs a combination of geometric and algebraic operations on the input data, which is represented in the form of irreducible representations (irreps) of the 3D rotation group O(3).

## Importing required libraries

The code starts by importing required libraries:

- `e3nn` is a library for equivariant neural networks.
- `torch` is the PyTorch library for deep learning.
- `torch_cluster`, `torch_scatter` are extensions of PyTorch for handling graphs and scatter operations.
- `numpy` is a library for numerical operations in Python.

```python
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph, knn_graph, knn
from torch_scatter import scatter, scatter_mean, scatter_max
import numpy as np
from e3nn.nn import BatchNorm
from .utils import get_timestep_embedding, GaussianSmearing, sinusoidal_embedding
```

## Class definition

The `TensorProductConvLayer` class is defined as a subclass of `torch.nn.Module`. This means it inherits properties and methods from the PyTorch `Module` class, which is the base class for all neural network layers in PyTorch.

```python
class TensorProductConvLayer(torch.nn.Module):
```

### Initialization

The `__init__` method initializes the layer with the given parameters. These parameters include irreps for input, output, and tensor product, as well as additional options like batch normalization, dropout, and node feature dimensions.

```python
def __init__(self, in_irreps, in_tp_irreps, out_tp_irreps,
             sh_irreps, out_irreps, n_edge_features,
             batch_norm=False, dropout=0.0, node_feature_dim=4,
             fc_dim=32, lin_self=False, attention=False):
```

#### Linear layers

The layer has several linear layers which transform input irreps and tensor product irreps:

- `self.lin_in`: Linear transformation of input irreps to input tensor product irreps.
- `self.lin_out`: Linear transformation of output tensor product irreps to output irreps.
- `self.lin_self`: Optional linear transformation of input irreps to output irreps.

```python
self.lin_in = o3.Linear(in_irreps, in_tp_irreps, internal_weights=True)
self.lin_out = o3.Linear(out_tp_irreps, out_irreps, internal_weights=True)
if lin_self:
    self.lin_self = o3.Linear(in_irreps, out_irreps, internal_weights=True)
else: self.lin_self = False
```

#### Tensor products

The layer has two tensor product layers:

- `self.tp`: Fully connected tensor product of input tensor product irreps and spherical harmonics irreps.
- `self.tp_k`: Fully connected tensor product of input tensor product irreps and key irreps, used for attention.

```python
self.tp = tp = o3.FullyConnectedTensorProduct(in_tp_irreps, sh_irreps, out_tp_irreps, shared_weights=False)
```

#### Attention mechanism

The layer has an optional attention mechanism, which uses additional linear layers and tensor products:

- `self.h_q`: Linear transformation of input tensor product irreps to key irreps.
- `self.tp_k`: Fully connected tensor product of input tensor product irreps and spherical harmonics irreps, used for attention.
- `self.fc_k`: Fully connected layer for computing attention weights.
- `self.dot`: Fully connected tensor product of key irreps, used for computing attention scores.

```python
if attention:
    self.attention = True
    key_irreps = [(mul//2, ir) for mul, ir in in_tp_irreps]
    self.h_q = o3.Linear(in_tp_irreps, key_irreps)
    self.tp_k = tp_k = o3.FullyConnectedTensorProduct(in_tp_irreps, sh_irreps, key_irreps, shared_weights=False)
    self.fc_k = self.fc = nn.Sequential(
        nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, fc_dim),
        nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, tp_k.weight_numel)
    )
    self.dot = o3.FullyConnectedTensorProduct(key_irreps, key_irreps, "0e")
else: self.attention = False
```

#### Fully connected layers

The layer has fully connected layers for processing edge features:

- `self.fc`: Fully connected layer for computing tensor product weights.

```python
self.fc = nn.Sequential(
    nn.Linear(n_edge_features, fc_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, fc_dim),
    nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_dim, tp.weight_numel)
)
```

#### Batch normalization

The layer has an optional batch normalization layer for normalizing output irreps:

- `self.batch_norm`: Batch normalization layer for output irreps.

```python
self.batch_norm = BatchNorm(out_irreps) if batch_norm else None
```

In summary, this code defines a custom neural network layer that performs tensor product operations on input data represented in the form of irreducible representations of the O(3) rotation group. It includes linear layers, tensor product layers, optional attention mechanism, fully connected layers, and optional batch normalization.


# Forward Method of TensorProductConvLayer

The `forward` method is the core function of the `TensorProductConvLayer` class. It computes the output of the layer given the input node attributes, edge indices, edge attributes, and spherical harmonics.

## Method Definition

The `forward` method takes the following input arguments:

- `node_attr`: Node attributes, which are the features associated with each node in the graph.
- `edge_index`: Edge indices, which indicate the connections between nodes in the graph.
- `edge_attr`: Edge attributes, which are the features associated with each edge in the graph.
- `edge_sh`: Spherical harmonics, which are mathematical functions used for geometric representation.
- `ones`: Optional tensor for padding.
- `residual`: Optional flag to enable/disable residual connections.
- `out_nodes`: Optional number of output nodes.
- `reduce`: Optional reduction method for scatter operation (default is 'mean').

```python
def forward(self, node_attr, edge_index, edge_attr, edge_sh, ones=None, residual=True, out_nodes=None, reduce='mean'):
```

### Attention Mechanism

The code first checks if the attention mechanism is enabled. If it is, the `ckpt_forward` function is defined with attention, otherwise, it is defined without attention.

```python
if self.attention:
    ...
else:
    ...
```

#### With Attention

The `ckpt_forward` function with attention performs the following operations:

1. Compute the query `q` using the linear layer `self.h_q`.
2. Compute the key `k` using the tensor product layer `self.tp_k` and the fully connected layer `self.fc_k`.
3. Compute the value `v` using the tensor product layer `self.tp` and the fully connected layer `self.fc`.
4. Compute the attention scores `a` using the fully connected tensor product `self.dot`.
5. Normalize the attention scores by subtracting the maximum value and applying the exponential function.
6. Compute the normalization factor `z` by scattering the attention scores.
7. Normalize the attention scores by dividing by the normalization factor `z`.
8. Compute the final output by multiplying the attention scores with the value `v` and scattering the result.

```python
def ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr):
    ...
```

#### Without Attention

The `ckpt_forward` function without attention performs the following operations:

1. Compute the tensor product `tp` using the tensor product layer `self.tp` and the fully connected layer `self.fc`.
2. Compute the final output by scattering the tensor product `tp`.

```python
def ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr):
    ...
```

### Checkpointing and Execution

The code uses PyTorch's `checkpoint` function to save memory during training. The `ckpt_forward` function is used in the checkpointing operation during training, and it is executed directly during inference.

```python
if self.training:
    out = torch.utils.checkpoint.checkpoint(ckpt_forward, node_attr_in, edge_src, edge_dst, edge_sh, edge_attr)
else:
    out = ckpt_forward(node_attr_in, edge_src, edge_dst, edge_sh, edge_attr)
```

### Output Transformation

The output is transformed using the linear layer `self.lin_out`. If residual connections are enabled, the output is added to the input node attributes or the output of the `self.lin_self` layer.

```python
out = self.lin_out(out)

if not residual:
    return out
if self.lin_self:
    out = out + self.lin_self(node_attr)
else:
    out = out + F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
```

### Batch Normalization

If the batch normalization layer is enabled, the output is passed through the `self.batch_norm` layer before being returned.

```python
if self.batch_norm:
    out = self.batch_norm(out)
return out
```

In summary, the `forward` method computes the output of the `TensorProductConvLayer` given the input node attributes, edge indices, edge attributes, and spherical harmonics. It supports optional attention mechanism, residual connections, and batch normalization.



# Explanation of ResiLevelTensorProductScoreModel

The code defines a class called `ResiLevelTensorProductScoreModel` which inherits from `torch.nn.Module` (a base class for neural network modules provided by the PyTorch library). This class represents a model that computes scores for some input data using tensor product-based convolutional layers.

## Initialization

The `__init__` method initializes the model with various layers and parameters based on the provided `args` (an object containing the arguments for the model configuration).

### Timestep Embedding

The model initializes a timestep embedding function with:

```python
self.t_emb_func = get_timestep_embedding(args.t_emb_type, args.t_emb_dim)
```

This function is used to embed the timestep information into the input data.

### Spherical Harmonics

The model creates spherical harmonics irreducible representations with:

```python
self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=args.sh_lmax)
```

These representations are used in the convolutional layers of the model.

### Node and Edge Embeddings

The model initializes two embedding layers for nodes and edges respectively:

```python
self.resi_node_embedding = nn.Sequential(...)
self.resi_edge_embedding = nn.Sequential(...)
```

These layers are used to process the node and edge attributes in the input data.

### Distance Expansion

The model initializes a distance expansion function based on the `args.radius_emb_type` and `args.no_radius_sqrt` arguments:

```python
if args.no_radius_sqrt:
    ...
else:
    ...
```

This function is used to process the distances between nodes in the input data.

### Convolutional Layers

The model initializes a list of tensor product-based convolutional layers:

```python
conv_layers = []
...
self.conv_layers = nn.ModuleList(conv_layers)
```

These layers are used to process the input data in the forward pass.

### Final Tensor Product

The model initializes a final tensor product layer:

```python
self.resi_final_tp = o3.FullyConnectedTensorProduct(...)
```

This layer is used to produce the final output scores of the model.

## Forward Pass

The `forward` method defines the forward pass of the model. It takes the input data and computes the output scores using the initialized layers and functions.

### Preprocessing

The method starts by normalizing the node and edge attributes in the input data:

```python
data['resi'].x = self.resi_node_norm(data['resi'].node_attr)
data['resi'].edge_attr = self.resi_edge_norm(data['resi'].edge_attr_)
```

### Building the Convolution Graph

The method builds the convolution graph using the `build_conv_graph` method:

```python
node_attr, edge_index, edge_attr, edge_sh = self.build_conv_graph(data, key='resi', knn=False, edge_pos_emb=True)
```

This graph is used as the input to the convolutional layers in the forward pass.

### Node and Edge Embeddings

The method applies the node and edge embedding layers to the input data:

```python
node_attr = self.resi_node_embedding(node_attr)
edge_attr = self.resi_edge_embedding(edge_attr)
```

### Convolutional Layers

The method applies the convolutional layers to the input data:

```python
for layer in self.conv_layers:
    ...
    node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh)
```

### Final Tensor Product and Output Scores

The method computes the final output scores using the final tensor product layer:

```python
resi_out = self.resi_final_tp(node_attr, node_attr)
```

If the `args.parity` argument is True, the output scores are averaged over the two parity representations:

```python
if self.args.parity:
    resi_out = resi_out.view(-1, 2, 3).mean(1)
```

Finally, the method scales the output scores using the `data.score_norm` attribute and returns the result:

```python
data['resi'].pred = resi_out
data.pred = resi_out
return resi_out
```

## Building the Convolution Graph

The `build_conv_graph` method is used to create the input graph for the convolutional layers. It takes the input data and extracts the node attributes, edge attributes, and spherical harmonics representations.

### Timestep Embedding and Node Attributes

The method computes the timestep embedding and concatenates it with the node attributes:

```python
node_t_emb = self.t_emb_func(node_t)
node_attr = torch.cat([node_t_emb, data[key].x], 1)
```

### Edge Attributes and Position Embeddings

The method computes the edge attributes and position embeddings, and concatenates them with the edge attributes:

```python
edge_t_emb = node_t_emb[edge_index[0].long()]
edge_attr = torch.cat([edge_t_emb, edge_attr], 1)
...
edge_attr = torch.cat([edge_pos_emb, edge_attr], 1)
```

### Distance Expansion

The method computes the distance expansion using the initialized distance expansion function:

```python
edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))
```

The expanded distances are concatenated with the edge attributes:

```python
edge_attr = torch.cat([edge_length_emb, edge_attr], 1)
```

### Spherical Harmonics

The method computes the spherical harmonics representations for the edge vectors:

```python
edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component').float()
```

The method returns the node attributes, edge indices, edge attributes, and spherical harmonics representations to be used in the forward pass of the model.