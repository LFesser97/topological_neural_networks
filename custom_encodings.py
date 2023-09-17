"""
custom_encodings.py

This file contains the following custom encodings:
    - shortest path relative positional encodings
    - Ollivier-Ricci curvature structural encodings
    - Ollivier-Ricci curvature shortest path relative positional encodings
"""

import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform

from GraphRicciCurvature.OllivierRicci import OllivierRicci


class ShortestPathGenerator:
    def __init__(self, directed=False):
        self.directed = directed

    def __call__(self, data):
        row = data.edge_index[0].numpy()
        col = data.edge_index[1].numpy()
        weight = np.ones_like(row)

        graph = csr_matrix((weight, (row, col)), shape=(len(data.x), len(data.x)))
        dist_matrix, _ = shortest_path(
            csgraph=graph, directed=self.directed, return_predecessors=True
        )

        data["distance"] = torch.from_numpy(dist_matrix)
        return data
    

class OneHotEdgeAttr:
    def __init__(self, max_range=4) -> None:
        self.max_range = max_range

    def __call__(self, data):
        x = data["edge_attr"]
        if len(x.shape) == 1:
            return data

        offset = torch.ones((1, x.shape[1]), dtype=torch.long)
        offset[:, 1:] = self.max_range
        offset = torch.cumprod(offset, dim=1)
        x = (x * offset).sum(dim=1)
        data["edge_attr"] = x
        return data
    

class LocalCurvatureProfile(BaseTransform):
    """
    This class computes the local curvature profile positional encoding for each node in a graph.
    """
    def __init__(self, attr_name = 'lcp_pe'):
        self.attr_name = attr_name
        

    def forward(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        for node in graph.nodes():
            for neighbor in neighbors[node]:
                print(orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"])
                print(type(orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"]))

        min_orc = [min([orc.G.edges[node, neighbor]["ricciCurvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_orc = [max([orc.G.edges[node, neighbor]["ricciCurvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_orc = [np.mean([orc.G.edges[node, neighbor]["ricciCurvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_orc = [np.std([orc.G.edges[node, neighbor]["ricciCurvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_orc = [np.median([orc.G.edges[node, neighbor]["ricciCurvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]

        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_orc, max_orc, mean_orc, std_orc, median_orc]).T
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data