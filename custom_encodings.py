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

from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.FormanRicci4 import FormanRicci4


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
        

    def compute_orc(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        orc = OllivierRicci(graph, alpha=0, verbose="ERROR")
        orc.compute_ricci_curvature()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_orc = [min([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_orc = [max([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_orc = [np.mean([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_orc = [np.std([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_orc = [np.median([orc.G.edges[node, neighbor]["ricciCurvature"]["rc_curvature"] for neighbor in neighbors[node]]) for node in graph.nodes()]
                                                                      
        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_orc, max_orc, mean_orc, std_orc, median_orc]).T
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

    def compute_afrc_3(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        afrc_3 = FormanRicci(graph)
        afrc_3.compute_ricci_curvature()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_afrc_3 = [min([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_afrc_3 = [max([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_afrc_3 = [np.mean([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_afrc_3 = [np.std([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_afrc_3 = [np.median([afrc_3.G[node][neighbor]['AFRC'] for neighbor in neighbors[node]]) for node in graph.nodes()] 
                                                                      
        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_afrc_3, max_afrc_3, mean_afrc_3, std_afrc_3, median_afrc_3]).T
    
        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data
    

    def compute_afrc_4(self, data: Data) -> Data:
        graph = to_networkx(data)
        
        # compute ORC
        afrc_4 = FormanRicci4(graph)
        afrc_4.compute_afrc_4()
    
        # get the neighbors of each node
        neighbors = [list(graph.neighbors(node)) for node in graph.nodes()]
    
        # compute the min, max, mean, std, and median of the ORC for each node
        min_afrc_4 = [min([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        max_afrc_4 = [max([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        mean_afrc_4 = [np.mean([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        std_afrc_4 = [np.std([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()]
        median_afrc_4 = [np.median([afrc_4.G[node][neighbor]['AFRC_4'] for neighbor in neighbors[node]]) for node in graph.nodes()] 
                                                                      
        # create a torch.tensor of dimensions (num_nodes, 5) containing the min, max, mean, std, and median of the ORC for each node
        lcp_pe = torch.tensor([min_afrc_4, max_afrc_4, mean_afrc_4, std_afrc_4, median_afrc_4]).T

        # add the local degree profile positional encoding to the data object
        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat((data.x, lcp_pe), dim=-1)
        else:
            data.x = torch.cat(lcp_pe, dim=-1)

        return data