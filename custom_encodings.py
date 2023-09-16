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