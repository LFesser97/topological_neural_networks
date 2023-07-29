import heapq
import importlib
import math
import time
import torch
import pandas as pd
# torch.multiprocessing.set_start_method('spawn')
_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import multiprocessing as mp
from functools import lru_cache

import networkit as nk
import networkx as nx
import numpy as np
import ot

from .util import logger


def _compute_afrc_edge(G: nx.Graph, ni: int, nj: int, t_num: int) -> float:
    """
    Computes the Augmented Forman-Ricci curvature of a given edge

    Parameters
    ----------
    G" Graph

    ni: node i

    nj: node j

    m: number of triangles containing the edge between node i and j

    Returns
    -------
    afrc : AFRC of the edge connecting nodes i and j
    """
    afrc = 4 - G.degree(ni) - G.degree(nj) + 3 * t_num
    return afrc


def _compute_afrc_edges(G: nx.Graph, weight="weight", edge_list=[]) -> dict:
    """Compute Augmented Forman-Ricci curvature for edges in  given edge lists.

    Parameters
    ----------
    G : A given directed or undirected NetworkX graph.
    
    weight : The edge weight used to compute the AFRC. (Default value = "weight")
    
    edge_list : The list of edges to compute the AFRC, set to [] to run for all edges in G.

    Returns
    -------
    output : A dictionary of AFRC values keyed by edge tuples,
        e.g. {(1,2): 1, (2,3): -2}
    """
    if edge_list == []:
        edge_list = G.edges()

    # Compute AFRC for all edges
    edge_afrc = {}
    for edge in edge_list:
        num_triangles = G.edges[edge]["triangles"]
        edge_afrc[edge] = _compute_afrc_edge(G, edge[0], edge[1], num_triangles)

    return edge_afrc


def _simple_cycles(G: nx.Graph, limit: int = 3):
    """
    Find simple cycles (elementary circuits) of a graph up to a given length.

    Parameters
    ----------
    G : An undirected graph.

    limit : Maximum length of cycles to find plus one.

    Returns
    -------
    cycles : A generator that produces lists of nodes, one for each cycle.
    """
    subG = type(G)(G.edges())
    sccs = list(nx.strongly_connected_components(subG))
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path = [startnode]
        blocked = set()
        blocked.add(startnode)
        stack = [(startnode, list(subG[startnode]))]

        while stack:
            thisnode, nbrs = stack[-1]

            if nbrs and len(path) < limit:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                elif nextnode not in blocked:
                    path.append(nextnode)
                    stack.append((nextnode, list(subG[nextnode])))
                    blocked.add(nextnode)
                    continue
            if not nbrs or len(path) >= limit:
                blocked.remove(thisnode)
                stack.pop()
                path.pop()
        subG.remove_node(startnode)
        H = subG.subgraph(scc)
        sccs.extend(list(nx.strongly_connected_components(H)))


def _compute_afrc(G: nx.Graph, weight: str="weight") -> nx.Graph:
    """
    Compute Augmented Forman-Ricci curvature for a given NetworkX graph.

    Parameters
    ----------
    G : A given directed or undirected NetworkX graph.

    weight : The edge weight used to compute the AFRC. (Default value = "weight")

    Returns
    -------
    G : A NetworkX graph with "AFRC" on edges and nodes.
    """
    # Compute AFRC for all edges
    edge_afrc = _compute_afrc_edges(G, weight=weight)

    # Assign edge AFRC from result to graph G
    nx.set_edge_attributes(G, edge_afrc, "AFRC")

    # Compute node AFRC
    for n in G.nodes():
        afrc_sum = 0
        if G.degree(n) > 1:
            for nbr in G.neighbors(n):
                if 'afrc' in G[n][nbr]:
                    afrc_sum += G[n][nbr]['afrc']

            # Assign the node AFRC to be the average of its incident edges
            G.nodes[n]["AFRC"] = afrc_sum / G.degree(n)

    return G 


class FormanRicci:
    """
    A class to compute Forman-Ricci curvature for a given NetworkX graph.
    """

    def __init__(self, G: nx.Graph, weight: str="weight"):
        """
        Initialize a container for Forman-Ricci curvature.
        """
        self.G = G
        self.weight = weight
        self.triangles = []     

        # Compute triangles
        for cycle in _simple_cycles(self.G.to_directed(), 4): # Only compute 2 and 3 cycles
            if len(cycle) == 3:
                self.triangles.append(cycle)

        for edge in list(self.G.edges()):
            u, v = edge
            self.edges[edge]["triangles"] = len([cycle for cycle in self.triangles if u in cycle and v in cycle])/2


        if not nx.get_edge_attributes(self.G, weight):
            logger.info('Edge weight not found. Set weight to 1.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][weight] = 1.0

        self_loop_edges = list(nx.selfloop_edges(self.G))
        if self_loop_edges:
            logger.info('Self-loop edge detected. Removing %d self-loop edges.' % len(self_loop_edges))
            self.G.remove_edges_from(self_loop_edges)


    def compute_afrc_edges(self, edge_list=None):
        """
        Compute Augmented Forman-Ricci curvature for edges in  given edge lists.

        Parameters
        ----------
        edge_list : The list of edges to compute the AFRC, set to None to run for all edges in G.

        Returns
        -------
        output : A dictionary of AFRC values keyed by edge tuples,
            e.g. {(1,2): 1, (2,3): -2}
        """
        if edge_list is None:
            edge_list = self.G.edges()
        else:
            edge_list = list(edge_list)

        return _compute_afrc_edges(self.G, self.weight, edge_list)
    

    def compute_ricci_curvature(self) -> nx.Graph:
        """
        Compute AFRC of edges and nodes.

        Returns
        -------
        G : A NetworkX graph with "AFRC" on nodes and edges.
        """
        self.G = _compute_afrc(self.G, self.weight)

        return self.G