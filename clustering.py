import sys
import numpy as np
import networkx as nx
import sklearn.cluster as cluster
import sklearn.manifold as manifold
import scipy.sparse as sp

SEED = 1

def spectral_clusters(g, n_clusters):
    spectral_clustering = cluster.SpectralClustering(n_clusters = n_clusters,
                                affinity = 'precomputed',
                                random_state = SEED)
    
    X = make_dense_adj_matrix(get_canonical(g))
    labels = spectral_clustering.fit_predict(X)
    return labels

def get_canonical(g):
    """
    Parameters
    ----------
    g: undirected graph (networkx.Graph)
    
    Return
    ------
    A canonical version of the input graph.
    """
    canonical =  nx.Graph()
    node_inv_idx = dict([(u, i) for i, u in enumerate(g.nodes())])
    edge_set = g.edges(data = True)
    for (u, v, d) in edge_set:
        i = node_inv_idx[u]
        j = node_inv_idx[v]
        canonical.add_edge(i, j, d)
    return canonical

def make_dense_adj_matrix(g):
    """
    g: networkx.Graph()
    
    return adjacency (nodes x nodes) matrix
    """
    X = np.matrix(np.zeros((len(g), len(g))))
    it = g.adjacency_iter()
    for adj_list in it:
        u = adj_list[0]
        for v in adj_list[1].keys():
            X[u, v] = 1
    assert(is_symmetric(X))
    return X

def is_symmetric(M):
    xdim, ydim = M.shape
    for x in range(xdim):
        for y in range(ydim):
            if M[x,y] != M[y,x]:
                return False
    return True