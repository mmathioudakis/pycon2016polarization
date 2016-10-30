import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.manifold as manifold
import scipy.sparse as sp

SEED = 1

def spectral_clusters(g, n_clusters):
    spectral_clustering = cluster.SpectralClustering(n_clusters = n_clusters,
                                affinity = 'precomputed',
                                random_state = SEED)
    
    X = make_sparse_adj_matrix(g)
    labels = spectral_clustering.fit_predict(X)
    return labels

def make_sparse_adj_matrix(g):
    """
    g: networkx.Graph()
    
    return adjacency matrix in scipy.csr sparse format
    """
    rows = []
    columns = []
    it = g.adjacency_iter()
    for adj_list in it:
        u = adj_list[0]
        for v in adj_list[1].keys():
            rows.append(u)
            columns.append(v)
    data = np.ones(len(rows))
    X = sp.csr_matrix((data, (rows, columns)))
    assert(is_symmetric(X))
    return X

def is_symmetric(M):
    xdim, ydim = M.shape
    for x in range(xdim):
        for y in range(ydim):
            if M[x,y] != M[y,x]:
                return False
    return True
