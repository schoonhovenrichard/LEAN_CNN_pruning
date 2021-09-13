import torch
import msd_pytorch as mp
import numpy as np
import torch.nn as nn
import torch.nn.utils.prune as prune
import networkx as nx

# For Rust implementation:
from n_longest_paths import mark_longest_paths
from n_longest_paths import fast_graph_mark_longest_paths


def longest_path_prune_fast(adj_list, perc, ignore_edges_arr=None):
    r"""Fast RUST code for LEAN path extraction. Given an adjacency list
     of the network graph, returns a boolean array of nodes to be pruned
     by keeping the longest value path. Uses the fast Rust implementation 
     made by Allard Hendriksen

    Args:
        - adj_list (np.array): a nr_edges x 3 numpy array with each entry
                        [input_index, output_index, edge_value].
        - perc (float): fraction of convolutions to keep.
        - ignore_edges_arr (np.array): A 1D array of length number of
                        edges filled with booleans. The i-th entry is
                        True if the i-th edge should not be prunable.
    """
    tot_convs = adj_list.shape[0]
    print(tot_convs)
    if ignore_edges_arr is not None:
        tot_convs -= ignore_edges_arr[:,2].sum()
        ignore_edges = ignore_edges_arr[:,2]
    else:
        ignore_edges = None
    convs_to_prune = int((1-perc)*tot_convs)

    srcs = adj_list[:,0]
    tgts = adj_list[:,1]
    lengths = adj_list[:,2]

    def wrap_mark_longest_paths(src, dest, length, num_to_mark):
        return mark_longest_paths(src.astype(np.uint64), dest.astype(np.uint64), length.astype(np.float32), num_to_mark)
    def wrap_skippable(src, dest, length, num_to_mark, length_type="multiplicative", skippable=None):
        if skippable is not None:
            skippable = skippable.astype(bool)
        return fast_graph_mark_longest_paths(
            src.astype(np.uint64),
            dest.astype(np.uint64),
            length.astype(np.float32),
            num_to_mark,
            length_type,
            skippable,
        )
    path = wrap_skippable(srcs, tgts, lengths, convs_to_prune, "multiplicative", ignore_edges)

    pruned_convs = np.where(path)
    print("Graph algorithm returned", pruned_convs[0].shape[0], "convolutions to keep.")
    return path


def longest_path_prune(norm_adj_matrix, perc, skip_connection_matrix=None, verbose=False):
    r"""Given an adjacency matrix of the network graph, returns a boolean array
        of nodes to be pruned by keeping the longest value path.

    Args:
        - norm_adj_matrix (np.ndarray): Adjacency matrix describing the CNN
            graph as a DAG. The metrics/norms of the convolutions are given
            on the edges.
        - perc (float): fraction of convolutions to be pruned in this pass,
            e.g. 0.2 means that 20% of the convolutions will be removed. 
            This means we will extract longest multiplicative paths until we 
            have extracted 80% of the nodes in the graph.
        - c_in (int): Number of input channels to the CNN. #TODO: REMOVE
        - skip_connection_matrix (np.ndarray (bool)): Matrix of booleans
            with True when an edge represents an unprunable skip connection.
        - verbose (bool): Algorithm is silent, or prints progress reports.
    """
    adj_matrix = norm_adj_matrix.copy()
    pruned = np.ones(shape=(adj_matrix.shape[0], adj_matrix.shape[1]),dtype=np.uint8)
    pruned[adj_matrix == 0] = 0
    tot_convs = (adj_matrix != 0).sum()
    if skip_connection_matrix is not None:
        #this is to avoid counting e.g. average pooling layers which are set to zero
        tot_convs -= (skip_connection_matrix[adj_matrix != 0]).sum()
        pruned[skip_connection_matrix] = 0
    if pruned.sum() != tot_convs:
        print(pruned.sum(), tot_convs, (adj_matrix != 0).sum(), skip_connection_matrix is None)
        raise Exception("Investigate this")
    convs_pruned = pruned.sum()/float(tot_convs)
    count_single_edge_paths = 0
    count_pruned = 0
    convs_to_prune = perc*tot_convs
    if verbose:
        print("Convolution to prune:", convs_to_prune,", percentage", perc)
    escape_bool = False
    while convs_pruned > perc:
        prune_path = dag_longest_path_mult(adj_matrix)
        full_skip_path = True
        if len(prune_path) == 2:
            if adj_matrix[prune_path[0], prune_path[1]] < 1:
                print(prune_path, adj_matrix[prune_path[0], prune_path[1]])
                escape_bool = True
        for k in range(len(prune_path)-1):
            nstart = prune_path[k]
            nend = prune_path[k+1]
            if skip_connection_matrix is not None:
                if not skip_connection_matrix[nstart,nend]:
                    full_skip_path = False
                    if pruned[nstart,nend] == 0:
                        raise Exception("Should already be pruned, how was it still in the graph?")
                    pruned[nstart,nend] = 0
                    adj_matrix[nstart,nend] = 0
                    count_pruned += 1
                    convs_pruned = pruned.sum()/float(tot_convs)
                else:
                    if adj_matrix[nstart, nend] == 0:
                        raise Exception("Non-existent skip connection pruned!")
            else:
                if pruned[nstart,nend] == 0:
                    raise Exception("Should already be pruned, how was it still in the graph?")
                pruned[nstart,nend] = 0
                adj_matrix[nstart,nend] = 0
                count_pruned += 1
                convs_pruned = pruned.sum()/float(tot_convs)
            if convs_pruned <= perc:
                break
            if verbose:
                print('{}\r'.format(convs_pruned), end="")

        if full_skip_path and skip_connection_matrix is not None:
            if verbose:
                print("BREAKING OUT because have found only a path consisting of skip connections.")
            break
        if len(prune_path) == 2:
            count_single_edge_paths += 1
            if escape_bool:
                if verbose:
                    print("BREAKING OUT because removed single edge with weight < 1")
                break
        else:
            count_single_edge_paths = 0
        if count_single_edge_paths >= 1000:
            if verbose:
                print("BREAKING OUT because been removing consecutive single edge paths")
            break
        if len(prune_path) == 1:
            if verbose:
                print("BREAKING OUT because only negative weights in edges left.")
            break

    if verbose:
        print("Convolution to prune:", convs_to_prune, "Kept {0} with path method, {1:.4f}%".format(count_pruned, 100*count_pruned/float(tot_convs - convs_to_prune)))
    # Need to switch off the skip connections again when computing threshold
    if skip_connection_matrix is not None:
        adj_matrix[skip_connection_matrix] = 0
    if convs_pruned > perc:
        # Found many consecutive single edge paths, do the rest of the pruning by efficient edge pruning
        all_norms = adj_matrix[adj_matrix!=0].copy().flatten()
        reduction_perc = perc/float(convs_pruned)
        threshold = np.percentile(all_norms, 100*reduction_perc)
        pruned[adj_matrix >= threshold] = 0
        if verbose:
            print("Threshold", threshold)
    convs_pruned = pruned.sum()/float(tot_convs)
    if verbose:
        print(convs_pruned, pruned.sum(), convs_to_prune)
    return pruned

def dag_longest_path_nx_mult(G, matrix):
    r"""This is one of the helper functions to test the correctness
        of the computationally efficient variant of the longest
        multiplicative path algorithm. This implementation uses
        topological sort from NetworkX but performs the DP itself.

    Args:
        - G (networkX.DiGraph): Input graph.
        - matrix (np.ndarray): Associated adjacency matrix.
    """
    if not G:
        return []
    if not G.is_directed():
        raise Exception("Not implemented for undirected graphs!")
    topo_order = nx.topological_sort(G)
    n = matrix.shape[0]
    dss = matrix.transpose().tolist()
    dist2 = [list(x) for x in zip([None]*n, [None]*n)]

    for v in topo_order:
        maxu = (None,None)
        preds = matrix[:,v].nonzero()[0].tolist()
        for u in G.pred[v]:
            if dist2[u][0] != 0:
                d = dist2[u][0] * dss[v][u]
            else:
                d = dss[v][u]
            if maxu[0] is None:
                maxu = (d,u)
            elif d > maxu[0]:
                maxu = (d,u)
        if maxu[0] is None:
            maxu = (0,v)
        if maxu[0] >= 1:
            dist2[v][0] = maxu[0]
            dist2[v][1] = maxu[1]
        else:
            dist2[v][0] = 0
            dist2[v][1] = v
    u = None
    v = argmax(dist2)
    path = []
    while u != v:
        path.append(v)
        u = v
        v = int(dist2[v][1])
    path.reverse()
    return path


def dag_longest_path_mult(matrix: np.ndarray):
    r"""This is a low-level, computationally efficient implementation
        of dag_longest_path_nx_mult (longest multiplicative path 
        algorithm). It only uses numpy arrays, and numba.jit for speedup.
        
        NOTE: This implementation has no safeguards to check for
                the DAG proprety.

    Args:
        - matrix (np.ndarray): Adjacency matrix of graph.
    """
    n = matrix.shape[0]
    # Efficiently compute predecessors and descendants of nodes.
    preds, descs = row_col_idx_nonzero_nb(matrix)

    # Efficiently determine topological order
    topo_order = []
    indegree_map = []
    for v in range(len(preds)):
        indegree_map.append(preds[v].shape[0])
    indegree_map = np.array(indegree_map)
    zero_indegree = [v for v in range(len(preds)) if preds[v].size == 0]
    while zero_indegree:
        node = zero_indegree.pop()
        children = descs[node]
        indegree_map[children] -= 1
        zero_indegree += list(children[np.where(indegree_map[children] == 0)[0]])
        topo_order.append(node)

    # Do the dynamic programming to figure out longest path
    dss = matrix.transpose()
    distancesDP = np.zeros(shape=(n,2),dtype=np.float64)
    for v in topo_order:
        predv = preds[v]
        maxu = (0,v)
        if len(predv) > 0:
            dssv = dss[v][predv]
            distancesv = distancesDP[predv,0]
            ds = dssv * distancesv # This is different because multiplicative
            ds[distancesv == 0] = dssv[distancesv == 0]
            ux = np.argmax(ds)
            maxu = (ds[ux],predv[ux])
        if maxu[0] >= 1:
            distancesDP[v][0] = maxu[0]
            distancesDP[v][1] = maxu[1]
        else:
            distancesDP[v][0] = 0
            distancesDP[v][1] = v
    
    # Return the longest multiplicative path
    u = -1
    v = int(np.argmax(distancesDP[:,0]))
    path = []
    while u != v:
        path.append(v)
        u = v
        v = int(distancesDP[v][1])
    path.reverse()
    return np.array(path)


###################################################################
### Next comes low-level implementation for performance purposes###
###################################################################


def convert_matr_to_adjlist(adjmat, ignoremat=None):
    r""" Helper function for the fast Rust implementation.

    Converts the adjacency matrix representation of a graph
     to an adjacency list representation.

    Args:
        - adjmat (np.ndarray): Adjacency matrix of graph.
        - ignoremat (np.ndarray): For LEAN pruning, there are
            edges that are marked as unprunable, these must be
            taken into account when running the graph algorithms.
    """
    print("Converting adjacency matrix to List format...")
    #edge_dtype = [('from', '<i8'), ('to', '<i8'), ('norm', '<f4')]# Compute edges
    edges = np.where(adjmat != 0)
    adj_list = []
    if ignoremat is not None:
        ignore_list = []
    for edge in range(edges[0].shape[0]):
        outidx = edges[0][edge]
        inidx = edges[1][edge]
        val = adjmat[outidx, inidx]
        if outidx <= inidx:
            print(inidx, outidx, val)
            raise Exception("PAUSE")
        if val == 0:
            print(inidx, outidx, val)
            raise Exception("PAUSE")
        adj_list.append([inidx, outidx, val])
        if ignoremat is not None:
            ival = ignoremat[outidx, inidx]
            ignore_list.append([inidx, outidx, ival])
    adj_list.sort(key=lambda tup: tup[0])
    adjarr = np.array(adj_list)
    #adjarr = np.array(adj_list, dtype=edge_dtype)
    if ignoremat is not None:
        ignore_list.sort(key=lambda tup: tup[0])
        ignorearr = np.array(ignore_list)
    if ignoremat is None:
        return adjarr
    else:
        return adjarr, ignorearr


# Code below is to make Python function as fast as possible,
#  it is till not nearly as fast as the Rust implementation.

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def topo_sort(preds, descs):
    indegree_map = {v: len(preds[v]) for v in range(len(preds)) if len(preds[v]) > 0}
    zero_indegree = [v for v in range(len(preds)) if len(preds[v]) == 0]
    order = []
    while zero_indegree:
        node = zero_indegree.pop()
        for child in descs[node]:
            try:
                indegree_map[child] -= 1
            except KeyError:
                raise RuntimeError("Graph changed during iteration")
            if indegree_map[child] == 0:
                zero_indegree.append(child)
                del indegree_map[child]
        order.append(node)
    if indegree_map:
        raise Exception("Graph contains a cycle or graph changed during iteration")
    return order

import numba as nb
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix

@nb.jit(nopython=True, nogil=True, cache=True, forceobj=False,
     parallel=False, fastmath=True, locals={})
def cumsum(arr: np.ndarray):
    result = np.empty_like(arr)
    cumsum = result[0] = arr[0]
    for i in range(1, len(arr)):
        cumsum += arr[i]
        result[i] = cumsum
    return result

@nb.jit(nopython=True, nogil=True, cache=True, forceobj=False,
     parallel=False, fastmath=True, locals={})
def count_nonzero(arr: np.ndarray):
    arr = arr.ravel()
    n = 0
    for x in arr:
        if x != 0:
            n += 1
    return n

@nb.jit(nopython=True, nogil=True, cache=True, forceobj=False,
     parallel=True, fastmath=True, locals={})
def row_col_nonzero_nb(arr: np.ndarray):
    n, m = arr.shape
    max_k = count_nonzero(arr)
    indices = np.empty((2, max_k), dtype=np.uint64)
    i_offset = np.zeros(n + 1, dtype=np.uint64)
    j_offset = np.zeros(m + 1, dtype=np.uint64)
    n, m = arr.shape
    k = 0
    for i in range(n):
        for j in range(m):
            if arr[i, j] != 0:
                indices[0, k] = i
                indices[1, k] = j
                i_offset[i + 1] += 1
                j_offset[j + 1] += 1
                k += 1
    return indices, cumsum(i_offset), cumsum(j_offset)

def row_col_idx_nonzero_nb(arr: np.ndarray):
    (ii, jj), jj_split, ii_split = row_col_nonzero_nb(arr)
    ii_ = np.argsort(jj)
    ii = ii[ii_]
    return np.split(ii, ii_split[1:-1]), np.split(jj, jj_split[1:-1])

def row_col_idx_sep(arr):
    return (
        [arr[:, j].nonzero()[0] for j in range(arr.shape[1])],
        [arr[i, :].nonzero()[0] for i in range(arr.shape[0])],)

def row_col_idx_sparse_lil(arr):
    lil_mat = lil_matrix(arr)
    p = lil_mat.T.rows
    d = lil_mat.rows
    for k in range(len(p)):
        p[k] = np.array(p[k], dtype=np.int)
        d[k] = np.array(d[k], dtype=np.int)
    return p, d

def row_col_idx_zip(arr):
    n, m = arr.shape
    ii = [[] for _ in range(n)]
    jj = [[] for _ in range(m)]
    x, y = np.nonzero(arr)
    for i, j in zip(x, y):
        ii[i].append(j)
        jj[j].append(i)
    for k in range(len(ii)):
        ii[k] = np.array(ii[k], dtype=np.int)
        jj[k] = np.array(jj[k], dtype=np.int)
    return jj, ii

def row_col_idx_sparse_coo(arr):
    coo_mat = coo_matrix(arr)
    csr_mat = coo_mat.tocsr()
    csc_mat = coo_mat.tocsc()
    return (
        np.split(csc_mat.indices, csc_mat.indptr)[1:-1],
        np.split(csr_mat.indices, csr_mat.indptr)[1:-1],)
