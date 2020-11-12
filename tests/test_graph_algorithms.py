### CVPR 2021 Submission #8167. Confidential review copy. Do not distribute.

import graph_algorithms as gu
import numpy as np
import networkx as nx
import random
import pytest

def test_longest_mult_path1(tries=100, n=100, p=0.5):
    """Test computation of multiplicative longest path 
        against in-built longest path algorithm of the
        networkX package. This test uses the logarithm
        of the weights of the graph to test against the
        additive longest-path algorithm in networkX.
        
    We check that:
    - The paths are the same.
    """
    for t in range(tries):
        # Create random DAG
        matrix, _, GLog = create_random_DAG_mat(n, p)
        
        # Compare against networkx longest path
        p1 = gu.dag_longest_path_mult(matrix)
        p2 = np.array(nx.dag_longest_path(GLog, weight='weight'))
        assert (p1==p2).all(),"Unit test failed for longest path algorithm!"

def test_longest_mult_path2(tries=100, n=100, p=0.5):
    """Test computation of multiplicative longest path 
        against a multiplicative version of the 
        longest path algorithm which makes use of (slower)
        networkx in-built functions.
        
    We check that:
    - The paths are the same.
    """
    for t in range(tries):
        # Create random DAG
        matrix, G, _ = create_random_DAG_mat(n, p)
        
        # Compare against networkx longest path
        p1 = gu.dag_longest_path_mult(matrix)
        p2 = np.array(gu.dag_longest_path_nx_mult(G, matrix))
        assert (p1==p2).all(),"Unit test failed for longest path algorithm!"


def create_random_DAG_mat(nrnodes, prob):
    random_graph = nx.fast_gnp_random_graph(nrnodes, prob, directed=True)
    
    # Make weighted DAG
    dag = nx.DiGraph([(u,v,{'weight':float(1+random.random()*10)}) for (u,v) in random_graph.edges() if u<v])

    # Extract  largest connected component:
    Gcc = sorted(nx.weakly_connected_components(dag), key=len, reverse=True)
    component = dag.subgraph(Gcc[0])

    # Get adjacency matrix
    adj_mat = nx.to_numpy_array(component).transpose()
    for row in range(adj_mat.shape[0]):
        adj_mat[row,row:] = 0.0 # Cannot have back edges
    Gr = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

    # Create logarithm matrix
    adj_log_mat = np.zeros_like(adj_mat)
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i,j] > 0:
                adj_log_mat[i,j] = np.log(adj_mat[i,j])
    Glog = nx.from_numpy_array(adj_log_mat, create_using=nx.DiGraph)
    return adj_mat, Gr, Glog

def test_longest_path_pruning(tries=10, n=70, p=0.5, eps=0.01):
    """Test computation of longest path pruning
       to see if the pruning ratio is correct.

    We check that:
    - The pruning method actually prunes the pruning ratio number of nodes
    """
    for t in range(tries):
        # Create random DAG
        matrix, _, _ = create_random_DAG_mat(n, p)
        
        # Draw random pruning ratio
        pratio = random.uniform(0.1, 0.99)

        tot_convs = (matrix != 0).sum()
        pruned_mat = gu.longest_path_prune(matrix, pratio, verbose=False)
        actually_pruned_ratio = pruned_mat.sum()/float(tot_convs)
        assert abs(actually_pruned_ratio-pratio)/pratio < eps,"Unit test failed for longest path pruning!"

def test_longest_path_pruning_skip(tries=10, n=50, p=0.8, eps=0.01):
    """Test computation of longest path pruning
       to see if the pruning ratio is correct
       when unprunable skip connections are involved.

    We check that:
    - The pruning method actually prunes the pruning ratio number of nodes
    """
    for t in range(tries):
        # Create random DAG
        matrix, _, _ = create_random_DAG_mat(n, p)
        
        # Draw random pruning ratio
        pratio = random.uniform(0.1, 0.99)

        # Generate random skip connection matrix
        #  We have to assign edges in current DAG
        #  as skip to avoid possible cycles.
        skips_mat = np.zeros(shape=matrix.shape, dtype=np.bool)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i,j] != 0:
                    if random.random() > 0.9:
                        matrix[i,j] = 1
                        skips_mat[i,j] = True

        tot_convs = (matrix != 0).sum() - skips_mat.sum()
        pruned_mat = gu.longest_path_prune(matrix, pratio, skip_connection_matrix=skips_mat, verbose=False)
        actually_pruned_ratio = pruned_mat.sum()/float(tot_convs)
        assert abs(actually_pruned_ratio-pratio)/pratio < eps,"Unit test failed for longest path pruning!"


if __name__ == '__main__':
    test_longest_path_pruning_skip()
    test_longest_path_pruning()
    test_longest_mult_path1()
    test_longest_mult_path2()
