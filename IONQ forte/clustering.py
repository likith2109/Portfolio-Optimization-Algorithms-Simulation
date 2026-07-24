import numpy as np
import community.community_louvain as community_louvain
import networkx as nx

def rmt_denoising(C_corr, T, N):
    """
    RMT denoising based on Marchenko-Pastur distribution.
    C_corr: Empirical correlation matrix
    T: number of time observations
    N: number of assets
    """
    vals, vecs = np.linalg.eigh(C_corr)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Marchenko-Pastur upper bound
    q = N / T
    lambda_plus = (1 + np.sqrt(q))**2
    
    # Identify modes
    # Global mode is usually the largest eigenvalue
    # Noise modes are those <= lambda_plus
    # Structured modes (C_star) are the ones in between
    
    C_star = np.zeros_like(C_corr)
    for i in range(1, len(vals)):
        if vals[i] > lambda_plus:
            v = vecs[:, i].reshape(-1, 1)
            C_star += vals[i] * (v @ v.T)
            
    # Optional: ensure diagonal is 1 for a correlation matrix, 
    # but here we just need the structure for clustering.
    return C_star

def hardware_aware_clustering(C_corr, Q_max, T, N):
    """
    Algorithm 1: Hardware-Aware Clustering (GetClusters)
    """
    # 1. RMT denoising
    C_star = rmt_denoising(C_corr, T, N)
    
    # 2. Initial clustering using Louvain
    # We build a graph where edge weights are absolute values of C_star
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    
    for i in range(N):
        for j in range(i+1, N):
            weight = np.abs(C_star[i, j])
            if weight > 1e-5: # threshold to avoid dense noisy graph
                G.add_edge(i, j, weight=weight)
                
    partition = community_louvain.best_partition(G, weight='weight')
    
    # Group assets by community
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
        
    initial_clusters = list(communities.values())
    
    final_clusters = []
    
    # 3. Size-bounded splitting
    for C_u in initial_clusters:
        if len(C_u) <= Q_max:
            final_clusters.append(C_u)
        else:
            R = set(C_u)
            while len(R) > 0:
                if len(R) <= Q_max:
                    final_clusters.append(list(R))
                    break
                
                # Compute restricted matrix C_R
                R_list = list(R)
                # Compute degrees di = sum_j |(C_R)_{ij}| - 1 (in our case we just sum |C_star_ij| for j in R)
                degrees = []
                for i in R_list:
                    deg = sum(np.abs(C_star[i, j]) for j in R_list if j != i)
                    degrees.append((deg, i))
                
                # Select seed i* with max degree
                degrees.sort(reverse=True, key=lambda x: x[0])
                i_star = degrees[0][1]
                
                # Compute similarities s_j = |(C_R)_{i*, j}|
                similarities = []
                for j in R_list:
                    if j != i_star:
                        similarities.append((np.abs(C_star[i_star, j]), j))
                
                similarities.sort(reverse=True, key=lambda x: x[0])
                
                # Form new cluster of size Q_max
                C_new = [i_star]
                for k in range(Q_max - 1):
                    if k < len(similarities):
                        C_new.append(similarities[k][1])
                        
                final_clusters.append(C_new)
                
                # Update R
                R = R - set(C_new)
                
    return final_clusters

if __name__ == "__main__":
    # Test with dummy data
    N = 30
    T = 1000
    np.random.seed(42)
    dummy_returns = np.random.randn(T, N)
    C_corr = np.corrcoef(dummy_returns, rowvar=False)
    
    clusters = hardware_aware_clustering(C_corr, Q_max=10, T=T, N=N)
    print(f"Number of clusters: {len(clusters)}")
    for i, c in enumerate(clusters):
        print(f"Cluster {i} size: {len(c)}, elements: {c}")
