import numpy as np

def compute_obj(x, Q, q):
    return x.T @ Q @ x + q.T @ x

def compute_grad(x, Q, q):
    return 2 * Q @ x + q

def two_phase_local_search(x_init, Q, q, K, max_iters=100, patience=5):
    """
    Algorithm 4: Two-phase local search with cardinality constraint.
    """
    x = np.copy(x_init).astype(float)
    N = len(x)
    w = np.sum(x)
    
    # Phase 1: Cardinality repair
    if w != K:
        g = compute_grad(x, Q, q)
        if w > K:
            # Too many 1s, need to flip to 0
            S = np.where(x == 1)[0]
            # sort S by gi descending
            S_sorted = S[np.argsort(-g[S])]
            flips = int(w - K)
            x[S_sorted[:flips]] = 0
        else:
            # Too many 0s, need to flip to 1
            S = np.where(x == 0)[0]
            # sort S by gi ascending
            S_sorted = S[np.argsort(g[S])]
            flips = int(K - w)
            x[S_sorted[:flips]] = 1
            
    # Phase 2: Swap local search
    f = compute_obj(x, Q, q)
    t = 0
    p = 0
    
    S0 = np.where(x == 0)[0].tolist()
    S1 = np.where(x == 1)[0].tolist()
    
    maxSwaps = min(100, len(S0) * len(S1))
    if maxSwaps == 0:
        return x
        
    while t < max_iters and p < patience:
        improved = False
        s = 0
        
        # randomly permute S0 and S1
        np.random.shuffle(S0)
        np.random.shuffle(S1)
        
        for i in S0:
            if improved or s >= maxSwaps:
                break
                
            for j in S1:
                if s >= maxSwaps:
                    break
                    
                # Try swap
                x[i] = 1
                x[j] = 0
                s += 1
                
                f_prime = compute_obj(x, Q, q)
                if f_prime < f:
                    f = f_prime
                    # Update S0 and S1
                    S0.remove(i)
                    S0.append(j)
                    S1.remove(j)
                    S1.append(i)
                    improved = True
                    p = 0
                    break
                else:
                    # Revert swap
                    x[i] = 0
                    x[j] = 1
                    
        if not improved:
            p += 1
        t += 1
        
    return x
