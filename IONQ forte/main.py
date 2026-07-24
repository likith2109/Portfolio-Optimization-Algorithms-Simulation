import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_collection import fetch_data
from clustering import hardware_aware_clustering
from quantum_optimizer import portfolio_to_ising, run_bf_dcqo
from local_search import two_phase_local_search

def evaluate_global_portfolio(x, mu, C):
    gamma = 0.0
    risk = x.T @ C @ x
    ret = mu.T @ x
    obj = (gamma - 1) / 2 * ret + (gamma + 1) / 2 * risk
    return obj, risk, ret

def main():
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "V", 
               "PG", "UNH", "HD", "MA", "DIS", "PYPL", "VZ", "ADBE", "NFLX", "INTC"]
    # We use 20 tickers for faster execution
    mu, C, C_corr, actual_tickers, log_returns = fetch_data(tickers, "2020-01-02", "2023-12-31")
    N = len(actual_tickers)
    K_global = N // 2
    
    # 1. Clustering
    Q_max = 8
    print(f"Running clustering with Q_max={Q_max}...")
    clusters = hardware_aware_clustering(C_corr, Q_max, 1000, N)
    print(f"Formed {len(clusters)} clusters: {[len(c) for c in clusters]}")
    
    # 2. Solve subproblems
    cluster_candidates = []
    
    for c_idx, cluster in enumerate(clusters):
        print(f"\nSolving cluster {c_idx} with {len(cluster)} assets...")
        K_sub = len(cluster) // 2
        
        # Extract subproblem mu and C
        mu_sub = mu[cluster]
        C_sub = C[np.ix_(cluster, cluster)]
        
        # Build QUBO and Ising
        h, J, offset = portfolio_to_ising(mu_sub, C_sub, K_sub, gamma=0.0)
        
        # Q and q for local search on subproblem
        Q_sub = (0.0 + 1) / 2 * C_sub
        q_sub = (0.0 - 1) / 2 * mu_sub
        
        # Run BF-DCQO
        # Using a short dt and few steps for fast simulation
        samples = run_bf_dcqo(h, J, len(cluster), R=3, nl=5, nsteps=2, dt=0.1)
        
        print(f"BF-DCQO found {len(samples)} unique candidates.")
        
        post_processed_cands = []
        for bitstring, _ in samples:
            x_cand = np.array([1 if b == '0' else 0 for b in bitstring[::-1]])
            x_improved = two_phase_local_search(x_cand, Q_sub, q_sub, K_sub, max_iters=20)
            post_processed_cands.append(x_improved)
            
        cluster_candidates.append(post_processed_cands)
        
    # 3. Recombination
    print("\nRecombining cluster candidates...")
    num_global_cands = 100
    global_candidates = []
    for _ in range(num_global_cands):
        global_x = np.zeros(N)
        for c_idx, cluster in enumerate(clusters):
            # randomly pick a candidate from this cluster
            if len(cluster_candidates[c_idx]) > 0:
                cand_idx = np.random.randint(len(cluster_candidates[c_idx]))
                sub_x = cluster_candidates[c_idx][cand_idx]
                for i, asset_idx in enumerate(cluster):
                    global_x[asset_idx] = sub_x[i]
            else:
                # Fallback if no valid candidates
                for i, asset_idx in enumerate(cluster):
                    global_x[asset_idx] = np.random.choice([0, 1])
        global_candidates.append(global_x)
        
    # 4. Global Post-processing
    print("Running global two-phase local search...")
    Q_global = (0.0 + 1) / 2 * C
    q_global = (0.0 - 1) / 2 * mu
    
    refined_candidates = []
    raw_objs = []
    refined_objs = []
    
    for x in global_candidates:
        raw_obj, _, _ = evaluate_global_portfolio(x, mu, C)
        raw_objs.append(raw_obj)
        
        x_refined = two_phase_local_search(x, Q_global, q_global, K_global, max_iters=50)
        refined_candidates.append(x_refined)
        
        ref_obj, _, _ = evaluate_global_portfolio(x_refined, mu, C)
        refined_objs.append(ref_obj)
        
    # Calculate returns and risks for the scatter plot
    refined_risks = []
    refined_returns = []
    for x in refined_candidates:
        _, risk, ret = evaluate_global_portfolio(x, mu, C)
        refined_risks.append(risk)
        refined_returns.append(ret)
        
    # Reference Random Search
    print("Running random search reference...")
    random_candidates = []
    random_objs = []
    for _ in range(num_global_cands):
        x = np.zeros(N)
        x[np.random.choice(N, K_global, replace=False)] = 1
        x_refined = two_phase_local_search(x, Q_global, q_global, K_global, max_iters=50)
        ref_obj, risk, ret = evaluate_global_portfolio(x_refined, mu, C)
        random_candidates.append({'obj': ref_obj, 'risk': risk, 'ret': ret})
        random_objs.append(ref_obj)
        
    # Plotting
    print("Generating plots...")
    
    # Fig 2: Energy Distribution
    plt.figure(figsize=(10, 5))
    # Add tiny jitter to avoid KDE crashing on identical values (zero variance)
    raw_objs_jitter = np.array(raw_objs) + np.random.normal(0, 1e-6, len(raw_objs))
    refined_objs_jitter = np.array(refined_objs) + np.random.normal(0, 1e-6, len(refined_objs))
    
    sns.histplot(raw_objs_jitter, color='orange', label='BF-DCQO (Recombined Raw)', kde=True, stat="density", bins=20)
    sns.histplot(refined_objs_jitter, color='blue', label='BF-DCQO + LS', kde=True, stat="density", bins=20)
    plt.xlabel('Objective Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Energy distribution of global candidate portfolios')
    plt.savefig('energy_distribution.png', bbox_inches='tight')
    plt.close()
    
    # Fig 3: Risk-Return Scatter Plot
    plt.figure(figsize=(8, 6))
    
    rand_risks = [c['risk'] for c in random_candidates]
    rand_rets = [c['ret'] for c in random_candidates]
    plt.scatter(rand_risks, rand_rets, marker='s', color='orange', label='Random selection + LS', alpha=0.6)
    
    sc = plt.scatter(refined_risks, refined_returns, c=refined_objs, cmap='viridis', marker='o', label='BF-DCQO + LS', alpha=0.8)
    plt.colorbar(sc, label='Global objective')
    
    plt.xlabel('Annual Risk')
    plt.ylabel('Annual Return')
    plt.legend()
    plt.title('Return-risk distribution of global candidate portfolios')
    plt.savefig('risk_return_scatter.png', bbox_inches='tight')
    plt.close()
    
    # Fig 4: Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(C_corr, xticklabels=actual_tickers, yticklabels=actual_tickers, cmap='coolwarm', center=0, annot=False)
    plt.title('Asset Correlation Matrix')
    plt.savefig('correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # Fig 5: Cumulative Returns Over Time
    best_x = refined_candidates[np.argmin(refined_objs)]
    
    # Convert log returns to simple returns for accurate portfolio return calculation
    simple_returns = np.exp(log_returns) - 1
    
    # Equal Weight Portfolio (All Assets)
    ew_simple_returns = simple_returns.mean(axis=1)
    ew_cum_returns = (1 + ew_simple_returns).cumprod()
    
    # Optimized Portfolio
    opt_simple_returns = simple_returns.loc[:, best_x == 1].mean(axis=1)
    opt_cum_returns = (1 + opt_simple_returns).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(ew_cum_returns.index, ew_cum_returns, label='Equal Weight Index (All Assets)', color='gray')
    plt.plot(opt_cum_returns.index, opt_cum_returns, label='BF-DCQO Optimized Portfolio', color='blue', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Historical Performance of Optimized Portfolio vs Equal Weight Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cumulative_returns.png', bbox_inches='tight')
    plt.close()
    
    print("Simulation completed successfully! Plots saved as 'energy_distribution.png', 'risk_return_scatter.png', 'correlation_heatmap.png', and 'cumulative_returns.png'")

if __name__ == "__main__":
    main()
