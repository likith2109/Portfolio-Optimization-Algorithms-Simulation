import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import scipy.optimize as opt
from itertools import combinations

def portfolio_to_ising(mu, C, K, gamma=0.0):
    """
    Map the portfolio optimization problem to an Ising Hamiltonian.
    Returns h (linear terms), J (quadratic terms), offset.
    """
    N = len(mu)
    
    # Q and q for the original unconstrained problem (without penalty)
    Q = (gamma + 1) / 2 * C
    q = (gamma - 1) / 2 * mu
    
    # Calculate penalty Lambda
    I = np.zeros(N)
    for i in range(N):
        I[i] = np.abs(q[i] + Q[i, i]) + sum(np.abs(Q[i, j]) for j in range(N) if j != i)
    Lambda = 2 * np.max(I)
    
    # Q' and q' with penalty
    Q_prime = np.zeros((N, N))
    q_prime = np.zeros(N)
    
    for i in range(N):
        q_prime[i] = q[i] + Q[i, i] + Lambda - 2 * K * Lambda
        for j in range(N):
            if i != j:
                Q_prime[i, j] = Q[i, j] + Lambda
                
    # Map to Ising
    h = np.zeros(N)
    J = np.zeros((N, N))
    
    for i in range(N):
        sum_Q_ij = sum(Q_prime[i, j] for j in range(N) if j != i)
        h[i] = -0.5 * q_prime[i] - 0.5 * sum_Q_ij
        
        for j in range(i + 1, N):
            J[i, j] = 0.5 * Q_prime[i, j]
            
    offset = Lambda * K**2 + 0.5 * np.sum(q_prime) + 0.25 * np.sum(Q_prime)
    
    return h, J, offset

def build_sparse_pauli_op_Hf(h, J, N):
    paulis = []
    coeffs = []
    
    for i in range(N):
        if h[i] != 0:
            op = ['I'] * N
            op[N - 1 - i] = 'Z'
            paulis.append("".join(op))
            coeffs.append(h[i])
            
    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] != 0:
                op = ['I'] * N
                op[N - 1 - i] = 'Z'
                op[N - 1 - j] = 'Z'
                paulis.append("".join(op))
                coeffs.append(J[i, j])
                
    if not paulis:
        return SparsePauliOp.from_list([("I" * N, 0.0)])
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

def build_sparse_pauli_op_Hi(h_b, N):
    paulis = []
    coeffs = []
    
    for i in range(N):
        # -X_i
        op = ['I'] * N
        op[N - 1 - i] = 'X'
        paulis.append("".join(op))
        coeffs.append(-1.0)
        
        # h_b * Z_i
        if h_b[i] != 0:
            op = ['I'] * N
            op[N - 1 - i] = 'Z'
            paulis.append("".join(op))
            coeffs.append(h_b[i])
            
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))

def compute_alpha(Hi, Hf, lam, N):
    """
    Compute alpha(lambda) = - ||[Hi, Hf]||^2 / ||[Had, [Hi, Hf]]||^2
    Using Hilbert-Schmidt norm (which is proportional to sum of squared coeffs).
    """
    comm_i_f = (Hi @ Hf - Hf @ Hi).simplify()
    if len(comm_i_f.coeffs) == 0 or np.allclose(comm_i_f.coeffs, 0):
        return 0.0
        
    Had = ((1 - lam) * Hi + lam * Hf).simplify()
    comm_ad_if = (Had @ comm_i_f - comm_i_f @ Had).simplify()
    
    norm_if = np.sum(np.abs(comm_i_f.coeffs)**2)
    norm_ad_if = np.sum(np.abs(comm_ad_if.coeffs)**2)
    
    if norm_ad_if < 1e-12:
        return 0.0
        
    return - norm_if / norm_ad_if

def dcqo_circuit(h, J, h_b, N, nsteps=2, dt=0.1, theta_cutoff=1e-5):
    """
    Algorithm 2: DCQO
    """
    T = nsteps * dt
    Hi = build_sparse_pauli_op_Hi(h_b, N)
    Hf = build_sparse_pauli_op_Hf(h, J, N)
    
    qc = QuantumCircuit(N)
    
    # Initial state preparation (ground state of Hi if it was just -X, we use H gates)
    # With bias, it's slightly different, but H is a good start
    for i in range(N):
        qc.h(i)
        
    comm_i_f = (Hi @ Hf - Hf @ Hi).simplify()
    # A_lambda is i * alpha(lambda) * comm_i_f
    
    for k in range(1, nsteps + 1):
        t = k * dt
        lam = np.sin(np.pi / 2 * np.sin(np.pi * t / (2 * T))**2)**2
        # Derivative of lam w.r.t t
        d_lam = (np.pi**2 / (4 * T)) * np.sin(np.pi * np.sin(np.pi*t/(2*T))**2) * np.sin(np.pi*t/T)
        
        alpha = compute_alpha(Hi, Hf, lam, N)
        
        # Hk = d_lam * A_lambda = d_lam * i * alpha * comm_i_f
        # Note: comm_i_f has purely imaginary coefficients because it's a commutator of Hermitian ops
        
        # Trotter step: exp(-i * dt * Hk)
        # Hk = sum r_j P_j
        # So we apply exp(-i * dt * r_j * P_j)
        
        for pauli, coeff in zip(comm_i_f.paulis, comm_i_f.coeffs):
            # coeff is purely imaginary, let's say coeff = i * c
            c = np.imag(coeff)
            r_j = d_lam * (-alpha) * c # since i * i * alpha * c = -alpha * c
            
            angle = 2 * dt * r_j  # 2 because R_P(theta) = exp(-i theta/2 P)
            
            if np.abs(angle) > theta_cutoff:
                # Apply Pauli rotation
                pauli_str = str(pauli)[::-1] # Reverse for Qiskit endianness
                
                # Find indices of non-I
                active_qubits = [i for i, p in enumerate(pauli_str) if p != 'I']
                
                if len(active_qubits) == 1:
                    q = active_qubits[0]
                    if pauli_str[q] == 'X':
                        qc.rx(angle, q)
                    elif pauli_str[q] == 'Y':
                        qc.ry(angle, q)
                    elif pauli_str[q] == 'Z':
                        qc.rz(angle, q)
                elif len(active_qubits) == 2:
                    q1, q2 = active_qubits
                    # We need to apply exp(-i angle/2 P1 P2)
                    # We can use Qiskit's PauliEvolutionGate or manually construct it
                    # Since comm_i_f from [Hi, Hf] only has Y and YZ terms, we can hardcode
                    if pauli_str[q1] == 'Y' and pauli_str[q2] == 'Z':
                        qc.rx(np.pi/2, q1)
                        qc.cx(q1, q2)
                        qc.rz(angle, q2)
                        qc.cx(q1, q2)
                        qc.rx(-np.pi/2, q1)
                    elif pauli_str[q1] == 'Z' and pauli_str[q2] == 'Y':
                        qc.rx(np.pi/2, q2)
                        qc.cx(q2, q1)
                        qc.rz(angle, q1)
                        qc.cx(q2, q1)
                        qc.rx(-np.pi/2, q2)
                    else:
                        # General fallback (not strictly needed for YZ, Y)
                        pass 
                        
    qc.measure_all()
    return qc

def run_bf_dcqo(h, J, N, R=5, nl=10, nsteps=2, dt=0.1, theta_cutoff=1e-5):
    """
    Algorithm 3: BF-DCQO
    """
    h_b = np.zeros(N)
    simulator = AerSimulator()
    
    best_samples = []
    
    for r in range(R):
        qc = dcqo_circuit(h, J, h_b, N, nsteps=nsteps, dt=dt, theta_cutoff=theta_cutoff)
        
        # Execute
        compiled_circuit = transpile(qc, simulator)
        result = simulator.run(compiled_circuit, shots=1000).result()
        counts = result.get_counts()
        
        # Compute classical energies
        energies = {}
        for bitstring, count in counts.items():
            # bitstring is like '0101' where left is qubit N-1, right is qubit 0
            # Qiskit endianness: bitstring[0] corresponds to qubit N-1
            z = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
            
            energy = 0
            for i in range(N):
                energy += h[i] * z[i]
                for j in range(i+1, N):
                    energy += J[i, j] * z[i] * z[j]
            energies[bitstring] = energy
            
        # Sort by energy
        sorted_samples = sorted(energies.items(), key=lambda x: x[1])
        top_samples = sorted_samples[:nl]
        
        # Save best bitstrings
        best_samples.extend(top_samples)
        
        # Update h_b
        # h_b^(r+1) = - <sigma_z>
        avg_z = np.zeros(N)
        for bitstring, _ in top_samples:
            z = np.array([1 if b == '0' else -1 for b in bitstring[::-1]])
            avg_z += z
        avg_z /= len(top_samples)
        
        h_b = -avg_z
        
    # Deduplicate and sort best samples
    unique_best = {}
    for bs, en in best_samples:
        if bs not in unique_best or unique_best[bs] > en:
            unique_best[bs] = en
            
    final_sorted = sorted(unique_best.items(), key=lambda x: x[1])
    return final_sorted[:nl]
