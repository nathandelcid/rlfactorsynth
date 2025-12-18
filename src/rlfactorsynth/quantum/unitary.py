"""Unitary matrix operations and distance metrics."""

import numpy as np
from typing import Optional, Tuple
from scipy.linalg import logm, expm


def unitary_distance(U: np.ndarray, V: np.ndarray, metric: str = "operator") -> float:
    """
    Compute distance between two unitary matrices.
    
    Args:
        U: First unitary matrix
        V: Second unitary matrix
        metric: Distance metric ('operator', 'frobenius', 'trace')
    
    Returns:
        Distance value
    """
    if metric == "operator":
        return operator_norm_distance(U, V)
    elif metric == "frobenius":
        return frobenius_distance(U, V)
    elif metric == "trace":
        return trace_distance(U, V)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def operator_norm_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Operator norm distance: ||U - V||_op"""
    diff = U - V
    return np.linalg.norm(diff, ord=2)


def frobenius_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Frobenius distance: ||U - V||_F"""
    diff = U - V
    return np.linalg.norm(diff, ord='fro')


def trace_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Trace distance: 1 - |Tr(U† V)| / d"""
    d = U.shape[0]
    trace = np.trace(U.conj().T @ V)
    return 1.0 - np.abs(trace) / d


def global_phase_invariant_distance(U: np.ndarray, V: np.ndarray) -> float:
    """
    Distance invariant to global phase.
    
    Computes: min_φ ||U - e^(iφ) V||
    """
    d = U.shape[0]
    trace = np.trace(U.conj().T @ V)
    
    # Optimal phase that minimizes distance
    optimal_phase = np.angle(trace)
    V_phased = np.exp(1j * optimal_phase) * V
    
    return operator_norm_distance(U, V_phased)


def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Check if a matrix is unitary."""
    d = U.shape[0]
    identity = np.eye(d, dtype=U.dtype)
    
    # Check U† U = I
    product = U.conj().T @ U
    return np.allclose(product, identity, atol=tol)


def random_unitary(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random unitary matrix (Haar measure).
    
    Uses QR decomposition of random complex matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = 2 ** n_qubits
    
    # Random complex matrix
    Z = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    
    # QR decomposition gives Haar random unitary
    Q, R = np.linalg.qr(Z)
    
    # Adjust phases to ensure uniform distribution
    Lambda = np.diag(R) / np.abs(np.diag(R))
    U = Q @ np.diag(Lambda)
    
    return U


def random_clifford_unitary(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random Clifford unitary.
    
    Simplified implementation using random Clifford gates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    from .gates import GateSet
    
    d = 2 ** n_qubits
    U = np.eye(d, dtype=np.complex128)
    
    # Apply random Clifford gates
    clifford_gates = ["H", "S", "Sdg", "Z"]
    n_gates = np.random.randint(10, 30)
    
    for _ in range(n_gates):
        gate_name = np.random.choice(clifford_gates)
        qubit = np.random.randint(n_qubits)
        gate = GateSet.get_gate(gate_name, (qubit,))
        U = GateSet.apply_gate(U, gate, n_qubits)
    
    return U


def compute_residual(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    Compute residual unitary: R = target @ current†
    """
    return target @ current.conj().T


def pauli_transfer_matrix(U: np.ndarray) -> np.ndarray:
    """
    Compute Pauli transfer matrix representation of a unitary.
    
    PTM[i,j] = Tr(P_i U P_j U†) / d
    where P_i are Pauli matrices.
    """
    from .gates import I, X, Y, Z
    
    paulis = [I, X, Y, Z]
    d = U.shape[0]
    
    ptm = np.zeros((4, 4), dtype=np.float64)
    
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            # Tr(P_i U P_j U†)
            product = Pi @ U @ Pj @ U.conj().T
            trace = np.trace(product)
            ptm[i, j] = np.real(trace) / d
    
    return ptm


def bloch_vector(state: np.ndarray) -> np.ndarray:
    """
    Compute Bloch vector representation of a single-qubit state.
    
    |ψ⟩ = α|0⟩ + β|1⟩ → (x, y, z) on Bloch sphere
    """
    from .gates import X, Y, Z
    
    # Density matrix
    rho = np.outer(state, state.conj())
    
    # Bloch coordinates
    x = np.real(np.trace(X @ rho))
    y = np.real(np.trace(Y @ rho))
    z = np.real(np.trace(Z @ rho))
    
    return np.array([x, y, z])


def low_rank_approximation(
    U: np.ndarray, rank: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute low-rank approximation of a unitary using SVD.
    
    Returns: U ≈ U_approx, (left, singular_values, right)
    """
    left, s, right = np.linalg.svd(U, full_matrices=False)
    
    # Truncate to rank
    left_trunc = left[:, :rank]
    s_trunc = s[:rank]
    right_trunc = right[:rank, :]
    
    return left_trunc, s_trunc, right_trunc
