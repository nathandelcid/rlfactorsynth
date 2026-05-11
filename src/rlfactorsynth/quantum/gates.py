"""Quantum gate definitions and operations for Clifford+T gate set."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)



# Hadamard gate
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

# Phase gates
S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)


@dataclass
class Gate:
    """Quantum gate representation."""
    
    name: str
    matrix: np.ndarray
    qubits: Tuple[int, ...]
    is_clifford: bool
    t_count: int
    
    def __repr__(self) -> str:
        qubit_str = ",".join(map(str, self.qubits))
        return f"{self.name}({qubit_str})"


class GateSet:
    """Clifford+T gate set with operations."""
    
    # Single-qubit gates
    SINGLE_QUBIT_GATES = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z,
        "H": H,
        "S": S,
        "Sdg": Sdg,
        "T": T,
        "Tdg": Tdg,
    }
    
    # Gate properties
    CLIFFORD_GATES = {"I", "X", "Y", "Z", "H", "S", "Sdg"}
    T_GATES = {"T", "Tdg"}
    
    @classmethod
    def get_gate(cls, name: str, qubits: Tuple[int, ...]) -> Gate:
        """Get a gate by name and target qubits."""
        if name not in cls.SINGLE_QUBIT_GATES:
            raise ValueError(f"Unknown gate: {name}")
        
        matrix = cls.SINGLE_QUBIT_GATES[name]
        is_clifford = name in cls.CLIFFORD_GATES
        t_count = 1 if name in cls.T_GATES else 0
        
        return Gate(name, matrix, qubits, is_clifford, t_count)
    
    @classmethod
    def get_primitive_gates(cls, n_qubits: int) -> List[str]:
        """Get list of primitive gate names for n qubits."""
        # Standard Clifford+T primitives (excluding I, X, Y for efficiency)
        primitives = ["H", "T", "Tdg", "S", "Sdg", "Z"]
        
        # Generate gates for each qubit
        gates = []
        for gate_name in primitives:
            for qubit in range(n_qubits):
                gates.append(f"{gate_name}_{qubit}")
        
        return gates
    
    @classmethod
    def cx_gate(cls, control: int, target: int) -> Gate:
        """Create a CNOT gate."""
        # 2-qubit CNOT matrix
        cx_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        
        return Gate("CX", cx_matrix, (control, target), is_clifford=True, t_count=0)
    
    @classmethod
    def apply_gate(cls, unitary: np.ndarray, gate: Gate, n_qubits: int) -> np.ndarray:
        """Apply a gate to a unitary matrix."""
        if len(gate.qubits) == 1:
            return cls._apply_single_qubit_gate(unitary, gate, n_qubits)
        elif len(gate.qubits) == 2:
            return cls._apply_two_qubit_gate(unitary, gate, n_qubits)
        else:
            raise ValueError(f"Unsupported gate: {gate}")
    
    @classmethod
    def _apply_single_qubit_gate(
        cls, unitary: np.ndarray, gate: Gate, n_qubits: int
    ) -> np.ndarray:
        """Apply a single-qubit gate to a unitary."""
        qubit = gate.qubits[0]
        
        # Build full gate matrix using tensor product
        full_gate = np.eye(1, dtype=np.complex128)
        for i in range(n_qubits):
            if i == qubit:
                full_gate = np.kron(full_gate, gate.matrix)
            else:
                full_gate = np.kron(full_gate, I)
        
        return full_gate @ unitary
    
    @classmethod
    def _apply_two_qubit_gate(
        cls, unitary: np.ndarray, gate: Gate, n_qubits: int
    ) -> np.ndarray:
        """Apply a two-qubit gate to a unitary."""
        # Simplified implementation for CNOT
        # Full implementation would handle arbitrary qubit positions
        return gate.matrix @ unitary


def decompose_rz(angle: float, precision: float = 1e-3) -> List[str]:
    """
    Decompose Rz(angle) into Clifford+T gates using gridsynth-like approach.
    
    This is a simplified heuristic version. Real implementation would use
    number-theoretic grid search.
    """
    # Normalize angle to [0, 2π)
    angle = angle % (2 * np.pi)
    
    # Simple heuristic decomposition
    gates = []
    
    # Approximate using T gates (π/4 rotations)
    n_t_gates = int(round(angle / (np.pi / 4)))
    
    if n_t_gates % 8 == 0:
        # Identity
        pass
    elif n_t_gates % 4 == 0:
        gates.append("Z")
    elif n_t_gates % 2 == 0:
        gates.append("S")
    else:
        # Use T gates
        for _ in range(n_t_gates % 8):
            gates.append("T")
    
    return gates


def count_t_gates(gates: List[str]) -> int:
    """Count T and Tdg gates in a gate sequence."""
    return sum(1 for g in gates if g.startswith("T"))


def count_clifford_gates(gates: List[str]) -> int:
    """Count Clifford gates in a gate sequence."""
    clifford_prefixes = ("H", "S", "Sdg", "Z", "X", "Y", "CX")
    return sum(1 for g in gates if any(g.startswith(p) for p in clifford_prefixes))

def make_rotation_gate(axis: str, angle: float):
    '''
    if axis == 'x':
        gate = np.array([])
    '''
    gate = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    return gate