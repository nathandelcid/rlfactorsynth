"""Tests for unitary distance metrics."""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rlfactorsynth.quantum.unitary import (
    unitary_distance,
    operator_norm_distance,
    frobenius_distance,
    trace_distance,
    is_unitary,
    random_unitary,
)
from rlfactorsynth.quantum.gates import I, H, T


class TestUnitaryDistance:
    """Test unitary distance metrics."""
    
    def test_identity_distance(self):
        """Distance between identical unitaries should be zero."""
        U = random_unitary(1, seed=42)
        
        assert operator_norm_distance(U, U) < 1e-10
        assert frobenius_distance(U, U) < 1e-10
        assert trace_distance(U, U) < 1e-10
    
    def test_orthogonal_distance(self):
        """Distance between orthogonal unitaries."""
        # H and T are not orthogonal but different
        dist = operator_norm_distance(H, T)
        assert dist > 0.1
    
    def test_is_unitary(self):
        """Test unitary checker."""
        assert is_unitary(I)
        assert is_unitary(H)
        assert is_unitary(T)
        
        # Random unitary
        U = random_unitary(2, seed=42)
        assert is_unitary(U)
        
        # Non-unitary
        M = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        assert not is_unitary(M)
    
    def test_random_unitary_properties(self):
        """Test random unitary generation."""
        for n_qubits in [1, 2, 3]:
            U = random_unitary(n_qubits, seed=42)
            
            # Check dimensions
            d = 2 ** n_qubits
            assert U.shape == (d, d)
            
            # Check unitarity
            assert is_unitary(U)
    
    def test_distance_metrics_consistency(self):
        """Different metrics should give consistent ordering."""
        U = random_unitary(1, seed=42)
        V1 = random_unitary(1, seed=43)
        V2 = random_unitary(1, seed=44)
        
        # Compute distances
        d1_op = operator_norm_distance(U, V1)
        d2_op = operator_norm_distance(U, V2)
        
        d1_frob = frobenius_distance(U, V1)
        d2_frob = frobenius_distance(U, V2)
        
        # Check consistency (same ordering)
        if d1_op < d2_op:
            assert d1_frob < d2_frob
        else:
            assert d1_frob >= d2_frob


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
