# benchmarks.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# ============================================================
# Benchmark metadata
# ============================================================

@dataclass(frozen=True)
class Benchmark:
    name: str
    category: str
    n_qubits: int
    generator: Callable[[], QuantumCircuit]
    description: str


# ============================================================
# GHZ circuits
# ============================================================

def make_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"GHZ_{n}")

    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)

    return qc


def GHZ_4() -> QuantumCircuit:
    return make_ghz(4)


def GHZ_8() -> QuantumCircuit:
    return make_ghz(8)


# ============================================================
# QAOA ring circuits
# ============================================================

def make_ring_ising_params(n: int):
    h = {i: 1.0 for i in range(n)}
    J = {(i, (i + 1) % n): 1.0 for i in range(n)}
    return h, J


def make_qaoa_ring(n: int, p: int) -> QuantumCircuit:
    h, J = make_ring_ising_params(n)

    qc = QuantumCircuit(n, name=f"QAOA_ring_{n}_p{p}")

    qc.h(range(n))

    for layer in range(p):
        gamma = Parameter(f"g_{layer}")
        beta = Parameter(f"b_{layer}")

        for i in h:
            qc.rz(h[i] * gamma, i)

        for i, j in J:
            qc.cx(i, j)
            qc.rz(J[(i, j)] * gamma, j)
            qc.cx(i, j)

        for i in range(n):
            qc.rx(beta, i)

    return qc


def QAOA_ring_4_p1() -> QuantumCircuit:
    return make_qaoa_ring(n=4, p=1)


def QAOA_ring_8_p2() -> QuantumCircuit:
    return make_qaoa_ring(n=8, p=2)


# ============================================================
# Hardware-efficient ansatz
# ============================================================

def make_hea(n: int, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"HEA_{n}_d{depth}")

    for layer in range(depth):
        for q in range(n):
            theta = Parameter(f"theta_{layer}_{q}")
            phi = Parameter(f"phi_{layer}_{q}")
            qc.ry(theta, q)
            qc.rz(phi, q)

        for q in range(n - 1):
            qc.cx(q, q + 1)

    return qc


def HEA_4_d2() -> QuantumCircuit:
    return make_hea(n=4, depth=2)


# ============================================================
# Random circuit
# ============================================================

def make_random_circuit(n: int, depth: int, seed: int = 42) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n, name=f"RAND_{n}_d{depth}")

    single_qubit_gates = ["h", "x", "s", "t"]

    for _ in range(depth):
        gate_type = rng.choice(["single", "cx"])

        if gate_type == "single":
            q = rng.randrange(n)
            gate = rng.choice(single_qubit_gates)

            if gate == "h":
                qc.h(q)
            elif gate == "x":
                qc.x(q)
            elif gate == "s":
                qc.s(q)
            elif gate == "t":
                qc.t(q)

        else:
            control, target = rng.sample(range(n), 2)
            qc.cx(control, target)

    return qc


def RAND_4_d10() -> QuantumCircuit:
    return make_random_circuit(n=4, depth=10, seed=42)


# ============================================================
# Barenco-style multi-controlled Toffoli
# ============================================================

def Barenco_Toffoli_4() -> QuantumCircuit:
    """
    4-control Toffoli-style circuit.

    Qubits:
        q0, q1, q2, q3: controls
        q4: target
    """
    qc = QuantumCircuit(5, name="Barenco_Toffoli_4")

    controls = [0, 1, 2, 3]
    target = 4

    qc.mcx(controls, target)

    return qc


# ============================================================
# Simple VBE-style 3-bit ripple-carry adder placeholder
# ============================================================

def VBE_Adder_3() -> QuantumCircuit:
    """
    Small 3-bit ripple-carry-adder-style benchmark.

    This is not a full textbook-clean VBE implementation with all
    ancilla management exposed. It is a compact arithmetic benchmark
    meant to stress Toffoli/CX-heavy compilation.

    Registers:
        a0,a1,a2: qubits 0,1,2
        b0,b1,b2: qubits 3,4,5
        carry:    qubit 6
    """
    qc = QuantumCircuit(7, name="VBE_Adder_3")

    a = [0, 1, 2]
    b = [3, 4, 5]
    carry = 6

    for i in range(3):
        qc.cx(a[i], b[i])
        qc.ccx(a[i], b[i], carry)

    for i in reversed(range(3)):
        qc.ccx(a[i], b[i], carry)
        qc.cx(a[i], b[i])

    return qc


# ============================================================
# Benchmark registry
# ============================================================

BENCHMARKS: Dict[str, Benchmark] = {
    "GHZ_4": Benchmark(
        name="GHZ_4",
        category="entanglement",
        n_qubits=4,
        generator=GHZ_4,
        description="4-qubit GHZ state preparation circuit.",
    ),
    "GHZ_8": Benchmark(
        name="GHZ_8",
        category="entanglement",
        n_qubits=8,
        generator=GHZ_8,
        description="8-qubit GHZ state preparation circuit.",
    ),
    "QAOA_ring_4_p1": Benchmark(
        name="QAOA_ring_4_p1",
        category="variational",
        n_qubits=4,
        generator=QAOA_ring_4_p1,
        description="QAOA ring circuit with 4 qubits and p=1.",
    ),
    "QAOA_ring_8_p2": Benchmark(
        name="QAOA_ring_8_p2",
        category="variational",
        n_qubits=8,
        generator=QAOA_ring_8_p2,
        description="QAOA ring circuit with 8 qubits and p=2.",
    ),
    "HEA_4_d2": Benchmark(
        name="HEA_4_d2",
        category="variational",
        n_qubits=4,
        generator=HEA_4_d2,
        description="4-qubit hardware-efficient ansatz with depth 2.",
    ),
    "RAND_4_d10": Benchmark(
        name="RAND_4_d10",
        category="random",
        n_qubits=4,
        generator=RAND_4_d10,
        description="4-qubit random Clifford+T/CX circuit with depth 10.",
    ),
    "Barenco_Toffoli_4": Benchmark(
        name="Barenco_Toffoli_4",
        category="arithmetic",
        n_qubits=5,
        generator=Barenco_Toffoli_4,
        description="4-control Toffoli-style arithmetic benchmark.",
    ),
    "VBE_Adder_3": Benchmark(
        name="VBE_Adder_3",
        category="arithmetic",
        n_qubits=7,
        generator=VBE_Adder_3,
        description="Small 3-bit ripple-carry-adder-style benchmark.",
    ),
}


def get_benchmark(name: str) -> QuantumCircuit:
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark: {name}")

    return BENCHMARKS[name].generator()


def list_benchmarks() -> list[str]:
    return list(BENCHMARKS.keys())


if __name__ == "__main__":
    for name in list_benchmarks():
        circuit = get_benchmark(name)
        print("=" * 80)
        print(name)
        print(circuit)