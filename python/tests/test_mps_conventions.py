"""Fermionic sign convention of MPS amplitudes, checked against determinants."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy.sparse

from qdk_chemistry import algorithms
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    Configuration,
    Hamiltonian,
    ModelOrbitals,
    MPSContainer,
    MPSSite,
    Wavefunction,
)
from qdk_chemistry.data import symmetry as sym

# Default spin-half physical basis; index = n_alpha + 2 * n_beta.
PHYSICAL_STATES = "0ud2"


def make_site(values):
    """Wrap a (left, physical, right) array as one trivial-symmetry block."""
    left, physical, right = values.shape
    product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
    tensor_type = sym.SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else sym.SymmetryBlockedTensorRank3
    tensor = tensor_type(
        [product] * 3,
        [{label: left}, {label: physical}, {label: right}],
        [((label,) * 3, values.reshape(left * physical, right))],
    )
    return MPSSite(tensor, [label], [label], [label])


def blocked_configuration(bits):
    """Spin-half configuration from occupation bits alpha_0 ... alpha_n-1 beta_0 ... beta_n-1."""
    n = len(bits) // 2
    return Configuration.from_spin_half_string(
        "".join(PHYSICAL_STATES[int(bits[orbital]) + 2 * int(bits[n + orbital])] for orbital in range(n))
    )


def reference_sites():
    """Represent (|2u0> + i|u02> - |udu> - i|02u>)/2 in chain order.

    Each bond channel carries one product state, so no decomposition, truncation, gauge choice or
    normalization enters the fixture.
    """
    first, middle, last = (np.zeros(shape, dtype=complex) for shape in [(1, 4, 4), (4, 4, 4), (4, 4, 1)])
    for bond, (states, coefficient) in enumerate(zip(["2u0", "u02", "udu", "02u"], [1, 1j, -1, -1j], strict=True)):
        physical = [PHYSICAL_STATES.index(state) for state in states]
        first[0, physical[0], bond] = coefficient / 2
        middle[bond, physical[1], bond] = 1
        last[bond, physical[2], 0] = 1
    return [make_site(values) for values in (first, middle, last)]


@pytest.mark.parametrize("serialization", [None, "json", "hdf5"])
@pytest.mark.parametrize(
    ("site_order", "expected"),
    [
        pytest.param(
            [0, 1, 2],
            {"110100": -0.5, "101001": 0.5j, "101010": 0.5, "011010": 0.5j},
            id="identity",
        ),
        pytest.param(
            [2, 0, 1],
            {"101001": 0.5, "011010": -0.5j, "011100": -0.5, "110100": 0.5j},
            id="permuted",
        ),
    ],
)
def test_signed_reference_amplitudes(site_order, expected, serialization, tmp_path):
    """Determinant amplitudes carry the documented fermionic sign.

    The expected amplitudes are hand-enumerated. For example, |2u0> creates a0 b0 a1 in identity order
    (one exchange to reach a0 a1 b0) but a2 b2 a0 for order [2,0,1] (two exchanges). These relative
    signs distinguish a fermionic mapping from a permutation of tensor axes or a global phase.
    """
    original = Wavefunction(MPSContainer(reference_sites(), ModelOrbitals(3), site_to_orbital_order=site_order))
    if serialization == "json":
        restored = Wavefunction.from_json(original.to_json())
    elif serialization == "hdf5":
        path = tmp_path / "signed.wavefunction.h5"
        original.to_file(path, "hdf5")
        restored = Wavefunction.from_file(path, "hdf5")
    else:
        restored = original
    for mask in range(64):
        bits = format(mask, "06b")
        actual = restored.get_coefficient(blocked_configuration(bits))
        assert actual == pytest.approx(expected.get(bits, 0), abs=1e-14), bits


def chain_annihilators(num_sites):
    """Annihilators on the chain-ordered Fock space of spatial-orbital sites.

    Mode 2*site + spin (alpha 0, beta 1). Bit m of a basis index is the occupation of mode m, and the
    basis state applies creation operators in ascending mode order, so annihilating mode m crosses
    every occupied mode below it.
    """
    dim = 4**num_sites
    operators = []
    for mode in range(2 * num_sites):
        occupied = [state for state in range(dim) if state >> mode & 1]
        signs = [(-1) ** bin(state & ((1 << mode) - 1)).count("1") for state in occupied]
        targets = [state ^ (1 << mode) for state in occupied]
        operators.append(scipy.sparse.csr_matrix((signs, (targets, occupied)), shape=(dim, dim)))
    return operators


def exact_mps_sites(amplitudes):
    """Split a (4,) * n amplitude tensor into sites by successive SVD, without truncation."""
    sites, rest = [], amplitudes.reshape(1, -1)
    for _ in range(amplitudes.ndim - 1):
        left = rest.shape[0]
        u, singular, vh = np.linalg.svd(rest.reshape(left * 4, -1), full_matrices=False)
        keep = singular > 1e-12
        sites.append(make_site(u[:, keep].reshape(left, 4, -1)))
        rest = singular[keep, None] * vh[keep]
    sites.append(make_site(rest.reshape(-1, 4, 1)))
    return sites


def test_coefficients_match_macis_determinants():
    """An exact MPS of the ground state reproduces MACIS's signed determinant coefficients.

    The reference state is diagonalized here in the chain-ordered Fock space with explicit
    Jordan-Wigner operators, so it shares no sign-convention code with get_coefficient or MACIS.
    """
    n, n_alpha, n_beta = 4, 2, 2
    site_order = [2, 0, 3, 1]
    rng = np.random.default_rng(7)
    h1 = rng.normal(size=(n, n))
    h1 = (h1 + h1.T) / 2
    eri = 0.3 * rng.normal(size=(n,) * 4)
    eri = eri + eri.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)

    annihilate = chain_annihilators(n)
    chain_site = np.argsort(site_order)

    def excitation(p, q):
        """Spin-summed a_p^dagger a_q, with orbitals placed on their chain sites."""
        return sum(annihilate[2 * chain_site[p] + s].T @ annihilate[2 * chain_site[q] + s] for s in (0, 1))

    excitations = {(p, q): excitation(p, q) for p in range(n) for q in range(n)}
    fock_hamiltonian = sum(h1[p, q] * excitations[p, q] for p, q in excitations)
    for (p, q), e_pq in excitations.items():
        for (r, s), e_rs in excitations.items():
            two_body = e_pq @ e_rs - (excitations[p, s] if q == r else 0)
            fock_hamiltonian = fock_hamiltonian + 0.5 * eri[p, q, r, s] * two_body

    def count(state, spin):
        return sum(state >> (2 * site + spin) & 1 for site in range(n))

    sector = [state for state in range(4**n) if count(state, 0) == n_alpha and count(state, 1) == n_beta]
    energies, vectors = np.linalg.eigh(fock_hamiltonian[sector][:, sector].toarray())
    assert energies[1] - energies[0] > 1e-6
    amplitudes = np.zeros((4,) * n)
    for index, state in enumerate(sector):
        local = tuple((state >> 2 * site & 1) + 2 * (state >> (2 * site + 1) & 1) for site in range(n))
        amplitudes[local] = vectors[index, 0]
    mps = Wavefunction(MPSContainer(exact_mps_sites(amplitudes), ModelOrbitals(n), site_to_orbital_order=site_order))

    hamiltonian = Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(h1, eri.reshape(-1), ModelOrbitals(n), 0.0, np.array([]))
    )
    energy, reference = algorithms.create("multi_configuration_calculator", "macis_cas").run(
        hamiltonian, n_alpha, n_beta
    )
    assert energy == pytest.approx(energies[0], abs=1e-9)
    determinants = reference.get_active_determinants()
    coefficients = np.asarray(reference.get_coefficients())
    assert np.linalg.norm(coefficients) == pytest.approx(1.0, abs=1e-8)

    signed = np.array([mps.get_coefficient(det) for det in determinants])
    phase = np.sign(np.dot(coefficients, signed))
    np.testing.assert_allclose(signed, phase * coefficients, atol=1e-8)

    # Negative control: the chain-ordered amplitudes without the fermionic sign do not match.
    unsigned = []
    for det in determinants:
        states = det.to_string()
        unsigned.append(amplitudes[tuple(PHYSICAL_STATES.index(states[orbital]) for orbital in site_order)])
    assert not np.allclose(np.abs(np.dot(coefficients, unsigned)), 1.0, atol=1e-6)
