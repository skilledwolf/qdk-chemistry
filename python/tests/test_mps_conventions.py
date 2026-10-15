"""Solver-independent signed MPS fixture in the documented fermionic convention."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import ModelOrbitals, MPSContainer, MPSSite, Wavefunction


def _reference_sites():
    """Represent (|2u0> + i|u02> - |udu> - i|02u>)/2 in chain order.

    Each bond channel carries one product state, so no solver, decomposition,
    truncation, gauge choice, or normalization algorithm enters this fixture.
    """
    first, middle, last = (np.zeros(shape, dtype=complex) for shape in [(1, 4, 4), (4, 4, 4), (4, 4, 1)])
    for bond, (states, coefficient) in enumerate(zip(["2u0", "u02", "udu", "02u"], [1, 1j, -1, -1j], strict=True)):
        physical = ["0ud2".index(state) for state in states]
        first[0, physical[0], bond] = coefficient / 2
        middle[bond, physical[1], bond] = 1
        last[bond, physical[2], 0] = 1
    return [MPSSite.from_dense(values) for values in (first, middle, last)]


def _blocked_amplitudes(container):
    """Contract this tiny fixture and apply creation operators to the vacuum.

    Blocked modes are (alpha_0, ..., alpha_n-1, beta_0, ..., beta_n-1).
    Operators act right to left; creating mode k crosses every occupied mode
    below k. This explicitly evaluates the Fock convention rather than using
    a solver's tensor conversion or the reference table to calculate signs.
    """
    chain = np.ones(1, dtype=complex)
    sites = container.sites
    for site in sites:
        chain = np.tensordot(chain, site.to_dense().reshape(site.shape), axes=(-1, 0))
    blocked = np.zeros(4 ** len(sites), dtype=complex)
    for physical_indices, amplitude in np.ndenumerate(chain[..., 0]):
        if amplitude == 0:
            continue
        created = []
        for site, physical, orbital in zip(sites, physical_indices, container.site_to_orbital_order, strict=True):
            alpha, beta = site.physical_basis[physical].to_bits(2)
            if alpha:
                created.append(orbital)
            if beta:
                created.append(len(sites) + orbital)
        occupied = [0] * (2 * len(sites))
        signed_amplitude = amplitude
        for mode in reversed(created):
            signed_amplitude *= (-1) ** sum(occupied[:mode])
            occupied[mode] = 1
        blocked[int("".join(map(str, occupied)), 2)] += signed_amplitude
    return blocked


@pytest.mark.parametrize("serialization", [None, "json", "hdf5"])
@pytest.mark.parametrize(
    ("site_order", "expected_amplitudes"),
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
def test_signed_determinant_fixture(site_order, expected_amplitudes, serialization, tmp_path):
    """Preserve scale, complex phase, local basis and signed orbital mapping.

    The expected amplitudes are hand-enumerated, not obtained from the helper.
    For example, |2u0> creates a0 b0 a1 in identity order (one swap), but a2 b2
    a0 for order [2,0,1] (two swaps). |u02> creates a0 a2 b2 (zero swaps) or
    a2 a1 b1 (one swap). These relative signs distinguish a fermionic mapping
    from a simple tensor-axis permutation or a change of global phase.
    """
    original = Wavefunction(MPSContainer(_reference_sites(), ModelOrbitals(3), site_to_orbital_order=site_order))
    if serialization == "json":
        restored = Wavefunction.from_json(original.to_json())
    elif serialization == "hdf5":
        path = tmp_path / "signed.wavefunction.h5"
        original.to_file(path, "hdf5")
        restored = Wavefunction.from_file(path, "hdf5")
    else:
        restored = original

    container = restored.get_container()
    assert container.is_complex()
    assert container.site_to_orbital_order == site_order
    assert restored.content_hash() == original.content_hash()
    for site, original_site in zip(container.sites, original.get_container().sites, strict=True):
        assert [state.to_bits(2) for state in site.physical_basis] == [[0, 0], [1, 0], [0, 1], [1, 1]]
        np.testing.assert_array_equal(site.to_dense(), original_site.to_dense())

    expected = np.zeros(64, dtype=complex)
    for bits, amplitude in expected_amplitudes.items():
        expected[int(bits, 2)] = amplitude
    np.testing.assert_array_equal(_blocked_amplitudes(container), expected)
