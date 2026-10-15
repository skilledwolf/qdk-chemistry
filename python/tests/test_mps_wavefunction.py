"""Basic Python API tests for MPS storage."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import (
    Configuration,
    ModelOrbitals,
    MPSContainer,
    MPSSite,
    StateVectorContainer,
    Wavefunction,
    WavefunctionContainer,
)
from qdk_chemistry.data import symmetry as sym


@pytest.fixture
def site_data():
    """Use reproducible random input rather than a particular physical state."""
    rng = np.random.default_rng(42)
    return [rng.random((1, 4, 3)), rng.random((3, 4, 1))]


def test_construction_and_accessors(site_data):
    """The container retains supplied sites and optional metadata."""
    sites = [MPSSite.from_dense(values) for values in site_data]
    orbitals = ModelOrbitals(2)
    counts = sym.SymmetryBlockedScalarCount([sym.SymmetryProduct([])], [((sym.SymmetryLabel([]),), 2)])
    wavefunction = Wavefunction(
        MPSContainer(
            sites,
            orbitals,
            total_num_particles=counts,
            orthogonality_center=1,
            site_to_orbital_order=[1, 0],
        )
    )
    assert "num_sites=2" in repr(wavefunction)
    container = wavefunction.get_container()
    del wavefunction
    assert isinstance(container, MPSContainer)
    assert container.num_sites == 2
    assert container.max_bond_dimension == 3
    assert not container.is_complex()
    assert container.orthogonality_center == 1
    assert container.site_to_orbital_order == [1, 0]
    assert container.total_num_particles().value(sym.SymmetryLabel([])) == 2
    assert not container.has_active_num_particles()
    with pytest.raises(RuntimeError, match="Active particle-count is not set"):
        container.active_num_particles()
    for site, values in zip(container.sites, site_data, strict=True):
        assert site.shape == values.shape
        np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("complex_values", [False, True])
def test_serialization(site_data, complex_values, tmp_path):
    """Python JSON and file factories preserve real and complex tensor data."""
    if complex_values:
        rng = np.random.default_rng(43)
        site_data = [values + 1j * rng.random(values.shape) for values in site_data]
    original = Wavefunction(MPSContainer([MPSSite.from_dense(values) for values in site_data], ModelOrbitals(2)))
    restored = [Wavefunction.from_json(original.to_json())]
    for extension, format_name in [("json", "json"), ("h5", "hdf5")]:
        path = tmp_path / f"mps.wavefunction.{extension}"
        original.to_file(path, format_name)
        restored.append(Wavefunction.from_file(path, format_name))
    for wavefunction in restored:
        assert wavefunction.content_hash() == original.content_hash()
        container = wavefunction.get_container()
        assert container.is_complex() is complex_values
        for site, values in zip(container.sites, site_data, strict=True):
            assert site.to_dense().dtype == values.dtype
            np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("dimension", [2, 4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_numpy_input_and_output_copies(dimension, dtype):
    """Strided input and exported arrays do not alias stored data."""
    rng = np.random.default_rng(42)
    values = rng.random((2, dimension, 3))
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * rng.random(values.shape)
    values = values.astype(dtype)[::-1, :, ::-1]
    expected = values.copy()
    site = MPSSite.from_dense(values)
    values[:] = 0
    exported = site.to_dense()
    assert exported.shape == (2 * dimension, 3)
    assert site.is_complex is np.iscomplexobj(values)
    assert exported.dtype == (np.complex128 if np.iscomplexobj(values) else np.float64)
    np.testing.assert_array_equal(exported.reshape(site.shape), expected)
    exported[:] = 1
    block = site.tensor.block((sym.SymmetryLabel([]),) * 3)
    block[:] = 2
    np.testing.assert_array_equal(site.to_dense().reshape(site.shape), expected)


@pytest.mark.parametrize("complex_values", [False, True])
def test_common_container_interface(complex_values):
    """MPS and state-vector containers support the same callable base interface."""

    def check_interface(value):
        assert value.is_complex() is complex_values
        assert value.total_num_particles().value(sym.SymmetryLabel([])) == 2
        assert value.active_num_particles().value(sym.SymmetryLabel([])) == 2

    coefficient = 1j if complex_values else 1.0
    orbitals = ModelOrbitals(2)
    counts = sym.SymmetryBlockedScalarCount([sym.SymmetryProduct([])], [((sym.SymmetryLabel([]),), 2)])
    first = np.array([0, 0, 0, coefficient]).reshape(1, 4, 1)
    second = np.array([1, 0, 0, 0], dtype=first.dtype).reshape(1, 4, 1)
    containers = [
        StateVectorContainer(np.array([coefficient]), [Configuration.from_spin_half_string("20")], orbitals),
        MPSContainer(
            [MPSSite.from_dense(first), MPSSite.from_dense(second)],
            orbitals,
            total_num_particles=counts,
            active_num_particles=counts,
        ),
    ]
    for container in containers:
        assert isinstance(container, WavefunctionContainer)
        check_interface(container)
        check_interface(Wavefunction(container))


def test_dense_site_custom_basis():
    """The factory forwards explicit local bases and copies Fortran-order arrays."""
    values = np.asfortranarray(np.arange(18).reshape(2, 3, 3))
    basis = [Configuration.from_spin_half_string(state) for state in ("d", "0", "u")]
    site = MPSSite.from_dense(values, physical_basis=basis)
    assert site.physical_basis == basis
    assert site.shape == values.shape
    np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("shape", [(4,), (4, 1), (1, 4, 1, 1)])
def test_dense_site_requires_rank_three(shape):
    with pytest.raises(ValueError, match="rank-3"):
        MPSSite.from_dense(np.zeros(shape))


@pytest.mark.parametrize("shape", [(0, 4, 1), (1, 0, 1), (1, 4, 0)])
def test_dense_site_requires_nonempty_axes(shape):
    with pytest.raises(ValueError, match="positive dimensions"):
        MPSSite.from_dense(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, complex(1, np.nan), complex(1, np.inf)])
def test_dense_site_rejects_nonfinite_values(value):
    values = np.zeros((1, 4, 1), dtype=complex if np.iscomplexobj(value) else float)
    values[0, 2, 0] = value
    with pytest.raises(ValueError, match="finite"):
        MPSSite.from_dense(values)


def test_dense_site_requires_numeric_dtype():
    with pytest.raises(TypeError, match="numeric array"):
        MPSSite.from_dense(np.full((1, 4, 1), "1"))


def test_dense_site_validates_physical_basis():
    values = np.ones((1, 3, 1))
    with pytest.raises(ValueError, match="basis"):
        MPSSite.from_dense(values)
    with pytest.raises(ValueError, match="basis"):
        MPSSite.from_dense(values, physical_basis=[Configuration.from_spin_half_string("0")])


@pytest.mark.parametrize("complex_values", [False, True])
def test_supplied_rdms(site_data, complex_values):
    """Each RDM keyword reaches its corresponding inherited getter."""
    rng = np.random.default_rng(42)
    one = [rng.random((2, 2)) for _ in range(3)]
    two = [rng.random(16) for _ in range(4)]
    if complex_values:
        one = [values + 1j * rng.random(values.shape) for values in one]
        two = [values + 1j * rng.random(values.shape) for values in two]
    wavefunction = Wavefunction(
        MPSContainer(
            [MPSSite.from_dense(values) for values in site_data],
            ModelOrbitals(2),
            one_rdm_spin_traced=one[0],
            one_rdm_aa=one[1],
            one_rdm_bb=one[2],
            two_rdm_spin_traced=two[0],
            two_rdm_aaaa=two[1],
            two_rdm_aabb=two[2],
            two_rdm_bbbb=two[3],
        )
    )
    assert wavefunction.has_one_rdm_spin_dependent()
    assert wavefunction.has_two_rdm_spin_dependent()
    actual_one = [wavefunction.get_active_one_rdm_spin_traced(), *wavefunction.get_active_one_rdm_spin_dependent()]
    actual_two = [wavefunction.get_active_two_rdm_spin_traced(), *wavefunction.get_active_two_rdm_spin_dependent()]
    for actual, expected in zip(actual_one + actual_two, one + two, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_natural_occupations_from_supplied_rdm(site_data):
    """Both public interfaces return descending eigenvalues, not the diagonal."""

    def check_occupations(value):
        for occupations in (value.active_orbital_occupations(), value.total_orbital_occupations()):
            actual = np.asarray(occupations.block([sym.SymmetryLabel([])])).ravel()
            np.testing.assert_allclose(actual, [1.25, 0.75])

    sites = [MPSSite.from_dense(values) for values in site_data]
    orbitals = ModelOrbitals(2)
    one_rdm = np.array([[1, 0.25j], [-0.25j, 1]])
    container = MPSContainer(sites, orbitals, one_rdm_spin_traced=one_rdm)
    check_occupations(container)
    check_occupations(Wavefunction(container))

    missing = MPSContainer(sites, orbitals)
    with pytest.raises(RuntimeError, match="RDM"):
        missing.active_orbital_occupations()
    with pytest.raises(RuntimeError, match="RDM"):
        Wavefunction(missing).active_orbital_occupations()


def _blocked_rdm(rank, blocks, *, equivalent):
    """Declare block relations independently of the associated orbital basis."""
    symmetry = sym.SymmetryProduct([sym.axes.spin(1, equivalent)])
    labels = {"a": sym.SymmetryLabel([sym.axes.alpha()]), "b": sym.SymmetryLabel([sym.axes.beta()])}
    scalar_suffix = "Complex" if any(np.iscomplexobj(values) for values in blocks.values()) else ""
    tensor_type = getattr(sym, f"SymmetryBlockedTensorRank{rank}{scalar_suffix}")
    return tensor_type(
        [symmetry] * rank,
        [{labels["a"]: 2, labels["b"]: 2}] * rank,
        [(tuple(labels[label] for label in key), values) for key, values in blocks.items()],
    )


@pytest.mark.parametrize("complex_values", [False, True])
def test_declared_rdm_equivalence_roundtrip(site_data, complex_values, tmp_path):
    """Explicit equivalence preserves distinct block orbits, values and occupations."""
    alpha = sym.SymmetryLabel([sym.axes.alpha()])
    beta = sym.SymmetryLabel([sym.axes.beta()])
    one = np.array([[0.5, 0.25], [0.25, 0.5]])
    same = np.arange(16, dtype=float)
    mixed = np.arange(16, dtype=float)[::-1].copy() + 2
    if complex_values:
        one = np.array([[0.5, 0.25j], [-0.25j, 0.5]])
        same = same.astype(complex)
        mixed = mixed + 1j * np.arange(16)
    container = MPSContainer(
        [MPSSite.from_dense(values) for values in site_data],
        ModelOrbitals(2, sym.SymmetryProduct([sym.axes.spin(1, False)])),
        active_one_rdm=_blocked_rdm(2, {"aa": one}, equivalent=True),
        active_two_rdm=_blocked_rdm(4, {"aaaa": same, "aabb": mixed}, equivalent=True),
    )
    original = Wavefunction(container)
    restored = [original, Wavefunction.from_json(original.to_json())]
    path = tmp_path / "equivalent-rdms.wavefunction.h5"
    original.to_file(path, "hdf5")
    restored.append(Wavefunction.from_file(path, "hdf5"))

    for wavefunction in restored:
        actual = wavefunction.get_container()
        for tensor in (actual.active_one_rdm(), actual.active_two_rdm()):
            assert all(slot.axis(sym.AxisName.Spin).equivalent() for slot in tensor.symmetries())
        for block in wavefunction.get_active_one_rdm_spin_dependent():
            np.testing.assert_array_equal(block, one)
        for labels, expected in [
            ((alpha,) * 4, same),
            ((beta,) * 4, same),
            ((alpha, alpha, beta, beta), mixed),
            ((beta, beta, alpha, alpha), mixed),
        ]:
            np.testing.assert_array_equal(actual.active_two_rdm().block(labels), expected)
        np.testing.assert_array_equal(wavefunction.get_active_one_rdm_spin_traced(), 2 * one)
        np.testing.assert_array_equal(wavefunction.get_active_two_rdm_spin_traced(), 2 * same + 2 * mixed)
        occupations = wavefunction.active_orbital_occupations()
        for label in (alpha, beta):
            np.testing.assert_allclose(occupations.block([label]), [0.75, 0.25])


def test_explicit_zero_rdm_block_is_preserved(site_data):
    """An independent zero block is available data, even with equivalent orbitals."""
    one = np.diag([0.8, 0.2])
    zero = np.zeros((2, 2))
    container = MPSContainer(
        [MPSSite.from_dense(values) for values in site_data],
        ModelOrbitals(2, sym.SymmetryProduct([sym.axes.spin(1, True)])),
        active_one_rdm=_blocked_rdm(2, {"aa": one, "bb": zero}, equivalent=False),
    )
    alpha, beta = Wavefunction(container).get_active_one_rdm_spin_dependent()
    np.testing.assert_array_equal(alpha, one)
    np.testing.assert_array_equal(beta, zero)


@pytest.mark.parametrize(
    ("rank", "blocks", "equivalent", "raw_inputs"),
    [
        (2, {"aa": np.eye(2)}, False, {}),
        (4, {"aaaa": np.zeros(16)}, True, {}),
        (2, {"aa": np.eye(2)}, True, {"one_rdm_aa": np.eye(2), "one_rdm_bb": np.eye(2)}),
        (
            4,
            {"aaaa": np.zeros(16), "aabb": np.ones(16)},
            True,
            {"two_rdm_aaaa": np.zeros(16), "two_rdm_aabb": np.ones(16), "two_rdm_bbbb": np.zeros(16)},
        ),
    ],
)
def test_invalid_blocked_rdm_inputs(site_data, rank, blocks, equivalent, raw_inputs):
    """Missing independent blocks and conflicting representations are rejected."""
    keyword = "active_one_rdm" if rank == 2 else "active_two_rdm"
    with pytest.raises(ValueError, match="RDM"):
        MPSContainer(
            [MPSSite.from_dense(values) for values in site_data],
            ModelOrbitals(2, sym.SymmetryProduct([sym.axes.spin(1, True)])),
            **{keyword: _blocked_rdm(rank, blocks, equivalent=equivalent)},
            **raw_inputs,
        )


@pytest.mark.parametrize(
    "rdms",
    [
        {"one_rdm_spin_traced": np.zeros((3, 3))},
        {"two_rdm_spin_traced": np.zeros(15)},
        {"one_rdm_spin_traced": np.full((2, 2), np.nan)},
        {"two_rdm_spin_traced": np.full(16, complex(1, np.inf))},
        {"one_rdm_aa": np.zeros((2, 2))},
        {"two_rdm_aabb": np.zeros(16)},
    ],
)
def test_invalid_supplied_rdms(site_data, rdms):
    """Malformed optional RDMs fail in the shared native constructor."""
    with pytest.raises(ValueError, match="RDM"):
        MPSContainer([MPSSite.from_dense(values) for values in site_data], ModelOrbitals(2), **rdms)


def test_invalid_constructor_arguments():
    """Native argument failures reach Python as ValueError."""
    label = sym.SymmetryLabel([])
    with pytest.raises(ValueError, match="MPS site tensor must not be null"):
        MPSSite(None, [label], [label], [label])
    with pytest.raises(ValueError, match="MPS requires nonempty sites"):
        MPSContainer([], ModelOrbitals(1))
