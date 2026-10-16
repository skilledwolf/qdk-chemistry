"""Basic Python API tests for MPS storage."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import itertools

import numpy as np
import pytest

from qdk_chemistry.data import Configuration, ModelOrbitals, MPSContainer, MPSSite, Wavefunction
from qdk_chemistry.data import symmetry as sym


def make_site(values):
    """Wrap an array as one trivial-symmetry block."""
    left, physical, right = values.shape
    product, label = sym.SymmetryProduct([]), sym.SymmetryLabel([])
    tensor_type = sym.SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else sym.SymmetryBlockedTensorRank3
    tensor = tensor_type(
        [product] * 3,
        [{label: left}, {label: physical}, {label: right}],
        [((label,) * 3, values.reshape(left * physical, right))],
    )
    return MPSSite(tensor, [label], [label], [label])


@pytest.fixture
def site_data():
    """Use reproducible random input rather than a particular physical state."""
    rng = np.random.default_rng(42)
    return [rng.random((1, 4, 3)), rng.random((3, 4, 1))]


def test_construction_and_accessors(site_data):
    """The container retains supplied sites and optional metadata."""
    sites = [make_site(values) for values in site_data]
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
    assert not container.is_complex
    assert container.orthogonality_center == 1
    assert container.site_to_orbital_order == [1, 0]
    assert container.total_num_particles.value(sym.SymmetryLabel([])) == 2
    assert not container.has_active_num_particles()
    with pytest.raises(RuntimeError, match="Active particle-count is not set"):
        _ = container.active_num_particles
    for site, values in zip(container.sites, site_data, strict=True):
        assert site.shape == values.shape
        np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("complex_values", [False, True])
def test_serialization(site_data, complex_values, tmp_path):
    """Python JSON and file factories preserve real and complex tensor data."""
    if complex_values:
        rng = np.random.default_rng(43)
        site_data = [values + 1j * rng.random(values.shape) for values in site_data]
    original = Wavefunction(MPSContainer([make_site(values) for values in site_data], ModelOrbitals(2)))
    restored = [Wavefunction.from_json(original.to_json())]
    for extension, format_name in [("json", "json"), ("h5", "hdf5")]:
        path = tmp_path / f"mps.wavefunction.{extension}"
        original.to_file(path, format_name)
        restored.append(Wavefunction.from_file(path, format_name))
    for wavefunction in restored:
        assert wavefunction.content_hash() == original.content_hash()
        container = wavefunction.get_container()
        assert container.is_complex is complex_values
        for site, values in zip(container.sites, site_data, strict=True):
            assert site.to_dense().dtype == values.dtype
            np.testing.assert_array_equal(site.to_dense().reshape(site.shape), values)


@pytest.mark.parametrize("dimension", [2, 4])
def test_numpy_input_and_output_copies(dimension):
    """Strided input and exported arrays do not alias stored data."""
    values = np.random.default_rng(42).random((2, dimension, 3))[:, :, ::-1]
    expected = values.copy()
    site = make_site(values)
    values[:] = 0
    exported = site.to_dense()
    assert exported.shape == (2 * dimension, 3)
    np.testing.assert_array_equal(exported.reshape(site.shape), expected)
    exported[:] = 1
    block = site.tensor.block((sym.SymmetryLabel([]),) * 3)
    block[:] = 2
    np.testing.assert_array_equal(site.to_dense().reshape(site.shape), expected)


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
            [make_site(values) for values in site_data],
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

    sites = [make_site(values) for values in site_data]
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
        [make_site(values) for values in site_data],
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


@pytest.mark.parametrize("complex_values", [False, True])
def test_independent_two_rdm_blocks_roundtrip(site_data, complex_values, tmp_path):
    """Explicitly stored mixed blocks are retained without an equivalence declaration."""
    labels = {"a": sym.SymmetryLabel([sym.axes.alpha()]), "b": sym.SymmetryLabel([sym.axes.beta()])}
    blocks = {
        key: np.arange(16, dtype=float) + 20 * index for index, key in enumerate(("aaaa", "aabb", "bbaa", "bbbb"))
    }
    if complex_values:
        blocks = {key: values + 1j * (index + 1) for index, (key, values) in enumerate(blocks.items())}
    original = Wavefunction(
        MPSContainer(
            [make_site(values) for values in site_data],
            ModelOrbitals(2),
            active_two_rdm=_blocked_rdm(4, blocks, equivalent=False),
        )
    )
    restored = [original, Wavefunction.from_json(original.to_json())]
    path = tmp_path / "independent-rdms.wavefunction.h5"
    original.to_file(path, "hdf5")
    restored.append(Wavefunction.from_file(path, "hdf5"))

    for wavefunction in restored:
        tensor = wavefunction.get_container().active_two_rdm()
        assert all(not slot.axis(sym.AxisName.Spin).equivalent() for slot in tensor.symmetries())
        for key, expected in blocks.items():
            np.testing.assert_array_equal(tensor.block(tuple(labels[label] for label in key)), expected)
        np.testing.assert_array_equal(wavefunction.get_active_two_rdm_spin_traced(), sum(blocks.values()))


def test_explicit_zero_rdm_block_is_preserved(site_data):
    """An independent zero block is available data, even with equivalent orbitals."""
    one = np.diag([0.8, 0.2])
    zero = np.zeros((2, 2))
    container = MPSContainer(
        [make_site(values) for values in site_data],
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
            [make_site(values) for values in site_data],
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
        MPSContainer([make_site(values) for values in site_data], ModelOrbitals(2), **rdms)


def test_provenance(site_data, tmp_path):
    """Producer provenance round-trips verbatim."""
    sites = [make_site(values) for values in site_data]
    provenance = {"producer": "test", "energy": -1.25, "bond_dimensions": [4, 8]}
    original = Wavefunction(MPSContainer(sites, ModelOrbitals(2), provenance=provenance))
    path = tmp_path / "provenance.wavefunction.h5"
    original.to_file(path, "hdf5")
    for wavefunction in [original, Wavefunction.from_json(original.to_json()), Wavefunction.from_file(path, "hdf5")]:
        assert wavefunction.get_container().provenance == provenance
    assert MPSContainer(sites, ModelOrbitals(2)).provenance == {}
    with pytest.raises(ValueError, match="JSON object"):
        MPSContainer(sites, ModelOrbitals(2), provenance=[1, 2])
    with pytest.raises(ValueError, match="JSON compliant"):
        MPSContainer(sites, ModelOrbitals(2), provenance={"energy": float("nan")})


def test_supplied_properties_define_content_hash(site_data):
    """Supplied RDMs and entropies change the hash; getters that fill derived caches do not."""
    sites = [make_site(values) for values in site_data]
    plain = Wavefunction(MPSContainer(sites, ModelOrbitals(2)))
    supplied = Wavefunction(MPSContainer(sites, ModelOrbitals(2), one_rdm_spin_traced=np.eye(2)))
    other = Wavefunction(MPSContainer(sites, ModelOrbitals(2), one_rdm_spin_traced=2 * np.eye(2)))
    entropies = Wavefunction(MPSContainer(sites, ModelOrbitals(2), entropies={"single_orbital": np.array([0.3, 0.7])}))
    provenance = Wavefunction(MPSContainer(sites, ModelOrbitals(2), provenance={"producer": "test"}))
    assert Wavefunction.from_json(provenance.to_json()).content_hash() == provenance.content_hash()
    hashes = [wavefunction.content_hash() for wavefunction in (plain, supplied, other, entropies, provenance)]
    assert len(set(hashes)) == len(hashes)

    one = 0.5 * np.eye(2)
    mixed = np.zeros(16)
    mixed[[0, 3, 12, 15]] = 0.5
    blocked = Wavefunction(
        MPSContainer(
            sites,
            ModelOrbitals(2, sym.SymmetryProduct([sym.axes.spin(1, False)])),
            one_rdm_aa=one,
            one_rdm_bb=one,
            two_rdm_aaaa=np.zeros(16),
            two_rdm_aabb=mixed,
            two_rdm_bbbb=np.zeros(16),
        )
    )
    untouched = blocked.content_hash()
    blocked.get_active_one_rdm_spin_traced()
    blocked.get_active_two_rdm_spin_traced()
    np.testing.assert_allclose(blocked.get_single_orbital_entropies(), np.log(2))
    assert blocked.content_hash() == untouched
    assert Wavefunction.from_json(blocked.to_json()).content_hash() == untouched


def test_invalid_constructor_arguments():
    """Native argument failures reach Python as ValueError."""
    label = sym.SymmetryLabel([])
    with pytest.raises(ValueError, match="MPS site tensor must not be null"):
        MPSSite(None, [label], [label], [label])
    with pytest.raises(ValueError, match="MPS requires nonempty sites"):
        MPSContainer([], ModelOrbitals(1))


def test_particle_number_and_spin_blocked_sites(tmp_path):
    """(N, 2Ms) sectors label the physical and bond indices of a spinful MPS."""
    product = sym.SymmetryProduct([sym.axes.particle_number(4), sym.axes.spin_projection(2)])

    def label(number, two_ms):
        return sym.SymmetryLabel([sym.axes.particle_number_value(number), sym.axes.spin_value(two_ms)])

    physical = [label(0, 0), label(1, 1), label(1, -1), label(2, 0)]
    vacuum, total = label(0, 0), label(2, 0)
    rng = np.random.default_rng(5)
    first_values, second_values = rng.random(4), rng.random(4)
    # Charges accumulate from the left, so each first-site sector pairs with its complement.
    first = MPSSite(
        sym.SymmetryBlockedTensorRank3(
            [product] * 3,
            [{vacuum: 1}, dict.fromkeys(physical, 1), dict.fromkeys(physical, 1)],
            [
                ((vacuum, sector, sector), np.array([[value]]))
                for sector, value in zip(physical, first_values, strict=True)
            ],
        ),
        [vacuum],
        physical,
        physical,
    )
    second = MPSSite(
        sym.SymmetryBlockedTensorRank3(
            [product] * 3,
            [dict.fromkeys(physical, 1), dict.fromkeys(physical, 1), {total: 1}],
            [
                ((sector, partner, total), np.array([[value]]))
                for sector, partner, value in zip(physical, reversed(physical), second_values, strict=True)
            ],
        ),
        physical,
        physical,
        [total],
    )
    original = Wavefunction(MPSContainer([first, second], ModelOrbitals(2)))
    path = tmp_path / "n_sz.wavefunction.h5"
    original.to_file(path, "hdf5")
    # |du> creates b0 a1, one exchange away from determinant order a1 b0.
    expected = {
        "02": first_values[0] * second_values[0],
        "ud": first_values[1] * second_values[1],
        "du": -first_values[2] * second_values[2],
        "20": first_values[3] * second_values[3],
    }
    for wavefunction in [original, Wavefunction.from_json(original.to_json()), Wavefunction.from_file(path, "hdf5")]:
        for states in itertools.product("0ud2", repeat=2):
            state = "".join(states)
            actual = wavefunction.get_coefficient(Configuration.from_spin_half_string(state))
            assert actual == pytest.approx(expected.get(state, 0.0), abs=1e-14), state


def test_supplied_entropies(site_data, tmp_path):
    """Producer-supplied entropies reach the wavefunction accessors and survive serialization."""
    single = np.array([0.3, 0.7])
    mutual = np.array([[0.0, 0.2], [0.2, 0.0]])
    sites = [make_site(values) for values in site_data]
    original = Wavefunction(
        MPSContainer(sites, ModelOrbitals(2), entropies={"single_orbital": single, "mutual_information": mutual})
    )
    path = tmp_path / "entropies.wavefunction.h5"
    original.to_file(path, "hdf5")
    for wavefunction in [original, Wavefunction.from_json(original.to_json()), Wavefunction.from_file(path, "hdf5")]:
        np.testing.assert_array_equal(wavefunction.get_single_orbital_entropies(), single)
        np.testing.assert_array_equal(wavefunction.get_mutual_information(), mutual)
    pairs = Wavefunction(
        MPSContainer(
            sites, ModelOrbitals(2), entropies={"single_orbital": single, "two_orbital": np.array([[0, 0.8], [0.8, 0]])}
        )
    )
    np.testing.assert_allclose(Wavefunction.from_json(pairs.to_json()).get_mutual_information(), mutual)
    with pytest.raises(ValueError, match="single-orbital entropies"):
        MPSContainer(sites, ModelOrbitals(2), entropies={"single_orbital": np.ones(3)})
    with pytest.raises(ValueError, match="require single-orbital entropies"):
        MPSContainer(sites, ModelOrbitals(2), entropies={"two_orbital": np.zeros((2, 2))})
