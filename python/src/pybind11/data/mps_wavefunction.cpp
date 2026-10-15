// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace qdk::chemistry::data;

namespace {

template <std::size_t Rank>
using RdmInput = std::variant<
    std::shared_ptr<const SymmetryBlockedTensor<Rank, double>>,
    std::shared_ptr<const SymmetryBlockedTensor<Rank, std::complex<double>>>>;

template <std::size_t Rank>
std::optional<SymmetryBlockedTensorVariant<Rank>> rdm_from_python(
    const std::optional<RdmInput<Rank>>& value) {
  if (!value) return std::nullopt;
  return std::visit(
      [](const auto& tensor)
          -> std::optional<SymmetryBlockedTensorVariant<Rank>> {
        if (!tensor) return std::nullopt;
        return *tensor;
      },
      *value);
}

template <typename Scalar>
MPSSite site_from_dense(const py::array& values,
                        std::vector<Configuration> physical_basis) {
  const auto array = py::array_t<Scalar, py::array::forcecast>::ensure(values);
  if (!array) {
    throw py::type_error(
        "MPS dense site requires a real or complex numeric array.");
  }
  const auto input = array.template unchecked<3>();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> packed(
      input.shape(0) * input.shape(1), input.shape(2));
  for (py::ssize_t left = 0; left < input.shape(0); ++left) {
    for (py::ssize_t physical = 0; physical < input.shape(1); ++physical) {
      for (py::ssize_t right = 0; right < input.shape(2); ++right) {
        packed(left * input.shape(1) + physical, right) =
            input(left, physical, right);
      }
    }
  }
  return MPSSite::from_dense(std::move(packed), input.shape(1),
                             std::move(physical_basis));
}

template <typename Scalar>
std::shared_ptr<MPSSite> site_from_tensor(
    std::shared_ptr<const SymmetryBlockedTensor<3, Scalar>> tensor,
    std::vector<SymmetryLabel> left_order,
    std::vector<SymmetryLabel> physical_order,
    std::vector<SymmetryLabel> right_order,
    std::vector<Configuration> physical_basis) {
  if (!tensor) {
    throw std::invalid_argument("MPS site tensor must not be null.");
  }
  return std::make_shared<MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(*tensor),
      MPSSite::SectorOrders{std::move(left_order), std::move(physical_order),
                            std::move(right_order)},
      std::move(physical_basis));
}

}  // namespace

void bind_mps_wavefunction(py::module& data) {
  py::class_<MPSSite, py::smart_holder>(
      data, "MPSSite",
      "An explicit rank-3 MPS tensor with (left, physical, right) slots. "
      "Each block packs local_left * physical_extent + local_physical into "
      "rows and local_right into columns.")
      .def(py::init(&site_from_tensor<double>), py::arg("tensor"),
           py::arg("left_sector_order"), py::arg("physical_sector_order"),
           py::arg("right_sector_order"),
           py::arg("physical_basis") = std::vector<Configuration>{})
      .def(py::init(&site_from_tensor<std::complex<double>>), py::arg("tensor"),
           py::arg("left_sector_order"), py::arg("physical_sector_order"),
           py::arg("right_sector_order"),
           py::arg("physical_basis") = std::vector<Configuration>{})
      .def_static(
          "from_dense",
          [](const py::array& values,
             std::vector<Configuration> physical_basis) {
            if (values.ndim() != 3) {
              throw py::value_error(
                  "MPS dense site requires a rank-3 (left, physical, right) "
                  "array.");
            }
            const auto kind = values.dtype().kind();
            if (kind == 'c') {
              return site_from_dense<std::complex<double>>(
                  values, std::move(physical_basis));
            }
            if (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f') {
              return site_from_dense<double>(values, std::move(physical_basis));
            }
            throw py::type_error(
                "MPS dense site requires a real or complex numeric array.");
          },
          py::arg("array"),
          py::arg("physical_basis") = std::vector<Configuration>{},
          "Copy a (left, physical, right) array into one trivial-symmetry "
          "block. Real and complex numeric arrays, including strided arrays, "
          "are stored as float64 or complex128. The default physical basis is "
          "(0,1) for dimension two or (0,u,d,2) for dimension four.")
      .def_property_readonly(
          "tensor",
          [](const MPSSite& site) {
            return std::visit(
                [](const auto& value) {
                  return py::cast(value, py::return_value_policy::copy);
                },
                site.tensor());
          },
          "A copy of the bound real or complex rank-3 symmetry-blocked tensor.")
      .def_property_readonly("sector_orders", &MPSSite::sector_orders)
      .def_property_readonly("left_sector_order", &MPSSite::left_sector_order)
      .def_property_readonly("physical_sector_order",
                             &MPSSite::physical_sector_order)
      .def_property_readonly("right_sector_order", &MPSSite::right_sector_order)
      .def_property_readonly("physical_basis", &MPSSite::physical_basis,
                             "One-mode configurations in physical-index order.")
      .def_property_readonly("physical_dimension", &MPSSite::physical_dimension)
      .def_property_readonly("left_bond_dimension",
                             &MPSSite::left_bond_dimension)
      .def_property_readonly("right_bond_dimension",
                             &MPSSite::right_bond_dimension)
      .def_property_readonly("is_complex", &MPSSite::is_complex)
      .def_property_readonly("shape",
                             [](const MPSSite& site) {
                               return py::make_tuple(
                                   site.left_bond_dimension(),
                                   site.physical_dimension(),
                                   site.right_bond_dimension());
                             })
      .def("to_dense", &MPSSite::to_dense,
           "Return a packed matrix of shape (left * physical, right). "
           "Use site.to_dense().reshape(site.shape) for three-dimensional "
           "indexing.");

  py::class_<MPSContainer, WavefunctionContainer, py::smart_holder>(
      data, "MPSContainer",
      "Container to store MPS over an orbital or model-mode basis. Sites may "
      "have different local bases. Scale, phase and gauge are preserved; "
      "particle counts and canonical center are optional, unverified producer "
      "metadata.")
      .def(
          py::init(
              [](std::vector<MPSContainer::SitePtr> sites,
                 std::shared_ptr<Orbitals> orbitals,
                 std::shared_ptr<const MPSContainer::ParticleCount>
                     total_num_particles,
                 std::shared_ptr<const MPSContainer::ParticleCount>
                     active_num_particles,
                 std::optional<std::size_t> orthogonality_center,
                 std::vector<std::size_t> site_to_orbital_order,
                 const std::optional<ContainerTypes::MatrixVariant>&
                     one_rdm_spin_traced,
                 const std::optional<ContainerTypes::MatrixVariant>& one_rdm_aa,
                 const std::optional<ContainerTypes::MatrixVariant>& one_rdm_bb,
                 const std::optional<ContainerTypes::VectorVariant>&
                     two_rdm_spin_traced,
                 const std::optional<ContainerTypes::VectorVariant>&
                     two_rdm_aaaa,
                 const std::optional<ContainerTypes::VectorVariant>&
                     two_rdm_aabb,
                 const std::optional<ContainerTypes::VectorVariant>&
                     two_rdm_bbbb,
                 const std::optional<RdmInput<2>>& active_one_rdm,
                 const std::optional<RdmInput<4>>& active_two_rdm) {
                return std::make_unique<MPSContainer>(
                    std::move(sites), std::move(orbitals),
                    std::move(total_num_particles),
                    std::move(active_num_particles), orthogonality_center,
                    std::move(site_to_orbital_order), one_rdm_spin_traced,
                    one_rdm_aa, one_rdm_bb, two_rdm_spin_traced, two_rdm_aaaa,
                    two_rdm_aabb, two_rdm_bbbb, rdm_from_python(active_one_rdm),
                    rdm_from_python(active_two_rdm));
              }),
          py::arg("sites"), py::arg("orbitals"),
          py::arg("total_num_particles") = nullptr,
          py::arg("active_num_particles") = nullptr,
          py::arg("orthogonality_center") = std::nullopt,
          py::arg("site_to_orbital_order") = std::vector<std::size_t>{},
          py::arg("one_rdm_spin_traced") = std::nullopt,
          py::arg("one_rdm_aa") = std::nullopt,
          py::arg("one_rdm_bb") = std::nullopt,
          py::arg("two_rdm_spin_traced") = std::nullopt,
          py::arg("two_rdm_aaaa") = std::nullopt,
          py::arg("two_rdm_aabb") = std::nullopt,
          py::arg("two_rdm_bbbb") = std::nullopt,
          py::arg("active_one_rdm") = std::nullopt,
          py::arg("active_two_rdm") = std::nullopt,
          "Store supplied sites and optional real/complex RDM arrays. RDMs "
          "use active-orbital order. Alternatively, active_one_rdm and "
          "active_two_rdm accept symmetry-blocked tensors in the supported "
          "spin-diagonal layout. Block equivalence must be explicitly "
          "declared by the tensor symmetry metadata; missing arrays, "
          "particle counts and orbital equivalence do not imply RDM block "
          "equivalence. Do not combine blocked and raw block inputs for "
          "the same RDM rank.")
      .def_property_readonly("sites", &MPSContainer::sites)
      .def_property_readonly("orbitals", &MPSContainer::get_orbitals)
      .def("has_total_num_particles", &MPSContainer::has_total_num_particles)
      .def("has_active_num_particles", &MPSContainer::has_active_num_particles)
      .def_property_readonly("orthogonality_center",
                             &MPSContainer::orthogonality_center,
                             "Asserted canonical center, or None if unknown.")
      .def_property_readonly("site_to_orbital_order",
                             &MPSContainer::site_to_orbital_order,
                             "Permutation of active-mode slots, not full "
                             "orbital indices.")
      .def_property_readonly("num_sites", &MPSContainer::num_sites)
      .def_property_readonly("max_bond_dimension",
                             &MPSContainer::max_bond_dimension)
      .def(
          "to_json",
          [](const MPSContainer& self) { return self.to_json().dump(); },
          "Serialize all metadata and tensors without normalization.")
      .def_static(
          "from_json",
          [](const std::string& json) {
            return MPSContainer::from_json(nlohmann::json::parse(json));
          },
          py::arg("json"), "Restore a versioned MPS container JSON string.");
}
