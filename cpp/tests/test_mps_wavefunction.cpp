// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

namespace {

template <typename Scalar>
MPSContainer::SitePtr make_site(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    std::size_t physical_dimension = 4) {
  using SBT = SymmetryBlockedTensor<3, Scalar>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryLabel label;
  typename SBT::ExtentsArray extents;
  extents[0][label] =
      static_cast<std::size_t>(packed.rows()) / physical_dimension;
  extents[1][label] = physical_dimension;
  extents[2][label] = static_cast<std::size_t>(packed.cols());
  typename SBT::BlockMap blocks{
      {{label, label, label},
       std::make_shared<const Tensor<3, Scalar>>(packed)}};
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({trivial, trivial, trivial}, extents, std::move(blocks)));
  return std::make_shared<const MPSSite>(
      tensor, MPSSite::SectorOrders{{{label}, {label}, {label}}});
}

template <std::size_t Rank, typename Scalar = double>
SymmetryBlockedTensor<Rank, Scalar> make_blocked_rdm(
    typename SymmetryBlockedTensor<Rank, Scalar>::BlockMap blocks,
    bool equivalent = false) {
  using SBT = SymmetryBlockedTensor<Rank, Scalar>;
  auto symmetry = testing::spin_symmetry(equivalent);
  typename SBT::SymmetriesArray symmetries;
  symmetries.fill(symmetry);
  typename SBT::ExtentsArray extents;
  extents.fill({{axes::alpha(), 2}, {axes::beta(), 2}});
  return SBT(symmetries, extents, std::move(blocks));
}

class MPSContainerTest : public ::testing::Test {
 protected:
  Eigen::MatrixXd random_matrix(Eigen::Index rows, Eigen::Index cols) {
    return Eigen::MatrixXd::NullaryExpr(rows, cols,
                                        [&] { return distribution(rng); });
  }

  void SetUp() override {
    orbitals = std::make_shared<ModelOrbitals>(2);
    data = {random_matrix(4, 3), random_matrix(12, 1)};
    sites = {make_site(data[0]), make_site(data[1])};
    counts = std::make_shared<const MPSContainer::ParticleCount>(
        MPSContainer::ParticleCount::SymmetriesArray{
            std::make_shared<const SymmetryProduct>(
                SymmetryProduct::trivial())},
        MPSContainer::ParticleCount::BlockMap{
            {{SymmetryLabel{}}, std::make_shared<const std::size_t>(2)}});
  }

  std::unique_ptr<MPSContainer> make_container() {
    return std::make_unique<MPSContainer>(sites, orbitals, counts, counts, 1,
                                          std::vector<std::size_t>{1, 0});
  }

  MPSContainer with_blocked_rdms(
      const std::optional<SymmetryBlockedTensorVariant<2>>& one,
      const std::optional<SymmetryBlockedTensorVariant<4>>& two =
          std::nullopt) {
    return MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt, one, two);
  }

  std::mt19937 rng{42};
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};
  std::shared_ptr<Orbitals> orbitals;
  std::shared_ptr<const MPSContainer::ParticleCount> counts;
  std::array<Eigen::MatrixXd, 2> data;
  std::vector<MPSContainer::SitePtr> sites;
};

}  // namespace

TEST_F(MPSContainerTest, BasicProperties) {
  auto container = make_container();
  EXPECT_EQ(container->get_container_type(), "mps");
  EXPECT_EQ(container->get_orbitals(), orbitals);
  EXPECT_EQ(container->num_sites(), 2u);
  EXPECT_EQ(container->max_bond_dimension(), 3u);
  EXPECT_FALSE(container->is_complex());
  EXPECT_EQ(container->orthogonality_center(), 1u);
  EXPECT_EQ(container->site_to_orbital_order(),
            (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(container->total_num_particles()->value(SymmetryLabel{}), 2u);
  EXPECT_EQ(container->active_num_particles()->value(SymmetryLabel{}), 2u);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    EXPECT_EQ(container->sites()[i], sites[i]);
    EXPECT_EQ(sites[i]->physical_dimension(), 4u);
    EXPECT_EQ(sites[i]->physical_basis()[1],
              Configuration::from_spin_half_string("u"));
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(sites[i]->to_dense())
                    .isApprox(data[i], testing::wf_tolerance));
  }
}

TEST_F(MPSContainerTest, BinaryLocalBasisAndOptionalMetadata) {
  const Eigen::MatrixXd binary = random_matrix(2, 1);
  MPSContainer container({make_site(binary, 2)},
                         std::make_shared<ModelOrbitals>(1));
  EXPECT_EQ(container.sites()[0]->physical_basis()[1],
            Configuration::from_bitstring("1"));
  EXPECT_EQ(container.max_bond_dimension(), 1u);
  EXPECT_EQ(container.site_to_orbital_order(), (std::vector<std::size_t>{0}));
  EXPECT_FALSE(container.orthogonality_center().has_value());
  EXPECT_FALSE(container.has_total_num_particles());
  EXPECT_FALSE(container.has_active_num_particles());
  EXPECT_FALSE(container.has_one_rdm_spin_traced());
  EXPECT_FALSE(container.has_two_rdm_spin_traced());
  EXPECT_THROW(container.total_num_particles(), std::runtime_error);
  EXPECT_THROW(container.active_num_particles(), std::runtime_error);
  auto restored = MPSContainer::from_json(container.to_json());
  EXPECT_FALSE(restored->has_total_num_particles());
  EXPECT_FALSE(restored->orthogonality_center().has_value());
  EXPECT_TRUE(std::get<Eigen::MatrixXd>(restored->sites()[0]->to_dense())
                  .isApprox(binary, testing::wf_tolerance));
}

TEST_F(MPSContainerTest, BlockedSitePreservesBlocksAndSectorOrder) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray extents;
  extents[0] = {{zero, 1}, {one, 2}};
  extents[1] = {{zero, 1}, {one, 2}, {two, 1}};
  extents[2] = {{one, 2}, {two, 1}};
  const Eigen::MatrixXd block = random_matrix(4, 2);
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({symmetry, symmetry, symmetry}, extents,
          {{{one, one, one}, std::make_shared<const Eigen::MatrixXd>(block)}}));
  MPSSite::SectorOrders orders{{{one, zero}, {two, one, zero}, {two, one}}};
  MPSSite site(tensor, orders,
               {Configuration::from_spin_half_string("2"),
                Configuration::from_spin_half_string("u"),
                Configuration::from_spin_half_string("d"),
                Configuration::from_spin_half_string("0")});
  EXPECT_EQ(site.sector_orders(), orders);
  const auto& stored = std::get<SBT>(site.tensor());
  EXPECT_EQ(stored.extents(), extents);
  EXPECT_EQ(stored.num_blocks(), 1u);
  EXPECT_TRUE(
      stored.block({one, one, one}).isApprox(block, testing::wf_tolerance));
  Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(12, 3);
  expected.block(1, 1, 2, 2) = block.topRows(2);
  expected.block(5, 1, 2, 2) = block.bottomRows(2);
  EXPECT_TRUE(std::get<Eigen::MatrixXd>(site.to_dense())
                  .isApprox(expected, testing::wf_tolerance));
}

TEST_F(MPSContainerTest, SerializationAndClone) {
  Wavefunction original(make_container());
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_serialization_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  std::vector<std::shared_ptr<Wavefunction>> copies{
      std::make_shared<Wavefunction>(original),
      Wavefunction::from_json(original.to_json()),
      Wavefunction::from_hdf5(file)};
  for (const auto& copy : copies) {
    ASSERT_TRUE(copy->has_container_type<MPSContainer>());
    const auto& container = copy->get_container<MPSContainer>();
    EXPECT_EQ(copy->content_hash(), original.content_hash());
    EXPECT_EQ(container.orthogonality_center(), 1u);
    EXPECT_EQ(container.site_to_orbital_order(),
              (std::vector<std::size_t>{1, 0}));
    EXPECT_EQ(container.total_num_particles()->value(SymmetryLabel{}), 2u);
    EXPECT_EQ(container.active_num_particles()->value(SymmetryLabel{}), 2u);
    ASSERT_EQ(container.sites().size(), data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
      EXPECT_EQ(container.sites()[i]->sector_orders(),
                sites[i]->sector_orders());
      EXPECT_EQ(container.sites()[i]->physical_basis(),
                sites[i]->physical_basis());
      EXPECT_TRUE(std::get<Eigen::MatrixXd>(container.sites()[i]->to_dense())
                      .isApprox(data[i], testing::wf_tolerance));
    }
  }
}

TEST_F(MPSContainerTest, BlockedSerializationPreservesIndexSpacesAndData) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray first_extents, second_extents;
  first_extents[0] = {{zero, 1}};
  first_extents[1] = {{zero, 1}, {one, 2}, {two, 1}};
  first_extents[2] = {{zero, 1}, {one, 2}};
  second_extents[0] = first_extents[2];
  second_extents[1] = first_extents[1];
  second_extents[2] = {{two, 1}};
  SBT::BlockMap first_blocks{
      {{zero, one, one},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 2))},
      {{zero, two, zero},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}};
  SBT::BlockMap second_blocks{
      {{one, one, two},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(4, 1))},
      {{zero, zero, two},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}};
  const std::vector<Configuration> basis{
      Configuration::from_spin_half_string("2"),
      Configuration::from_spin_half_string("u"),
      Configuration::from_spin_half_string("d"),
      Configuration::from_spin_half_string("0")};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, first_extents, first_blocks)),
      MPSSite::SectorOrders{{{zero}, {two, one, zero}, {one, zero}}}, basis);
  auto second = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, second_extents, second_blocks)),
      MPSSite::SectorOrders{{{one, zero}, {two, one, zero}, {two}}}, basis);
  MPSContainer original({first, second}, orbitals);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_blocked_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  auto json_copy = MPSContainer::from_json(original.to_json());
  auto hdf_copy = MPSContainer::from_hdf5(file);
  for (const auto* copy : {json_copy.get(), hdf_copy.get()}) {
    ASSERT_EQ(copy->num_sites(), original.num_sites());
    for (std::size_t i = 0; i < original.num_sites(); ++i) {
      const auto& expected_site = *original.sites()[i];
      const auto& actual_site = *copy->sites()[i];
      EXPECT_EQ(actual_site.sector_orders(), expected_site.sector_orders());
      EXPECT_EQ(actual_site.physical_basis(), basis);
      const auto& expected = std::get<SBT>(expected_site.tensor());
      const auto& actual = std::get<SBT>(actual_site.tensor());
      EXPECT_EQ(actual.extents(), expected.extents());
      for (std::size_t slot = 0; slot < 3; ++slot) {
        EXPECT_EQ(*actual.symmetries()[slot], *expected.symmetries()[slot]);
      }
      ASSERT_EQ(actual.num_blocks(), expected.num_blocks());
      for (const auto& [labels, block] : expected.blocks()) {
        ASSERT_TRUE(actual.has_block(labels));
        EXPECT_TRUE(
            actual.block(labels).isApprox(*block, testing::wf_tolerance));
      }
    }
  }
}

TEST_F(MPSContainerTest, RejectsIncorrectBlockPacking) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryLabel label;
  SBT::ExtentsArray extents;
  extents[0][label] = 2;
  extents[1][label] = 2;
  extents[2][label] = 3;
  // SBT accepts the 12 entries; MPS requires a 4x3 matrix rather than 2x6.
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({symmetry, symmetry, symmetry}, extents,
          {{{label, label, label},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 6))}}));
  const MPSSite::SectorOrders orders{{{label}, {label}, {label}}};
  EXPECT_THROW(MPSSite(tensor, orders), std::invalid_argument);
}

TEST_F(MPSContainerTest, RejectsDifferentBondSectorsWithEqualDimensions) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray first_extents;
  first_extents[0] = {{zero, 1}};
  first_extents[1] = {{zero, 2}};
  first_extents[2] = {{zero, 1}, {one, 2}};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, first_extents,
          {{{zero, zero, one},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 2))}})),
      MPSSite::SectorOrders{{{zero}, {zero}, {zero, one}}});
  for (const std::string mismatch : {"none", "labels", "order"}) {
    SCOPED_TRACE(mismatch);
    const auto label = mismatch == "labels" ? two : zero;
    SBT::ExtentsArray extents;
    extents[0] = {{label, 1}, {one, 2}};
    extents[1] = {{zero, 2}};
    extents[2] = {{two, 1}};
    const std::vector<SymmetryLabel> order =
        mismatch == "order" ? std::vector<SymmetryLabel>{one, label}
                            : std::vector<SymmetryLabel>{label, one};
    auto second = std::make_shared<const MPSSite>(
        std::make_shared<const MPSSite::TensorVariant>(SBT(
            {symmetry, symmetry, symmetry}, extents,
            {{{one, zero, two},
              std::make_shared<const Eigen::MatrixXd>(random_matrix(4, 1))}})),
        MPSSite::SectorOrders{{order, {zero}, {two}}});
    ASSERT_EQ(first->right_bond_dimension(), second->left_bond_dimension());
    if (mismatch == "none") {
      EXPECT_NO_THROW(MPSContainer({first, second}, orbitals));
    } else {
      EXPECT_THROW(MPSContainer({first, second}, orbitals),
                   std::invalid_argument);
    }
  }
}

TEST_F(MPSContainerTest, ComplexSerialization) {
  using Complex = std::complex<double>;
  const Eigen::MatrixXcd first =
      data[0].cast<Complex>() +
      Complex(0, 1) * random_matrix(4, 3).cast<Complex>();
  const Eigen::MatrixXcd second =
      data[1].cast<Complex>() +
      Complex(0, 1) * random_matrix(12, 1).cast<Complex>();
  MPSContainer original({make_site(first), make_site(second)}, orbitals);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_complex_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  auto json_copy = MPSContainer::from_json(original.to_json());
  auto hdf_copy = MPSContainer::from_hdf5(file);
  for (const auto* copy : {&original, json_copy.get(), hdf_copy.get()}) {
    EXPECT_TRUE(copy->is_complex());
    EXPECT_TRUE(std::get<Eigen::MatrixXcd>(copy->sites()[0]->to_dense())
                    .isApprox(first, testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::MatrixXcd>(copy->sites()[1]->to_dense())
                    .isApprox(second, testing::wf_tolerance));
  }
}

TEST_F(MPSContainerTest, SuppliedRdms) {
  for (bool complex : {false, true}) {
    std::array<ContainerTypes::MatrixVariant, 3> one;
    std::array<ContainerTypes::VectorVariant, 4> two;
    for (auto& channel : one) {
      Eigen::MatrixXd real = random_matrix(2, 2);
      channel = real;
      if (complex)
        channel = Eigen::MatrixXcd(
            real.cast<std::complex<double>>() +
            std::complex<double>(0, 1) *
                random_matrix(2, 2).cast<std::complex<double>>());
    }
    for (auto& channel : two) {
      Eigen::VectorXd real = random_matrix(16, 1);
      channel = real;
      if (complex)
        channel = Eigen::VectorXcd(
            real.cast<std::complex<double>>() +
            std::complex<double>(0, 1) *
                random_matrix(16, 1).cast<std::complex<double>>());
    }
    MPSContainer original(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                          one[0], one[1], one[2], two[0], two[1], two[2],
                          two[3]);
    H5::FileAccPropList access;
    access.setCore(4096, false);
    H5::H5File file("mps_rdms_in_memory.h5", H5F_ACC_TRUNC,
                    H5::FileCreatPropList::DEFAULT, access);
    original.to_hdf5(file);
    std::vector<std::unique_ptr<WavefunctionContainer>> copies;
    // Raw convenience inputs and canonical blocked inputs must preserve the
    // same independent blocks and explicitly supplied traced companions.
    copies.push_back(std::make_unique<MPSContainer>(
        sites, orbitals, nullptr, nullptr, std::nullopt,
        std::vector<std::size_t>{}, one[0], std::nullopt, std::nullopt, two[0],
        std::nullopt, std::nullopt, std::nullopt, original.active_one_rdm(),
        original.active_two_rdm()));
    copies.push_back(original.clone());
    copies.push_back(MPSContainer::from_json(original.to_json()));
    copies.push_back(MPSContainer::from_hdf5(file));
    for (const auto& copy : copies) {
      copy->clear_caches();
      ASSERT_TRUE(copy->has_one_rdm_spin_dependent());
      ASSERT_TRUE(copy->has_two_rdm_spin_dependent());
      const auto [aa, bb] = copy->get_active_one_rdm_spin_dependent();
      const auto [aaaa, aabb, bbbb] = copy->get_active_two_rdm_spin_dependent();
      const std::array<ContainerTypes::MatrixVariant, 3> actual_one{
          copy->get_active_one_rdm_spin_traced(), aa, bb};
      const std::array<ContainerTypes::VectorVariant, 4> actual_two{
          copy->get_active_two_rdm_spin_traced(), aaaa, aabb, bbbb};
      for (std::size_t i = 0; i < one.size(); ++i) {
        std::visit(
            [&](const auto& expected) {
              using Matrix = std::decay_t<decltype(expected)>;
              EXPECT_TRUE(std::get<Matrix>(actual_one[i])
                              .isApprox(expected, testing::wf_tolerance));
            },
            one[i]);
      }
      for (std::size_t i = 0; i < two.size(); ++i) {
        std::visit(
            [&](const auto& expected) {
              using Vector = std::decay_t<decltype(expected)>;
              EXPECT_TRUE(std::get<Vector>(actual_two[i])
                              .isApprox(expected, testing::wf_tolerance));
            },
            two[i]);
      }
    }
  }
}

TEST_F(MPSContainerTest, RejectsInvalidSuppliedRdmDimensionsAndEntries) {
  for (int channel = 0; channel < 7; ++channel) {
    for (bool wrong_shape : {false, true}) {
      SCOPED_TRACE(channel);
      SCOPED_TRACE(wrong_shape);
      std::array<std::optional<ContainerTypes::MatrixVariant>, 3> one;
      std::array<std::optional<ContainerTypes::VectorVariant>, 4> two;
      for (auto& value : one) value = Eigen::MatrixXd::Identity(2, 2).eval();
      for (auto& value : two) value = Eigen::VectorXd::Zero(16).eval();
      if (channel < 3) {
        Eigen::MatrixXd invalid =
            Eigen::MatrixXd::Zero(wrong_shape ? 3 : 2, wrong_shape ? 3 : 2);
        if (!wrong_shape)
          invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
        one[channel] = invalid;
      } else {
        Eigen::VectorXd invalid = Eigen::VectorXd::Zero(wrong_shape ? 81 : 16);
        if (!wrong_shape) invalid(0) = std::numeric_limits<double>::infinity();
        two[channel - 3] = invalid;
      }
      EXPECT_THROW(
          MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                       one[0], one[1], one[2], two[0], two[1], two[2], two[3]),
          std::invalid_argument);
    }
  }
  Eigen::MatrixXcd invalid = Eigen::MatrixXcd::Identity(2, 2);
  invalid(0, 0) = {0, std::numeric_limits<double>::infinity()};
  EXPECT_THROW(MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                            ContainerTypes::MatrixVariant{invalid}),
               std::invalid_argument);
}

TEST_F(MPSContainerTest, ValidatesRdmsAfterJsonAndHdf5Deserialization) {
  const ContainerTypes::MatrixVariant rdm =
      Eigen::MatrixXd::Identity(2, 2).eval();
  MPSContainer original(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        rdm);
  auto json = original.to_json();
  json["rdms"]["one_rdm_spin_traced"] = {
      {1., 0., 0.}, {0., 1., 0.}, {0., 0., 0.}};
  EXPECT_THROW(MPSContainer::from_json(json), std::invalid_argument);

  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_invalid_rdms_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  auto dataset = file.openDataSet("rdms/one_rdm_spin_traced");
  const std::array<double, 4> invalid{std::numeric_limits<double>::quiet_NaN(),
                                      0., 0., 1.};
  dataset.write(invalid.data(), H5::PredType::NATIVE_DOUBLE);
  EXPECT_THROW(MPSContainer::from_hdf5(file), std::invalid_argument);

  const ContainerTypes::VectorVariant two = Eigen::VectorXd::Zero(16).eval();
  MPSContainer spin(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                    std::nullopt, rdm, rdm, std::nullopt, two, two, two);
  auto malformed_two = spin.to_json();
  auto& block = malformed_two["rdms"]["active_two_rdm"]["blocks"][0]["block"];
  block["rows"] = 15;
  block["data"] = std::vector<double>(15, 0.);
  EXPECT_THROW(MPSContainer::from_json(malformed_two), std::invalid_argument);
  auto incomplete_spin = spin.to_json();
  incomplete_spin["rdms"]["active_one_rdm"]["blocks"].erase(0);
  EXPECT_THROW(MPSContainer::from_json(incomplete_spin), std::invalid_argument);
}

TEST_F(MPSContainerTest, NaturalOccupationsFromSuppliedRdms) {
  auto spin_orbitals = testing::create_test_orbitals(2, 2);
  Eigen::MatrixXd aa(2, 2), bb(2, 2);
  aa << .5, .3, .3, .5;
  bb << .25, 0., 0., .75;
  MPSContainer spin(sites, spin_orbitals, nullptr, nullptr, std::nullopt, {},
                    std::nullopt, ContainerTypes::MatrixVariant{aa},
                    ContainerTypes::MatrixVariant{bb});
  const auto occupations = spin.active_orbital_occupations();
  const Eigen::Vector2d expected_alpha(.8, .2), expected_beta(.75, .25);
  EXPECT_TRUE(occupations->block({axes::alpha()}).isApprox(expected_alpha));
  EXPECT_TRUE(occupations->block({axes::beta()}).isApprox(expected_beta));
  StateVectorContainer reference(
      ContainerTypes::VectorVariant{Eigen::VectorXd::Constant(2, 1.).eval()},
      {Configuration::from_spin_half_string("20"),
       Configuration::from_spin_half_string("02")},
      spin_orbitals, std::nullopt, ContainerTypes::MatrixVariant{aa},
      ContainerTypes::MatrixVariant{bb}, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt);
  for (const auto& channel : {axes::alpha(), axes::beta()}) {
    EXPECT_TRUE(occupations->block({channel}).isApprox(
        reference.active_orbital_occupations()->block({channel})));
  }
  Eigen::MatrixXcd complex_aa = aa.cast<std::complex<double>>();
  complex_aa(0, 1) = {0., .3};
  complex_aa(1, 0) = {0., -.3};
  const Eigen::MatrixXcd complex_bb = bb.cast<std::complex<double>>();
  MPSContainer complex_spin(sites, spin_orbitals, nullptr, nullptr,
                            std::nullopt, {}, std::nullopt,
                            ContainerTypes::MatrixVariant{complex_aa},
                            ContainerTypes::MatrixVariant{complex_bb});
  EXPECT_TRUE(complex_spin.active_orbital_occupations()
                  ->block({axes::alpha()})
                  .isApprox(expected_alpha));
  EXPECT_TRUE(complex_spin.active_orbital_occupations()
                  ->block({axes::beta()})
                  .isApprox(expected_beta));

  Eigen::MatrixXcd traced(2, 2);
  traced << 1., std::complex<double>(0., .6), std::complex<double>(0., -.6), 1.;
  MPSContainer spinless(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        ContainerTypes::MatrixVariant{traced});
  const Eigen::Vector2d expected_traced(1.6, .4);
  EXPECT_TRUE(spinless.active_orbital_occupations()
                  ->block({SymmetryLabel{}})
                  .isApprox(expected_traced));
  EXPECT_TRUE(spinless.total_orbital_occupations()
                  ->block({SymmetryLabel{}})
                  .isApprox(expected_traced));
  EXPECT_THROW(MPSContainer(sites, orbitals).active_orbital_occupations(),
               std::runtime_error);
  aa(0, 1) = .9;
  MPSContainer nonhermitian(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                            ContainerTypes::MatrixVariant{aa});
  EXPECT_THROW(nonhermitian.active_orbital_occupations(), std::runtime_error);
}

TEST_F(MPSContainerTest, TotalOccupationsEmbedActiveAndInactiveOrbitals) {
  const auto symmetry = testing::spin_symmetry(false);
  const std::unordered_map<SymmetryLabel, std::size_t> extents{
      {axes::alpha(), 4}, {axes::beta(), 4}};
  auto indices = [&](const std::vector<std::uint32_t>& selected) {
    return std::make_shared<const SymmetryBlockedIndexSet>(
        symmetry, extents,
        std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>{
            {axes::alpha(), selected}, {axes::beta(), selected}});
  };
  auto spin_orbitals =
      std::make_shared<ModelOrbitals>(indices({1, 3}), indices({0}));
  Eigen::MatrixXd aa(2, 2), bb(2, 2);
  aa << .5, .3, .3, .5;
  bb << .25, 0., 0., .75;
  MPSContainer container(sites, spin_orbitals, nullptr, nullptr, std::nullopt,
                         {}, std::nullopt, ContainerTypes::MatrixVariant{aa},
                         ContainerTypes::MatrixVariant{bb});
  const auto occupations = container.total_orbital_occupations();
  const Eigen::Vector4d expected_alpha(1., .8, 0., .2);
  const Eigen::Vector4d expected_beta(1., .75, 0., .25);
  EXPECT_TRUE(occupations->block({axes::alpha()}).isApprox(expected_alpha));
  EXPECT_TRUE(occupations->block({axes::beta()}).isApprox(expected_beta));

  const auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  auto spinless_indices = [&](const std::vector<std::uint32_t>& selected) {
    return std::make_shared<const SymmetryBlockedIndexSet>(
        trivial,
        std::unordered_map<SymmetryLabel, std::size_t>{{SymmetryLabel{}, 4}},
        std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>{
            {SymmetryLabel{}, selected}});
  };
  auto spinless_orbitals = std::make_shared<ModelOrbitals>(
      spinless_indices({1, 3}), spinless_indices({0}));
  MPSContainer spinless(sites, spinless_orbitals, nullptr, nullptr,
                        std::nullopt, {}, ContainerTypes::MatrixVariant{aa});
  EXPECT_TRUE(spinless.total_orbital_occupations()
                  ->block({SymmetryLabel{}})
                  .isApprox(expected_alpha));
}

TEST_F(MPSContainerTest, RejectsIncompleteSpinRdmGroups) {
  const ContainerTypes::MatrixVariant one =
      Eigen::MatrixXd::Identity(2, 2).eval();
  const ContainerTypes::VectorVariant two = Eigen::VectorXd::Zero(16).eval();
  for (bool alpha : {false, true}) {
    EXPECT_THROW(
        MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                     std::nullopt, alpha ? std::optional(one) : std::nullopt,
                     alpha ? std::nullopt : std::optional(one)),
        std::invalid_argument);
  }
  for (int mask = 1; mask < 7; ++mask) {
    EXPECT_THROW(
        MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                     std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                     mask & 1 ? std::optional(two) : std::nullopt,
                     mask & 2 ? std::optional(two) : std::nullopt,
                     mask & 4 ? std::optional(two) : std::nullopt),
        std::invalid_argument);
  }
}

TEST_F(MPSContainerTest, ExplicitRdmEquivalenceSurvivesCopiesAndSerialization) {
  orbitals = testing::create_test_orbitals(2, 2);
  auto check = [&]<typename Scalar>() {
    using Matrix = Tensor<2, Scalar>;
    using Vector = Tensor<4, Scalar>;
    Matrix one = Matrix::Identity(2, 2) * Scalar(0.5);
    if constexpr (std::is_same_v<Scalar, double>) {
      one(0, 1) = one(1, 0) = 0.25;
    } else {
      one(0, 1) = Scalar(0, 0.25);
      one(1, 0) = Scalar(0, -0.25);
    }
    auto one_ptr = std::make_shared<const Matrix>(one);
    auto zero = std::make_shared<const Vector>(Vector::Zero(16));
    const Vector mixed = Eigen::VectorXd::LinSpaced(16, 0., 0.3).cast<Scalar>();
    auto mixed_ptr = std::make_shared<const Vector>(mixed);
    const auto one_rdm = make_blocked_rdm<2, Scalar>(
        {{{axes::alpha(), axes::alpha()}, one_ptr}}, true);
    const auto two_rdm = make_blocked_rdm<4, Scalar>(
        {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}, zero},
         {{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()},
          mixed_ptr}},
        true);
    auto original = with_blocked_rdms(one_rdm, two_rdm);
    H5::FileAccPropList access;
    access.setCore(4096, false);
    H5::H5File file("mps_equivalent_rdms_in_memory.h5", H5F_ACC_TRUNC,
                    H5::FileCreatPropList::DEFAULT, access);
    original.to_hdf5(file);
    std::vector<std::unique_ptr<WavefunctionContainer>> copies;
    copies.push_back(original.clone());
    copies.push_back(MPSContainer::from_json(original.to_json()));
    copies.push_back(MPSContainer::from_hdf5(file));
    for (const auto& copy : copies) {
      const auto& restored_one =
          std::get<SymmetryBlockedTensor<2, Scalar>>(copy->active_one_rdm());
      const auto& restored_two =
          std::get<SymmetryBlockedTensor<4, Scalar>>(copy->active_two_rdm());
      EXPECT_TRUE(
          restored_one.symmetries()[0]->axis(AxisName::Spin).equivalent());
      EXPECT_TRUE(
          restored_two.symmetries()[0]->axis(AxisName::Spin).equivalent());
      EXPECT_EQ(restored_one.block_ptr({axes::alpha(), axes::alpha()}),
                restored_one.block_ptr({axes::beta(), axes::beta()}));
      EXPECT_EQ(restored_two.block_ptr(
                    {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}),
                restored_two.block_ptr({axes::beta(), axes::beta(),
                                        axes::alpha(), axes::alpha()}));
      EXPECT_TRUE(
          restored_two
              .block({axes::beta(), axes::beta(), axes::beta(), axes::beta()})
              .isZero());
      EXPECT_TRUE(std::get<Matrix>(copy->get_active_one_rdm_spin_traced())
                      .isApprox(2 * one));
      // The deliberately asymmetric mixed block distinguishes a declared
      // bbaa alias from the index-pair transpose used for three raw blocks.
      EXPECT_TRUE(std::get<Vector>(copy->get_active_two_rdm_spin_traced())
                      .isApprox(2 * mixed));
      const Eigen::Vector2d expected(0.75, 0.25);
      const auto occupations = copy->active_orbital_occupations();
      EXPECT_TRUE(occupations->block({axes::alpha()}).isApprox(expected));
      EXPECT_TRUE(occupations->block({axes::beta()}).isApprox(expected));
    }
  };
  check.template operator()<double>();
  check.template operator()<std::complex<double>>();
}

TEST_F(MPSContainerTest, BlockedRdmsRequireIndependentRepresentatives) {
  orbitals = testing::create_test_orbitals(2, 2);
  ASSERT_TRUE(orbitals->is_restricted());
  auto one =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(2, 2));
  const auto incomplete_one =
      make_blocked_rdm<2>({{{axes::alpha(), axes::alpha()}, one}});
  EXPECT_THROW(with_blocked_rdms(incomplete_one), std::invalid_argument);
  const auto declared_one =
      make_blocked_rdm<2>({{{axes::alpha(), axes::alpha()}, one}}, true);
  EXPECT_NO_THROW(with_blocked_rdms(declared_one));

  auto zero =
      std::make_shared<const Eigen::VectorXd>(Eigen::VectorXd::Zero(16));
  const auto incomplete_two = make_blocked_rdm<4>(
      {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}, zero}},
      true);
  EXPECT_THROW(with_blocked_rdms(declared_one, incomplete_two),
               std::invalid_argument);
  const auto declared_two = make_blocked_rdm<4>(
      {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}, zero},
       {{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}, zero}},
      true);
  auto complete = with_blocked_rdms(declared_one, declared_two);
  auto invalid = complete.to_json();
  invalid["rdms"]["active_two_rdm"] = incomplete_two.to_json();
  EXPECT_THROW(MPSContainer::from_json(invalid), std::invalid_argument);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_missing_block_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  complete.to_hdf5(file);
  ASSERT_GE(H5Ldelete(file.getId(), "rdms/active_two_rdm", H5P_DEFAULT), 0);
  auto group = file.createGroup("rdms/active_two_rdm");
  const hbool_t complex = false;
  group
      .createAttribute("is_complex", H5::PredType::NATIVE_HBOOL,
                       H5::DataSpace(H5S_SCALAR))
      .write(H5::PredType::NATIVE_HBOOL, &complex);
  incomplete_two.to_hdf5(group);
  EXPECT_THROW(MPSContainer::from_hdf5(file), std::invalid_argument);
}

TEST_F(MPSContainerTest, IndependentFourthRdmBlockSurvivesSerialization) {
  auto check = [&]<typename Scalar>() {
    using Vector = Tensor<4, Scalar>;
    using SBT = SymmetryBlockedTensor<4, Scalar>;
    const typename SBT::Labels aaaa{axes::alpha(), axes::alpha(), axes::alpha(),
                                    axes::alpha()};
    const typename SBT::Labels aabb{axes::alpha(), axes::alpha(), axes::beta(),
                                    axes::beta()};
    const typename SBT::Labels bbaa{axes::beta(), axes::beta(), axes::alpha(),
                                    axes::alpha()};
    const typename SBT::Labels bbbb{axes::beta(), axes::beta(), axes::beta(),
                                    axes::beta()};
    const Vector same = Vector::Constant(16, Scalar(0.25));
    const Vector mixed = Eigen::VectorXd::LinSpaced(16, 0., 0.3).cast<Scalar>();
    Vector fourth = Eigen::VectorXd::LinSpaced(16, 0.4, 0.8).cast<Scalar>();
    if constexpr (!std::is_same_v<Scalar, double>) {
      fourth *= Scalar(1., 0.5);
    }
    const auto tensor = make_blocked_rdm<4, Scalar>(
        {{aaaa, std::make_shared<const Vector>(same)},
         {aabb, std::make_shared<const Vector>(mixed)},
         {bbaa, std::make_shared<const Vector>(fourth)},
         {bbbb, std::make_shared<const Vector>(same)}});
    auto original = with_blocked_rdms(std::nullopt, tensor);
    H5::FileAccPropList access;
    access.setCore(4096, false);
    H5::H5File file("mps_four_rdm_blocks_in_memory.h5", H5F_ACC_TRUNC,
                    H5::FileCreatPropList::DEFAULT, access);
    original.to_hdf5(file);
    std::vector<std::unique_ptr<WavefunctionContainer>> copies;
    copies.push_back(original.clone());
    copies.push_back(MPSContainer::from_json(original.to_json()));
    copies.push_back(MPSContainer::from_hdf5(file));
    for (const auto& copy : copies) {
      const auto& restored = std::get<SBT>(copy->active_two_rdm());
      EXPECT_FALSE(restored.symmetries()[0]->axis(AxisName::Spin).equivalent());
      EXPECT_NE(restored.block_ptr(aabb), restored.block_ptr(bbaa));
      EXPECT_TRUE(restored.block(bbaa).isApprox(fourth));
      // Synthetic data distinguishes explicit storage from both an alias and
      // the pair-transpose convention used when only three blocks are supplied.
      EXPECT_TRUE(std::get<Vector>(copy->get_active_two_rdm_spin_traced())
                      .isApprox(2 * same + mixed + fourth));
    }
  };
  check.template operator()<double>();
  check.template operator()<std::complex<double>>();
}

TEST_F(MPSContainerTest, RejectsConflictingOrUnsupportedBlockedRdms) {
  auto one =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(2, 2));
  const auto declared_one =
      make_blocked_rdm<2>({{{axes::alpha(), axes::alpha()}, one}}, true);
  auto zero =
      std::make_shared<const Eigen::VectorXd>(Eigen::VectorXd::Zero(16));
  const auto declared_two = make_blocked_rdm<4>(
      {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}, zero},
       {{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}, zero}},
      true);
  EXPECT_THROW(
      MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                   std::nullopt, ContainerTypes::MatrixVariant{*one},
                   ContainerTypes::MatrixVariant{*one}, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, declared_one),
      std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                            std::nullopt, std::nullopt, std::nullopt,
                            std::nullopt, ContainerTypes::VectorVariant{*zero},
                            ContainerTypes::VectorVariant{*zero},
                            ContainerTypes::VectorVariant{*zero}, std::nullopt,
                            declared_two),
               std::invalid_argument);
  const auto coherences =
      make_blocked_rdm<2>({{{axes::alpha(), axes::alpha()}, one},
                           {{axes::beta(), axes::beta()}, one},
                           {{axes::alpha(), axes::beta()}, one}});
  EXPECT_THROW(with_blocked_rdms(coherences), std::invalid_argument);
  const auto unsupported_two = make_blocked_rdm<4>(
      {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}, zero},
       {{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}, zero},
       {{axes::beta(), axes::beta(), axes::beta(), axes::beta()}, zero},
       {{axes::alpha(), axes::beta(), axes::alpha(), axes::beta()},
        std::make_shared<const Eigen::VectorXd>(*zero)}});
  EXPECT_THROW(with_blocked_rdms(std::nullopt, unsupported_two),
               std::invalid_argument);
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryBlockedTensor<2> unsupported(
      {trivial, trivial}, {{{{SymmetryLabel{}, 2}}, {{SymmetryLabel{}, 2}}}},
      {{{SymmetryLabel{}, SymmetryLabel{}}, one}});
  EXPECT_THROW(with_blocked_rdms(unsupported), std::invalid_argument);
}

TEST_F(MPSContainerTest, ProvenanceRoundTrips) {
  const nlohmann::json provenance{
      {"producer", "test"},
      {"energy", -1.25},
      {"bond_dimensions", nlohmann::json::array({4, 8})}};
  auto with_provenance = [&](const nlohmann::json& value) {
    return MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, OrbitalEntropies{}, value);
  };
  const auto original = with_provenance(provenance);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_provenance_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  const std::shared_ptr<const MPSContainer> json_copy =
      MPSContainer::from_json(original.to_json());
  const std::shared_ptr<const MPSContainer> hdf_copy =
      MPSContainer::from_hdf5(file);
  for (const auto* copy : {&original, json_copy.get(), hdf_copy.get()}) {
    EXPECT_EQ(copy->provenance(), provenance);
  }
  const MPSContainer plain(sites, orbitals);
  EXPECT_TRUE(plain.provenance().is_object());
  EXPECT_TRUE(plain.provenance().empty());
  auto without = original.to_json();
  without.erase("provenance");
  EXPECT_TRUE(MPSContainer::from_json(without)->provenance().empty());
  EXPECT_THROW(with_provenance(nlohmann::json::array({1, 2})),
               std::invalid_argument);
}

TEST_F(MPSContainerTest, SuppliedPropertiesDefineContentHash) {
  auto hash = [](const MPSContainer& container) {
    return Wavefunction(container.clone()).content_hash();
  };
  const ContainerTypes::MatrixVariant rdm =
      Eigen::MatrixXd::Identity(2, 2).eval();
  const ContainerTypes::MatrixVariant other_rdm =
      (2 * Eigen::MatrixXd::Identity(2, 2)).eval();
  const ContainerTypes::VectorVariant two_rdm =
      Eigen::VectorXd::Zero(16).eval();
  OrbitalEntropies single, other_single, mutual;
  single.single_orbital = Eigen::Vector2d(0.3, 0.7);
  other_single.single_orbital = Eigen::Vector2d(0.3, 0.6);
  mutual = single;
  mutual.mutual_information = Eigen::Matrix2d{{0., 0.2}, {0.2, 0.}};
  auto with_entropies = [&](const OrbitalEntropies& entropies) {
    return MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, entropies);
  };
  const MPSContainer with_provenance(
      sites, orbitals, nullptr, nullptr, std::nullopt, {}, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, OrbitalEntropies{},
      nlohmann::json{{"producer", "test"}});
  EXPECT_EQ(hash(*MPSContainer::from_json(with_provenance.to_json())),
            hash(with_provenance));
  const std::vector<std::string> hashes{
      hash(MPSContainer(sites, orbitals)),
      hash(MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        rdm)),
      hash(MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        other_rdm)),
      hash(MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        std::nullopt, std::nullopt, std::nullopt, two_rdm)),
      hash(with_entropies(single)),
      hash(with_entropies(other_single)),
      hash(with_entropies(mutual)),
      hash(with_provenance)};
  EXPECT_EQ(std::set<std::string>(hashes.begin(), hashes.end()).size(),
            hashes.size());
}

TEST_F(MPSContainerTest, ContentHashIgnoresLazilyDerivedProperties) {
  orbitals = testing::create_test_orbitals(2, 2);
  // RDMs of (|20> + |02>)/sqrt(2), whose single-orbital entropies are finite.
  const Eigen::MatrixXd one = 0.5 * Eigen::MatrixXd::Identity(2, 2);
  const Eigen::VectorXd same = Eigen::VectorXd::Zero(16);
  Eigen::VectorXd mixed = Eigen::VectorXd::Zero(16);
  for (const Eigen::Index index : {0, 3, 12, 15}) mixed(index) = 0.5;
  OrbitalEntropies entropies;
  entropies.mutual_information = Eigen::Matrix2d{{0., 0.5}, {0.5, 0.}};
  auto one_ptr = std::make_shared<const Eigen::MatrixXd>(one);
  auto same_ptr = std::make_shared<const Eigen::VectorXd>(same);
  auto mixed_ptr = std::make_shared<const Eigen::VectorXd>(mixed);
  using Labels = SymmetryBlockedTensor<4>::Labels;
  const Labels aaaa{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()};
  const Labels aabb{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()};
  const Labels bbaa{axes::beta(), axes::beta(), axes::alpha(), axes::alpha()};
  const Labels bbbb{axes::beta(), axes::beta(), axes::beta(), axes::beta()};
  auto blocked = [&](const SymmetryBlockedTensorVariant<2>& one_rdm,
                     const SymmetryBlockedTensorVariant<4>& two_rdm) {
    return MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                        std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                        std::nullopt, std::nullopt, std::nullopt, one_rdm,
                        two_rdm, entropies);
  };
  const std::vector<std::pair<std::string, MPSContainer>> cases{
      {"three raw blocks",
       MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                    std::nullopt, ContainerTypes::MatrixVariant{one},
                    ContainerTypes::MatrixVariant{one}, std::nullopt,
                    ContainerTypes::VectorVariant{same},
                    ContainerTypes::VectorVariant{mixed},
                    ContainerTypes::VectorVariant{same}, std::nullopt,
                    std::nullopt, entropies)},
      {"four blocks",
       blocked(make_blocked_rdm<2>({{{axes::alpha(), axes::alpha()}, one_ptr},
                                    {{axes::beta(), axes::beta()}, one_ptr}}),
               make_blocked_rdm<4>({{aaaa, same_ptr},
                                    {aabb, mixed_ptr},
                                    {bbaa, mixed_ptr},
                                    {bbbb, same_ptr}}))},
      {"declared equivalence",
       blocked(
           make_blocked_rdm<2>({{{axes::alpha(), axes::alpha()}, one_ptr}},
                               true),
           make_blocked_rdm<4>({{aaaa, same_ptr}, {aabb, mixed_ptr}}, true))}};
  for (const auto& [name, container] : cases) {
    SCOPED_TRACE(name);
    const auto untouched = Wavefunction(container.clone()).content_hash();
    auto used = container.clone();
    used->get_active_one_rdm_spin_traced();
    used->get_active_two_rdm_spin_traced();
    used->active_orbital_occupations();
    used->total_orbital_occupations();
    EXPECT_NEAR(used->get_single_orbital_entropies()(0), std::log(2.), 1e-12);
    used->get_two_orbital_entropies();
    used->get_mutual_information();
    const auto json = used->to_json();
    // The getters filled caches that serialization now writes.
    ASSERT_TRUE(json.at("rdms").contains("one_rdm_spin_traced"));
    ASSERT_TRUE(json.at("rdms").contains("two_rdm_spin_traced"));
    ASSERT_TRUE(json.contains("single_orbital_entropies"));
    H5::FileAccPropList access;
    access.setCore(4096, false);
    H5::H5File file("mps_hash_caches_in_memory.h5", H5F_ACC_TRUNC,
                    H5::FileCreatPropList::DEFAULT, access);
    used->to_hdf5(file);
    EXPECT_EQ(Wavefunction(MPSContainer::from_json(json)).content_hash(),
              untouched);
    EXPECT_EQ(Wavefunction(MPSContainer::from_hdf5(file)).content_hash(),
              untouched);
    EXPECT_EQ(Wavefunction(std::move(used)).content_hash(), untouched);
  }
}

TEST_F(MPSContainerTest, RejectsInvalidStructure) {
  EXPECT_THROW(MPSContainer({}, orbitals), std::invalid_argument);
  EXPECT_THROW(MPSContainer({nullptr, sites[1]}, orbitals),
               std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, nullptr), std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, orbitals, nullptr, nullptr, 2),
               std::invalid_argument);
  EXPECT_THROW(
      MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {0, 0}),
      std::invalid_argument);
  const Eigen::MatrixXd wrong_bond = random_matrix(8, 1);
  EXPECT_THROW(MPSContainer({sites[0], make_site(wrong_bond)}, orbitals),
               std::invalid_argument);
  const Eigen::MatrixXd nonfinite =
      Eigen::MatrixXd::Constant(4, 1, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(make_site(nonfinite), std::invalid_argument);
  auto json = make_container()->to_json();
  json["version"] = "9.0.0";
  EXPECT_THROW(MPSContainer::from_json(json), std::runtime_error);
}

TEST_F(MPSContainerTest, EqualSpinCountsDoNotImplyClosedShellRdms) {
  auto restricted = testing::create_test_orbitals(2, 2);
  ASSERT_TRUE(restricted->is_restricted());
  using Count = MPSContainer::ParticleCount;
  auto equal_counts = std::make_shared<const Count>(
      Count::SymmetriesArray{testing::spin_symmetry(false)},
      Count::BlockMap{
          {{axes::alpha()}, std::make_shared<const std::size_t>(1)},
          {{axes::beta()}, std::make_shared<const std::size_t>(1)}});
  ContainerTypes::MatrixVariant traced = random_matrix(2, 2);
  MPSContainer traced_only(sites, restricted, equal_counts, equal_counts,
                           std::nullopt, {}, traced);
  EXPECT_TRUE(traced_only.has_one_rdm_spin_traced());
  EXPECT_FALSE(traced_only.has_one_rdm_spin_dependent());
  EXPECT_THROW(traced_only.active_one_rdm(), std::runtime_error);
  EXPECT_THROW(traced_only.active_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(traced_only.total_orbital_occupations(), std::runtime_error);
  Eigen::MatrixXd aa = random_matrix(2, 2), bb = random_matrix(2, 2);
  Eigen::VectorXd aaaa = random_matrix(16, 1), aabb = random_matrix(16, 1),
                  bbbb = random_matrix(16, 1);
  MPSContainer supplied(
      sites, restricted, equal_counts, equal_counts, std::nullopt, {},
      std::nullopt, ContainerTypes::MatrixVariant{aa},
      ContainerTypes::MatrixVariant{bb}, std::nullopt,
      ContainerTypes::VectorVariant{aaaa}, ContainerTypes::VectorVariant{aabb},
      ContainerTypes::VectorVariant{bbbb});
  Eigen::VectorXd expected = aaaa + bbbb + aabb;
  for (std::size_t p = 0; p < 2; ++p)
    for (std::size_t q = 0; q < 2; ++q)
      for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t s = 0; s < 2; ++s)
          expected(p * 8 + q * 4 + r * 2 + s) +=
              aabb(r * 8 + s * 4 + p * 2 + q);
  EXPECT_TRUE(
      std::get<Eigen::MatrixXd>(supplied.get_active_one_rdm_spin_traced())
          .isApprox((aa + bb).eval(), testing::wf_tolerance));
  EXPECT_TRUE(
      std::get<Eigen::VectorXd>(supplied.get_active_two_rdm_spin_traced())
          .isApprox(expected, testing::wf_tolerance));
}

namespace {

/// Spin-half configuration from blocked occupation bits written
/// alpha_0 ... alpha_{n-1} beta_0 ... beta_{n-1}.
Configuration from_blocked_bits(const std::string& bits) {
  const auto n = bits.size() / 2;
  std::string states;
  for (std::size_t orbital = 0; orbital < n; ++orbital) {
    const bool alpha = bits[orbital] == '1';
    const bool beta = bits[n + orbital] == '1';
    states += alpha ? (beta ? '2' : 'u') : (beta ? 'd' : '0');
  }
  return Configuration::from_spin_half_string(states);
}

/// Sites of (|2u0> + i|u02> - |udu> - i|02u>)/2 in chain order. Each bond
/// channel carries one product state, so no decomposition enters the fixture.
std::vector<MPSContainer::SitePtr> signed_reference_sites() {
  using Complex = std::complex<double>;
  const std::array<std::string, 4> states{"2u0", "u02", "udu", "02u"};
  const std::array<Complex, 4> coefficients{
      Complex(0.5, 0.0), Complex(0.0, 0.5), Complex(-0.5, 0.0),
      Complex(0.0, -0.5)};
  const std::string order = "0ud2";
  Eigen::MatrixXcd first = Eigen::MatrixXcd::Zero(4, 4);
  Eigen::MatrixXcd middle = Eigen::MatrixXcd::Zero(16, 4);
  Eigen::MatrixXcd last = Eigen::MatrixXcd::Zero(16, 1);
  for (std::size_t bond = 0; bond < states.size(); ++bond) {
    const auto b = static_cast<Eigen::Index>(bond);
    auto local = [&](std::size_t site) {
      return static_cast<Eigen::Index>(order.find(states[bond][site]));
    };
    first(local(0), b) = coefficients[bond];
    middle(b * 4 + local(1), b) = 1.0;
    last(b * 4 + local(2), 0) = 1.0;
  }
  return {make_site(first), make_site(middle), make_site(last)};
}

}  // namespace

TEST(MPSConventionTest, SignedReferenceAmplitudes) {
  using Complex = std::complex<double>;
  // Hand-enumerated rather than computed: |2u0> creates a0 b0 a1 in identity
  // order (one exchange to reach a0 a1 b0) but a2 b2 a0 for order [2,0,1]
  // (two exchanges). Bits are alpha_0 alpha_1 alpha_2 beta_0 beta_1 beta_2.
  const std::vector<
      std::pair<std::vector<std::size_t>, std::map<std::string, Complex>>>
      cases{{{0, 1, 2},
             {{"110100", {-0.5, 0.0}},
              {"101001", {0.0, 0.5}},
              {"101010", {0.5, 0.0}},
              {"011010", {0.0, 0.5}}}},
            {{2, 0, 1},
             {{"101001", {0.5, 0.0}},
              {"011010", {0.0, -0.5}},
              {"011100", {-0.5, 0.0}},
              {"110100", {0.0, 0.5}}}}};
  for (const auto& [order, expected] : cases) {
    const MPSContainer original(signed_reference_sites(),
                                std::make_shared<ModelOrbitals>(3), nullptr,
                                nullptr, std::nullopt, order);
    const std::shared_ptr<const MPSContainer> restored =
        MPSContainer::from_json(original.to_json());
    for (const auto* container : {&original, restored.get()}) {
      for (unsigned mask = 0; mask < 64; ++mask) {
        std::string bits(6, '0');
        for (std::size_t i = 0; i < bits.size(); ++i) {
          if ((mask >> i) & 1u) bits[i] = '1';
        }
        const auto found = expected.find(bits);
        const Complex reference =
            found == expected.end() ? Complex{} : found->second;
        const auto actual = std::get<Complex>(
            container->get_coefficient(from_blocked_bits(bits)));
        EXPECT_NEAR(std::abs(actual - reference), 0.0, testing::wf_tolerance)
            << bits;
      }
    }
  }
}

TEST_F(MPSContainerTest, CoefficientsContractBlockedSitesInBasisOrder) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(4)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  // Physical index order (2,u,d,0) groups the sectors as two, one, zero.
  const std::string states = "2ud0";
  std::vector<Configuration> basis;
  for (const char state : states) {
    basis.push_back(Configuration::from_spin_half_string(std::string{state}));
  }
  const std::vector<SymmetryLabel> physical_order{two, one, zero};
  SBT::ExtentsArray first_extents, second_extents;
  first_extents[0] = {{zero, 1}};
  first_extents[1] = {{two, 1}, {one, 2}, {zero, 1}};
  first_extents[2] = {{one, 2}, {two, 1}};
  second_extents[0] = first_extents[2];
  second_extents[1] = first_extents[1];
  second_extents[2] = {{two, 1}};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, first_extents,
          {{{zero, one, one},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 2))},
           {{zero, two, two},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}})),
      MPSSite::SectorOrders{{{zero}, physical_order, {two, one}}}, basis);
  auto second = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, second_extents,
          {{{one, one, two},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(4, 1))},
           {{two, zero, two},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}})),
      MPSSite::SectorOrders{{{two, one}, physical_order, {two}}}, basis);
  const MPSContainer container({first, second}, orbitals);
  const auto left = std::get<Eigen::MatrixXd>(first->to_dense());
  const auto right = std::get<Eigen::MatrixXd>(second->to_dense());
  std::size_t nonzero = 0;
  for (Eigen::Index p0 = 0; p0 < 4; ++p0) {
    for (Eigen::Index p1 = 0; p1 < 4; ++p1) {
      double expected = 0.0;
      for (Eigen::Index bond = 0; bond < 3; ++bond) {
        expected += left(p0, bond) * right(bond * 4 + p1, 0);
      }
      // Chain order creates a0 b0 a1 b1, so one exchange is needed exactly
      // when site 0 holds a beta electron and site 1 an alpha electron.
      const char s0 = states[static_cast<std::size_t>(p0)];
      const char s1 = states[static_cast<std::size_t>(p1)];
      if ((s0 == 'd' || s0 == '2') && (s1 == 'u' || s1 == '2')) {
        expected = -expected;
      }
      if (expected != 0.0) ++nonzero;
      const auto det =
          Configuration::from_spin_half_string(std::string{s0, s1});
      EXPECT_NEAR(std::get<double>(container.get_coefficient(det)), expected,
                  testing::wf_tolerance)
          << s0 << s1;
    }
  }
  EXPECT_EQ(nonzero, 5u);
}

TEST_F(MPSContainerTest, SpinlessCoefficientsCarryPermutationSign) {
  const Eigen::MatrixXd first = random_matrix(2, 2);
  const Eigen::MatrixXd second = random_matrix(4, 1);
  // Chain site 0 holds orbital 1, so |11> in chain order is a1^ a0^ |vac>.
  const MPSContainer container({make_site(first, 2), make_site(second, 2)},
                               orbitals, nullptr, nullptr, std::nullopt,
                               {1, 0});
  for (Eigen::Index orbital0 = 0; orbital0 < 2; ++orbital0) {
    for (Eigen::Index orbital1 = 0; orbital1 < 2; ++orbital1) {
      const Eigen::Index p0 = orbital1, p1 = orbital0;
      double expected = 0.0;
      for (Eigen::Index bond = 0; bond < 2; ++bond) {
        expected += first(p0, bond) * second(bond * 2 + p1, 0);
      }
      if (orbital0 == 1 && orbital1 == 1) expected = -expected;
      const auto det = Configuration::from_bitstring(std::to_string(orbital0) +
                                                     std::to_string(orbital1));
      EXPECT_NEAR(std::get<double>(container.get_coefficient(det)), expected,
                  testing::wf_tolerance);
    }
  }
}

TEST_F(MPSContainerTest, CoefficientRejectsMismatchedConfigurations) {
  const auto container = make_container();
  EXPECT_THROW(container->get_coefficient(Configuration::from_bitstring("10")),
               std::invalid_argument);
  // A mode outside the active sites lies outside the represented space.
  EXPECT_EQ(std::get<double>(container->get_coefficient(
                Configuration::from_spin_half_string("u0u"))),
            0.0);
}

TEST_F(MPSContainerTest, ParticleNumberAndSpinBlockedSites) {
  using SBT = SymmetryBlockedTensor<3>;
  // Physical and bond indices carry (N, 2Ms). Both parities of 2Ms occur, so
  // these sectors need spin_projection rather than a spin multiplet axis.
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(4), axes::spin_projection(2)}));
  auto label = [](std::size_t number, int two_ms) {
    return SymmetryLabel(
        {axes::particle_number_value(number), axes::spin_value(two_ms)});
  };
  // The default physical basis (0,u,d,2) has one state per sector.
  const std::vector<SymmetryLabel> physical{label(0, 0), label(1, 1),
                                            label(1, -1), label(2, 0)};
  const auto vacuum = label(0, 0);
  const auto total = label(2, 0);
  SBT::ExtentsArray first_extents, second_extents;
  first_extents[0][vacuum] = 1;
  second_extents[2][total] = 1;
  SBT::BlockMap first_blocks, second_blocks;
  // Charges accumulate from the left: q_left + q_physical = q_right, so each
  // first-site sector pairs with its complement on the second site.
  for (std::size_t i = 0; i < physical.size(); ++i) {
    first_extents[1][physical[i]] = first_extents[2][physical[i]] = 1;
    second_extents[0][physical[i]] = second_extents[1][physical[i]] = 1;
    first_blocks[{vacuum, physical[i], physical[i]}] =
        std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1));
    second_blocks[{physical[i], physical[physical.size() - 1 - i], total}] =
        std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1));
  }
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, first_extents, first_blocks)),
      MPSSite::SectorOrders{{{vacuum}, physical, physical}});
  auto second = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, second_extents, second_blocks)),
      MPSSite::SectorOrders{{physical, physical, {total}}});
  const MPSContainer original({first, second}, orbitals);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_n_sz_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  const std::shared_ptr<const MPSContainer> json_copy =
      MPSContainer::from_json(original.to_json());
  const std::shared_ptr<const MPSContainer> hdf_copy =
      MPSContainer::from_hdf5(file);
  const auto left = std::get<Eigen::MatrixXd>(first->to_dense());
  const auto right = std::get<Eigen::MatrixXd>(second->to_dense());
  const std::string states = "0ud2";
  for (const auto* copy : {&original, json_copy.get(), hdf_copy.get()}) {
    for (std::size_t slot = 0; slot < 3; ++slot) {
      EXPECT_EQ(*std::get<SBT>(copy->sites()[0]->tensor()).symmetries()[slot],
                *symmetry);
    }
    std::size_t nonzero = 0;
    for (Eigen::Index p0 = 0; p0 < 4; ++p0) {
      for (Eigen::Index p1 = 0; p1 < 4; ++p1) {
        double expected = 0.0;
        for (Eigen::Index bond = 0; bond < 4; ++bond) {
          expected += left(p0, bond) * right(bond * 4 + p1, 0);
        }
        const char s0 = states[static_cast<std::size_t>(p0)];
        const char s1 = states[static_cast<std::size_t>(p1)];
        if ((s0 == 'd' || s0 == '2') && (s1 == 'u' || s1 == '2')) {
          expected = -expected;
        }
        if (expected != 0.0) ++nonzero;
        EXPECT_NEAR(std::get<double>(copy->get_coefficient(
                        Configuration::from_spin_half_string({s0, s1}))),
                    expected, testing::wf_tolerance);
      }
    }
    EXPECT_EQ(nonzero, 4u);
  }
}

TEST_F(MPSContainerTest, SuppliedEntropiesRoundTrip) {
  Eigen::VectorXd single(2);
  single << 0.3, 0.7;
  Eigen::MatrixXd mutual(2, 2);
  mutual << 0.0, 0.2, 0.2, 0.0;
  OrbitalEntropies entropies;
  entropies.single_orbital = single;
  entropies.mutual_information = mutual;
  const MPSContainer original(
      sites, orbitals, nullptr, nullptr, std::nullopt, {}, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, entropies);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_entropies_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  const std::shared_ptr<const MPSContainer> json_copy =
      MPSContainer::from_json(original.to_json());
  const std::shared_ptr<const MPSContainer> hdf_copy =
      MPSContainer::from_hdf5(file);
  for (const auto* copy : {&original, json_copy.get(), hdf_copy.get()}) {
    ASSERT_TRUE(copy->has_single_orbital_entropies());
    ASSERT_TRUE(copy->has_mutual_information());
    EXPECT_TRUE(copy->get_single_orbital_entropies().isApprox(single));
    EXPECT_TRUE(copy->get_mutual_information().isApprox(mutual));
    // Two-orbital entropies follow from single-orbital entropies and mutual
    // information, so entropy-based selection can consume them directly.
    EXPECT_TRUE(copy->has_two_orbital_entropies());
  }

  // Two-orbital entropies are stored as the mutual information they imply,
  // so the pair information survives serialization.
  Eigen::MatrixXd pair(2, 2);
  pair << 0.0, 0.8, 0.8, 0.0;
  OrbitalEntropies two_orbital;
  two_orbital.single_orbital = single;
  two_orbital.two_orbital = pair;
  const MPSContainer from_pairs(
      sites, orbitals, nullptr, nullptr, std::nullopt, {}, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, two_orbital);
  const auto restored_pairs = MPSContainer::from_json(from_pairs.to_json());
  EXPECT_TRUE(restored_pairs->get_mutual_information().isApprox(mutual));
  EXPECT_TRUE(restored_pairs->get_two_orbital_entropies().isApprox(pair));
  two_orbital.single_orbital.reset();
  EXPECT_THROW(
      MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, two_orbital),
      std::invalid_argument);

  OrbitalEntropies wrong_size;
  wrong_size.single_orbital = Eigen::VectorXd::Ones(3);
  EXPECT_THROW(
      MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, wrong_size),
      std::invalid_argument);
  OrbitalEntropies nonfinite;
  nonfinite.mutual_information =
      Eigen::MatrixXd::Constant(2, 2, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(
      MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, nonfinite),
      std::invalid_argument);
}
