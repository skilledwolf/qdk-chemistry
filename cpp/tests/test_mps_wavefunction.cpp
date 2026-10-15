// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
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

TEST_F(MPSContainerTest, DenseFactoryPreservesStorageAndOwnsInput) {
  auto input = data[0];
  const auto real = MPSSite::from_dense(input);
  input.setZero();
  EXPECT_FALSE(real.is_complex());
  EXPECT_EQ(real.left_bond_dimension(), 1u);
  EXPECT_EQ(real.right_bond_dimension(), 3u);
  EXPECT_TRUE(std::get<Eigen::MatrixXd>(real.to_dense()).isApprox(data[0]));

  const Eigen::MatrixXcd complex =
      data[1].cast<std::complex<double>>() * std::complex<double>(-2, 3);
  const auto site = MPSSite::from_dense(complex);
  EXPECT_TRUE(site.is_complex());
  EXPECT_EQ(site.left_bond_dimension(), 3u);
  EXPECT_TRUE(std::get<Eigen::MatrixXcd>(site.to_dense()).isApprox(complex));

  const std::vector<Configuration> basis{
      Configuration::from_spin_half_string("u"),
      Configuration::from_spin_half_string("d")};
  const auto spin_site = MPSSite::from_dense(random_matrix(2, 1), 2, basis);
  EXPECT_EQ(spin_site.physical_basis(), basis);
}

TEST_F(MPSContainerTest, DenseFactoryRejectsInvalidPackingAndEntries) {
  EXPECT_THROW(MPSSite::from_dense(data[0], 0), std::invalid_argument);
  EXPECT_THROW(MPSSite::from_dense(data[0], 3), std::invalid_argument);
  EXPECT_THROW(MPSSite::from_dense(Eigen::MatrixXd(0, 1)),
               std::invalid_argument);
  EXPECT_THROW(MPSSite::from_dense(Eigen::MatrixXd(4, 0)),
               std::invalid_argument);
  EXPECT_THROW(MPSSite::from_dense(random_matrix(3, 1), 3),
               std::invalid_argument);
  auto nonfinite = data[0];
  nonfinite(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(MPSSite::from_dense(nonfinite), std::invalid_argument);
  EXPECT_THROW(
      MPSSite::from_dense(data[0], 4, {Configuration::from_bitstring("0")}),
      std::invalid_argument);
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

TEST_F(MPSContainerTest, RdmCachesDoNotChangeStateIdentityHash) {
  MPSContainer without(sites, orbitals);
  const ContainerTypes::MatrixVariant rdm =
      Eigen::MatrixXd::Identity(2, 2).eval();
  const ContainerTypes::MatrixVariant other_rdm =
      (2 * Eigen::MatrixXd::Identity(2, 2)).eval();
  MPSContainer first(sites, orbitals, nullptr, nullptr, std::nullopt, {}, rdm);
  MPSContainer second(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                      other_rdm);
  const Wavefunction without_rdms(without.clone());
  const Wavefunction with_rdms(first.clone());
  const Wavefunction with_other_rdms(second.clone());
  EXPECT_EQ(without_rdms.content_hash(), with_rdms.content_hash());
  EXPECT_EQ(with_rdms.content_hash(), with_other_rdms.content_hash());
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
  const auto independent_bbaa = make_blocked_rdm<4>(
      {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}, zero},
       {{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}, zero},
       {{axes::beta(), axes::beta(), axes::beta(), axes::beta()}, zero},
       {{axes::beta(), axes::beta(), axes::alpha(), axes::alpha()},
        std::make_shared<const Eigen::VectorXd>(*zero)}});
  EXPECT_THROW(with_blocked_rdms(std::nullopt, independent_bbaa),
               std::invalid_argument);
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryBlockedTensor<2> unsupported(
      {trivial, trivial}, {{{{SymmetryLabel{}, 2}}, {{SymmetryLabel{}, 2}}}},
      {{{SymmetryLabel{}, SymmetryLabel{}}, one}});
  EXPECT_THROW(with_blocked_rdms(unsupported), std::invalid_argument);
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
