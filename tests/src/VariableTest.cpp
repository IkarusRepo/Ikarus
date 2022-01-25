//
// Created by Alex on 21.04.2021.
//
//#include <gmock/gmock.h>
//#include <gtest/gtest.h>

#include "testHelpers.h"

#include <array>
#include <fstream>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "ikarus/Manifolds/RealTuple.h"
#include "ikarus/Variables/InterfaceVariable.h"
#include "ikarus/Variables/VariableDefinitions.h"
#include "ikarus/utils/utils/algorithms.h"
#include <catch2/catch_test_macros.hpp>
TEST_CASE("DefaultVariableTest: RealTupleDisplacement", "[1]") {
  using namespace Ikarus::Variable;
  auto a = VariableFactory::createVariable(VariableTags::displacement3d);
  auto q = a;

  const auto p = std::move(q);
  a += Eigen::Vector<double, 3>::UnitX();
  CHECK ((Eigen::Vector<double, 3>::UnitX()) == getValue(a));

  a += Eigen::Vector<double, 3>::UnitY();
  CHECK ((Eigen::Vector<double, 3>(1.0, 1.0, 0.0)) == getValue(a));

  auto d = a + Eigen::Vector<double, 3>::UnitY();
  CHECK ((Eigen::Vector<double, 3>(1.0, 2.0, 0.0)) == getValue(d));

  auto b{a};
  CHECK ((Eigen::Vector<double, 3>(1.0, 1.0, 0.0)) == getValue(b));

  DISPLACEMENT3D c{DISPLACEMENT3D{Eigen::Vector<double, 3>::UnitZ()}};  // move constructor test
  CHECK ((Eigen::Vector<double, 3>(0.0, 0.0, 1.0)) == c.getValue());

  c.setValue(Eigen::Vector<double, 3>(13.0, -5.0, 1.0));
  CHECK ((Eigen::Vector<double, 3>(13.0, -5.0, 1.0)) == c.getValue());

  b = a;
  CHECK (a == b);
  CHECK (getValue(a) == getValue(b));
  auto testVec = Eigen::Vector<double, 3>(127.0, -5.0, 1.0);
  setValue(b, testVec);

  CHECK (testVec == getValue(b));
  for (auto&& varTag : Ikarus::Variable::AllVariableTags)
    auto h = VariableFactory::createVariable(varTag);  // check if all variables can be created

  // creating variables with unknown tag is illegal
  CHECK_THROWS_AS (VariableFactory::createVariable(static_cast<Ikarus::Variable::VariableTags>(-15)), std::logic_error);

  std::stringstream testStream;
  testStream << b;
  CHECK ("127  -5   1\n Tag: displacement3d\n" == testStream.str());

  std::stringstream testStream2;
  testStream2 << c;
  CHECK ("13 -5  1\n" == testStream2.str());
}

static constexpr double tol = 1e-15;

TEST_CASE("DefaultVariableTest: UnitVectorDirector", "[1]") {
  using namespace Ikarus::Variable;
  DIRECTOR3D a{DIRECTOR3D::CoordinateType::UnitZ()};
  a.update(Eigen::Vector<double, 2>::UnitX());
  const auto aExpected = Eigen::Vector<double, 3>(1.0 / sqrt(2), 0.0, 1.0 / sqrt(2));
  CHECK_THAT (a.getValue(), EigenApproxEqual(aExpected, tol));

  a = update(a, Eigen::Vector<double, 2>::UnitY());
  CHECK_THAT (a.getValue(), EigenApproxEqual(Eigen::Vector<double, 3>(0.5, 1.0 / sqrt(2), 0.5), tol));

  const auto d = update(a, Eigen::Vector<double, 2>::UnitY());

  CHECK_THAT(d.getValue(), EigenApproxEqual(Eigen::Vector<double, 3>(0.18688672392660707344, 0.97140452079103167815,
                                                                      -0.14644660940672621363),
                                             tol));

  DIRECTOR3D b{a};

  CHECK_THAT (b.getValue(), EigenApproxEqual(Eigen::Vector<double, 3>(0.5, 1.0 / sqrt(2), 0.5), tol));

  DIRECTOR3D c{DIRECTOR3D{Eigen::Vector<double, 3>::UnitZ() * 2.0}};  // move constructor test
  CHECK_THAT (c.getValue(), EigenApproxEqual(Eigen::Vector<double, 3>(0.0, 0.0, 1.0), tol));

  c.setValue(Eigen::Vector<double, 3>(13.0, -5.0, 1.0));
  CHECK_THAT (c.getValue(), EigenApproxEqual(Eigen::Vector<double, 3>(13.0, -5.0, 1.0).normalized(), tol));

  b = a;
  CHECK (a == b);
  const auto testVec = Eigen::Vector<double, 3>(127.0, -5.0, 1.0);
  b.setValue(testVec);

  CHECK_THAT (b.getValue(), EigenApproxEqual(testVec.normalized(), tol));

  auto e{std::move(a)};

  e.setValue(Eigen::Vector<double, 3>(0.0, 0.0, -1.0));

  e = update(e, Eigen::Vector<double, 2>::UnitY());

  const auto eExpected = Eigen::Vector<double, 3>(0, 1.0 / sqrt(2), -1.0 / sqrt(2));
  CHECK_THAT (e.getValue(), EigenApproxEqual(eExpected, tol));
}

TEST_CASE("VariableTest: GenericVariableVectorTest", "[1]") {
  using namespace Ikarus::Variable;
  DISPLACEMENT3D a;
  DISPLACEMENT2D b;
  DISPLACEMENT1D c;
  DISPLACEMENT3D d;
  DISPLACEMENT2D e;
  DIRECTOR3D f{DIRECTOR3D::CoordinateType::UnitZ()};

  std::vector<IVariable> varVec;
  varVec.emplace_back(a);
  varVec.emplace_back(b);
  varVec.emplace_back(c);
  varVec.emplace_back(d);
  varVec.emplace_back(e);

  CHECK (11 == valueSize(varVec));

  CHECK (11 == correctionSize(varVec));
  varVec.emplace_back(f);
  CHECK (14 == valueSize(varVec));
  CHECK (13 == correctionSize(varVec));

  Eigen::Vector2d corr;
  corr << 1, 1;
  varVec[5] += corr;

  Eigen::Vector3d varVec4Expected({1, 1, 1});
  varVec4Expected.normalize();
  CHECK_THAT (getValue(varVec[5]), EigenApproxEqual(varVec4Expected, tol));

  Ikarus::utils::makeUniqueAndSort(varVec);

  CHECK (9 == valueSize(varVec));
  CHECK (8 == correctionSize(varVec));

  Eigen::VectorXd correction(correctionSize(varVec));

  correction << 1, 2, 3, 4, 5, 6, 7, 8;
  update(varVec, correction);

  CHECK_THAT (getValue(varVec[0]), EigenApproxEqual(Eigen::Matrix<double, 1, 1>(1), tol));
  CHECK_THAT (getValue(varVec[1]), EigenApproxEqual(Eigen::Vector2d(2, 3), tol));
  CHECK_THAT (getValue(varVec[2]), EigenApproxEqual(Eigen::Vector3d(4, 5, 6), tol));
  CHECK_THAT(
      getValue(varVec[3]),
      EigenApproxEqual(Eigen::Vector3d(0.41279806929140905325, 0.50645665044957854928, -0.75703329861022516933), tol));
}
