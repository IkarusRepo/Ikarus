//
// Created by Alex on 21.04.2021.
//

//#include <gmock/gmock.h>
//#include <gtest/gtest.h>

#include "testHelpers.h"

#include <array>
#include <complex>
#include <vector>

#include <Eigen/Core>

#include "ikarus/AnsatzFunctions/Lagrange.h"
#include "ikarus/Geometries/GeometryWithExternalInput.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>


const double tol = 1e-15;

TEST_CASE("GeometryTest: CreateJacobianDeterminantAndJacobian2DFlat", "[1]") {
  using namespace Ikarus::Geometry;
  const Eigen::Matrix<double, 2, 1> xieta({0.0, 0.0});

  Eigen::Matrix<double, 4, 2> dN = Ikarus::LagrangeCube<double, 2, 1>::evaluateJacobian(xieta);

  Eigen::Matrix<double, 2, 4> x;
  x.col(0) << 0, 0;
  x.col(1) << 4, 0;
  x.col(2) << 0, 4;
  x.col(3) << 4, 4;
  Eigen::Matrix<double, 2, 2> JT = ExternalPlaneGeometry<double>::jacobianTransposed(dN, x);

  Eigen::Matrix2d JTexpected;
  JTexpected << 4, 0, 0, 4;
  CHECK (JT == JTexpected);

  Eigen::Matrix2d JTinv = ExternalPlaneGeometry<double>::jacobianInverseTransposed(dN, x);

  Eigen::Matrix2d JTinvexpected;
  JTinvexpected << 0.25, 0.0, 0.0, 0.25;

  CHECK (JTinv == JTinvexpected);

  const double detJ = ExternalPlaneGeometry<double>::determinantJacobian(dN, x);

  CHECK (detJ == 16.0);
}

#include <ikarus/Geometries/GeometricElementDefinitions.h>

TEST_CASE("GeometryTest: WithInteralAnsatzandVertices", "[1]") {
  using namespace Ikarus::Geometry;

  const Eigen::Matrix<double, 3, 1> xieta({0.0, 0.0, 0.0});

  Eigen::Matrix<double, 3, 8> x;
  x.col(0) << 0, 0, 0;
  x.col(1) << 4, 0, 0;
  x.col(2) << 0, 4, 0;
  x.col(3) << 4, 4, 0;
  x.col(4) << 0, 0, 1;
  x.col(5) << 4, 0, 1;
  x.col(6) << 0, 4, 1;
  x.col(7) << 4, 4, 1;
  LinearBrickGeometry<double> geoEle(x);

  Eigen::Matrix<double, 3, 3> JT = geoEle.jacobianTransposed(xieta);

  Eigen::Matrix3d JTexpected;
  JTexpected << 4, 0, 0, 0, 4, 0, 0, 0, 1;

  REQUIRE_THAT  (JT , EigenApproxEqual(JTexpected, tol));

  Eigen::Matrix3d JTinv = geoEle.jacobianInverseTransposed(xieta);

  Eigen::Matrix3d JTinvexpected;
  JTinvexpected << 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0;

  REQUIRE_THAT  (JTinv,  EigenApproxEqual(JTinvexpected, tol));

  double detJ = geoEle.determinantJacobian(xieta);

  CHECK (detJ == 16.0);
}

TEST_CASE("GeometryTest: WithInteralAnsatzandVerticesQuadratic", "[1]") {
  using namespace Ikarus::Geometry;

  const Eigen::Matrix<double, 2, 1> xieta({
      0.325,
      0.9,
  });

  Eigen::Matrix<double, 2, 9> x;
  x.col(0) << 0, 0;
  x.col(1) << 3, 0;
  x.col(2) << 7, 0;
  x.col(3) << 0, 1;
  x.col(4) << 2, 2;
  x.col(5) << 4, 3;
  x.col(6) << -1, 1;
  x.col(7) << 2, 7;
  x.col(8) << 10, 10;
  QuadraticPlaneGeometry<double> geoEle(x);

  Eigen::Matrix<double, 2, 2> JT = geoEle.jacobianTransposed(xieta);

  Eigen::Matrix2d JTexpected;
  JTexpected << 6.336000000, 8.712000000, -2.06700000, 8.34725000;

  CHECK_THAT (JT, EigenApproxEqual(JTexpected, tol));

  Eigen::Matrix2d JTinv = geoEle.jacobianInverseTransposed(xieta);

  Eigen::Matrix2d JTinvexpected;
  JTinvexpected << 0.1177395639915888, -0.1228844327766296, 0.02915543188123201, 0.08937049656482153;
  CHECK_THAT (JTinv, EigenApproxEqual(JTinvexpected, tol));

  double detJ = geoEle.determinantJacobian(xieta);
  CHECK (detJ == Catch::Approx(70.895880000000005));

  auto dNCart = geoEle.transformCurvLinearDerivativesToCartesian(xieta);

  Eigen::Matrix<double, 9, 2> dNCartExpected;
  dNCartExpected << 0.12624897551105789618e-1, 0.10920614574095834274e-1, -0.10396974453851826743e-1,
      0.88743757452822190506e-1, -0.22279230972539628740e-2, -0.84966090236083447561e-2, -0.56812038979976053284e-1,
      -0.67091418924707847485e-1, 0.46786385042333220342e-1, -.46601333523387006079, 0.10025653937642832933e-1,
      0.46876684807592948154e-1, -.11362407795995210656, .18889292229356298362, 0.93572770084666440706e-1,
      .26796901006332354172, 0.20051307875285665863e-1, -0.61801626009211245234e-1;

  CHECK_THAT (dNCart, EigenApproxEqual(dNCartExpected, tol));
}

TEST_CASE("GeometryTest: CreateJacobianDeterminantAndJacobian2DSurfIn3D", "[1]") {
  const Eigen::Matrix<double, 2, 1> xieta({-1.0, -1.0});
  Eigen::Matrix<double, 4, 2> dN = Ikarus::LagrangeCube<double, 2, 1>::evaluateJacobian(xieta);

  Eigen::Matrix<double, 2, 4> dNCorrect;
  Eigen::Matrix<double, 3, 4> x;
  x.col(0) << 0, 0, 0;
  x.col(1) << 4, 0, 0;
  x.col(2) << 0, 4, 0;
  x.col(3) << 4, 4, 0;
  Eigen::Matrix<double, 2, 3> JT = Ikarus::Geometry::ExternalSurfaceGeometry<double>::jacobianTransposed(dN, x);
  Eigen::Matrix<double, 2, 3> JTexpected;
  JTexpected << 4, 0, 0, 0, 4, 0;

  REQUIRE (JT == JTexpected);
}
