//
// Created by Alex on 21.04.2021.
//
#include <config.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common.hh"
#include "testHelpers.hh"

#include <array>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <complex>
#include <vector>

#include <dune/common/classname.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/grid/yaspgrid.hh>

#include <Eigen/Core>

#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/localFunctions/expressions.hh>
#include <ikarus/localFunctions/impl/projectionBasedLocalFunction.hh>
#include <ikarus/localFunctions/impl/standardLocalFunction.hh>
#include <ikarus/localFunctions/localFunctionName.hh>
#include <ikarus/manifolds/realTuple.hh>
#include <ikarus/manifolds/unitVector.hh>
#include <ikarus/utils/functionSanityChecks.hh>

using namespace Dune::Functions::BasisFactory;

template <typename LF, bool isCopy = false>
void testLocalFunction(const LF& lf) {
  spdlog::info("Testing: " + std::string(isCopy ? "Copy " : "") + Ikarus::localFunctionName(lf));

  const double tol = 1e-13;
  using namespace Ikarus::DerivativeDirections;
  using namespace autodiff;
  using namespace Ikarus;
  const auto& coeffs     = lf.coefficientsRef();
  const size_t coeffSize = coeffs.size();

  constexpr int gridDim                = LF::gridDim;
  using Manifold                       = typename std::remove_cvref_t<decltype(coeffs)>::value_type;
  constexpr int localFunctionValueSize = LF::valueSize;
  constexpr int coeffValueSize         = Manifold::valueSize;
  using ctype                          = typename Manifold::ctype;
  constexpr int coeffCorrectionSize    = Manifold::correctionSize;

  for (const auto& [ipIndex, ip] : lf.viewOverIntegrationPoints()) {
    /// Check spatial derivatives
    /// Check spatial derivatives return sizes
    {
      if constexpr (requires {lf.evaluateDerivative(ipIndex, Ikarus::wrt(spatialAll));}) {
        const auto spatialAllDerivative = lf.evaluateDerivative(ipIndex, Ikarus::wrt(spatialAll));
        static_assert(spatialAllDerivative.cols() == gridDim);
        static_assert(spatialAllDerivative.rows() == localFunctionValueSize);
      }
      if constexpr(requires { lf.evaluateDerivative(ipIndex, Ikarus::wrt(spatial(0)));}) {
        const auto spatialSingleDerivative = lf.evaluateDerivative(ipIndex, Ikarus::wrt(spatial(0)));
        static_assert(spatialSingleDerivative.cols() == 1);
        static_assert(spatialSingleDerivative.rows() == localFunctionValueSize);
      }

      /// Check if spatial derivatives are really derivatives
      /// Perturb in a random direction in the elements parameter space and check spatial derivative
      auto func = [&](auto& gpOffset_) { return lf.evaluateFunction(toFieldVector(gpOffset_)); };
      auto spatialDerivAll
          = [&](auto& gpOffset_) { return lf.evaluateDerivative(toFieldVector(gpOffset_), Ikarus::wrt(spatialAll)); };
      auto derivDerivSingle = [&](auto gpOffset_, int spatialIndex) {
        return lf.evaluateDerivative(toFieldVector(gpOffset_), Ikarus::wrt(spatial(spatialIndex)));
      };
      Eigen::Vector<double, gridDim> ipOffset = (Eigen::Vector<double, gridDim>::Random()).normalized();
      auto nonLinOpSpatialAll
          = Ikarus::NonLinearOperator(linearAlgebraFunctions(func, spatialDerivAll), parameter(ipOffset));
      EXPECT_TRUE((checkJacobian<decltype(nonLinOpSpatialAll), Eigen::Vector<double, gridDim>>(
          nonLinOpSpatialAll, {.draw = false, .writeSlopeStatement = false, .tolerance = 1e-2})));

      /// Perturb each spatial direction and check with derivative value
      for (int i = 0; i < gridDim; ++i) {
        Eigen::Vector<double, 1> ipOffsetSingle(ipOffset[i]);
        auto derivDerivSingleI = [&](auto gpOffset_) {
          auto offSetSingle = ipOffset;
          offSetSingle[i] += gpOffset_[0];
          return lf.evaluateDerivative(toFieldVector(offSetSingle), Ikarus::wrt(spatial(i)));
        };

        auto funcSingle = [&](const auto& gpOffset_) {
          auto offSetSingle = ipOffset;
          offSetSingle[i] += gpOffset_[0];
          return Eigen::Vector<ctype, localFunctionValueSize>(lf.evaluateFunction(toFieldVector(offSetSingle)));
        };

        auto nonLinOpSpatialSingle = Ikarus::NonLinearOperator(linearAlgebraFunctions(funcSingle, derivDerivSingleI),
                                                               parameter(ipOffsetSingle));
        EXPECT_TRUE((checkJacobian<decltype(nonLinOpSpatialSingle), Eigen::Vector<double, 1>>(
            nonLinOpSpatialSingle, {.draw = false, .writeSlopeStatement = false, .tolerance = 1e-2})));
      }
    }

    /// Check coeff and spatial derivatives
    Eigen::VectorXdual xv(correctionSize(coeffs));
    xv.setZero();

    const Eigen::Vector<double, localFunctionValueSize> alongVec
        = localFunctionValueSize == 1 ? Eigen::Vector<double, localFunctionValueSize>::Ones().eval()
                                      : Eigen::Vector<double, localFunctionValueSize>::Random().eval();
    const Eigen::Matrix<double, localFunctionValueSize, gridDim> alongMat
        = localFunctionValueSize == 1 ? Eigen::Matrix<double, localFunctionValueSize, gridDim>::Ones().eval()
                                      : Eigen::Matrix<double, localFunctionValueSize, gridDim>::Random().eval();
    /// Rebind local function to second order dual number
    auto lfDual2nd                   = lf.rebindClone(dual2nd());
    auto lfDual2ndLeafNodeCollection = collectLeafNodeLocalFunctions(lfDual2nd);

    auto localFdual2nd = [&](const auto& x) {
      lfDual2ndLeafNodeCollection.addToCoeffs(x);
      auto value = (transpose(lfDual2nd.evaluateFunction(ipIndex)) * alongVec).trace();
      lfDual2ndLeafNodeCollection.addToCoeffs(-x);
      return value;
    };

    auto localFdual2ndSpatialSingle = [&](const auto& x, int i) {
      lfDual2ndLeafNodeCollection.addToCoeffs(x);
      auto value = (lfDual2nd.evaluateDerivative(ipIndex, Ikarus::wrt(spatial(i))).transpose() * alongVec).trace();
      lfDual2ndLeafNodeCollection.addToCoeffs(-x);
      return value;
    };

    auto localFdual2ndSpatialAll = [&](const auto& x) {
      lfDual2ndLeafNodeCollection.addToCoeffs(x);
      auto value = (lfDual2nd.evaluateDerivative(ipIndex, Ikarus::wrt(spatialAll)).transpose() * alongMat).trace();
      lfDual2ndLeafNodeCollection.addToCoeffs(-x);
      return value;
    };

    Eigen::VectorXdual2nd xvr(correctionSize(coeffs));
    xvr.setZero();

    dual2nd u;
    Eigen::VectorXd gradienWRTCoeffs;
    Eigen::MatrixXd hessianWRTCoeffs;
    autodiff::hessian(localFdual2nd, autodiff::wrt(xvr), autodiff::at(xvr), u, gradienWRTCoeffs, hessianWRTCoeffs);

    Eigen::VectorXd gradienWRTCoeffsSpatialAll;
    Eigen::MatrixXd hessianWRTCoeffsSpatialAll;
    autodiff::hessian(localFdual2ndSpatialAll, autodiff::wrt(xvr), autodiff::at(xvr), u, gradienWRTCoeffsSpatialAll,
                      hessianWRTCoeffsSpatialAll);

    std::array<Eigen::MatrixXd, gridDim> hessianWRTCoeffsTwoTimesSingleSpatial;
    std::array<Eigen::VectorXd, gridDim> gradientWRTCoeffsTwoTimesSingleSpatial;
    for (int d = 0; d < gridDim; ++d) {
      autodiff::hessian(localFdual2ndSpatialSingle, autodiff::wrt(xvr), autodiff::at(xvr, d), u,
                        gradientWRTCoeffsTwoTimesSingleSpatial[d], hessianWRTCoeffsTwoTimesSingleSpatial[d]);
    }

    for (size_t i = 0; i < coeffSize; ++i) {
      const auto jacobianWRTCoeffslf = lf.evaluateDerivative(ipIndex, Ikarus::wrt(coeff(i)));
      static_assert(jacobianWRTCoeffslf.ColsAtCompileTime == coeffCorrectionSize);
      static_assert(jacobianWRTCoeffslf.RowsAtCompileTime == localFunctionValueSize);
      const auto jacobianWRTCoeffs = ((alongVec.transpose() * jacobianWRTCoeffslf).transpose()).eval();
      static_assert(jacobianWRTCoeffs.cols() == 1);
      static_assert(jacobianWRTCoeffs.rows() == coeffCorrectionSize);
      EXPECT_THAT(
          jacobianWRTCoeffs,
          EigenApproxEqual(gradienWRTCoeffs.template segment<coeffCorrectionSize>(i * coeffCorrectionSize), tol));

      for (int d = 0; d < gridDim; ++d) {
        const auto jacoWrtCoeffAndSpatiallf = lf.evaluateDerivative(ipIndex, Ikarus::wrt(coeff(i), spatial(d)));
        static_assert(jacoWrtCoeffAndSpatiallf.ColsAtCompileTime == coeffCorrectionSize);
        static_assert(jacoWrtCoeffAndSpatiallf.RowsAtCompileTime == localFunctionValueSize);
        const auto jacoWrtCoeffAndSpatial = ((alongVec.transpose() * jacoWrtCoeffAndSpatiallf).transpose()).eval();
        const auto jacoWrtSpatialAndCoeff
            = ((alongVec.transpose() * lf.evaluateDerivative(ipIndex, Ikarus::wrt(spatial(d), coeff(i)))).transpose())
                  .eval();

        static_assert(jacoWrtSpatialAndCoeff.cols() == 1);
        static_assert(jacoWrtSpatialAndCoeff.rows() == coeffCorrectionSize);
        static_assert(jacoWrtCoeffAndSpatial.rows() == jacoWrtCoeffAndSpatial.rows()
                      and jacoWrtCoeffAndSpatial.cols() == jacoWrtCoeffAndSpatial.cols());

        EXPECT_THAT(jacoWrtCoeffAndSpatial, EigenApproxEqual(jacoWrtSpatialAndCoeff, tol));

        EXPECT_THAT(jacoWrtCoeffAndSpatial,
                    EigenApproxEqual(gradientWRTCoeffsTwoTimesSingleSpatial[d].template segment<coeffCorrectionSize>(
                                         i * coeffCorrectionSize),
                                     tol));
      }

      const auto jacoWrtSpatialAllAndCoeff = lf.evaluateDerivative(ipIndex, Ikarus::wrt(spatialAll, coeff(i)));

      static_assert(jacoWrtSpatialAllAndCoeff[0].ColsAtCompileTime == coeffCorrectionSize);
      static_assert(jacoWrtSpatialAllAndCoeff[0].RowsAtCompileTime == localFunctionValueSize);

      Eigen::Matrix<double, 1, coeffCorrectionSize> jacoWrtSpatialAllAndCoeffProd;
      jacoWrtSpatialAllAndCoeffProd.setZero();
      //    std::cout<<"alongMat"<<std::endl;
      //    std::cout<<alongMat<<std::endl;

      for (int d = 0; d < gridDim; ++d)
        jacoWrtSpatialAllAndCoeffProd += (alongMat.col(d).transpose() * jacoWrtSpatialAllAndCoeff[d]).eval();

      EXPECT_THAT(
          jacoWrtSpatialAllAndCoeffProd,
          EigenApproxEqual(
              gradienWRTCoeffsSpatialAll.template segment<coeffCorrectionSize>(i * coeffCorrectionSize).transpose(),
              tol));

      for (size_t j = 0; j < coeffSize; ++j) {
        const auto jacobianWRTCoeffsTwoTimes
            = lf.evaluateDerivative(ipIndex, Ikarus::wrt(coeff(i, j)), Ikarus::along(alongVec));
        static_assert(jacobianWRTCoeffsTwoTimes.cols() == coeffCorrectionSize);
        static_assert(jacobianWRTCoeffsTwoTimes.rows() == coeffCorrectionSize);
        const auto jacobianWRTCoeffsTwoTimesExpected
            = hessianWRTCoeffs.template block<coeffCorrectionSize, coeffCorrectionSize>(i * coeffCorrectionSize,
                                                                                        j * coeffCorrectionSize);
        EXPECT_THAT(jacobianWRTCoeffsTwoTimes, EigenApproxEqual(jacobianWRTCoeffsTwoTimesExpected, tol));

        const auto jacobianWRTCoeffsTwoTimesSpatialAll
            = lf.evaluateDerivative(ipIndex, Ikarus::wrt(coeff(i, j), spatialAll), Ikarus::along(alongMat));
        static_assert(jacobianWRTCoeffsTwoTimesSpatialAll.cols() == coeffCorrectionSize);
        static_assert(jacobianWRTCoeffsTwoTimesSpatialAll.rows() == coeffCorrectionSize);
        const auto jacobianWRTCoeffsTwoTimesSpatialAllExpected
            = hessianWRTCoeffsSpatialAll.template block<coeffCorrectionSize, coeffCorrectionSize>(
                i * coeffCorrectionSize, j * coeffCorrectionSize);

        /// if the order of the function value is less then quadratic then this should yield a vanishing derivative
        if constexpr (lf.order() < quadratic) {
          EXPECT_TRUE(jacobianWRTCoeffsTwoTimesSpatialAll.norm() < tol);
          EXPECT_TRUE(jacobianWRTCoeffsTwoTimesSpatialAllExpected.norm() < tol);
        } else {
          EXPECT_THAT(jacobianWRTCoeffsTwoTimesSpatialAll,
                      EigenApproxEqual(jacobianWRTCoeffsTwoTimesSpatialAllExpected, tol));
        }

        for (int d = 0; d < gridDim; ++d) {
          const auto jacobianWRTCoeffsTwoTimesSingleSpatial
              = lf.evaluateDerivative(ipIndex, Ikarus::wrt(coeff(i, j), spatial(d)), Ikarus::along(alongVec));
          static_assert(jacobianWRTCoeffsTwoTimesSingleSpatial.cols() == coeffCorrectionSize);
          static_assert(jacobianWRTCoeffsTwoTimesSingleSpatial.rows() == coeffCorrectionSize);
          const auto jacobianWRTCoeffsTwoTimesSingleSpatialExpected
              = hessianWRTCoeffsTwoTimesSingleSpatial[d].template block<coeffCorrectionSize, coeffCorrectionSize>(
                  i * coeffCorrectionSize, j * coeffCorrectionSize);
          EXPECT_THAT(jacobianWRTCoeffsTwoTimesSingleSpatial,
                      EigenApproxEqual(jacobianWRTCoeffsTwoTimesSingleSpatialExpected, tol));
        }
      }
    }
  }
  spdlog::info("done. ");
  if constexpr (not isCopy) {  //  test the cloned local function
    const auto lfCopy     = lf.clone();
    const auto& coeffCopy = lfCopy.coefficientsRef();
    for (size_t i = 0; i < coeffSize; ++i)
      EXPECT_EQ(coeffs[i], coeffCopy[i]);  // since the coeffs are copied the values should be the same

    EXPECT_NE(&coeffs, &coeffCopy);  // since the coeffs are copied the adresses should differ

    testLocalFunction<std::remove_cvref_t<decltype(lfCopy)>, true>(lfCopy);
  }
}

TEST(LocalFunctionTests, TestExpressions) {
  constexpr int sizeD = 3;
  using Manifold      = Ikarus::RealTuple<double, sizeD>;
  using Manifold2     = Ikarus::UnitVector<double, sizeD>;
  using Manifold3      = Ikarus::RealTuple<double, 2>;
  using VectorType    = Eigen::Vector<double, sizeD>;
  using MatrixType    = Eigen::Matrix<double, sizeD, sizeD>;
  constexpr int size  = Manifold::valueSize;
  using namespace Ikarus;
  using namespace Dune::Indices;
  auto grid             = createGrid<Grids::Yasp>(2, 1);
  constexpr int gridDim = std::remove_reference_t<decltype(*grid)>::dimension;

  auto gridView        = grid->leafGridView();
  const auto& indexSet = gridView.indexSet();
  auto basis           = makeBasis(gridView, lagrange<2>());
  Dune::BlockVector<Manifold> vBlocked(basis.size());
  Dune::BlockVector<Manifold> vBlocked2(basis.size());
  Dune::BlockVector<Manifold2> vBlocked3(basis.size());
  Dune::BlockVector<Manifold3> vBlocked4(basis.size());
  for (auto& vsingle : vBlocked)
    vsingle.setValue(0.1 * Eigen::Vector<double, size>::Random() + Eigen::Vector<double, size>::UnitX());

  for (auto& vsingle : vBlocked2)
    vsingle.setValue(0.5 * Eigen::Vector<double, size>::Random() + 7 * Eigen::Vector<double, size>::UnitX());

  for (auto& vsingle : vBlocked3)
    vsingle.setValue(0.25 * Eigen::Vector<double, size>::Random() + 37 * Eigen::Vector<double, size>::UnitX());

  for (auto& vsingle : vBlocked4)
    vsingle.setValue(0.25 * Eigen::Vector<double, size>::Random() + 37 * Eigen::Vector<double, size>::UnitX());

  auto localView = basis.localView();
  for (auto& ele : elements(gridView)) {
    std::cout << "Test on Element: " << indexSet.index(ele) << std::endl;
    std::cout << "==============================" << std::endl;
    localView.bind(ele);
    const auto& fe   = localView.tree().finiteElement();
    auto localBasis  = Ikarus::LocalBasis(localView.tree().finiteElement().localBasis());
    const auto& rule = Dune::QuadratureRules<double, 2>::rule(localView.element().type(), 3);
    localBasis.bind(rule, bindDerivatives(0, 1));
    Dune::BlockVector<Manifold> vBlockedLocal(fe.size());
    Dune::BlockVector<Manifold> vBlockedLocal2(fe.size());
    Dune::BlockVector<Manifold2> vBlockedLocal3(fe.size());
    Dune::BlockVector<Manifold3> vBlockedLocal4(fe.size());

    for (size_t i = 0; i < fe.size(); ++i) {
      auto globalIndex  = localView.index(localView.tree().localIndex(i));
      vBlockedLocal[i]  = vBlocked[globalIndex[0]];
      vBlockedLocal2[i] = vBlocked2[globalIndex[0]];
      vBlockedLocal3[i] = vBlocked3[globalIndex[0]];
      vBlockedLocal4[i] = vBlocked4[globalIndex[0]];
    }

    auto f = Ikarus::StandardLocalFunction(localBasis, vBlockedLocal);
    {
      auto localBasisNotBound = Ikarus::LocalBasis(localView.tree().finiteElement().localBasis());
      auto fNotBound          = Ikarus::StandardLocalFunction(localBasisNotBound, vBlockedLocal);
      auto h                  = f + fNotBound;
      EXPECT_DEBUG_DEATH(h.viewOverIntegrationPoints(), "The basis of the leaf nodes are not in the same state.");

      const auto& ruleHigher                 = Dune::QuadratureRules<double, 2>::rule(localView.element().type(), 7);
      auto localBasisBoundButToDifferentRule = Ikarus::LocalBasis(localView.tree().finiteElement().localBasis());
      localBasisBoundButToDifferentRule.bind(ruleHigher, bindDerivatives(0, 1));
      auto fBoundButHigher = Ikarus::StandardLocalFunction(localBasisBoundButToDifferentRule, vBlockedLocal);
      auto h2              = f + fBoundButHigher;
      EXPECT_DEBUG_DEATH(h.viewOverIntegrationPoints(), "The basis of the leaf nodes are not in the same state.");
    }
    static_assert(f.order() == linear);
    auto g = Ikarus::StandardLocalFunction(localBasis, vBlockedLocal);
    static_assert(g.order() == linear);
    auto gP = Ikarus::ProjectionBasedLocalFunction(localBasis, vBlockedLocal3);
    static_assert(gP.order() == nonLinear);
    static_assert(countNonArithmeticLeafNodes(f) == 1);
    static_assert(countNonArithmeticLeafNodes(g) == 1);
    using namespace Ikarus::DerivativeDirections;
    auto h   = f + g;
    auto ft2 = 2 * f;
    auto f23 = 2 * f * 3;
    auto mf  = -f;
    static_assert(h.order() == linear);
    static_assert(ft2.order() == linear);
    static_assert(f23.order() == linear);
    static_assert(f.order() == mf.order());

    auto a     = collectNonArithmeticLeafNodes(h);
    auto hLeaf = collectLeafNodeLocalFunctions(h);
    static_assert(std::tuple_size_v<decltype(a)> == 2);

    static_assert(countNonArithmeticLeafNodes(h) == 2);
    static_assert(std::is_same_v<decltype(h)::Ids, std::tuple<Dune::index_constant<0>, Dune::index_constant<0>>>);

    for (size_t i = 0; i < fe.size(); ++i) {
      EXPECT_TRUE(h.coefficientsRef(_0)[i] == vBlockedLocal[i]);
      EXPECT_TRUE(h.coefficientsRef(_1)[i] == vBlockedLocal[i]);
    }
    testLocalFunction(f);
    testLocalFunction(ft2);
    testLocalFunction(f23);
    testLocalFunction(mf);
    testLocalFunction(h);
    for (int gpIndex = 0; auto& gp : rule) {
      //      testLocalFunction(gP, gpIndex);

      EXPECT_THAT(f.evaluateFunction(gpIndex) + g.evaluateFunction(gpIndex),
                  EigenApproxEqual(h.evaluateFunction(gpIndex), 1e-15));

      ++gpIndex;
    }

    auto k = -dot(f + f, 3.0 * (g / 5.0) * 5.0);
    //    auto k = -dot(f + f, g);
    static_assert(k.order() == quadratic);
    auto b = collectNonArithmeticLeafNodes(k);
    static_assert(std::tuple_size_v<decltype(b)> == 3);

    //    std::cout<<Dune::className(a)<<std::endl;
    //    std::cout << Dune::className(b) << std::endl;
    static_assert(countNonArithmeticLeafNodes(k) == 3);
    //    static_assert(std::is_same_v<decltype(k)::Ids, std::tuple<Dune::index_constant<0>, Dune::index_constant<0>,
    //                                                              Ikarus::Arithmetic, Dune::index_constant<0>>>);

    const double tol = 1e-13;

    auto dotff     = dot(f, g);
    auto sqrtdotff = sqrt(dotff);
    auto normSq    = normSquared(f);
    auto logg      = log(dotff);
    auto powf      = pow<3>(dotff);

    auto f2d = Ikarus::StandardLocalFunction(localBasis, vBlockedLocal4);
    auto epsVoigt      = linearStrains(f2d);
    static_assert(normSq.order() == quadratic);

    static_assert(countNonArithmeticLeafNodes(dotff) == 2);
    static_assert(dotff.order() == quadratic);
    static_assert(std::is_same_v<decltype(dotff)::Ids, std::tuple<Dune::index_constant<0>, Dune::index_constant<0>>>);

    testLocalFunction(dotff);
    testLocalFunction(sqrtdotff);
    testLocalFunction(k);
    testLocalFunction(normSq);
    testLocalFunction(logg);
    testLocalFunction(powf);
    testLocalFunction(epsVoigt);
    for (int gpIndex = 0; auto& gp : rule) {
      EXPECT_DOUBLE_EQ((-2 * 3) * f.evaluateFunction(gpIndex).dot(g.evaluateFunction(gpIndex)),
                       k.evaluateFunction(gpIndex)[0]);

      ++gpIndex;
    }

    auto f2 = Ikarus::StandardLocalFunction(localBasis, vBlockedLocal, _0);
    auto g2 = Ikarus::StandardLocalFunction(localBasis, vBlockedLocal2, _1);
    static_assert(countNonArithmeticLeafNodes(f2) == 1);
    static_assert(countNonArithmeticLeafNodes(g2) == 1);

    auto k2 = dot(f2 + g2, g2);
    static_assert(countNonArithmeticLeafNodes(k2) == 3);
    static_assert(
        std::is_same_v<decltype(k2)::Ids,
                       std::tuple<Dune::index_constant<0>, Dune::index_constant<1>, Dune::index_constant<1>>>);

    auto b2 = collectNonArithmeticLeafNodes(k2);
    static_assert(std::tuple_size_v<decltype(b2)> == 3);

    for (int gpIndex = 0; auto& gp : rule) {
      //      testLocalFunction(k2,gpIndex);
      const auto& N  = localBasis.evaluateFunction(gpIndex);
      const auto& dN = localBasis.evaluateJacobian(gpIndex);
      EXPECT_DOUBLE_EQ((f2.evaluateFunction(gpIndex) + g2.evaluateFunction(gpIndex)).dot(g2.evaluateFunction(gpIndex)),
                       k2.evaluateFunction(gpIndex)[0]);
      auto resSingleSpatial
          = ((f2.evaluateDerivative(gpIndex, wrt(spatial(0))) + g2.evaluateDerivative(gpIndex, wrt(spatial(0))))
                     .transpose()
                 * g2.evaluateFunction(gpIndex)
             + (f2.evaluateFunction(gpIndex) + g2.evaluateFunction(gpIndex)).transpose()
                   * g2.evaluateDerivative(gpIndex, wrt(spatial(0))))
                .eval();
      EXPECT_THAT(resSingleSpatial, EigenApproxEqual(k2.evaluateDerivative(gpIndex, wrt(spatial(0))), tol));
      auto resSpatialAll
          = (((f2.evaluateDerivative(gpIndex, wrt(spatialAll)) + g2.evaluateDerivative(gpIndex, wrt(spatialAll)))
                  .transpose()
              * g2.evaluateFunction(gpIndex))
                 .transpose()
             + (f2.evaluateFunction(gpIndex) + g2.evaluateFunction(gpIndex)).transpose()
                   * g2.evaluateDerivative(gpIndex, wrt(spatialAll)))
                .eval();
      static_assert(resSpatialAll.cols() == 2);
      static_assert(resSpatialAll.rows() == 1);

      EXPECT_THAT(resSpatialAll, EigenApproxEqual(k2.evaluateDerivative(gpIndex, wrt(spatialAll)), tol));

      for (size_t i = 0; i < fe.size(); ++i) {
        const VectorType dfdi = g2.evaluateFunction(gpIndex) * N[i];

        const VectorType dkdi = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, i)));

        EXPECT_THAT(dfdi, EigenApproxEqual(dkdi, tol));

        for (size_t j = 0; j < fe.size(); ++j) {
          const MatrixType dkdij         = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, i, _1, j)));
          const MatrixType dkdijExpected = N[j] * N[i] * MatrixType::Identity();
          EXPECT_THAT(dkdijExpected, EigenApproxEqual(dkdij, tol));

          const MatrixType dkdij2         = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, i, _0, j)));
          const MatrixType dkdijExpected2 = MatrixType::Zero();
          EXPECT_THAT(dkdijExpected2, EigenApproxEqual(dkdij2, tol));
          const MatrixType dkdij3         = k2.evaluateDerivative(gpIndex, wrt(coeff(_1, i, _1, j)));
          const MatrixType dkdijExpected3 = 2 * N[i] * N[j] * MatrixType::Identity();
          EXPECT_THAT(dkdijExpected3, EigenApproxEqual(dkdij3, tol));

          const MatrixType dkdSij         = k2.evaluateDerivative(gpIndex, wrt(spatial(0), coeff(_0, i, _1, j)));
          const MatrixType dkdSijR        = k2.evaluateDerivative(gpIndex, wrt(coeff(_0, i, _1, j), spatial(0)));
          const MatrixType dkdSijExpected = (dN(j, 0) * N[i] + N[j] * dN(i, 0)) * MatrixType::Identity();
          EXPECT_THAT(dkdSijR, EigenApproxEqual(dkdSij, tol));
          EXPECT_THAT(dkdSijExpected, EigenApproxEqual(dkdSij, tol));
        }
      }
      ++gpIndex;
    }
  }
}

using namespace Dune::Functions::BasisFactory;