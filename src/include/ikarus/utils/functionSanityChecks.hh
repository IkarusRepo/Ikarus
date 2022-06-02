//
// Created by alex on 3/8/22.
//

#pragma once
#include "findLineSegment.hh"

#include <dune/common/float_cmp.hh>

#include <spdlog/spdlog.h>
namespace Ikarus {
  double drawResultAndReturnSlope(std::string&& functionName, const std::function<double(double)>& ftfunc, bool draw);

  struct CheckFlags {
    bool draw                = true;
    bool writeSlopeStatement = true;
    double tolerance         = 1e-2;
  };

  /*
   * The checkgradient function is inspired by http://sma.epfl.ch/~nboumal/book/  Chapter 4.8 and
   * https://github.com/NicolasBoumal/manopt/blob/master/manopt/tools/checkdiff.m
   */
  template <typename NonlinearOperator, typename UpdateType = typename NonlinearOperator::template ParameterValue<0>>
  bool checkGradient(
      NonlinearOperator& nonLinOp, CheckFlags checkFlags = CheckFlags(),
      std::function<void(typename NonlinearOperator::template ParameterValue<0>&, const UpdateType&)> p_updateFunction
      = [](typename NonlinearOperator::template ParameterValue<0>& a, const UpdateType& b) { a += b; }) {
    auto& x         = nonLinOp.firstParameter();
    const auto xOld = x;
    UpdateType b;
    if constexpr (not std::is_floating_point_v<UpdateType>) {
      b.resizeLike(nonLinOp.derivative());
      b.setRandom();
      b /= b.norm();
    } else
      b = 1;

    nonLinOp.template updateAll();
    const auto e = nonLinOp.value();

    double gradfv;
    if constexpr (not std::is_floating_point_v<UpdateType>)
      gradfv = nonLinOp.derivative().dot(b);
    else
      gradfv = nonLinOp.derivative() * b;

    auto ftfunc = [&](auto t) {
      p_updateFunction(x, t * b);
      nonLinOp.template update<0>();
      auto value = std::abs(nonLinOp.value() - e - t * gradfv);
      x          = xOld;
      return value;
    };

    const double slope = drawResultAndReturnSlope("Gradient", ftfunc, checkFlags.draw);

    const bool checkPassed = Dune::FloatCmp::le(2.0, slope, checkFlags.tolerance);

    if (checkFlags.writeSlopeStatement) {
      spdlog::info("Gradient check:");
      spdlog::info("The slope should be 2. It seems to be {}.", slope);
      if (checkPassed)
        spdlog::info("We consider this as sufficient.");
      else
        spdlog::info("The gradient seems wrong.");
    }

    nonLinOp.template updateAll();
    return checkPassed;
  }

  /*
   * The checkgradient function is inspired by http://sma.epfl.ch/~nboumal/book/  Chapter 4.8 and
   * https://github.com/NicolasBoumal/manopt/blob/master/manopt/tools/checkdiff.m
   */
  template <typename NonlinearOperator, typename UpdateType = typename NonlinearOperator::template ParameterValue<0>>
  bool checkJacobian(
      NonlinearOperator& nonLinOp, CheckFlags checkFlags = CheckFlags(),
      std::function<void(typename NonlinearOperator::template ParameterValue<0>&, const UpdateType&)> p_updateFunction
      = [](typename NonlinearOperator::template ParameterValue<0>& a, const UpdateType& b) { a += b; }) {
    auto& x         = nonLinOp.firstParameter();
    const auto xOld = x;
    UpdateType b;
    b.resizeLike(nonLinOp.derivative().row(0).transpose());
    b.setRandom();
    b /= b.norm();

    nonLinOp.template updateAll();
    const auto e = nonLinOp.value();

    const auto jacofv = nonLinOp.derivative() * b;

    auto ftfunc = [&](auto t) {
      p_updateFunction(x, t * b);
      nonLinOp.template update<0>();
      auto value = (nonLinOp.value() - e - t * jacofv).norm();
      x          = xOld;
      return value;
    };

    const double slope = drawResultAndReturnSlope("Jacobian", ftfunc, checkFlags.draw);

    const bool checkPassed = Dune::FloatCmp::le(2.0, slope, checkFlags.tolerance);

    if (checkFlags.writeSlopeStatement) {
      spdlog::info("Jacobian check:");
      spdlog::info("The slope should be 2. It seems to be {}.", slope);
      if (checkPassed)
        spdlog::info("We consider this as sufficient.");
      else
        spdlog::info("The Jacobian seems wrong.");
    }
    nonLinOp.template updateAll();
    return checkPassed;
  }

  /*
   * The checkgradient function is inspired by http://sma.epfl.ch/~nboumal/book/  Chapter 6.8 and
   * https://github.com/NicolasBoumal/manopt/blob/master/manopt/tools/checkhessian.m
   */
  template <typename NonlinearOperator, typename UpdateType = typename NonlinearOperator::template ParameterValue<0>>
  bool checkHessian(
      NonlinearOperator& nonLinOp, CheckFlags checkFlags = CheckFlags(),
      std::function<void(typename NonlinearOperator::template ParameterValue<0>&, const UpdateType&)> p_updateFunction
      = [](typename NonlinearOperator::template ParameterValue<0>& a, const UpdateType& b) { a += b; }) {
    auto& x         = nonLinOp.firstParameter();
    const auto xOld = x;
    UpdateType b;
    if constexpr (not std::is_floating_point_v<UpdateType>) {
      b.resizeLike(nonLinOp.derivative());
      b.setRandom();
      b /= b.norm();
    } else
      b = 1;

    nonLinOp.template updateAll();
    const auto e = nonLinOp.value();

    double gradfv, vhessv;
    if constexpr (not std::is_floating_point_v<UpdateType>) {
      gradfv = nonLinOp.derivative().dot(b);
      vhessv = (nonLinOp.secondDerivative() * b).dot(b);
    } else {
      gradfv = nonLinOp.derivative() * b;
      vhessv = nonLinOp.secondDerivative() * b * b;
    }

    auto ftfunc = [&](auto t) {
      p_updateFunction(x, t * b);
      nonLinOp.template update<0>();
      auto value = std::abs(nonLinOp.value() - e - t * gradfv - 0.5 * t * t * vhessv);
      x          = xOld;
      return value;
    };

    const double slope = drawResultAndReturnSlope("Hessian", ftfunc, checkFlags.draw);

    const bool checkPassed = Dune::FloatCmp::le(3.0, slope, checkFlags.tolerance);

    if (checkFlags.writeSlopeStatement) {
      spdlog::info("Hessian check:");
      spdlog::info("The slope should be 3. It seems to be {}.", slope);
      if (checkPassed)
        spdlog::info("We consider this as sufficient.");
      else
        spdlog::info("The Hessian seems wrong.");
    }
    nonLinOp.template updateAll();
    return checkPassed;
  }
}  // namespace Ikarus