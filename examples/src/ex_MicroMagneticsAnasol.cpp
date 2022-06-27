//
// Created by Alex on 21.07.2021.
//

#include <config.h>

#include <autodiff/common/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <cmath>
#include <numbers>

#include <dune/common/power.hh>

#include <spdlog/spdlog.h>

#include <Eigen/Core>

#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/drawing/matplotHelper.hh>
#include <ikarus/utils/functionSanityChecks.hh>
#include <ikarus/utils/integrators/AdaptiveIntegrator.hpp>

template <typename F, typename DF>
auto exchangeEnergy(F&& f, DF&& df, double rho, double L) -> decltype(f(rho)) {
  const double pi = std::numbers::pi;
  if (rho == 0.0)
    return 0.0;
  else
    return 2. * pi * L * ((pow(df(rho), 2) * rho * rho + pow(sin(f(rho)), 2)) / (2.0 * rho));
}

template <typename F, typename DF>
auto magnetoStaticEnergy(F&& f, DF&& df, const double rho, const double R, const double L, double tol)
    -> decltype(f(rho)) {
  const decltype(f(rho)) mzrho = cos(f(rho));

  const double pi = std::numbers::pi;

  auto magnetoStaticEnergyF = [&](auto rhoS) -> decltype(f(rho)) {
    const decltype(f(rho)) mzrhoPrime = cos(f(rhoS));
    const double denom                = sqrt(4. * L * L + rho * rho + 2. * rho * rhoS + rhoS * rhoS);
    double nom                        = 2.0 * sqrt(rho) * sqrt(rhoS);
    const double fac                  = rho + rhoS;

    const double test  = std::comp_ellint_1(nom / fac);
    const double test2 = std::comp_ellint_1(nom / denom);
    if (std::isnan(test) or std::isnan(test2)) nom -= 1e-13;  // circumvent std::comp_ellint_1(1)== infinity
    if (rhoS == 0. and rho == 0.)
      return 0;
    else
      return ((-fac) * (std::comp_ellint_1(nom / denom)) + std::comp_ellint_1(nom / fac) * denom) * mzrhoPrime * rhoS
             / (denom * fac) ;
  };
  AdaptiveIntegrator::IntegratorC integrator;
  return 8.0 * pi * mzrho * rho * integrator.integrate(magnetoStaticEnergyF, 0, R, tol)/(4 * pi);
}

template <typename ScalarType>
auto fourierAnsatz(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
  ScalarType res = 0;

  const double pi = std::numbers::pi;

  for (int i = 0; i < d.size(); ++i)
    res += sin(rho * (2.0 * i + 1.0) * pi / R / 2.0) * d[i];

  return res;
}

template <typename ScalarType>
auto fourierAnsatzDerivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
  ScalarType res = 0;

  const double pi = std::numbers::pi;

  for (int i = 0; i < d.size(); ++i)
    res += ((2. * i + 1.) * pi * cos(rho * (2. * i + 1.) * pi / (2.0 * R)) * d[i]) / (2.0 * R);

  return res;
}

template <typename F, typename DF>
auto energyIntegrator(F&& f, DF&& df, const double R, const double L, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) {
    return exchangeEnergy(f, df, rho, L) + magnetoStaticEnergy(f, df, rho, R, L, tol);
  };
  AdaptiveIntegrator::IntegratorC integrator;
  return integrator.integrate(exE, 0, R, tol);
}

template <typename F, typename DF>
auto energyIntegratorEX(F&& f, DF&& df, const double R, const double L, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) { return exchangeEnergy(f, df, rho, L); };
  AdaptiveIntegrator::IntegratorC integrator;
  return integrator.integrate(exE, 0, R, tol);
}

template <typename F, typename DF>
auto energyIntegratorMag(F&& f, DF&& df, const double R, const double L, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) { return magnetoStaticEnergy(f, df, rho, R, L, tol); };
  AdaptiveIntegrator::IntegratorC integrator;
  return integrator.integrate(exE, 0, R, tol);
}

int main(int argc, char** argv) {
  Eigen::VectorXd radii(11);
  Eigen::Matrix3Xd results(3,11);
  radii<<0.5, 1, 2,3,4,5,6,7,7.5,8,10;
  for (int i = 0; i < radii.size(); ++i) {
    const double R = radii[i];
    const double L = 0.4;
    std::cout << "R: " << R << " L: " << L << std::endl;
    Eigen::VectorXd xd(10);

    for (int i = 0; i < xd.size(); ++i)
      xd[i] = 1.0 / (i * i * i + 0.5);
    std::cout << std::setprecision(17) << std::endl;
    std::cout << "Starting coeffs: \n" << xd << std::endl;

    auto f  = [&](auto x) { return fourierAnsatz<double>(xd, x, R); };
    auto df = [&](auto x) { return fourierAnsatzDerivative<double>(xd, x, R); };

    //  Ikarus::plot::drawFunction(f, {0,R}, 100);
    //  Ikarus::plot::drawFunction(df, {0,R}, 100);
    const double tol = 1e-8;
    auto energy      = [&](auto& d) {
      auto fdual  = [&](auto x) { return fourierAnsatz(d, x, R); };
      auto dfdual = [&](auto x) { return fourierAnsatzDerivative(d, x, R); };
      return energyIntegrator(fdual, dfdual, R, L, tol);
    };

    auto grad = [&](auto&& d) {
      auto xdR = d.template cast<autodiff::dual>().eval();
      return autodiff::gradient(energy, wrt(xdR), at(xdR));
    };
    Eigen::SparseMatrix<double> hSparse;
    auto hess = [&](auto&& d) {
      auto xdR = d.template cast<autodiff::dual2nd>().eval();
      hSparse  = autodiff::hessian(energy, wrt(xdR), at(xdR)).sparseView();
      return hSparse;
    };

    auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energy, grad, hess), parameter(xd));

    //  double rhoTest = R/2;
    //  auto nonLinOpTest = Ikarus::NonLinearOperator(linearAlgebraFunctions(f, df), parameter(rhoTest));

    //  checkGradient(nonLinOpTest, {.draw = true, .writeSlopeStatement = true});
    //  checkHessian(nonLinOp, {.draw = false, .writeSlopeStatement = true});

    auto tr = Ikarus::makeTrustRegion(nonLinOp);
    tr->setup({.verbosity = 1, .Delta0 = 1});
    const auto solverInfo = tr->solve();

    std::cout << "Resulting coeffs: \n" << xd << std::endl;

    //  std::cout << "First we approximate the integral of f(x) = x^2 on [0,2]" << std::endl;
    //  AdaptiveIntegrator::IntegratorC integrator;
    std::cout << std::setprecision(17) << std::endl;
    std::cout << "ExchangeEnergy: " << energyIntegratorEX(f, df, R, L, tol) << std::endl;
    std::cout << "MagnetoStaticEnergy: " << energyIntegratorMag(f, df, R, L, tol) << std::endl;
    nonLinOp.update<0>();
    results(0,i)= nonLinOp.value();
    results(1,i)= energyIntegratorEX(f, df, R, L, tol);
    results(2,i)=  energyIntegratorMag(f, df, R, L, tol);
  }
  std::cout<<results.transpose()<<std::endl;
  //  integrator.reset();
  //  std::cout<<"IntVal: "<<integrator.integrate(f, 0, R, 1e-8)<<std::endl;
  //  integrator.reset();

//  Eigen::VectorXdual xdual(xd.size());
//  xdual             = xd;
//  Eigen::VectorXd g = gradient(energy, autodiff::wrt(xdual), autodiff::at(xdual));

  //  const int tols = 5;
  //  std::vector<double> result1(tols);
  //  std::vector<double> error1(tols);
  //  std::vector<double> tolerance(tols);
  ////  for (int i = 0; i<tols; i++)
  ////  {
  ////    tolerance[i]=pow(10,-i);
  //////    result1 [i] = helper(0, R, f,tolerance[i] );
  ////    std::cout<<"IntVal: "<<integrator.integrate(f, 0, R, tolerance[i])<<std::endl;
  ////    integrator.reset();
  ////    std::cout << "(" << result1 [i] << "," << tolerance[i] << ")" << std::endl;
  //////    error1 [i] = fabs(true_value1 - result1 [i]);
  ////  }
  //
  //  std::cout << "(Approximate integral of x^2, tolerance, error )" << std::endl;
  //  for (int i = 0; i<tols; i++)
  //  {
  //
  //  }

//  auto mz = [&](auto x) { return cos(fourierAnsatz<double>(xd, x, R)); };
//  Ikarus::plot::drawFunction(mz, {0, R}, 100);
}