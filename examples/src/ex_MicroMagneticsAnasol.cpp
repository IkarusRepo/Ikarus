//
// Created by Alex on 21.07.2021.
//

#include <config.h>
#include <chrono>

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
auto exchangeEnergy(F&& f, DF&& df, double rho, double H) -> decltype(f(rho)) {
  const double pi = std::numbers::pi;
  if (rho == 0.0)
    return 0.0;
  else
    return 2. * pi * H * ((pow(df(rho), 2) * rho * rho + pow(sin(f(rho)), 2)) / (2.0 * rho));
}

template <typename F, typename DF>
auto magnetoStaticEnergy(F&& f, DF&& df, const double rho, const double R, const double H, double tol)
    -> decltype(f(rho)) {
  const decltype(f(rho)) mzrho = cos(f(rho));

  const double pi = std::numbers::pi;
  const double delta = H/2;

  auto magnetoStaticEnergyF = [&](auto rhoS) -> decltype(f(rho)) {
    const decltype(f(rho)) mzrhoPrime = cos(f(rhoS));
    //std::comp_ellint_1 is defined for k and not for m= k^2 thus we have to insert the sqrt
    double sqrtb                 = sqrt(4.0 * rho*rhoS);
    const double ahat                  = rho + rhoS;
    const double denom                = sqrt(4. * std::pow(delta,2) + std::pow(ahat,2));

    const double test  = std::comp_ellint_1(sqrtb / ahat);
    const double test2 = std::comp_ellint_1(sqrtb / denom);
    if (std::isnan(test) or std::isnan(test2)) sqrtb -= 1e-13;  // circumvent std::comp_ellint_1(1)== infinity
    if (Dune::FloatCmp::eq(rhoS, 0.) or Dune::FloatCmp::eq(rho, 0.)) // https://www.wolframalpha.com/input?i=limit%28%281%2F%28x%2By%29*K%284*x*y%2F%28x%2By%29%5E2%29-1%2Fsqrt%28%28x%2By%29%5E2%2B4*d%5E2%29*K%284*x*y%2F%284*d%5E2%2B%28x%2By%29%5E2%29%29%29*x*y%2C+as+x-%3E0%29
      return 0;
    else
      return (1/ahat * std::comp_ellint_1(sqrtb / ahat) - 1/denom * std::comp_ellint_1(sqrtb / denom))  * mzrhoPrime * rhoS ;
  };
  AdaptiveIntegrator::IntegratorC integrator;
  return 4  * mzrho * rho * integrator.integrate(magnetoStaticEnergyF, 0, rho, tol);
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
auto energyIntegrator(F&& f, DF&& df, const double R, const double H, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) {
    return exchangeEnergy(f, df, rho, H) + magnetoStaticEnergy(f, df, rho, R, H, tol);
  };
  AdaptiveIntegrator::IntegratorC integrator;
  return integrator.integrate(exE, 0, R, tol);
}

template <typename F, typename DF>
auto energyIntegratorEX(F&& f, DF&& df, const double R, const double H, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) { return exchangeEnergy(f, df, rho, H); };
  AdaptiveIntegrator::IntegratorC integrator;
  return integrator.integrate(exE, 0, R, tol);
}

template <typename F, typename DF>
auto energyIntegratorMag(F&& f, DF&& df, const double R, const double H, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) { return magnetoStaticEnergy(f, df, rho, R, H, tol); };
  AdaptiveIntegrator::IntegratorC integrator;
  return integrator.integrate(exE, 0, R, tol);
}

int main(int argc, char** argv) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  Eigen::VectorXd radii(11);
  Eigen::Matrix<double,5,Eigen::Dynamic> results(5,radii.size()*radii.size());
//  radii<<0.5, 1, 2,3,4;
  double oldEnergy=1;
  double newEnergy=0;
  int terms=0;
  Eigen::VectorXd xdOld(1);
  xdOld(0)=1;
  while(Dune::FloatCmp::gt(std::abs(oldEnergy-newEnergy),1e-10)){
    ++terms;
//    for (int j = 0; j < 1; ++j) {
      const double R = 8;
      const double H = 0.5;
      std::cout << "R: " << R << " H: " << H << std::endl;
      Eigen::VectorXd xd(terms);
      std::cout<<xd.size()<<" "<<xdOld.size()<<std::endl;
      if(terms>1) {
        xd.setZero();
        xd.head(terms - 1) = xdOld;
      } else
        xd.setOnes();
      std::cout<<"Coeffs Begin: "<<xd<<std::endl;

//      for (int i = 0; i < xd.size(); ++i)
//        xd[i] = 1.0/(i*i*i + 0.5);
      std::cout << std::setprecision(17) << std::endl;
//    std::cout << "Starting coeffs: \n" << xd << std::endl;

      auto f = [&](auto x) { return fourierAnsatz<double>(xd, x, R); };
      auto df = [&](auto x) { return fourierAnsatzDerivative<double>(xd, x, R); };

      //  Ikarus::plot::drawFunction(f, {0,R}, 100);
      //  Ikarus::plot::drawFunction(df, {0,R}, 100);
      const double tol = 1e-8;
      auto energy = [&](auto &d) {
        auto fdual = [&](auto x) { return fourierAnsatz(d, x, R); };
        auto dfdual = [&](auto x) { return fourierAnsatzDerivative(d, x, R); };
        return energyIntegrator(fdual, dfdual, R, H, tol);
      };

      auto grad = [&](auto &&d) {
        auto xdR = d.template cast<autodiff::dual>().eval();
        return autodiff::gradient(energy, wrt(xdR), at(xdR));
      };
      Eigen::SparseMatrix<double> hSparse;
      auto hess = [&](auto &&d) {
        auto xdR = d.template cast<autodiff::dual2nd>().eval();
        hSparse = autodiff::hessian(energy, wrt(xdR), at(xdR)).sparseView();
        return hSparse;
      };

      auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energy, grad, hess), parameter(xd));

      //  double rhoTest = R/2;
      //  auto nonLinOpTest = Ikarus::NonLinearOperator(linearAlgebraFunctions(f, df), parameter(rhoTest));

      //  checkGradient(nonLinOpTest, {.draw = true, .writeSlopeStatement = true});
      //  checkHessian(nonLinOp, {.draw = false, .writeSlopeStatement = true});
//      nonLinOp.update<0>();
//      oldEnergy=nonLinOp.value();
      auto tr = Ikarus::makeTrustRegion(nonLinOp);

      tr->setup({.verbosity = 1, .grad_tol=1e-10, .corr_tol=1e-10, .Delta0 = 1});
      const auto solverInfo = tr->solve();

      std::cout << "Resulting coeffs: \n" << xd << std::endl;
      xdOld=xd;

      //  std::cout << "First we approximate the integral of f(x) = x^2 on [0,2]" << std::endl;
      //  AdaptiveIntegrator::IntegratorC integrator;
      std::cout << std::setprecision(17) << std::endl;
      std::cout << "ExchangeEnergy: " << energyIntegratorEX(f, df, R, H, tol) << std::endl;
      std::cout << "MagnetoStaticEnergy: " << energyIntegratorMag(f, df, R, H, tol) << std::endl;
      nonLinOp.update<0>();
      oldEnergy=newEnergy;
      newEnergy=nonLinOp.value();
      std::cout<<"oldEnergy: "<<oldEnergy<<std::endl;
      std::cout<<"newEnergy: "<<newEnergy<<std::endl;
      std::cout<<"diff: "<<oldEnergy-newEnergy<<std::endl;
//      results(0, i*radii.size()+j) = R;
//      results(1, i*radii.size()+j) = H;
//      results(2, i*radii.size()+j) = nonLinOp.value();
//      results(3, i*radii.size()+j) = energyIntegratorEX(f, df, R, H, tol);
//      results(4, i*radii.size()+j) = energyIntegratorMag(f, df, R, H, tol);
//      auto mz = [&](auto x) { return cos(fourierAnsatz<double>(xd, x, R)); };
//      Ikarus::plot::drawFunction(mz, {0, R}, 100);
//    }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
      std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
      std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds> (end - begin).count() << "[s]" << std::endl;
//      std::cout<<results.transpose()<<std::endl;
  }

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


}