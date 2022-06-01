//
// Created by Alex on 21.07.2021.
//

#include <config.h>
#include <numbers>


#include <spdlog/spdlog.h>

#include <Eigen/Core>


#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/drawing/matplotHelper.hh>
#include <ikarus/utils/functionSanityChecks.hh>
#include <ikarus/utils/integrators/AdaptiveIntegrator.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/common/eigen.hpp>
#include <dune/common/power.hh>

template<typename ScalarType,typename F, typename DF>
auto exchangeEnergy(F&& f, DF&& df, const ScalarType rho)
{
  return (Dune::power(df(rho),2)*rho*rho - Dune::power(cos(f(rho)),2) + 1.0)/(2.0*rho);
}


template<typename ScalarType>
auto fourierAnsatz(const Eigen::VectorX<ScalarType>& d,double rho,double R)
{
  ScalarType res=0;

  const double pi = std::numbers::pi;

  for (int i = 0; i < d.size(); ++i)
    res+=sin(rho*(2.0*i+1.0)*pi/R/2.0)*d[i];

  return res;
}

template<typename ScalarType >
auto fourierAnsatzDerivative(const Eigen::VectorX<ScalarType>& d,double rho,double R)
{
  ScalarType res=0;

  const double pi = std::numbers::pi;

  for (int i = 0; i < d.size(); ++i)
    res+=((2*i + 1)*pi*cos(rho*(2*i + 1)*pi/(2*R))*d[i])/(2.0*R);

  return res;

}


template<typename F, typename DF>
auto energyIntegrator(F&& f, DF&& df, const double R, const double tol)
{
  auto exE= [&] (auto rho) { return exchangeEnergy( f,  df, rho);};
  return AdaptiveIntegrator::integrate(exE, 0, R, tol);
}


int main(int argc, char **argv) {

  const double R = 1;
  Eigen::VectorXd xd(2);
  xd << 1,0;

  auto f = [&](auto x ){return fourierAnsatz<double>(xd,x,R);};
  auto df = [&](auto x ){return fourierAnsatzDerivative<double>(xd,x,R);};

  Ikarus::plot::drawFunction(f, {0,R}, 100);
  Ikarus::plot::drawFunction(df, {0,R}, 100);


  std::cout << "First we approximate the integral of f(x) = x^2 on [0,2]" << std::endl;

  std::cout<<"IntVal: "<<AdaptiveIntegrator::integrate(f, 0, R, 1e-8)<<std::endl;
  std::cout<<"IntVal: "<<AdaptiveIntegrator::integrate(f, 0, R, 1e-8)<<std::endl;






  auto energy = [&](auto& d){
    auto fdual = [&](auto x ){return fourierAnsatz(d,x,R);};
    auto dfdual = [&](auto x ){return fourierAnsatzDerivative(d,x,R);};
    return energyIntegrator(fdual,dfdual,R,1e-8);};

  Eigen::VectorXdual xdual(xd.size());
  xdual =xd;
  Eigen::VectorXd g = gradient(energy, autodiff::wrt(xdual), autodiff::at(xdual));

  const int tols = 5;
  std::vector<double> result1(tols);
  std::vector<double> error1(tols);
  std::vector<double> tolerance(tols);
  for (int i = 0; i<tols; i++)
  {
    tolerance[i]=pow(10,-i);
//    result1 [i] = helper(0, R, f,tolerance[i] );
    std::cout<<"IntVal: "<<AdaptiveIntegrator::integrate(f, 0, R, tolerance[i])<<std::endl;
    std::cout << "(" << result1 [i] << "," << tolerance[i] << ")" << std::endl;
//    error1 [i] = fabs(true_value1 - result1 [i]);
  }

  std::cout << "(Approximate integral of x^2, tolerance, error )" << std::endl;
  for (int i = 0; i<tols; i++)
  {

  }


}