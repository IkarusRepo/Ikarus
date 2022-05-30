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
#include <autodiff/common/eigen.hpp>


template<typename ScalarType,typename ScalarType2>
auto fourierAnsatz(const Eigen::VectorX<ScalarType>& d,ScalarType2 rho,double R)
{
  std::common_type_t<ScalarType2,ScalarType> res;

  const double pi = std::numbers::pi;

  for (int i = 0; i < d.size(); ++i)
    res+=sin(rho*(2.0*i+1.0)*pi/R/2.0)*d[i];

  return res;
}

template<typename ScalarType >
auto fourierAnsatzDerivative(const Eigen::VectorX<ScalarType>& d,double rho,double R)
{
  autodiff::dual rhod=rho;

  const autodiff::dual func = fourierAnsatz(d,rhod,R);

  return func.grad;

}


const double R = 1;
Eigen::VectorXd xd(2);


auto f(double x)
{
  xd << 1, 1;
  return fourierAnsatz<double>(xd,x,R);
}

auto df(double x)
{
  xd << 1, 1;
  return fourierAnsatzDerivative(xd,x,R);
}

int main(int argc, char **argv) {

  Ikarus::plot::drawFunction(f, {0,R}, 100);
  Ikarus::plot::drawFunction(df, {0,R}, 100);


  std::cout << "First we approximate the integral of f(x) = x^2 on [0,2]" << std::endl;

  AdaptiveIntegrator<double(const double)> test;
  std::cout<<"IntVal: "<<test.integrate(f, 0, R, 1e-8)<<std::endl;


  const int tols = 5;
  std::vector<double> result1(tols);
  std::vector<double> error1(tols);
  std::vector<double> tolerance(tols);
  for (int i = 0; i<tols; i++)
  {
    tolerance[i]=pow(10,-i);
//    result1 [i] = helper(0, R, f,tolerance[i] );
    std::cout<<"IntVal: "<<test.integrate(f, 0, R, tolerance[i])<<std::endl;
    std::cout << "(" << result1 [i] << "," << tolerance[i] << ")" << std::endl;
//    error1 [i] = fabs(true_value1 - result1 [i]);
  }

  std::cout << "(Approximate integral of x^2, tolerance, error )" << std::endl;
  for (int i = 0; i<tols; i++)
  {

  }


}