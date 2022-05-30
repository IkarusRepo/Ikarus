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
//helper functions

double coarse_helper(double a, double b, const std::function<double(double)>& f)
{
  return 0.5*(b - a)*(f(a) + f(b)); //by definition of coarse approx
}


double fine_helper(double a, double b, const std::function<double(double)>& f)
{
  double c = (a+b)/2.0;
  return 0.25*(b - a)*(f(a) + 2*f(c) + f(b)); //by definition of fine approx
}



double trap_rule(double a, double b, const std::function<double(double)>& f,double tolerance, int count)
{
  double coarse = coarse_helper(a,b, f); //getting the coarse and fine approximations from the helper functions
  double fine = fine_helper(a,b,f);
  const int minLevel = 2;
  const int maxLevel = 100;

  if ((fabs(coarse - fine) <= 3.0*tolerance) && (count >= minLevel))
  //return fine if |c-f| <3*tol, (fine is "good") and if count above
  //required minimum level
  {
    return fine;
  }
  else if (count >= maxLevel)
  //maxLevel is the maximum number of recursion we can go through
  {
    return fine;

  }
  else
  {
    //if none of these conditions are satisfied, split [a,b] into [a,c] and [c,b] performing trap_rule
    //on these intervals -- using recursion to adaptively approach a tolerable |coarse-fine| level
    //here, (a+b)/2 = c
    ++count;
    return  (trap_rule(a, (a+b)/2.0, f, tolerance/2.0, count) + trap_rule((a+b)/2.0, b, f, tolerance/2.0, count));
  }
}


    //test function
 double function_1(double a)
{
  return pow(a,2);
}

double analytic_first(double a,double b)
{
  return 1.0/3.0*(pow(b,3)-pow(a,3));
}

//"true" integral for comparison and tolerances



double helper(double a, double b, const std::function<double(double)>& f, double tol)
{
  return trap_rule(a, b, f, tol, 1);
}




template<typename ScalarType >
auto fourierAnsatz(const Eigen::VectorX<ScalarType>& d,double rho,double R)
{
  ScalarType res;

  const double pi = std::numbers::pi;

  for (int i = 0; i < d.size(); ++i)
    res+=sin(rho*(2.0*i+1.0)*pi/R/2.0)*d[i];

  return res;
}

const double R = 1;

auto f(double x)
{
  Eigen::VectorXd xd(2);
  xd << 1, 1;
  return std::cos(fourierAnsatz<double>(xd,x,R));
}

int main(int argc, char **argv) {

//  Ikarus::plot::drawFunction(func, {0,R}, 100);


  std::cout << "First we approximate the integral of f(x) = x^2 on [0,2]" << std::endl;

  AdaptiveIntegrator<double(const double)> test;
  std::cout<<"IntVal: "<<test.integrate(f, 0, R, 1e-8)<<std::endl;

  const double true_value1 = analytic_first(0,R);
  const int tols = 5;
  std::vector<double> result1(tols);
  std::vector<double> error1(tols);
  std::vector<double> tolerance(tols);
  for (int i = 0; i<tols; i++)
  {
    tolerance[i]=pow(10,-i);
    result1 [i] = helper(0, R, f,tolerance[i] );
    std::cout<<"IntVal: "<<test.integrate(f, 0, R, tolerance[i])<<std::endl;
    std::cout << "(" << result1 [i] << "," << tolerance[i] << ")" << std::endl;
//    error1 [i] = fabs(true_value1 - result1 [i]);
  }

  std::cout << "(Approximate integral of x^2, tolerance, error )" << std::endl;
  for (int i = 0; i<tols; i++)
  {

  }


}