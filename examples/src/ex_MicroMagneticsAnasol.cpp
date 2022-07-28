//
// Created by Alex on 21.07.2021.
//

#include <config.h>

#include <autodiff/common/eigen.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <chrono>
#include <cmath>
#include <numbers>

#include <dune/common/power.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/quadraturerules/compositequadraturerule.hh>
#include <dune/geometry/type.hh>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <Eigen/Core>

#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/drawing/matplotHelper.hh>
#include <ikarus/utils/functionSanityChecks.hh>
#include <ikarus/utils/integrators/AdaptiveIntegrator.hpp>

template <typename F>
auto ownIntegrator(F&& f, const double R, const double tol) {
  const auto& rule
      = Dune::QuadratureRules<double, 1>::rule(Dune::GeometryTypes::line, 5, Dune::QuadratureType::GaussLegendre);
  using ScalarType  = decltype(f(std::declval<double>()));
  ScalarType res    = 0;
  ScalarType resOld = 1;
  int refine        = 1310720;
//  std::cout<<"ownIntegrator"<<std::endl;
  while (abs(resOld - res) > tol) {
    Dune::CompositeQuadratureRule rC(rule, Dune::RefinementIntervals(refine));
    resOld = res;
    res    = 0;
//    std::cout<<rC.size()<<std::endl;
    int count = 0;
    for (auto& gp : rC) {
      res += f(R * gp.position()[0]) * gp.weight() * R;
      ++count;
    }

    refine*=2;
    //    std::cout << "count" << count << std::endl;
    //    std::cout << "OldEnergy: " << resOld << std::endl;
    //    std::cout << "Energy: " << res << std::endl;
  }
  return res;
}

template <typename F, typename DF>
auto exchangeEnergy(F&& f, DF&& df, double rho, double H) -> decltype(f(rho)) {
  const double pi = std::numbers::pi;
  if (rho == 0.0)
    return 0.0;
  else
    return 2. * pi * H * ((pow(df(rho), 2) * rho * rho + pow(sin(f(rho)), 2)) / (2.0 * rho));
}

auto kernel(double rho, double rhoS, double delta) {
  // std::comp_ellint_1 is defined for k and not for m= k^2 thus we have to insert the sqrt
  double sqrtb       = sqrt(4.0 * rho * rhoS);
  const double ahat  = rho + rhoS;
  const double denom = sqrt(4. * std::pow(delta, 2) + std::pow(ahat, 2));

  double test  = std::comp_ellint_1(sqrtb / ahat);
  double test2 = std::comp_ellint_1(sqrtb / denom);
  if (std::isnan(test) or std::isnan(test2)) sqrtb -= 1e-15;  // circumvent std::comp_ellint_1(1)== infinity
  test  = std::comp_ellint_1(sqrtb / ahat);
  test2 = std::comp_ellint_1(sqrtb / denom);
  if (std::isnan(test) or std::isnan(test2)) sqrtb -= 1e-15;  // circumvent std::comp_ellint_1(1)== infinity
  test  = std::comp_ellint_1(sqrtb / ahat);
  test2 = std::comp_ellint_1(sqrtb / denom);
  if (std::isnan(test) or std::isnan(test2)) sqrtb -= 1e-15;  // circumvent std::comp_ellint_1(1)== infinity

  if (Dune::FloatCmp::eq(rhoS, 0.)
      or Dune::FloatCmp::eq(
          rho,
          0.))  // https://www.wolframalpha.com/input?i=limit%28%281%2F%28x%2By%29*K%284*x*y%2F%28x%2By%29%5E2%29-1%2Fsqrt%28%28x%2By%29%5E2%2B4*d%5E2%29*K%284*x*y%2F%284*d%5E2%2B%28x%2By%29%5E2%29%29%29*x*y%2C+as+x-%3E0%29
    return 0.;
  else
    return 1 / ahat * std::comp_ellint_1(sqrtb / ahat) - 1 / denom * std::comp_ellint_1(sqrtb / denom);
}

template <typename F, typename DF>
auto magnetoStaticEnergy(F&& f, DF&& df, const double rho, const double R, const double H, double tol)
    -> decltype(f(rho)) {
  //  const double pi    = std::numbers::pi;
  const double delta = H / 2;

  auto magnetoStaticEnergyF
      = [&](auto rhoS) -> decltype(f(rho)) { return kernel(rho, rhoS, delta) * cos(f(rhoS)) * rhoS; };

  try {
    AdaptiveIntegrator::IntegratorC integrator;
    return 4 * cos(f(rho)) * rho * integrator.integrate(magnetoStaticEnergyF, 0, rho, tol);
  }
  catch( const std::invalid_argument& e ) {
    AdaptiveIntegrator::IntegratorC integrator;
    return decltype(f(rho)){0};
  }

}

struct FourierAnsatz
{
  template <typename ScalarType>
  static auto value(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    const double pi = std::numbers::pi;

    for (int i = 0; i < d.size(); ++i)
      res += sin(rho * (2.0 * i + 1.0) * pi / R / 2.0) * d[i];

    return res;
  }

  template <typename ScalarType>
  static auto derivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    const double pi = std::numbers::pi;

    for (int i = 0; i < d.size(); ++i)
      res += ((2. * i + 1.) * pi * cos(rho * (2. * i + 1.) * pi / (2.0 * R)) * d[i]) / (2.0 * R);

    return res;
  }

  template <typename ScalarType>
  static auto secondDerivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    const double pi = std::numbers::pi;

    for (int i = 0; i < d.size(); ++i)
      res -= (Dune::power(2. * i + 1., 2) * Dune::power(pi, 2) * sin(rho * (2. * i + 1.) * pi / (2.0 * R)) * d[i])
          / (4.0 * Dune::power(R, 2));

    return res;
  }

};


struct PowerAnsatz
{
  template <typename ScalarType>
  static auto value(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    for (int i = 0; i < d.size(); ++i)
      res += d[i]*Dune::power(rho,i+1);

    return res;
  }

  template <typename ScalarType>
  static auto derivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    for (int i = 0; i < d.size(); ++i)
      res += (i+1)*d[i]*Dune::power(rho,i);

    return res;
  }

  template <typename ScalarType>
  static auto secondDerivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    for (int i = 0; i < d.size(); ++i)
      res += (i+1)*i*d[i]*Dune::power(rho,i-1);

    return res;
  }

};


struct UsovAnsatz
{
  template <typename ScalarType>
  static auto value(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    const ScalarType a = d[0];
    return rho<a ? 2*a*rho/ (a*a+rho*rho): ScalarType{1};
  }

  template <typename ScalarType>
  static auto derivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    const ScalarType a = d[0];
    return rho<a ? 2*a/(a*a + rho*rho) - 4*a*rho*rho/pow(a*a + rho*rho,2): ScalarType{0};
  }

  template <typename ScalarType>
  static auto secondDerivative(const Eigen::VectorX<ScalarType>& d, double rho, double R) {
    ScalarType res = 0;

    for (int i = 0; i < d.size(); ++i)
      res += (i+1)*i*d[i]*Dune::power(rho,i-1);

    return res;
  }

};





template <typename F, typename DF>
auto energyIntegrator(F&& f, DF&& df, const double R, const double H, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) {
    return exchangeEnergy(f, df, rho, H) + magnetoStaticEnergy(f, df, rho, R, H, tol);
  };
  try {
    AdaptiveIntegrator::IntegratorC integrator;
    return integrator.integrate(exE, 0, R, tol);
  }
  catch( const std::invalid_argument& e ) {
    AdaptiveIntegrator::IntegratorC integrator;
    return decltype(integrator.integrate(exE, 0, R, tol)){0};
  }

}

template <typename F, typename DF, typename DDF>
auto residualIntegrator(F&& f, DF&& df, DDF&& ddf, const double R, const double H, const double tol) {
  const double pi    = std::numbers::pi;
  const double delta = H / 2;
  auto exE           = [&](auto rho) -> decltype(f(rho)) {
    auto mag = [&](auto rhoS) -> decltype(f(rho)) { return cos(f(rhoS)) * rhoS * kernel(rho, rhoS, delta); };
    AdaptiveIntegrator::IntegratorC integrator;
    return abs(-4*sin(f(rho))*rho*integrator.integrate(mag,0,R,tol)+((rho==0)? 0.0 : 2*pi*H*(sin(f(rho))*cos(f(rho))/rho-df(rho)-ddf(rho)*rho)));
  };
  try {
    AdaptiveIntegrator::IntegratorC integrator;
    return integrator.integrate(exE, 0, R, tol);
  }
  catch( const std::invalid_argument& e ) {
    AdaptiveIntegrator::IntegratorC integrator;
    return decltype(integrator.integrate(exE, 0, R, tol)){0};
  }
}

template <typename F, typename DF>
auto energyIntegratorEX(F&& f, DF&& df, const double R, const double H, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) { return exchangeEnergy(f, df, rho, H); };

  return ownIntegrator(exE, R, tol);
}

template <typename F, typename DF>
auto energyIntegratorMag(F&& f, DF&& df, const double R, const double H, const double tol) {
  auto exE = [&](auto rho) -> decltype(f(rho)) { return magnetoStaticEnergy(f, df, rho, R, H, tol); };
  try {
    AdaptiveIntegrator::IntegratorC integrator;
    return integrator.integrate(exE, 0, R, tol);
  }
  catch( const std::invalid_argument& e ) {
    AdaptiveIntegrator::IntegratorC integrator;
    return decltype(integrator.integrate(exE, 0, R, tol)){0};
  }
}

int main(int argc, char** argv) {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::trace);
//  console_sink->set_pattern("[%^%l%$] %v");

//  using Ansatz = FourierAnsatz;
  using Ansatz = UsovAnsatz;
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_st>("logs/Micromagnetics.txt", false);
  file_sink->set_level(spdlog::level::trace);
//  file_sink->set_pattern("[%^%l%$] %v");

  //    spdlog::logger logger("multi_sink", {console_sink, file_sink});
  spdlog::logger logger("multi_sink", {console_sink,file_sink});
  spdlog::set_default_logger(std::make_shared<spdlog::logger>(logger));
  spdlog::flush_every(std::chrono::seconds(1));
//  spdlog::set_pattern("*** [%H:%M:%S %z] [thread %t] %v ***");

  Eigen::IOFormat HeavyFmt(Eigen::FullPrecision, 0, " ", "\n", "", "", "\n", "");
  using Ansatz = FourierAnsatz;
//  using Ansatz = PowerAnsatz;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  Eigen::VectorXd radii(11);
  Eigen::VectorXd heights(11);
  Eigen::Matrix<double, 8, Eigen::Dynamic> results(8, radii.size() * heights.size());
    radii<<0.5, 1, 2,3,4,5,6,7,8,9,10;
  heights<<0.5, 1, 2,3,4,5,6,7,8,9,10;
//  radii << 0.5,8;
  double oldEnergy = 1;
  double newEnergy = 0;
  int terms0       = 1;
  int terms        = terms0;

  for (int i = 0; i < radii.size(); ++i) {
    for (int j = 0; j < heights.size(); ++j) {
      terms        = terms-1;
      Eigen::VectorXd xdOld(terms);
      for (int k = 0; k < xdOld.size(); ++k) {
        xdOld[k] = 1.0/(k*k*k + 0.5);
      }
//      std::cout << xdOld << std::endl;
      double strongError = 0;
      double R, H, magE, exE;
      oldEnergy=1;
      newEnergy = 0;
//      while (Dune::FloatCmp::gt(std::abs(oldEnergy - newEnergy), 1e6)) {
//        ++terms;
        R = radii[i] * sqrt(2);
        H = heights[j]* sqrt(2);

        Eigen::VectorXd xd(terms);
//        std::cout << xd.size() << " " << xdOld.size() << std::endl;
        if (terms > 1) {
          xd.setZero();
          xd.head(terms - 1) = xdOld;
        } else
          xd.setOnes();
        auto f   = [&](auto x) { return Ansatz::value<double>(xd, x, R); };
        spdlog::info("R: {:.16f} H: {:.16f}" , R , H );
        spdlog::info("Coeffs Begin: {}",xd.format(HeavyFmt)  );


//        std::cout << std::setprecision(17) << std::endl;

        auto df  = [&](auto x) { return Ansatz::derivative<double>(xd, x, R); };
        auto ddf = [&](auto x) { return Ansatz::secondDerivative<double>(xd, x, R); };

//          Ikarus::plot::drawFunction(f, {0,R}, 100);
        //  Ikarus::plot::drawFunction(df, {0,R}, 100);
        const double tol = 1e-8;
        auto energy      = [&](auto& d) {
          auto fdual  = [&](auto x) { return Ansatz::value(d, x, R); };
          auto dfdual = [&](auto x) { return Ansatz::derivative(d, x, R); };
          return energyIntegrator(fdual, dfdual, R, H, tol);
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

        auto tr = Ikarus::makeTrustRegion<decltype(nonLinOp),Ikarus::PreConditioner::IncompleteCholesky>(nonLinOp);

        tr->setup({.verbosity = 1, .grad_tol=1e-10, .corr_tol=1e-10, .Delta0 = 1});

        try {
          const auto solverInfo = tr->solve();
          if(!solverInfo.sucess)
          {
            results(0, i * radii.size() + j) = R;
            results(1, i * radii.size() + j) = H;
//            results(2, i * radii.size() + j) = std::numeric_limits<double>::infinity();
//            results(3, i * radii.size() + j) = std::numeric_limits<double>::infinity();
            magE = std::numeric_limits<double>::infinity();
            strongError = std::numeric_limits<double>::infinity();
            results(6, i * radii.size() + j) = terms;

            spdlog::info("Failed since TrustRegion didn't converge" );
            break;
          }
        }
        catch( const std::invalid_argument& e ) {
          results(0, i * radii.size() + j) = R;
          results(1, i * radii.size() + j) = H;
          results(2, i * radii.size() + j) = std::numeric_limits<double>::infinity();
          results(3, i * radii.size() + j) = std::numeric_limits<double>::infinity();
          results(4, i * radii.size() + j) = std::numeric_limits<double>::infinity();
          results(5, i * radii.size() + j) = std::numeric_limits<double>::infinity();
          results(6, i * radii.size() + j) = terms;
          const auto& curcols = results.leftCols(i * radii.size() + j+1).transpose();

          spdlog::info("Result: {}",curcols.format(HeavyFmt)  );
          break;
        }





        //  double rhoTest = R/2;
        //  auto nonLinOpTest = Ikarus::NonLinearOperator(linearAlgebraFunctions(f, df), parameter(rhoTest));
        strongError = residualIntegrator(f, df, ddf, R, H, tol);
        spdlog::info("Strong Error: {0:.16f}" , strongError );
        spdlog::info("Resulting coeffs: " );
        spdlog::info("Resulting Begin: {}",xd.format(HeavyFmt)  );
        xdOld = xd;



        //  std::cout << "First we approximate the integral of f(x) = x^2 on [0,2]" << std::endl;
        //  AdaptiveIntegrator::IntegratorC integrator;
//        std::cout << std::setprecision(17) << std::endl;
        exE = energyIntegratorEX(f, df, R, H, tol);
        spdlog::info("ExchangeEnergy: {0:.16f}" , exE );



        magE = energyIntegratorMag(f, df, R, H, tol);
        spdlog::info("MagnetoStaticEnergy: {0:.16f}" , magE );
        nonLinOp.update<0>();
        oldEnergy = newEnergy;
        newEnergy = nonLinOp.value();
        spdlog::info("oldEnergy: {0:.16f}" , oldEnergy );
        spdlog::info("newEnergy: {0:.16f}" , newEnergy );
        spdlog::info("diff: {0:.16f}" , oldEnergy - newEnergy );

        auto mz            = [&](auto x) { return cos(Ansatz::value<double>(xd, x, R)); };
        auto magEFunc          = [&](auto rho) -> decltype(f(rho)) { return magnetoStaticEnergy(f, df, rho, R, H, tol); };
        auto exEFunc           = [&](auto rho) -> decltype(f(rho)) { return exchangeEnergy(f, df, rho, H); };
        const double delta = H / 2;
        const double pi    = std::numbers::pi;

        auto RexE = [&](auto rho) -> decltype(f(rho)) {
          auto mag = [&](auto rhoS) -> decltype(f(rho)) { return cos(f(rhoS)) * rhoS * kernel(rho, rhoS, delta); };
          AdaptiveIntegrator::IntegratorC integrator;

        return abs(-4*sin(f(rho))*rho*integrator.integrate(mag,0,R,tol)+((rho==0)? 0.0 : 2*pi*H*(sin(f(rho))*cos(f(rho))/rho-df(rho)-ddf(rho)*rho)));
      };

        spdlog::info("MzatR: {0:.16f}" , mz(R) );
//              Ikarus::plot::drawFunction(mz, {0, R}, 100);
        //      Ikarus::plot::drawFunction(magEFunc, {0, R}, 100);
        //      Ikarus::plot::drawFunction(exEFunc, {0, R}, 100);
//              Ikarus::plot::drawFunction(RexE, {0, R}, 100);
        //    }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
//                  << "[µs]" << std::endl;
//        spdlog::info("ime difference = {0:.16f} [µs]" , std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() );
//        spdlog::info("ime difference = {0:.16f} [ns]" , std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() );
        spdlog::info("Time difference = {} [s]" , std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() );
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()
//                  << "[ns]" << std::endl;
//        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count()
//                  << "[s]" << std::endl;
        //      std::cout<<results.transpose()<<std::endl;
//        logger.flush();
        if(exE<1e-16) {
          spdlog::info("Early exit due to vanishing exchange energy."  );
          terms=1;
          break;
        }

        if(strongError<1e-12) {
          spdlog::info("Early exit due to vanishing strongError."  );
          break;
        }

      }
      results(0, i * radii.size() + j) = R;
      results(1, i * radii.size() + j) = H;
      results(2, i * radii.size() + j) = newEnergy;
      results(3, i * radii.size() + j) = exE;
      results(4, i * radii.size() + j) = magE;
      results(5, i * radii.size() + j) = strongError;
      results(6, i * radii.size() + j) = terms;
      const auto& curcols = results.leftCols(i * radii.size() + j+1).transpose();

      spdlog::info("Result: {}",curcols.format(HeavyFmt)  );
//      std::cout<<results.leftCols(i * radii.size() + j+1).transpose()<<std::endl;

    }
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