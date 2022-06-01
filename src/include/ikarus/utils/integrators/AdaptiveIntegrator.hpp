#ifndef ADAPTIVEINTEGRATOR_HPP_
#define ADAPTIVEINTEGRATOR_HPP_

#include "Integrator.hpp"
#include <cmath>
using std::sqrt;
using std::fabs;
#include <limits>
using std::numeric_limits;
#include <iostream>
using std::cerr;

//template <class T>
//class AdaptiveIntegrator : Integrator<T> {
//private:
//    const static double alpha, beta, x1, x2, x3;
//    const static double x[12];
//
//    bool terminated;
//public:
//    AdaptiveIntegrator();
//
//    void set_tol(double tol);
//
//    double integrate(T &func, const double a, const double b, const double tol);
//
//    double adaptlobstp(T &func, const double a, const double b, const double fa,
//         const double fb, const double is);
//};
//
//
//template <class T>
//AdaptiveIntegrator<T>::AdaptiveIntegrator() : terminated(false) {}

namespace AdaptiveIntegrator
{
  const double alpha = sqrt(2./3.);
  const double beta = 1./sqrt(5.);
  const double x1 = .94288241569547971905635175843185720232;
  const double x2 = .64185334234578130578123554132903188354;
  const double x3 = .23638319966214988028222377349205292599;


template <class T>
  auto adaptlobstp(T &&f, const double a, const double b,
                   const double fa, const double fb, const double is )
{
  double m, h;
  m = (a+b)/2.; h = (b-a)/2.;

  auto mll = m - alpha*h;
  auto ml = m - beta*h;
  auto mr = m + beta*h;
  auto mrr = m + alpha*h;

  auto fmll = f(mll);
  auto fml = f(ml);
  auto fm = f(m);
  auto fmr = f(mr);
  auto fmrr = f(mrr);

  auto i2 = (h/6.)*(fa + fb + 5.*(fml+fmr));
  auto i1 = (h/1470.)*(77*(fa+fb) + 432.*(fmll + fmrr) + 625.*(fml + fmr) + 672.*fm);

  if (is + (i1-i2) == is or mll <= a or b <= mrr)
  {
//    if ( (m <= a or b <= m) and !terminated)
    if ( (m <= a or b <= m))
    {
      cerr << "No machine number in the interval. Requested tolerance may not be met.\n";
//      terminated = true;
    }
    return i1;
  }
  else
  {
    return adaptlobstp(f,a,mll,fa,fmll,is)
           + adaptlobstp(f,mll,ml,fmll,fml,is)
           + adaptlobstp(f,ml,m,fml,fm,is)
           + adaptlobstp(f,m,mr,fm,fmr,is)
           + adaptlobstp(f,mr,mrr,fmr,fmrr,is)
           + adaptlobstp(f,mrr,b,fmrr,fb,is);
  }
}




template <class T>
auto integrate(T &&f, const double a, const double b, const double tol_)
{



  double tol, eps;
    eps = numeric_limits<double>::epsilon();
    tol = (tol_ < eps) ?  eps : tol_;

    double m, h;
    m = (a+b)/2.; h = (b-a)/2.;
    std::array<decltype(f(a)),13> y = {f(a),f(m-x1*h),f(m-alpha*h),f(m-x2*h),f(m-beta*h),f(m-x3*h),f(m),
                    f(m+x3*h),f(m+beta*h),f(m+x2*h),f(m+alpha*h),f(m+x1*h) ,f(b)};

    auto fa = y[0];
    auto fb = y[12];


    auto i2 = (h/6.)*(y[0] + y[12] + 5.*(y[4] + y[8]));
    auto i1 = (h/1470.)*(77.*(y[0]+y[12]) + 432.*(y[2]+y[10]) + 625.*(y[4]+y[8]) + 672.*y[6]);
    auto is = h*(.0158271919734802*(y[0]+y[12]) + .0942738402188500*(y[1]+y[11])
           + .155071987336585*(y[2]+y[10]) + .188821573960182*(y[3]+y[9])
           + .199773405226859*(y[4]+y[8]) + .224926465333340*(y[5]+y[7])
           + .242611071901408*y[6]);    

    auto erri1 = abs(i1 - is);
    auto erri2 = abs(i2 - is);
    auto R = (erri2 != 0.) ? erri1/erri2 : 1.;
    
    tol = (R > 0. and R < 1.) ? tol_/R : tol_; 
    is = fabs(is)*tol/eps;
    if (is == 0.) is = b-a;

    return adaptlobstp(f, a, b, fa, fb, is);

}


}


#endif // ADAPTIVEINTEGRATOR_HPP_
