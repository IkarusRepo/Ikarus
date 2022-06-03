//
// Created by ac126718 on 11.03.2022.
//

#pragma once
#include <matplot/matplot.h>

#include <Eigen/Core>
namespace Ikarus::plot {
  void draw_xy(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

  template <typename FunctionType>
  void drawFunction(FunctionType&& f, std::pair<double, double>&& xRange, int eValuationPoints = 100) {
    std::vector<double> x = matplot::linspace(xRange.first, xRange.second, eValuationPoints);
    std::vector<double> y = matplot::transform(x, [&f](auto x) { return f(x); });
    matplot::plot(x, y, "-o");
//    matplot::hold(matplot::on);
    matplot::show();
  }

}  // namespace Ikarus::plot