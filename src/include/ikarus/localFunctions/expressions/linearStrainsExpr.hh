//
// Created by lex on 4/25/22.
//

#pragma once
#include "rebind.hh"

#include <ikarus/localFunctions/expressions/unaryExpr.hh>
#include <ikarus/manifolds/realTuple.hh>
#include <ikarus/utils/linearAlgebraHelper.hh>
namespace Ikarus {

  template <typename E1>
  class LinearStrainsExpr : public UnaryLocalFunctionExpression<LinearStrainsExpr, E1> {
  public:
    using Base = UnaryLocalFunctionExpression<LinearStrainsExpr, E1>;
    using Base::UnaryLocalFunctionExpression;
    using Traits = LocalFunctionTraits<LinearStrainsExpr>;


    /** \brief Type used for coordinates */
    using ctype                    = typename Traits::ctype;
    static constexpr int valueSize = Traits::valueSize;
    constexpr int strainSize = valueSize==1 ?  1 : valueSize==2 ? : 3 : 6;
    static constexpr int gridDim   = Traits::gridDim;


    template <size_t ID_ = 0>
    static constexpr int orderID = std::min(2 * Base::E1Raw::template order<ID_>(), nonLinear);

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs &lfArgs) const {
      const auto gradM = m.evaluateDerivative(gpIndex, wrt(spatialAll), transformWith(Jinv_));
      if constexpr (valueSize==1)
      {


        return
      }
      const auto m = evaluateFunctionImpl(this->m(), lfArgs);
      return Eigen::Vector<ctype, 1>((m.transpose() * m).trace());
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs &lfArgs) const {
      const auto u = evaluateFunctionImpl(this->m(), lfArgs);
      if constexpr (DerivativeOrder == 1)  // d(squaredNorm(u))/dx = 2 * u_x * u
      {
        const auto u_x = evaluateDerivativeImpl(this->m(), lfArgs);
        return Ikarus::eval(2 * u.transpose() * u_x);
      } else if constexpr (DerivativeOrder == 2) {  // dd(squaredNorm(u))/(dxdy) =  2 *u_{x,y} * u + 2*u_x*u_y
        const auto &[u_x, u_y] = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
        if constexpr (LFArgs::hasNoSpatial and LFArgs::hasTwoCoeff) {
          const auto alonguArgs = replaceAlong(lfArgs, along(u));
          const auto u_xyAlongu = evaluateDerivativeImpl(this->m(), alonguArgs);

          return Ikarus::eval(2 * (u_xyAlongu + transpose(u_x) * u_y));
        } else if constexpr (LFArgs::hasOneSpatial and LFArgs::hasSingleCoeff) {
          const auto u_xy = evaluateDerivativeImpl(this->m(), lfArgs);
          if constexpr (LFArgs::hasOneSpatialSingle and LFArgs::hasSingleCoeff) {
            return Ikarus::eval(2 * (transpose(u) * u_xy + transpose(u_x) * u_y));
          } else if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
            std::array<std::remove_cvref_t<decltype(Ikarus::eval(transpose(u) * u_xy[0]))>, gridDim> res;
            for (int i = 0; i < gridDim; ++i)
              res[i] = 2 * (transpose(u) * u_xy[i] + transpose(u_x.col(i)) * u_y);
            return res;
          }
        }
      } else if constexpr (DerivativeOrder == 3) {  // dd(squaredNorm(u))/(dxdydz) = 2*( u_{x,y,z} * v + u_{x,y} * v_z +
                                                    // u_{x,z}*v_y + u_x*v_{y,z})
        if constexpr (LFArgs::hasOneSpatialSingle) {
          const auto argsForDyz = lfArgs.extractSecondWrtArgOrFirstNonSpatial();

          const auto &[u_x, u_y, u_z] = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
          const auto &[u_xy, u_xz]    = evaluateSecondOrderDerivativesImpl(this->m(), lfArgs);

          const auto alonguArgs             = replaceAlong(lfArgs, along(u));
          const auto argsForDyzalongu_xArgs = replaceAlong(argsForDyz, along(u_x));

          const auto u_xyzAlongu = evaluateDerivativeImpl(this->m(), alonguArgs);
          const auto u_yzAlongux = evaluateDerivativeImpl(this->m(), argsForDyzalongu_xArgs);

          return Ikarus::eval(2 * (u_xyzAlongu + transpose(u_xy) * u_z + transpose(u_xz) * u_y + u_yzAlongux));
        } else if constexpr (LFArgs::hasOneSpatialAll) {
          // check that the along argument has the correct size
          const auto &alongMatrix = std::get<0>(lfArgs.alongArgs.args);
          static_assert(alongMatrix.ColsAtCompileTime == gridDim);
          static_assert(alongMatrix.RowsAtCompileTime == 1);

          const auto uTimesA = eval(u * alongMatrix);
          static_assert(uTimesA.RowsAtCompileTime == Base::E1Raw::valueSize);
          static_assert(uTimesA.ColsAtCompileTime == gridDim);

          const auto &[gradu, u_c0, u_c1]  = evaluateFirstOrderDerivativesImpl(this->m(), lfArgs);
          const auto &[gradu_c0, gradu_c1] = evaluateSecondOrderDerivativesImpl(this->m(), lfArgs);

          const auto graduTimesA = (gradu * alongMatrix.transpose()).eval();
          static_assert(graduTimesA.RowsAtCompileTime == Base::E1Raw::valueSize);
          static_assert(graduTimesA.ColsAtCompileTime == 1);

          const auto argsForDyz = lfArgs.extractSecondWrtArgOrFirstNonSpatial();

          const auto alonguAArgs          = replaceAlong(lfArgs, along(uTimesA));
          const auto alonggraduTimesAArgs = replaceAlong(argsForDyz, along(graduTimesA));

          const auto u_xyzAlongu            = evaluateDerivativeImpl(this->m(), alonguAArgs);
          const auto u_c0c1AlongGraduTimesA = evaluateDerivativeImpl(this->m(), alonggraduTimesAArgs);
          decltype(eval(u_xyzAlongu)) res;

          res = u_xyzAlongu + u_c0c1AlongGraduTimesA;
          for (int i = 0; i < gridDim; ++i)
            res += (transpose(u_c1) * gradu_c0[i] + transpose(u_c0) * gradu_c1[i]) * alongMatrix(0, i);

          res *= 2;
          return res;
        } else
          static_assert(
              LFArgs::hasOneSpatialSingle or LFArgs::hasOneSpatialAll,
              "Only a spatial single direction or all spatial directions are supported. You should not end up here.");
      } else
        static_assert(DerivativeOrder > 3 or DerivativeOrder < 1,
                      "Only first, second and third order derivatives are supported.");
    }

  };

  template <typename E1>
  struct LocalFunctionTraits<LinearStrainsExpr<E1>> {
    using E1Raw = std::remove_cvref_t<E1>;
    /** \brief Size of the function value */
    static constexpr int valueSize = E1Raw::valueSize;
    /** \brief Type for the points for evaluation, usually the integration points */
    using DomainType = std::common_type_t<typename E1Raw::DomainType>;
    /** \brief Type used for coordinates */
    using ctype = std::common_type_t<typename E1Raw::ctype>;
    /** \brief Dimension of the grid */
    static constexpr int gridDim = E1Raw::gridDim;
  };

  template <typename E1>
  requires IsLocalFunction<E1>
  constexpr auto linearStrains(E1 &&u) { return LinearStrainsExpr<E1>(std::forward<E1>(u)); }

}  // namespace Ikarus