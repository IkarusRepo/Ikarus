//
// Created by lex on 4/25/22.
//

#pragma once
#include "rebind.hh"

#include <ikarus/localFunctions/expressions/unaryExpr.hh>
#include <ikarus/manifolds/realTuple.hh>
#include <ikarus/utils/linearAlgebraHelper.hh>
#include <ikarus/finiteElements/physicsHelper.hh>
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
    static constexpr int displacementSize = Base::E1Raw::valueSize;
    static constexpr int gridDim   = Traits::gridDim;


    template <size_t ID_ = 0>
    static constexpr int orderID = std::min(2 * Base::E1Raw::template order<ID_>(), nonLinear);

    template <typename LFArgs>
    auto evaluateValueOfExpression(const LFArgs &lfArgs) const {
      const auto gradArgs = replaceWrt(lfArgs,wrt(DerivativeDirections::spatialAll));
      const auto gradM = evaluateDerivativeImpl(this->m(), gradArgs);

        const auto E      = (0.5 * (gradM.transpose() + gradM)).eval();
        const auto EVoigt = toVoigt(E);
        return EVoigt;
    }

    template <int DerivativeOrder, typename LFArgs>
    auto evaluateDerivativeOfExpression(const LFArgs &lfArgs) const {

      if constexpr (DerivativeOrder == 1 and LFArgs::hasSingleCoeff)
      {
        Eigen::Matrix<double, valueSize, gridDim> bopI;
        const auto gradArgs = addWrt(lfArgs,wrt(DerivativeDirections::spatialAll));
        const auto gradUdI = evaluateDerivativeImpl(this->m(), gradArgs);
        if constexpr (displacementSize==1)
        {
          bopI(0,0) = gradUdI[0].diagonal()(0);
        }else if constexpr (displacementSize==2) {

          bopI.row(0) << gradUdI[0].diagonal()(0), 0;
          bopI.row(1) << 0, gradUdI[1].diagonal()(1);
          bopI.row(2) << gradUdI[1].diagonal()(0), gradUdI[0].diagonal()(1);

          return bopI;
        }else if constexpr (displacementSize==3) {
          bopI.row(0) << gradUdI[0].diagonal()(0), 0,0;
          bopI.row(1) << 0, gradUdI[1].diagonal()(1),0;
          bopI.row(2) << 0, 0, gradUdI[2].diagonal()(2);
          bopI.row(3) <<0,gradUdI[2].diagonal()(1),0, gradUdI[1].diagonal()(2);
          bopI.row(4) << gradUdI[2].diagonal()(0),0, gradUdI[0].diagonal()(2);
          bopI.row(5) << gradUdI[1].diagonal()(0), gradUdI[0].diagonal()(1),0;

          return bopI;
        }

      } else if constexpr (DerivativeOrder == 1 and LFArgs::hasOneSpatialAll)
      {
#warning This artifically returns a zero
        return Eigen::Matrix<ctype ,valueSize,gridDim>::Zero();
      } else if constexpr (DerivativeOrder == 1 and LFArgs::hasOneSpatialSingle)
      {
#warning This artifically returns a zero
        return Eigen::Matrix<ctype ,valueSize,1>::Zero();
      }
      else if constexpr (DerivativeOrder == 2) {
        if constexpr (LFArgs::hasNoSpatial and LFArgs::hasTwoCoeff) {

          return Eigen::Matrix<ctype ,displacementSize,displacementSize>::Zero();
        } else if constexpr (LFArgs::hasOneSpatial and LFArgs::hasSingleCoeff) {
          if constexpr (LFArgs::hasOneSpatialSingle and LFArgs::hasSingleCoeff) {
            return Eigen::Matrix<ctype ,valueSize,displacementSize>::Zero();
          } else if constexpr (LFArgs::hasOneSpatialAll and LFArgs::hasSingleCoeff) {
            std::array<typename Eigen::Matrix<ctype ,displacementSize,displacementSize>::Zero, gridDim> res;
            return res;
          }
        }
      } else if constexpr (DerivativeOrder == 3) {
        if constexpr (LFArgs::hasOneSpatialSingle) {
          return typename Eigen::Matrix<ctype ,displacementSize,displacementSize>::Zero();
        } else if constexpr (LFArgs::hasOneSpatialAll) {
          std::array<typename Eigen::Matrix<ctype ,displacementSize,displacementSize>::Zero, gridDim> res;
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
    static constexpr int valueSize =  E1Raw::valueSize==1 ?  1 : ( E1Raw::valueSize==2 ? 3 : 6);
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