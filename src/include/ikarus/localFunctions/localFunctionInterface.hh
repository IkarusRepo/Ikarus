//
// Created by alex on 3/17/22.
//

#pragma once
#include "leafNodeCollection.hh"
#include "localFunctionArguments.hh"

#include <concepts>

#include <ikarus/localBasis/localBasis.hh>
#include <ikarus/localFunctions/expressions/exprChecks.hh>
#include <ikarus/utils/traits.hh>

namespace Ikarus {

  template <typename LocalFunctionImpl>
  class LocalFunctionInterface {
  public:
    using Traits                 = LocalFunctionTraits<LocalFunctionImpl>;
    using DomainType             = typename Traits::DomainType;
    static constexpr int gridDim = Traits::gridDim;

    template <typename WrtType>
    static constexpr bool hasTwoCoeff = DerivativeDirections::HasTwoCoeff<WrtType>;
    template <typename WrtType>
    static constexpr bool hasSingleCoeff = DerivativeDirections::HasSingleCoeff<WrtType>;
    template <typename WrtType>
    static constexpr bool hasNoCoeff = DerivativeDirections::HasNoCoeff<WrtType>;

    template <std::size_t ID_ = 0>
    static constexpr auto order(Dune::index_constant<ID_> = Dune::index_constant<ID_>()) {
      return LocalFunctionImpl::template orderID<ID_>;
    }

    /** \brief Return the function value*/
    template <typename DomainTypeOrIntegrationPointIndex, typename... TransformArgs>
    requires IsIntegrationPointIndexOrIntegrationPointPosition<DomainTypeOrIntegrationPointIndex, DomainType>
    auto evaluateFunction(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                          const TransformWith<TransformArgs...>& transArgs = transformWith()) const {
      const LocalFunctionEvaluationArgs evalArgs(ipIndexOrPosition, wrt(), along(), transArgs);
      return evaluateFunctionImpl(*this, evalArgs);
    }

    /** \brief Deligation function to calculate derivatives */
    template <typename... WrtArgs, typename... TransformArgs, typename... AlongArgs,
              typename DomainTypeOrIntegrationPointIndex>
    auto evaluateDerivative(const DomainTypeOrIntegrationPointIndex& localOrIpId, Wrt<WrtArgs...>&& args,
                            Along<AlongArgs...>&& along,
                            TransformWith<TransformArgs...>&& transArgs = transformWith()) const {
      const LocalFunctionEvaluationArgs evalArgs(localOrIpId, std::forward<Wrt<WrtArgs...>>(args),
                                                 std::forward<Along<AlongArgs...>>(along), transArgs);
      return evaluateDerivativeImpl(*this, evalArgs);
    }

    /** \brief Deligation function to calculate derivatives, without providing along arguments */
    template <typename... WrtArgs, typename... TransformArgs, typename DomainTypeOrIntegrationPointIndex>
    auto evaluateDerivative(const DomainTypeOrIntegrationPointIndex& localOrIpId, Wrt<WrtArgs...>&& args,
                            TransformWith<TransformArgs...>&& transArgs = transformWith()) const {
      return evaluateDerivative(localOrIpId, std::forward<Wrt<WrtArgs...>>(args), along(),
                                std::forward<TransformWith<TransformArgs...>>(transArgs));
    }

    /** Return the view of the integration points of the bound Basis with id I */
    template <std::size_t I = 0>
    auto viewOverIntegrationPoints(Dune::index_constant<I> = Dune::index_constant<I>()) const {
      assert(checkIfAllLeafNodeHaveTheSameBasisState(impl())
             && "The basis of the leaf nodes are not in the same state.");
      auto leafNodeCollection = collectLeafNodeLocalFunctions(impl());
      auto& node               = leafNodeCollection.node(Dune::index_constant<I>());
      return node.basis().viewOverIntegrationPoints();
    }

    /** Return the a non const reference of the coefficients if the leaf node with id tag I is unique. Otherwise this
     * function is deactivated */
    template <std::size_t I = 0>
    requires(Std::countType<typename LocalFunctionImpl::Ids, Dune::index_constant<I>>()
             == 1) auto& coefficientsRef(Dune::index_constant<I> = Dune::index_constant<I>()) {
      auto leafNodeCollection = collectLeafNodeLocalFunctions(impl());
      return collectLeafNodeLocalFunctions(impl()).coefficientsRef(Dune::index_constant<I>());
    }

    /** Return the a non const reference of the coefficients of the leaf node with id tag I. */
    template <std::size_t I = 0>
    const auto& coefficientsRef(Dune::index_constant<I> = Dune::index_constant<I>()) const {
      return collectLeafNodeLocalFunctions(impl()).coefficientsRef(Dune::index_constant<I>());
    }

    /** \brief Forward the binding to the local basis */
    template <typename IntegrationRule, typename... Ints>
    requires std::conjunction_v<std::is_convertible<int, Ints>...>
    void bind(IntegrationRule&& p_rule, Impl::Derivatives<Ints...>&& ints) {
      impl().basis.bind(std::forward<IntegrationRule>(p_rule), std::forward<Impl::Derivatives<Ints...>>(ints));
    }

  protected:
    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename... AlongArgs, typename... TransformArgs>
    auto evaluateSecondDerivativeWRTCoeffsImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                               const std::array<size_t, 2>& coeffsIndex,
                                               const Along<AlongArgs...>& alongArgs,
                                               const TransformWith<TransformArgs...>& transArgs) const {
      return Eigen::Matrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::correctionSize,
                           LocalFunctionImpl::correctionSize>::Zero();
    }

    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename... AlongArgs, typename... TransformArgs>
    auto evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialSingleImpl(
        const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, const std::array<size_t, 2>& coeffsIndex,
        const int spatialIndex, const Along<AlongArgs...>& alongArgs,
        const TransformWith<TransformArgs...>& transArgs) const {
      return Eigen::Matrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::correctionSize,
                           LocalFunctionImpl::correctionSize>::Zero();
    }

    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename... TransformArgs>
    auto evaluateDerivativeWRTSpaceAllImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                           const TransformWith<TransformArgs...>& transArgs) const {
      return typename LocalFunctionImpl::Jacobian::Zero();
    }

    /* Default implementation returns Zero expression if they are not overloaded */
    template <typename DomainTypeOrIntegrationPointIndex, typename... TransformArgs>
    auto evaluateDerivativeWRTCoeffsImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, int coeffsIndex,
                                         const TransformWith<TransformArgs...>& transArgs) const {
      return Eigen::Matrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::valueSize,
                           LocalFunctionImpl::correctionSize>::Zero();
    }

    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename... TransformArgs>
    auto evaluateDerivativeWRTCoeffsANDSpatialImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                                   int coeffsIndex,
                                                   const TransformWith<TransformArgs...>& transArgs) const {
      return std::array<Ikarus::DerivativeDirections::DerivativeNoOp, gridDim>();
    }

    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename... TransformArgs>
    auto evaluateDerivativeWRTCoeffsANDSpatialSingleImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                                         int coeffsIndex, int spatialIndex,
                                                         const TransformWith<TransformArgs...>& transArgs) const {
      return Eigen::Matrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::valueSize,
                           LocalFunctionImpl::correctionSize>::Zero();
    }
    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename... AlongArgs, typename... TransformArgs>
    auto evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialImpl(
        const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition, const std::array<size_t, 2>& coeffsIndex,
        const Along<AlongArgs...>& alongArgs, const TransformWith<TransformArgs...>& transArgs) const {
      return Eigen::Matrix<typename LocalFunctionImpl::ctype, LocalFunctionImpl::correctionSize,
                           LocalFunctionImpl::correctionSize>::Zero();
    }

    /* Default implementation returns Zero expression if they are not overloaded  */
    template <typename DomainTypeOrIntegrationPointIndex, typename... TransformArgs>
    auto evaluateDerivativeWRTSpaceSingleImpl(const DomainTypeOrIntegrationPointIndex& ipIndexOrPosition,
                                              int spaceIndex, const TransformWith<TransformArgs...>& transArgs) const {
      return typename Eigen::internal::plain_col_type<typename LocalFunctionImpl::Jacobian>::type::Zero();
    }

  private:
    template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl_>
    friend auto evaluateDerivativeImpl(const LocalFunctionInterface<LocalFunctionImpl_>& f,
                                       const LocalFunctionEvaluationArgs_& localFunctionArgs);

    template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl_>
    friend auto evaluateFunctionImpl(const LocalFunctionInterface<LocalFunctionImpl_>& f,
                                     const LocalFunctionEvaluationArgs_& localFunctionArgs);

    template <typename LF>
    requires LocalFunction<LF>
    friend auto collectNonArithmeticLeafNodes(LF&& a);

    constexpr LocalFunctionImpl const& impl() const  // CRTP
    {
      return static_cast<LocalFunctionImpl const&>(*this);
    }

    constexpr LocalFunctionImpl& impl()  // CRTP
    {
      return static_cast<LocalFunctionImpl&>(*this);
    }
  };

  template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl>
  auto evaluateFunctionImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                            const LocalFunctionEvaluationArgs_& localFunctionArgs) {
    if constexpr (LocalFunctionImpl::isLeaf)
      return f.impl().evaluateFunctionImpl(localFunctionArgs.integrationPointOrIndex,
                                           localFunctionArgs.transformWithArgs);
    else {
      return f.impl().evaluateValueOfExpression(localFunctionArgs);
    }
  }

  template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl>
  auto evaluateDerivativeImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                              const LocalFunctionEvaluationArgs_& localFunctionArgs) {
    using namespace Dune::Indices;
    if constexpr (LocalFunctionImpl::isLeaf) {
      if constexpr (localFunctionArgs.hasNoCoeff) {
        if constexpr (localFunctionArgs.hasOneSpatialSingle) {
          return f.impl().evaluateDerivativeWRTSpaceSingleImpl(localFunctionArgs.integrationPointOrIndex,
                                                               localFunctionArgs.spatialPartialIndices,
                                                               localFunctionArgs.transformWithArgs);
        } else if constexpr (localFunctionArgs.hasOneSpatialAll) {
          return f.impl().evaluateDerivativeWRTSpaceAllImpl(localFunctionArgs.integrationPointOrIndex,
                                                            localFunctionArgs.transformWithArgs);
        }
      } else if constexpr (localFunctionArgs.hasSingleCoeff) {
        if constexpr (decltype(localFunctionArgs.coeffsIndices[_0][_0])::value != LocalFunctionImpl::Ids::value)
          return DerivativeDirections::DerivativeNoOp();
        else if constexpr (localFunctionArgs.hasNoSpatial) {
          return f.impl().evaluateDerivativeWRTCoeffsImpl(localFunctionArgs.integrationPointOrIndex,
                                                          localFunctionArgs.coeffsIndices[_0][1],
                                                          localFunctionArgs.transformWithArgs);
        } else if constexpr (localFunctionArgs.hasOneSpatialSingle) {
          return f.impl().evaluateDerivativeWRTCoeffsANDSpatialSingleImpl(
              localFunctionArgs.integrationPointOrIndex, localFunctionArgs.coeffsIndices[_0][1],
              localFunctionArgs.spatialPartialIndices, localFunctionArgs.transformWithArgs);
        } else if constexpr (localFunctionArgs.hasOneSpatialAll) {
          return f.impl().evaluateDerivativeWRTCoeffsANDSpatialImpl(localFunctionArgs.integrationPointOrIndex,
                                                                    localFunctionArgs.coeffsIndices[_0][1],
                                                                    localFunctionArgs.transformWithArgs);
        }
      } else if constexpr (localFunctionArgs.hasTwoCoeff) {
        if constexpr (localFunctionArgs.hasNoSpatial) {
          return f.impl().evaluateSecondDerivativeWRTCoeffsImpl(
              localFunctionArgs.integrationPointOrIndex,
              {localFunctionArgs.coeffsIndices[_0][1], localFunctionArgs.coeffsIndices[_1][1]},
              localFunctionArgs.alongArgs, localFunctionArgs.transformWithArgs);
        } else if constexpr (localFunctionArgs.hasOneSpatialSingle) {
          return f.impl().evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialSingleImpl(
              localFunctionArgs.integrationPointOrIndex,
              {localFunctionArgs.coeffsIndices[_0][1], localFunctionArgs.coeffsIndices[_1][1]},
              localFunctionArgs.spatialPartialIndices, localFunctionArgs.alongArgs,
              localFunctionArgs.transformWithArgs);
        } else if constexpr (localFunctionArgs.hasOneSpatialAll) {
          return f.impl().evaluateThirdDerivativeWRTCoeffsTwoTimesAndSpatialImpl(
              localFunctionArgs.integrationPointOrIndex,
              {localFunctionArgs.coeffsIndices[_0][1], localFunctionArgs.coeffsIndices[_1][1]},
              localFunctionArgs.alongArgs, localFunctionArgs.transformWithArgs);
        }
      }
    } else {
      return f.impl().template evaluateDerivativeOfExpression<localFunctionArgs.derivativeOrder>(localFunctionArgs);
    }
  }

  template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl>
  auto evaluateFirstOrderDerivativesImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                                         const LocalFunctionEvaluationArgs_& localFunctionArgs) {
    if constexpr (localFunctionArgs.derivativeOrder == 3) {
      const auto argsForDx              = localFunctionArgs.extractSpatialOrFirstWrtArg();
      const auto [argsForDy, argsForDz] = extractWrtArgsTwoCoeffsToSingleCoeff(localFunctionArgs);
      auto dfdx                         = evaluateDerivativeImpl(f, argsForDx);
      auto dfdy                         = evaluateDerivativeImpl(f, argsForDy);
      auto dfdz                         = evaluateDerivativeImpl(f, argsForDz);
      return std::make_tuple(dfdx, dfdy, dfdz);
    } else if constexpr (localFunctionArgs.derivativeOrder == 2) {
      const auto [argsForDx, argsForDy] = extractFirstTwoArgs(localFunctionArgs);
      auto dfdx                         = evaluateDerivativeImpl(f, argsForDx);
      auto dfdy                         = evaluateDerivativeImpl(f, argsForDy);
      return std::make_tuple(dfdx, dfdy);
    }
  }

  template <typename LocalFunctionEvaluationArgs_, typename LocalFunctionImpl>
  auto evaluateSecondOrderDerivativesImpl(const LocalFunctionInterface<LocalFunctionImpl>& f,
                                          const LocalFunctionEvaluationArgs_& localFunctionArgs) {
    if constexpr (localFunctionArgs.derivativeOrder == 3) {
      const auto argsForDx              = localFunctionArgs.extractSpatialOrFirstWrtArg();
      const auto [argsForDy, argsForDz] = extractWrtArgsTwoCoeffsToSingleCoeff(localFunctionArgs);
      const auto argsForDxy             = joinWRTArgs(argsForDx, argsForDy);
      const auto argsForDxz             = joinWRTArgs(argsForDx, argsForDz);
      const auto df_dxy                 = evaluateDerivativeImpl(f, argsForDxy);
      const auto df_dxz                 = evaluateDerivativeImpl(f, argsForDxz);
      return std::make_tuple(df_dxy, df_dxz);
    } else
      static_assert(localFunctionArgs.derivativeOrder == 3);
  }

}  // namespace Ikarus
