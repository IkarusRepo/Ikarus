//
// Created by alex on 3/17/22.
//

#pragma once
#include "meta.hh"

#include <concepts>

#include <ikarus/localBasis/localBasis.hh>
#include <ikarus/utils/traits.hh>

namespace Ikarus {

  template <typename Derived>
  struct LocalFunctionTraits;

  template <typename LocalFunctionImpl>
  class LocalFunctionInterface;

  template <typename DomainTypeOrIntegrationPointIndex, typename DomainType>
  concept IsIntegrationPointIndexOrIntegrationPointPosition
      = std::is_same_v<DomainTypeOrIntegrationPointIndex,
                       DomainType> or std::numeric_limits<DomainTypeOrIntegrationPointIndex>::is_integer;

  template <typename TypeListOne, typename TypeListTwo, typename TypeListThree,
            typename DomainTypeOrIntegrationPointIndex>
  struct LocalFunctionEvaluationArgs {
  public:
    LocalFunctionEvaluationArgs(const DomainTypeOrIntegrationPointIndex&, [[maybe_unused]] const TypeListOne& l1,
                                [[maybe_unused]] const TypeListTwo& l2, [[maybe_unused]] const TypeListThree& l3) {
      static_assert(!sizeof(TypeListOne),
                    "This type should not be instantiated. Check that your arguments satisfies the template below");
    }
  };

  /** This class contains all the arguments a local function evaluation consumes */
  template <typename... WrtArgs, typename... TransformArgs, typename... AlongArgs,
            typename DomainTypeOrIntegrationPointIndex>
  struct LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                     DomainTypeOrIntegrationPointIndex> {
    template <typename, typename, typename, typename>
    friend class LocalFunctionEvaluationArgs;

    LocalFunctionEvaluationArgs(const DomainTypeOrIntegrationPointIndex& localOrIpId, const Wrt<WrtArgs...>& args,
                                const Along<AlongArgs...>& along, const TransformWith<TransformArgs...>& transArgs)
        : integrationPointOrIndex{localOrIpId}, wrtArgs{args}, alongArgs{along}, transformWithArgs{transArgs} {
      const auto coeffIndicesOfArgs = Ikarus::DerivativeDirections::extractCoeffIndices(args);
      Dune::Hybrid::forEach(Dune::Hybrid::integralRange(Dune::index_constant<coeffIndicesOfArgs.size()>{}),
                            [&](auto&& i) { coeffsIndices[i]._data = coeffIndicesOfArgs[i]._data; });

      spatialPartialIndices = Ikarus::DerivativeDirections::extractSpatialPartialIndices(args);
    }

    // Constructor that does not calculate extractCoeffIndices and extractSpatialPartialIndices
    LocalFunctionEvaluationArgs(const DomainTypeOrIntegrationPointIndex& localOrIpId, const Wrt<WrtArgs...>& args,
                                const Along<AlongArgs...>& along, const TransformWith<TransformArgs...>& transArgs,
                                bool)
        : integrationPointOrIndex{localOrIpId}, wrtArgs{args}, alongArgs{along}, transformWithArgs{transArgs} {}

  public:
    auto extractSpatialOrFirstWrtArg() const {
      if constexpr (hasOneSpatial) {
        if constexpr (hasOneSpatialSingle)
          return extractWrtArgsWithGivenType<DerivativeDirections::SpatialPartial>();
        else if constexpr (hasOneSpatialAll)
          return extractWrtArgsWithGivenType<DerivativeDirections::SpatialAll>();
      } else
        return extractWrtArgs<0>();
    }

    auto extractSecondWrtArgOrFirstNonSpatial() const {
      if constexpr (!hasOneSpatial)
        return extractWrtArgs<0>();
      else {
        static_assert(!hasNoCoeff, " There needs to be a coeff wrt argument!");

        if constexpr (hasSingleCoeff) {
          return extractWrtArgsWithGivenType<DerivativeDirections::SingleCoeff>();
        } else if constexpr (hasTwoCoeff) {
          return extractWrtArgsWithGivenType<DerivativeDirections::TwoCoeff>();
        }
      }
    }

    template <std::size_t... I>
    auto extractWrtArgs() const {
      auto wrt_lambda = [](auto... args) { return wrt(args...); };
      auto wrtArg
          = std::apply(wrt_lambda, Std::makeTupleFromTupleIndices<const decltype(wrtArgs.args)&, I...>(wrtArgs.args));
      return LocalFunctionEvaluationArgs<decltype(wrtArg), Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                         DomainTypeOrIntegrationPointIndex>(integrationPointOrIndex, wrtArg, alongArgs,
                                                                            transformWithArgs);
    }

    template <typename DerivativeDirection>
    auto extractWrtArgsWithGivenType() const {
      if constexpr (std::is_same_v<DerivativeDirection, DerivativeDirections::SpatialPartial>) {
        auto wrtArg = wrt(DerivativeDirections::spatial(spatialPartialIndices));
        return LocalFunctionEvaluationArgs<decltype(wrtArg), Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                           DomainTypeOrIntegrationPointIndex>(integrationPointOrIndex, wrtArg,
                                                                              alongArgs, transformWithArgs);
      } else if constexpr (std::is_same_v<DerivativeDirection, DerivativeDirections::SpatialAll>) {
        auto wrtArg = wrt(DerivativeDirections::spatialAll);
        return LocalFunctionEvaluationArgs<decltype(wrtArg), Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                           DomainTypeOrIntegrationPointIndex>(integrationPointOrIndex, wrtArg,
                                                                              alongArgs, transformWithArgs);
      }
    }

    template <template <auto...> class DerivativeDirection>
    auto extractWrtArgsWithGivenType() const {
      using namespace Dune::Indices;
      if constexpr (Std::isTemplateSame_v<DerivativeDirections::TwoCoeff, DerivativeDirection>) {
        auto wrtArg = wrt(DerivativeDirections::coeff(coeffsIndices[_0][_0], coeffsIndices[_0][1],
                                                      coeffsIndices[_1][_0], coeffsIndices[_1][1]));
        return LocalFunctionEvaluationArgs<decltype(wrtArg), Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                           DomainTypeOrIntegrationPointIndex>(integrationPointOrIndex, wrtArg,
                                                                              alongArgs, transformWithArgs);
      } else if constexpr (Std::isTemplateSame_v<DerivativeDirections::SingleCoeff, DerivativeDirection>) {
        auto wrtArg = wrt(DerivativeDirections::coeff(coeffsIndices[_0][_0], coeffsIndices[_0][1]));
        return LocalFunctionEvaluationArgs<decltype(wrtArg), Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                           DomainTypeOrIntegrationPointIndex>(integrationPointOrIndex, wrtArg,
                                                                              alongArgs, transformWithArgs);
      }
    }

    static constexpr DerivativeDirections::ConstExprCounter derivativeCounter
        = DerivativeDirections::countDerivativesType<Wrt<WrtArgs...>>();
    static constexpr int derivativeOrder
        = DerivativeDirections::countDerivativesType<Wrt<WrtArgs...>>().orderOfDerivative();

    static constexpr bool hasTwoCoeff         = DerivativeDirections::HasTwoCoeff<Wrt<WrtArgs...>>;
    static constexpr bool hasSingleCoeff      = DerivativeDirections::HasSingleCoeff<Wrt<WrtArgs...>>;
    static constexpr bool hasNoCoeff          = DerivativeDirections::HasNoCoeff<Wrt<WrtArgs...>>;
    static constexpr bool hasNoSpatial        = DerivativeDirections::HasNoSpatial<Wrt<WrtArgs...>>;
    static constexpr bool hasOneSpatialAll    = DerivativeDirections::HasOneSpatialAll<Wrt<WrtArgs...>>;
    static constexpr bool hasOneSpatialSingle = DerivativeDirections::HasOneSpatialSingle<Wrt<WrtArgs...>>;
    static constexpr bool hasOneSpatial       = hasOneSpatialAll or hasOneSpatialSingle;

    DomainTypeOrIntegrationPointIndex integrationPointOrIndex{};

    const Wrt<WrtArgs&&...> wrtArgs;
    const Along<AlongArgs&&...> alongArgs;
    const TransformWith<TransformArgs&&...> transformWithArgs;
    decltype(DerivativeDirections::extractCoeffIndices<Wrt<WrtArgs...>>(std::declval<Wrt<WrtArgs...>>())) coeffsIndices;
    decltype(DerivativeDirections::extractSpatialPartialIndices<Wrt<WrtArgs...>>(
        std::declval<Wrt<WrtArgs...>>())) spatialPartialIndices;
  };

  template <typename... WrtArgs, typename... OtherWrtArgs, typename... TransformArgs, typename... AlongArgs,
            typename DomainTypeOrIntegrationPointIndex>
  auto joinWRTArgs(
      const LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& a,
      const LocalFunctionEvaluationArgs<Wrt<OtherWrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& b) {
    auto wrt_lambda = [](auto... args) { return wrt(args...); };
    auto wrtArg     = std::apply(wrt_lambda, std::tuple_cat(a.wrtArgs.args, b.wrtArgs.args));
    return LocalFunctionEvaluationArgs<decltype(wrtArg), Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                       DomainTypeOrIntegrationPointIndex>(a.integrationPointOrIndex, wrtArg,
                                                                          a.alongArgs, a.transformWithArgs);
  }

  template <typename... WrtArgs, typename... OtherWrtArgs, typename... TransformArgs, typename... AlongArgs,
            typename DomainTypeOrIntegrationPointIndex>
  auto extractWrtArgsTwoCoeffsToSingleCoeff(
      const LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& a) {
    using namespace Dune::Indices;
    const auto coeffArg = Std::getSpecialization<DerivativeDirections::TwoCoeff>(a.wrtArgs.args);
    auto wrtArg0        = wrt(DerivativeDirections::coeff(a.coeffsIndices[_0][_0], a.coeffsIndices[_0][1]));
    auto wrtArg1        = wrt(DerivativeDirections::coeff(a.coeffsIndices[_1][_0], a.coeffsIndices[_1][1]));

    return std::make_pair(
        LocalFunctionEvaluationArgs(a.integrationPointOrIndex, wrtArg0, a.alongArgs, a.transformWithArgs),
        LocalFunctionEvaluationArgs(a.integrationPointOrIndex, wrtArg1, a.alongArgs, a.transformWithArgs));
  }

  // This function returns the first two args and returns the spatial derivative argument always as first
  template <typename... WrtArgs, typename... OtherWrtArgs, typename... TransformArgs, typename... AlongArgs,
            typename DomainTypeOrIntegrationPointIndex>
  auto extractFirstTwoArgs(
      const LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& a) {
    if constexpr (std::tuple_size_v<decltype(a.wrtArgs.args)> == 2) {
      if constexpr (DerivativeDirections::isSpatial<std::tuple_element_t<0, decltype(a.wrtArgs.args)>>)
        return std::make_pair(a.template extractWrtArgs<0>(), a.template extractWrtArgs<1>());
      else if constexpr (DerivativeDirections::isSpatial<std::tuple_element_t<1, decltype(a.wrtArgs.args)>>)
        return std::make_pair(a.template extractWrtArgs<1>(), a.template extractWrtArgs<0>());
    } else
      return extractWrtArgsTwoCoeffsToSingleCoeff(a);
  }

  /* This functions takes localfunction arguments and replaces the "along" argument with the given one */
  template <typename... WrtArgs, typename... TransformArgs, typename... AlongArgs, typename... AlongArgsOther,
            typename DomainTypeOrIntegrationPointIndex>
  auto replaceAlong(
      const LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& args,
      const Along<AlongArgsOther...>& alongArgs) {
    auto newArgs = LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgsOther...>,
                                               TransformWith<TransformArgs...>, DomainTypeOrIntegrationPointIndex>(
        args.integrationPointOrIndex, args.wrtArgs, alongArgs, args.transformWithArgs, false);

    using namespace Dune::Indices;
    std::get<1>(newArgs.coeffsIndices[_0]._data) = std::get<1>(args.coeffsIndices[_0]._data);
    std::get<1>(newArgs.coeffsIndices[_1]._data) = std::get<1>(args.coeffsIndices[_1]._data);
    newArgs.spatialPartialIndices                = args.spatialPartialIndices;

    return newArgs;
  }

  /* This functions takes localfunction arguments and replaces the "along" argument with the given one */
  template <typename... WrtArgs, typename... TransformArgs, typename... AlongArgs, typename... WRTArgsOther,
            typename DomainTypeOrIntegrationPointIndex>
  auto addWrt(
      const LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& args,
      const Wrt<WRTArgsOther...>& wrtArgs) {

    auto newWrtArgs =std::apply(Ikarus::wrt<std::remove_cvref_t<WrtArgs>...,std::remove_cvref_t<WRTArgsOther>...>,std::tuple_cat(args.wrtArgs.args,wrtArgs.args));


    auto newArgs = LocalFunctionEvaluationArgs<decltype(newWrtArgs), Along<AlongArgs...>,
                                               TransformWith<TransformArgs...>, DomainTypeOrIntegrationPointIndex>(
        args.integrationPointOrIndex,newWrtArgs, args.alongArgs, args.transformWithArgs, false);

    using namespace Dune::Indices;
    if constexpr(newArgs.hasSingleCoeff)
      std::get<1>(newArgs.coeffsIndices[_0]._data) = std::get<1>(args.coeffsIndices[_0]._data);
    if constexpr(newArgs.hasTwoCoeff)
      std::get<1>(newArgs.coeffsIndices[_1]._data) = std::get<1>(args.coeffsIndices[_1]._data);
    newArgs.spatialPartialIndices                = args.spatialPartialIndices;

    return newArgs;
  }

  template <typename... WrtArgs, typename... TransformArgs, typename... AlongArgs, typename... WRTArgsOther,
            typename DomainTypeOrIntegrationPointIndex>
  auto replaceWrt(
      const LocalFunctionEvaluationArgs<Wrt<WrtArgs...>, Along<AlongArgs...>, TransformWith<TransformArgs...>,
                                        DomainTypeOrIntegrationPointIndex>& args,
      const Wrt<WRTArgsOther...>& wrtArgs) {

    auto newArgs = LocalFunctionEvaluationArgs<Wrt<WRTArgsOther...>, Along<AlongArgs...>,
                                               TransformWith<TransformArgs...>, DomainTypeOrIntegrationPointIndex>(
        args.integrationPointOrIndex,wrtArgs, args.alongArgs, args.transformWithArgs, false);

    return newArgs;
  }

}  // namespace Ikarus
