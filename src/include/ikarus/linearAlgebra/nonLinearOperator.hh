/*
 * This file is part of the Ikarus distribution (https://github.com/IkarusRepo/Ikarus).
 * Copyright (c) 2022. The Ikarus developers.
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#pragma once
#include <dune/common/hybridutilities.hh>

#include <ikarus/utils/traits.hh>

template <typename Fun, typename... Args>
using ReturnType = std::invoke_result_t<Fun, Args...>;

namespace Impl {
  template <class F, class Tuple, std::size_t... I>
  constexpr decltype(auto) applyAndRemoveRefererenceWrapper(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return std::invoke(
        std::forward<F>(f),
        std::get<I>(std::forward<Tuple>(t)).get()...);  //.get gets the impl type of std::referenceWrapper
  }

  template <class F, class Tuple>
  constexpr decltype(auto) applyAndRemoveReferenceWrapper(F&& f, Tuple&& t) {
    return applyAndRemoveRefererenceWrapper(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
  }

  template <typename T>
  auto forwardasReferenceWrapperIfIsReference(T&& t) {
    if constexpr (std::is_lvalue_reference_v<decltype(t)>)
      return std::ref(t);
    else
      return t;
  }

  template <class Pars, class Tuple, std::size_t... I>
  constexpr decltype(auto) makeTupleOfValuesAndReferences(Tuple&& t, Pars&& p, std::index_sequence<I...>) {
    return std::make_tuple(
        forwardasReferenceWrapperIfIsReference(applyAndRemoveReferenceWrapper(std::get<I>(t), p.args))...);
  }

  template <typename... Args>
  struct LinearAlgebraFunctions {
    std::tuple<std::reference_wrapper<std::remove_reference_t<Args>>...> args;
  };

  template <typename... Args>
  struct Parameter {
    std::tuple<std::reference_wrapper<std::remove_reference_t<Args>>...> args;
  };

}  // namespace Impl

template <typename... Args>
auto parameter(Args&&... args) {
  return Impl::Parameter<Args&&...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

template <typename... Args>
auto linearAlgebraFunctions(Args&&... args) {
  return Impl::LinearAlgebraFunctions<Args&&...>{std::forward_as_tuple(std::forward<Args>(args)...)};
}

namespace Ikarus {

  template <typename... DerivativeArgs, typename... ParameterArgs>
  auto initResults(const Impl::LinearAlgebraFunctions<DerivativeArgs...>& derivativesFunctions,
                   const Impl::Parameter<ParameterArgs...>& parameterI) {
    return Impl::makeTupleOfValuesAndReferences(
        derivativesFunctions.args, parameterI,
        std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<decltype(derivativesFunctions.args)>>>{});
  }

  template <typename TypeListOne, typename TypeListTwo>
  class NonLinearOperator {
  public:
    NonLinearOperator([[maybe_unused]] const TypeListOne& derivativesFunctions,
                      [[maybe_unused]] const TypeListTwo& args) {
      static_assert(!sizeof(TypeListOne),
                    "This type should not be instantiated. check that your arguments satisfies the template below");
    }
  };

  /* NonLinearOperator is a class taking linear algebra function and their arguments.
   * The fcuntion are assumed to be derivatvies of each other. */
  template <typename... DerivativeArgs, typename... ParameterArgs>
  class NonLinearOperator<Impl::LinearAlgebraFunctions<DerivativeArgs...>, Impl::Parameter<ParameterArgs...>> {
    using FunctionReturnValues = std::tuple<ReturnType<DerivativeArgs, ParameterArgs&...>...>;
    using ParameterValues      = std::tuple<ParameterArgs...>;

  public:
    template <int n>
    using FunctionReturnType = std::tuple_element_t<n, FunctionReturnValues>;

    template <int n>
    using ParameterValue = std::remove_cvref_t<std::tuple_element_t<n, ParameterValues>>;

    using ValueType      = std::remove_cvref_t<std::tuple_element_t<0, FunctionReturnValues>>;
    using DerivativeType = std::remove_cvref_t<std::tuple_element_t<1, FunctionReturnValues>>;

    explicit NonLinearOperator(const Impl::LinearAlgebraFunctions<DerivativeArgs...>& derivativesFunctions,
                               const Impl::Parameter<ParameterArgs...>& parameterI)
        : derivatives_{derivativesFunctions.args},
          args_{parameterI.args},
          derivativesEvaluated_(initResults(derivativesFunctions, parameterI)) {
      updateAll();
    }

    /* Evaluates all functions. Usually called if the parameters changes */
    void updateAll() {
      Dune::Hybrid::forEach(
          Dune::Hybrid::integralRange(Dune::index_constant<sizeof...(DerivativeArgs)>()), [&](const auto i) {
            std::get<i>(derivativesEvaluated_) = Impl::applyAndRemoveReferenceWrapper(std::get<i>(derivatives_), args_);
          });
    }

    /* Evaluates the n-th function */
    template <int n>
    void update() {
      std::get<n>(derivativesEvaluated_) = Impl::applyAndRemoveReferenceWrapper(std::get<n>(derivatives_), args_);
    }

    /* Returns the value of the zeros function, e.g. the energy value */
    auto& value() requires(sizeof...(DerivativeArgs) > 0) { return nthDerivative<0>(); }

    /* Returns the derivative value, e.g. the gradient of an energy */
    auto& derivative() & requires(sizeof...(DerivativeArgs) > 1) { return nthDerivative<1>(); }

    /* Returns the second derivative value, e.g. the Hessian of an energy */
    auto& secondDerivative() requires(sizeof...(DerivativeArgs) > 2) { return nthDerivative<2>(); }

    /* Returns the n-th derivative value */
    template <int n>
    auto& nthDerivative() requires(sizeof...(DerivativeArgs) > n) {
      if constexpr (requires { std::get<n>(derivativesEvaluated_).get(); })
        return std::get<n>(derivativesEvaluated_).get();
      else
        return std::get<n>(derivativesEvaluated_);
    }

    /* Returns the last parameter value */
    auto& lastParameter() { return nthParameter<sizeof...(ParameterArgs) - 1>(); }
    /* Returns the first parameter value */
    auto& firstParameter() requires(sizeof...(ParameterArgs) > 0) { return nthParameter<0>(); }
    /* Returns the second parameter value */
    auto& secondParameter() requires(sizeof...(ParameterArgs) > 1) { return nthParameter<1>(); }
    /* Returns the n-th parameter value */
    template <int n>
    auto& nthParameter() requires(sizeof...(ParameterArgs) >= n) {
      return std::get<n>(args_).get();
    }

    /* Returns a new NonLinearOperator from the given indices */
    template <int... Derivatives>
    auto subOperator() {
      return Ikarus::NonLinearOperator(linearAlgebraFunctions(std::get<Derivatives>(derivatives_)...),
                                       Impl::applyAndRemoveReferenceWrapper(parameter<ParameterArgs...>, args_));
    }

  private:
    using FunctionReturnValuesWrapper = std::tuple<
        std::conditional_t<std::is_reference_v<ReturnType<DerivativeArgs, ParameterArgs&...>>,
                           std::reference_wrapper<std::remove_cvref_t<ReturnType<DerivativeArgs, ParameterArgs&...>>>,
                           std::remove_cvref_t<ReturnType<DerivativeArgs, ParameterArgs&...>>>...>;
    std::tuple<std::conditional_t<std::is_reference_v<DerivativeArgs>,
                                  std::reference_wrapper<std::remove_cvref_t<DerivativeArgs>>,
                                  std::remove_cvref_t<DerivativeArgs>>...>
        derivatives_;
    std::tuple<std::conditional_t<std::is_reference_v<ParameterArgs>,
                                  std::reference_wrapper<std::remove_cvref_t<ParameterArgs>>,
                                  std::remove_cvref_t<ParameterArgs>>...>
        args_;
    FunctionReturnValuesWrapper derivativesEvaluated_{};
  };
}  // namespace Ikarus