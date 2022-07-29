#pragma once
#include <ranges>
#include <set>
#include <vector>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>

#include <Eigen/Core>

#include <ikarus/utils/concepts.hh>

namespace Ikarus {

  namespace Impl {
    template <typename... Args>
    struct Derivatives {
      std::set<int> args;
    };
  }  // namespace Impl

  /* Helper function to pass integers. These indicate which derivatives should be precomputed */
  template <typename... Ints>
  requires std::conjunction_v<std::is_convertible<int, Ints>...>
  auto bindDerivatives(Ints&&... ints) { return Impl::Derivatives<Ints&&...>({std::forward<Ints>(ints)...}); }

  /* Convenient wrapper to store a dune local basis. It is possible to precompute derivatives */
  template <Concepts::DuneLocalBasis DuneLocalBasis>
  class LocalBasis {
    using RangeDuneType    = typename DuneLocalBasis::Traits::RangeType;
    using JacobianDuneType = typename DuneLocalBasis::Traits::JacobianType;

  public:
    constexpr explicit LocalBasis(const DuneLocalBasis& p_basis) : duneLocalBasis{&p_basis} {}
    LocalBasis() = default;

    static constexpr int gridDim = DuneLocalBasis::Traits::dimDomain;
    static_assert(gridDim<=3, "This local Basis only works for grids with dimensions<=3");
    using DomainType             = typename DuneLocalBasis::Traits::DomainType;

    using DomainFieldType = typename DuneLocalBasis::Traits::DomainFieldType;
    using RangeFieldType  = typename DuneLocalBasis::Traits::RangeFieldType;

    using JacobianType       = Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>;
    using AnsatzFunctionType = Eigen::VectorX<RangeFieldType>;

    /* Evaluates the ansatz functions into the given Eigen Vector N */
    template <typename Derived>
    void evaluateFunction(const DomainType& local, Eigen::PlainObjectBase<Derived>& N) const;

    /* Evaluates the ansatz functions derivatives into the given Eigen Matrix dN */
    template <typename Derived>
    void evaluateJacobian(const DomainType& local, Eigen::PlainObjectBase<Derived>& dN) const;

    /* Evaluates the ansatz functions second derivatives into the given Eigen Matrix ddN */
    template <typename Derived>
    void evaluateSecondDerivatives(const DomainType& local, Eigen::PlainObjectBase<Derived>& ddN) const;

    /* Evaluates the ansatz functions and derivatives into the given Eigen Vector/Matrix N,dN */
    template <typename Derived1, typename Derived2>
    void evaluateFunctionAndJacobian(const DomainType& local, Eigen::PlainObjectBase<Derived1>& N,
                                     Eigen::PlainObjectBase<Derived2>& dN) const;

    /* Returns the number of ansatz functions */
    unsigned int size() { return duneLocalBasis->size(); }

    /* Returns the number of integration points if the basis is bound */
    unsigned int integrationPointSize() const {
      if (not Nbound) throw std::logic_error("You have to bind the basis first");
      return Nbound.value().size();
    }

    /* Binds this basis to a given integration rule */
    template <typename IntegrationRule, typename... Ints>
    requires std::conjunction_v<std::is_convertible<int, Ints>...>
    void bind(IntegrationRule&& p_rule, Impl::Derivatives<Ints...>&& ints);

    /* Returns a reference to the ansatz functions evaluated at the given integration point index */
    const auto& evaluateFunction(long unsigned ipIndex) const {
      if (not Nbound) throw std::logic_error("You have to bind the basis first");
      return Nbound.value()[ipIndex];
    }

    /* Returns a reference to the ansatz functions derivatives evaluated at the given integration point index */
    const auto& evaluateJacobian(long unsigned i) const {
      if (not dNbound) throw std::logic_error("You have to bind the basis first");
      return dNbound.value()[i];
    }

    template <typename Derived>
    void evaluateSecondDerivatives(long unsigned i) const {
      if (not ddNbound) throw std::logic_error("You have to bind the basis first");
      return ddNbound.value()[i];
    }

    /* Returns true if the local basis is currently bound to an integration rule */
    bool isBound() const { return (dNbound and Nbound); }

    struct FunctionAndJacobian {
      long unsigned index{};
      const Dune::QuadraturePoint<DomainFieldType, gridDim>& ip{};
      const Eigen::VectorX<RangeFieldType>& N{};
      const Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>& dN{};
    };

    /* Returns a view over the integration point index, the point itself, and the ansatz function and ansatz function
     * derivatives at the very same point */
    auto viewOverFunctionAndJacobian() const {
      assert(Nbound.value().size() == dNbound.value().size()
             && "Number of intergrationpoint evaluations does not match.");
      if (Nbound and dNbound)
        return std::views::iota(0UL, Nbound.value().size()) | std::views::transform([&](auto&& i_) {
                 return FunctionAndJacobian(i_, rule.value()[i_], getFunction(i_), getJacobian(i_));
               });
      else {
        assert(false && "You need to call bind first");
        __builtin_unreachable();
      }
    }

    struct IntegrationPointsAndIndex {
      long unsigned index{};
      const Dune::QuadraturePoint<DomainFieldType, gridDim>& ip{};
    };

    /* Returns a view over the integration point index and the point itself */
    auto viewOverIntegrationPoints() const {  // FIXME dont construct this on the fly
      assert(Nbound && "You have to bind the basis first");
      assert(Nbound.value().size() == dNbound.value().size()
             && "Number of intergrationpoint evaluations does not match.");
      if (Nbound and dNbound) {
        auto res = std::views::iota(0UL, Nbound.value().size())
                   | std::views::transform([&](auto&& i_) { return IntegrationPointsAndIndex(i_, rule.value()[i_]); });
        return res;
      } else {
        assert(false && "You need to call bind first");
        __builtin_unreachable();
      }
    }

  private:
    mutable std::vector<JacobianDuneType> dNdune{};
    mutable std::vector<JacobianDuneType> ddNdune{};
    mutable std::vector<RangeDuneType> Ndune{};
    DuneLocalBasis const* duneLocalBasis;  // FIXME pass shared_ptr around
    std::optional<std::set<int>> boundDerivatives;
    std::optional<std::vector<Eigen::VectorX<RangeFieldType>>> Nbound{};
    std::optional<std::vector<Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>>> dNbound{};
    std::optional<std::vector<Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim== 1? 1: gridDim== 2 ? 3: 6>>> ddNbound{};
    std::optional<Dune::QuadratureRule<DomainFieldType, gridDim>> rule;
  };

}  // namespace Ikarus

#include "localBasis.inl"