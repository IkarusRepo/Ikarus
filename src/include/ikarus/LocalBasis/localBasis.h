#pragma once
#include <ranges>
#include <set>
#include <vector>

#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/geometry/quadraturerules.hh>

#include <Eigen/Core>

#include <ikarus/utils/concepts.h>

namespace Ikarus {

  template <typename... Args>
  struct Derivatives {
    std::set<int> args;
  };

  template <typename... Ints>
  requires std::conjunction_v<std::is_convertible<int, Ints>...>
  auto bindDerivatives(Ints&&... ints) { return Derivatives<Ints&&...>({std::forward<Ints>(ints)...}); }

  template <Concepts::DuneLocalBasis DuneLocalBasis>
  class LocalBasis {
    using RangeDuneType    = typename DuneLocalBasis::Traits::RangeType;
    using JacobianDuneType = typename DuneLocalBasis::Traits::JacobianType;

  public:
    explicit LocalBasis(const DuneLocalBasis& p_basis) : duneLocalBasis{&p_basis} {}
    LocalBasis() = default;

    static constexpr int gridDim = DuneLocalBasis::Traits::dimDomain;
    using DomainType             = typename DuneLocalBasis::Traits::DomainType;

    using DomainFieldType = typename DuneLocalBasis::Traits::DomainFieldType;
    using RangeFieldType  = typename DuneLocalBasis::Traits::RangeFieldType;

    using JacobianType       = Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>;
    using AnsatzFunctionType = Eigen::VectorX<RangeFieldType>;

    template <typename Derived>
    void evaluateFunction(const DomainType& local, Eigen::PlainObjectBase<Derived>& N) const {
      duneLocalBasis->evaluateFunction(local, Ndune);
      N.resize(Ndune.size(), 1);
      N.setZero();
      for (size_t i = 0; i < Ndune.size(); ++i)
        N[i] = Ndune[i][0];
    }

    template <typename Derived>
    void evaluateJacobian(const DomainType& local, Eigen::PlainObjectBase<Derived>& dN) const {
      duneLocalBasis->evaluateJacobian(local, dNdune);
      dN.setZero();
      dN.resize(dNdune.size(), gridDim);

      for (auto i = 0U; i < dNdune.size(); ++i)
        for (int j = 0; j < gridDim; ++j)
          dN(i, j) = dNdune[i][0][j];
    }

    template <typename Derived1, typename Derived2>
    void evaluateFunctionAndJacobian(const DomainType& local, Eigen::PlainObjectBase<Derived1>& N,
                                     Eigen::PlainObjectBase<Derived2>& dN) const {
      evaluateFunction(local, N);
      evaluateJacobian(local, dN);
    }

    unsigned int size() { return duneLocalBasis->size(); }

    template <typename IntegrationRule, typename... Ints>
    requires std::conjunction_v<std::is_convertible<int, Ints>...>
    void bind(IntegrationRule&& p_rule, Derivatives<Ints...>&& ints) {
      rule             = p_rule;
      boundDerivatives = ints.args;
      Nbound           = std::make_optional(std::vector<Eigen::VectorX<RangeFieldType>>{});
      dNbound          = std::make_optional(std::vector<Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>>{});
      dNbound.value().resize(rule.value().size());
      Nbound.value().resize(rule.value().size());

      for (int i = 0; auto& gp : rule.value()) {
        if (boundDerivatives.value().contains(0)) evaluateFunction(gp.position(), Nbound.value()[i]);
        if (boundDerivatives.value().contains(1)) evaluateJacobian(gp.position(), dNbound.value()[i]);
        ++i;
      }
    }

    const auto& evaluateFunction(long unsigned i) const {
      if (not Nbound) throw std::logic_error("You have to bind the basis first");
      return Nbound.value()[i];
    }

    const auto& evaluateJacobian(long unsigned i) const {
      if (not dNbound) throw std::logic_error("You have to bind the basis first");
      return dNbound.value()[i];
    }

    bool isBound() const { return (dNbound and Nbound); }

    struct FunctionAndJacobian {
      long unsigned index{};
      const Dune::QuadraturePoint<DomainFieldType, gridDim>& ip{};
      const Eigen::VectorX<RangeFieldType>& N{};
      const Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>& dN{};
    };
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
    auto viewOverIntegrationPoints() const {  // FIXME dont construct this on the fly
      assert(Nbound.value().size() == dNbound.value().size()
             && "Number of intergrationpoint evaluations does not match.");
      if (Nbound and dNbound)
        return std::views::iota(0UL, Nbound.value().size())
               | std::views::transform([&](auto&& i_) { return IntegrationPointsAndIndex(i_, rule.value()[i_]); });
      else {
        assert(false && "You need to call bind first");
        __builtin_unreachable();
      }
    }

  private:
    mutable std::vector<JacobianDuneType> dNdune{};
    mutable std::vector<RangeDuneType> Ndune{};
    DuneLocalBasis const* duneLocalBasis;
    std::optional<std::set<int>> boundDerivatives;
    std::optional<std::vector<Eigen::VectorX<RangeFieldType>>> Nbound{};
    std::optional<std::vector<Eigen::Matrix<RangeFieldType, Eigen::Dynamic, gridDim>>> dNbound{};
    std::optional<Dune::QuadratureRule<DomainFieldType, gridDim>> rule;
  };

}  // namespace Ikarus