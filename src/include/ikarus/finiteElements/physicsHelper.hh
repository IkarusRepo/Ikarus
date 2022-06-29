//
// Created by lex on 04/02/2022.
//

#pragma once
#include <dune/common/float_cmp.hh>

#include <Eigen/Core>
namespace Ikarus {
  template <typename ST, int size>
  requires(size > 0 and size <= 3) auto toVoigt(const Eigen::Matrix<ST, size, size>& E) {
    Eigen::Vector<ST, (size * (size + 1)) / 2> EVoigt;
    EVoigt.setZero();
    for (int i = 0; i < size; ++i)
      EVoigt(i) = E(i, i);

    if constexpr (size == 2) EVoigt(2) = E(0, 1) * 2;
    else if constexpr (size == 3) {
      EVoigt(size) = E(1, 2) * 2;
      EVoigt(size+1) = E(0, 2) * 2;
      EVoigt(size+2) = E(0, 1) * 2;
    }
    return EVoigt;
  }

  template <typename LocalView>
  struct TraitsFromLocalView {
    using GridEntity = typename LocalView::Element;
    /** \brief Dimension of the world space */
    static constexpr int worlddim = GridEntity::Geometry::coorddimension;

    /** \brief Dimension of the geometry */
    static constexpr int mydim = GridEntity::mydimension;

    /** \brief Dimension of the grid */
    static constexpr int dimension = GridEntity::dimension;

    /** \brief Type of the internal forces */
    using VectorType = Eigen::VectorXd;

    /** \brief Type of the stiffness matrix */
    using MatrixType = Eigen::MatrixXd;

    /** \brief Type of the stiffness matrix */
    using ScalarType = typename GridEntity::Geometry::ctype;
  };

  /// see https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
  struct YoungsModulusAndPoissonsRatio {
    double emodul;
    double nu;
  };

  struct YoungsModulusAndShearModulus {
    double emodul;
    double mu;
  };

  struct YoungsModulusAndBulkModulus {
    double emodul;
    double K;
  };

  struct YoungsModulusAndLamesFirstParameter {
    double emodul;
    double lambda;
  };

  struct BulkModulusAndLamesFirstParameter {
    double K;
    double lambda;
  };

  template <typename ValuePair>
  struct ConvertLameConstants {
    constexpr inline double toLamesFirstParameter() requires(
        !std::is_same_v<
            ValuePair,
            YoungsModulusAndLamesFirstParameter> and !std::is_same_v<ValuePair, BulkModulusAndLamesFirstParameter>) {
      if constexpr (std::is_same_v<ValuePair, YoungsModulusAndPoissonsRatio>) {
        const auto& E  = vp.emodul;
        const auto& nu = vp.nu;
        return Dune::FloatCmp::eq(nu, 0.5) ? std::numeric_limits<double>::infinity()
                                           : E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndShearModulus>) {
        const auto& E  = vp.emodul;
        const auto& mu = vp.mu;
        return mu * (E - 2.0 * mu) / (3.0 * mu - E);
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndBulkModulus>) {
        const auto& E = vp.emodul;
        const auto& K = vp.K;
        return 3.0 * K * (3.0 * K - E) / (9.0 * K - E);
      } else
        assert(false && "Your LameParameter request is not implemented");
    }

    constexpr inline double toBulkModulus() requires(
        !std::is_same_v<
            ValuePair, YoungsModulusAndBulkModulus> and !std::is_same_v<ValuePair, BulkModulusAndLamesFirstParameter>) {
      if constexpr (std::is_same_v<ValuePair, YoungsModulusAndPoissonsRatio>) {
        const auto& E  = vp.emodul;
        const auto& nu = vp.nu;
        return E / (3.0 * (1.0 - 2.0 * nu));
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndShearModulus>) {
        const auto& E  = vp.emodul;
        const auto& mu = vp.mu;
        return E * mu / (3.0 * (3.0 * mu - E));
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndLamesFirstParameter>) {
        const auto& E      = vp.emodul;
        const auto& lambda = vp.lambda;
        return (E + 3.0 * lambda + calcR(vp)) / 6.0;
      } else
        assert(false && "Your LameParameter request is not implemented");
    }

    constexpr inline double toShearModulus() requires(!std::is_same_v<ValuePair, YoungsModulusAndShearModulus>) {
      if constexpr (std::is_same_v<ValuePair, YoungsModulusAndPoissonsRatio>) {
        const auto& E  = vp.emodul;
        const auto& nu = vp.nu;
        return E / (2.0 * (1.0 + nu));
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndBulkModulus>) {
        const auto& E = vp.emodul;
        const auto& K = vp.K;
        return 3.0 * K * E / (9.0 * K - E);
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndLamesFirstParameter>) {
        const auto& E      = vp.emodul;
        const auto& lambda = vp.lambda;
        return (E - 3.0 * lambda + calcR(vp)) / 4.0;
      } else if constexpr (std::is_same_v<ValuePair, BulkModulusAndLamesFirstParameter>) {
        const auto& K      = vp.K;
        const auto& lambda = vp.lambda;
        return 3.0 * (K - lambda) / 2.0;
      } else
        assert(false && "Your LameParameter request is not implemented");
    }

    constexpr inline double toPWaveModulus() {
      if constexpr (std::is_same_v<ValuePair, YoungsModulusAndPoissonsRatio>) {
        const auto& E  = vp.emodul;
        const auto& nu = vp.nu;
        return E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndShearModulus>) {
        const auto& E  = vp.emodul;
        const auto& mu = vp.mu;
        return mu * (4.0 * mu - E) / (3.0 * mu - E);
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndBulkModulus>) {
        const auto& E = vp.emodul;
        const auto& K = vp.K;
        return 3.0 * K * (3.0 * K + E) / (9.0 * K - E);
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndLamesFirstParameter>) {
        const auto& E      = vp.emodul;
        const auto& lambda = vp.lambda;
        return (E - lambda + calcR(vp)) / 2.0;
      } else if constexpr (std::is_same_v<ValuePair, BulkModulusAndLamesFirstParameter>) {
        const auto& K      = vp.K;
        const auto& lambda = vp.lambda;
        return 3.0 * K - 2.0 * lambda;
      } else
        assert(false && "Your LameParameter request is not implemented");
    }

    constexpr inline double toPoissonsRatio() requires(!std::is_same_v<ValuePair, YoungsModulusAndPoissonsRatio>) {
      if constexpr (std::is_same_v<ValuePair, YoungsModulusAndShearModulus>) {
        const auto& E  = vp.emodul;
        const auto& mu = vp.mu;
        return E / (2.0 * mu) - 1.0;
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndBulkModulus>) {
        const auto& E = vp.emodul;
        const auto& K = vp.K;
        return (3.0 * K - E) / (6.0 * K);
      } else if constexpr (std::is_same_v<ValuePair, YoungsModulusAndLamesFirstParameter>) {
        const auto& E      = vp.emodul;
        const auto& lambda = vp.lambda;
        return 2.0 * lambda / (E + lambda + calcR(vp));
      } else if constexpr (std::is_same_v<ValuePair, BulkModulusAndLamesFirstParameter>) {
        const auto& K      = vp.K;
        const auto& lambda = vp.lambda;
        return lambda / (3 * K - lambda);
      } else
        assert(false && "Your LameParameter request is not implemented");
    }

    constexpr inline double toYoungsModulus() requires(
        !std::is_same_v<
            ValuePair,
            YoungsModulusAndPoissonsRatio> and !std::is_same_v<ValuePair, YoungsModulusAndShearModulus> and !std::is_same_v<ValuePair, YoungsModulusAndBulkModulus> and !std::is_same_v<ValuePair, YoungsModulusAndLamesFirstParameter>) {
      if constexpr (std::is_same_v<ValuePair, BulkModulusAndLamesFirstParameter>) {
        return 9.0 * vp.K * (vp.K - vp.lambda) / (3.0 * vp.K - vp.lambda);
      } else
        assert(false && "Your LameParameter request is not implemented");
    }

  private : friend ConvertLameConstants<YoungsModulusAndPoissonsRatio> convertLameConstants(
                const YoungsModulusAndPoissonsRatio& p_vp);
    friend ConvertLameConstants<YoungsModulusAndShearModulus> convertLameConstants(
        const YoungsModulusAndShearModulus& p_vp);

    friend ConvertLameConstants<YoungsModulusAndBulkModulus> convertLameConstants(
        const YoungsModulusAndBulkModulus& p_vp);

    friend ConvertLameConstants<BulkModulusAndLamesFirstParameter> convertLameConstants(
        const BulkModulusAndLamesFirstParameter& p_vp);
    ConvertLameConstants(ValuePair&& p_vp) : vp(p_vp) {}
    ConvertLameConstants(const ValuePair& p_vp) : vp(p_vp) {}

    double calcR(const YoungsModulusAndLamesFirstParameter& vp_) {
      const auto& E      = vp_.emodul;
      const auto& lambda = vp_.lambda;
      return std::sqrt(E * E + 9 * lambda * lambda + 2 * E * lambda);
    }
    ValuePair vp;
  };
  ConvertLameConstants<YoungsModulusAndPoissonsRatio> convertLameConstants(const YoungsModulusAndPoissonsRatio& p_vp);
  ConvertLameConstants<YoungsModulusAndShearModulus> convertLameConstants(const YoungsModulusAndShearModulus& p_vp);
  ConvertLameConstants<YoungsModulusAndBulkModulus> convertLameConstants(const YoungsModulusAndBulkModulus& p_vp);
  ConvertLameConstants<BulkModulusAndLamesFirstParameter> convertLameConstants(
      const BulkModulusAndLamesFirstParameter& p_vp);

}  // namespace Ikarus
