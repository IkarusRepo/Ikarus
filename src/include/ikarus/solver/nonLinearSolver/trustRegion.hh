//
// Created by lex on 30/12/2021.
//

/*
 * This code is heavily inspired by the trust-region implementation of
 * https://github.com/NicolasBoumal/manopt/blob/master/manopt/solvers/trustregions/trustregions.m
 */

#pragma once
#include <iosfwd>

#include <dune/common/float_cmp.hh>

#include <spdlog/spdlog.h>

#include <Eigen/Sparse>

#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/linearAlgebra/truncatedConjugateGradient.hh>
#include <ikarus/utils/linearAlgebraHelper.hh>
#include <ikarus/utils/observer/observer.hh>
#include <ikarus/utils/observer/observerMessages.hh>
#include <ikarus/utils/traits.hh>

namespace Ikarus {

  struct TRSolverInformation {
    bool sucess{false};
    double gradienNorm{0.0};
    int iterations{0};
  };

  enum class PreConditioner { IncompleteCholesky, IdentityPreconditioner, DiagonalPreconditioner };

  struct TrustRegionSettings {
    int verbosity    = 5;
    double maxtime   = std::numeric_limits<double>::infinity();
    int miniter      = 3;
    int maxiter      = 1000;
    int debug        = 0;
    double grad_tol  = 1e-6;
    double corr_tol  = 1e-6;
    double rho_prime = 0.01;
    bool useRand     = false;
    double rho_reg   = 1e6;
    double Delta_bar = std::numeric_limits<double>::infinity();
    double Delta0    = 10;
  };

  enum class StopReason {
    gradientNormTolReached,
    correctionNormTolReached,
    maximumTimeReached,
    maximumIterationsReached,
    dontStop
  };
  struct AlgoInfo {
    int consecutive_TRplus   = 0;
    int consecutive_TRminus  = 0;
    int consecutive_Rejected = 0;
    std::string stopReasonString;
    std::string trstr;
    std::string accstr;
    std::string randomString;
    std::string cauchystr = "                  ";
    bool acceptProposal;
    bool used_cauchy = false;

    StopReason stop{StopReason::dontStop};
    std::string reasonString;
  };

  struct Stats {
    double gradNorm{1};
    double etaNorm{1};
    double time{0};
    double energy{0};
    double energyProposal{0};
    double rho{0};
    int outerIter{0};
    int innerIterSum{0};
  };

  template <typename NonLinearOperatorImpl, PreConditioner preConditioner = PreConditioner::IncompleteCholesky,
            typename UpdateType
            = std::conditional_t<std::is_floating_point_v<typename NonLinearOperatorImpl::template ParameterValue<0>>,
                                 typename NonLinearOperatorImpl::template ParameterValue<0>, Eigen::VectorXd>>
  class TrustRegion : public IObservable<NonLinearSolverMessages> {
  public:
    using ResultType         = typename NonLinearOperatorImpl::template ParameterValue<0>;
    using UpdateFunctionType = std::function<void(ResultType&, const UpdateType&)>;
    using ScalarType         = std::remove_cvref_t<typename NonLinearOperatorImpl::template FunctionReturnType<0>>;
    using MatrixType         = std::remove_cvref_t<typename NonLinearOperatorImpl::template FunctionReturnType<2>>;

    explicit TrustRegion(
        const NonLinearOperatorImpl& p_nonLinearOperator,
        std::function<void(ResultType&, const UpdateType&)> p_updateFunction =
            [](ResultType& a, const UpdateType& b) {
              using Ikarus::operator+=;
              a += b;
            })
        : nonLinearOperator_{p_nonLinearOperator},
          updateFunction{p_updateFunction},
          xOld{nonLinearOperator().firstParameter()} {
      eta.setZero(gradient().size());
      Heta.setZero(gradient().size());
    }

    using NonLinearOperator = NonLinearOperatorImpl;

    void setup(const TrustRegionSettings& p_settings) {
      options = p_settings;
      assert(options.rho_prime < 0.25 && "options.rho_prime must be strictly smaller than 1/4.");
      assert(options.Delta_bar > 0 && "options.Delta_bar must be positive.");
      assert(options.Delta0 > 0 && options.Delta0 < options.Delta_bar
             && "options.Delta0 must be positive and smaller than Delta_bar.");
    }

    struct NoPredictor {};
    template <typename SolutionType = NoPredictor>
    requires std::is_same_v<SolutionType, NoPredictor> || std::is_convertible_v<
        SolutionType, std::remove_cvref_t<typename NonLinearOperatorImpl::ValueType>>
        TRSolverInformation solve(const SolutionType& dx_predictor = NoPredictor{}) {
      this->notify(NonLinearSolverMessages::INIT);
      stats = Stats{};
      info  = AlgoInfo{};

      this->notify(NonLinearSolverMessages::INIT);
      TRSolverInformation solverInformation;
      nonLinearOperator().updateAll();
      stats.energy   = energy();
      auto& x        = nonLinearOperator().firstParameter();
      xOld           = x;
      stats.gradNorm = norm(gradient());
      if constexpr (not std::is_same_v<SolutionType, NoPredictor>) updateFunction(x, dx_predictor);
      truncatedConjugateGradient.analyzePattern(hessian());

      innerInfo.Delta = options.Delta0;
      spdlog::info(
          "        | iter | inner_i |   rho |   energy | energy_p | energy_inc |  norm(g) |    Delta | norm(corr) | "
          "InnerBreakReason");
      spdlog::info("{:-^143}", "-");
      while (not stoppingCriterion()) {
        this->notify(NonLinearSolverMessages::ITERATION_STARTED);
        if (options.useRand) {
          if (stats.outerIter == 0) {
            eta.setRandom();
            while (eta.dot(hessian() * eta) > innerInfo.Delta * innerInfo.Delta)
              eta *= eps;  // eps is sqrt(sqrt(maschine-precision))
          } else
            eta.setZero();
        } else
          eta.setZero();

        solveInnerProblem();
        stats.innerIterSum += innerInfo.numInnerIter;

        info.stopReasonString = tcg_stop_reason[static_cast<int>(innerInfo.stop_tCG)];
        Heta                  = hessian() * eta;
        if (options.useRand and stats.outerIter == 0) {
          info.used_cauchy  = false;
          info.randomString = " Used Random correction predictor";
          info.cauchystr    = "                  ";
          double tau_c;
          // Check the curvature
          const Eigen::VectorXd Hg = hessian() * gradient();
          const auto g_Hg          = (gradient().dot(Hg));
          if (g_Hg <= 0)
            tau_c = 1;
          else
            tau_c = std::min(Dune::power(stats.gradNorm, 3) / (innerInfo.Delta * g_Hg), 1.0);

          // generate the Cauchy point.
          const Eigen::VectorXd eta_c  = -tau_c * innerInfo.Delta / stats.gradNorm * gradient();
          const Eigen::VectorXd Heta_c = -tau_c * innerInfo.Delta / stats.gradNorm * Hg;

          const double mdle  = stats.energy + gradient().dot(eta) + .5 * Heta.dot(eta);
          const double mdlec = stats.energy + gradient().dot(eta_c) + .5 * Heta_c.dot(eta_c);
          if (mdlec < mdle && stats.outerIter == 0) {
            eta              = eta_c;
            Heta             = Heta_c;
            info.used_cauchy = true;

            info.cauchystr = " USED CAUCHY POINT!";
          }

        } else
          info.cauchystr = "                  ";

        stats.etaNorm = eta.norm();

        updateFunction(x, eta);

        // Calculate energy of our proposed update step
        nonLinearOperator().template update<0>();
        stats.energyProposal = energy();

        // Will we accept the proposal or not?
        // Check the performance of the quadratic model against the actual energy.
        auto rhonum = stats.energy - stats.energyProposal;
        auto rhoden = -eta.dot(gradient() + 0.5 * Heta);

        /*  Close to convergence the proposed energy and the real energy almost coincide.
         *  Therefore, the performance check of our model becomes ill-conditioned
         *  The regularisation fixes this */
        const auto rho_reg = std::max(1.0, abs(stats.energy)) * eps * options.rho_reg;
        rhonum             = rhonum + rho_reg;
        rhoden             = rhoden + rho_reg;

        const bool model_decreased = rhoden > 0.0;

        if (!model_decreased) info.stopReasonString.append(", model did not decrease");

        stats.rho = rhonum / rhoden;
        stats.rho = stats.rho < 0.0 ? -1.0 : stats.rho;  // move rho to the domain [-1.0,inf]

        info.trstr = "   ";

        // measure if energy decreased
        const bool energyDecreased = Dune::FloatCmp::ge(stats.energy - stats.energyProposal, -1e-12);

        // If the model behaves badly or if the energy increased we reduce the trust region size
        if (stats.rho < 1e-4 || not model_decreased || std::isnan(stats.rho) || not energyDecreased) {
          info.trstr = "TR-";
          innerInfo.Delta /= 4.0;
          info.consecutive_TRplus = 0;
          info.consecutive_TRminus++;
          if (info.consecutive_TRminus >= 5 && options.verbosity >= 1) {
            info.consecutive_TRminus = -std::numeric_limits<int>::infinity();
            spdlog::info(" +++ Detected many consecutive TR- (radius decreases).");
            spdlog::info(" +++ Consider decreasing options.Delta_bar by an order of magnitude.");
          }

        } else if (stats.rho > 0.99
                   && (innerInfo.stop_tCG == Eigen::TCGStopReason::negativeCurvature
                       || innerInfo.stop_tCG == Eigen::TCGStopReason::exceededTrustRegion)) {
          info.trstr               = "TR+";
          innerInfo.Delta          = std::min(3.5 * innerInfo.Delta, options.Delta_bar);
          info.consecutive_TRminus = 0;
          info.consecutive_TRplus++;
          if (info.consecutive_TRplus >= 5 && options.verbosity >= 1) {
            info.consecutive_TRplus = -std::numeric_limits<int>::infinity();
            spdlog::info(" +++ Detected many consecutive TR+ (radius increases)");
            spdlog::info(" +++ Consider increasing options.Delta_bar by an order of magnitude");

          } else {
            info.consecutive_TRplus  = 0;
            info.consecutive_TRminus = 0;
          }
        }

        if (model_decreased && stats.rho > options.rho_prime && energyDecreased) {
          if (stats.energyProposal > stats.energy)
            spdlog::info(
                "Energy function increased by {} (step size: {}). Since this is small we accept the step and hope for "
                "convergence of the gradient norm.",
                stats.energyProposal - stats.energy, stats.etaNorm);

          info.acceptProposal       = true;
          info.accstr               = "acc";
          info.consecutive_Rejected = 0;
        } else {
          info.acceptProposal = false;
          info.accstr         = "REJ";

          if (info.consecutive_Rejected >= 5)
            innerInfo.Delta /= 2;
          else
            innerInfo.Delta = std::min(innerInfo.Delta, stats.etaNorm / 2.0);
          ++info.consecutive_Rejected;
        }

        stats.outerIter++;

        if (options.verbosity == 1) logState();

        info.randomString = "";

        if (info.acceptProposal) {
          stats.energy = stats.energyProposal;
          nonLinearOperator_.updateAll();
          xOld = x;
          this->notify(NonLinearSolverMessages::CORRECTIONNORM_UPDATED, stats.etaNorm);
          this->notify(NonLinearSolverMessages::RESIDUALNORM_UPDATED, stats.gradNorm);
          this->notify(NonLinearSolverMessages::SOLUTION_CHANGED);
        } else {
          x = xOld;
          eta.setZero();
        }
        nonLinearOperator_.updateAll();
        stats.gradNorm = gradient().norm();
        this->notify(NonLinearSolverMessages::ITERATION_ENDED);
      }
      spdlog::info("{}", info.reasonString);
      spdlog::info("Total iterations: {} Total CG Iterations: {}", stats.outerIter, stats.innerIterSum);

      solverInformation.sucess
          =  (info.stop == StopReason::gradientNormTolReached);

      solverInformation.iterations  = stats.outerIter;
      solverInformation.gradienNorm = stats.gradNorm;
      this->notify(NonLinearSolverMessages::FINISHED_SUCESSFULLY, solverInformation.iterations,
                   solverInformation.gradienNorm, options.grad_tol);
      return solverInformation;
    }

    auto& nonLinearOperator() { return nonLinearOperator_; }

  private:
    void logState() const {
      spdlog::info(
          "{:>3s} {:>3s} {:>6d} {:>9d}  {:>6.2f}  {:>9.2e}  {:>9.2e}  {:>11.2e}  {:>9.2e}  {:>9.2e}  {:>11.2e}   "
          "{:<73}",
          info.accstr, info.trstr, stats.outerIter, innerInfo.numInnerIter, stats.rho, stats.energy,
          stats.energyProposal, stats.energyProposal - stats.energy, stats.gradNorm, innerInfo.Delta, stats.etaNorm,
          info.stopReasonString + info.cauchystr + info.randomString);
    }

    void logFinalState() {
      spdlog::info("{:>3s} {:>3s} {:>6d} {:>9d}  {: ^6}  {: ^9}  {: ^9}  {: ^11}  {:>9.2e}  {: ^9}  {: ^11}   {:<73}",
                   info.accstr, info.trstr, stats.outerIter, innerInfo.numInnerIter, " ", " ", " ", " ", stats.gradNorm,
                   " ", " ", info.stopReasonString + info.cauchystr + info.randomString);
    }

    inline const auto& energy() { return nonLinearOperator().value(); }
    inline const auto& gradient() { return nonLinearOperator().derivative(); }
    inline const auto& hessian() { return nonLinearOperator().secondDerivative(); }

    bool stoppingCriterion() {
      std::ostringstream stream;
      /** Gradient correction tolerance reached  */
      if (stats.gradNorm < options.grad_tol && stats.outerIter != 0) {
        logFinalState();
        spdlog::info("CONVERGENCE:  Energy: {:1.16e}    norm(gradient): {:1.16e}", nonLinearOperator().value(),
                     stats.gradNorm);
        stream << "Gradient norm tolerance reached; options.tolerance = " << options.grad_tol;

        info.reasonString = stream.str();

        info.stop = StopReason::gradientNormTolReached;
        return true;
      } else if (stats.etaNorm < options.corr_tol && stats.outerIter != 0) {
        logFinalState();
        spdlog::info("CONVERGENCE:  Energy: {:1.16e}    norm(correction): {:1.16e}", nonLinearOperator().value(),
                     stats.etaNorm);
        stream << "Displacement norm tolerance reached;  = " << options.corr_tol << "." << std::endl;

        info.reasonString = stream.str();
        info.stop         = StopReason::correctionNormTolReached;
        return true;
      }

      /** Maximum Time reached  */
      if (stats.time >= options.maxtime) {
        logFinalState();
        stream << "Max time exceeded; options.maxtime = " << options.maxtime << ".";
        info.reasonString = stream.str();
        info.stop         = StopReason::maximumTimeReached;
        return true;
      }

      /** Maximum Iterations reached  */
      if (stats.outerIter >= options.maxiter) {
        logFinalState();
        stream << "Max iteration count reached; options.maxiter = " << options.maxiter << ".";
        info.reasonString = stream.str();
        info.stop         = StopReason::maximumIterationsReached;
        return true;
      }
      return false;
    }

    void solveInnerProblem() {
      truncatedConjugateGradient.setInfo(innerInfo);
      if(hessian().norm()>1e-10)
        truncatedConjugateGradient.factorize(hessian());
      else
        eta.setZero();
      // If the preconditioner is IncompleteCholesky the factorization may fail if we have negative diagonal entries and
      // the initial shift is too small. Therefore, if the factorization fails we increase the intial shift by a factor
      // of 5.
      if constexpr (preConditioner == PreConditioner::IncompleteCholesky) {
        int attempts = 0;
        while (truncatedConjugateGradient.info() != Eigen::Success) {
          choleskyInitialShift *= 5;
          truncatedConjugateGradient.preconditioner().setInitialShift(choleskyInitialShift);
          truncatedConjugateGradient.factorize(hessian());
          if (attempts > 5) throw std::logic_error("Factorization of preconditioner failed!");
          ++attempts;
        }
        if (truncatedConjugateGradient.info() == Eigen::Success) choleskyInitialShift = 1e-3;
      }
      eta       = truncatedConjugateGradient.solveWithGuess(-gradient(), eta);
      innerInfo = truncatedConjugateGradient.getInfo();
    }

    NonLinearOperatorImpl nonLinearOperator_;
    UpdateFunctionType updateFunction;
    typename NonLinearOperatorImpl::template ParameterValue<0> xOld;
    UpdateType eta;
    UpdateType Heta;
    TrustRegionSettings options;
    AlgoInfo info;
    double choleskyInitialShift = 1e-3;
    Eigen::TCGInfo<double> innerInfo;
    Stats stats;
    static constexpr double eps = 0.0001220703125;  // 0.0001220703125 is sqrt(sqrt(maschine-precision))
    std::array<std::string, 6> tcg_stop_reason{
        {"negative curvature", "exceeded trust region", "reached target residual-kappa (linear)",
         "reached target residual-theta (superlinear)", "maximum inner iterations", "model increased"}};

    using PreConditionerType
        = std::conditional_t<preConditioner == PreConditioner::IdentityPreconditioner, Eigen::IdentityPreconditioner,
                             std::conditional_t<preConditioner == PreConditioner::DiagonalPreconditioner,
                                                typename Eigen::DiagonalPreconditioner<ScalarType>,
                                                typename Eigen::IncompleteCholesky<ScalarType>>>;
    Eigen::TruncatedConjugateGradient<MatrixType, Eigen::Lower | Eigen::Upper, PreConditionerType>
        truncatedConjugateGradient;
  };

  template <typename NonLinearOperatorImpl, PreConditioner preConditioner = PreConditioner::IncompleteCholesky,
            typename UpdateType
            = std::conditional_t<std::is_floating_point_v<typename NonLinearOperatorImpl::template ParameterValue<0>>,
                                 typename NonLinearOperatorImpl::template ParameterValue<0>, Eigen::VectorXd>>
  std::shared_ptr<TrustRegion<NonLinearOperatorImpl, preConditioner, UpdateType>> makeTrustRegion(
      const NonLinearOperatorImpl& p_nonLinearOperator,
      std::function<void(typename NonLinearOperatorImpl::template ParameterValue<0>&, const UpdateType&)>
          p_updateFunction
      = [](typename NonLinearOperatorImpl::template ParameterValue<0>& a, const UpdateType& b) {
          using Ikarus::operator+=;
          a += b;
        }) {
    return std::make_shared<TrustRegion<NonLinearOperatorImpl, preConditioner, UpdateType>>(p_nonLinearOperator,
                                                                                            p_updateFunction);
  }

}  // namespace Ikarus