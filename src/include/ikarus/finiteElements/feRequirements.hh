//
// Created by Alex on 08.04.2022.
//

#pragma once

#include <iosfwd>
#include <map>
#include <set>
#include <vector>

#include <dune/common/exceptions.hh>

#include <Eigen/Core>

namespace Ikarus {

  // clang-format off
  enum class ScalarAffordances {
    noAffordance,
    mechanicalPotentialEnergy,
    microMagneticPotentialEnergy,
    microMagneticExchangeEnergy,
    microMagneticTotalMagnetoStaticEnergy,
    microMagneticInsideMagnetoStaticEnergy,
    microMagneticInsideFromMTimesHMagnetoStaticEnergy
  };

  enum class VectorAffordances {
    noAffordance,
    forces,
    microMagneticForces
  };

  enum class MatrixAffordances {
    noAffordance,
    stiffness,
    materialstiffness,
    geometricstiffness,
    stiffnessdiffBucklingVector,
    microMagneticHessian,
    mass
  };

  enum class FEParameter {
    noParameter,
    loadfactor,
    time
  };

  enum class FESolutions {
    noSolution,
    displacement,
    velocity,
    director,
    magnetizationAndVectorPotential
  };


  enum class ResultType {
    noType,
    magnetization,
    gradientNormOfMagnetization,
    vectorPotential,
    divergenceOfVectorPotential,
    BField,
    HField,
    cauchyStress,
    director
  };
  // clang-format on
  std::string getResultType(const ResultType &res);

  struct AffordanceCollection {
    ScalarAffordances scalarAffordances{ScalarAffordances::noAffordance};
    VectorAffordances vectorAffordances{VectorAffordances::noAffordance};
    MatrixAffordances matrixAffordances{MatrixAffordances::noAffordance};
  };

  template <typename Type>
  concept FEAffordance
      = std::is_same_v<std::remove_cvref_t<Type>, ScalarAffordances> or std::is_same_v<std::remove_cvref_t<Type>, VectorAffordances> or std::is_same_v<
      std::remove_cvref_t<Type>, MatrixAffordances> or std::is_same_v<AffordanceCollection, MatrixAffordances>;

  inline constexpr VectorAffordances forces = VectorAffordances::forces;

  inline constexpr MatrixAffordances stiffness                   = MatrixAffordances::stiffness;
  inline constexpr MatrixAffordances stiffnessdiffBucklingVector = MatrixAffordances::stiffnessdiffBucklingVector;
  inline constexpr MatrixAffordances mass                        = MatrixAffordances::mass;

  inline constexpr ScalarAffordances potentialEnergy = ScalarAffordances::mechanicalPotentialEnergy;

  inline constexpr AffordanceCollection elastoStatics
      = {ScalarAffordances::mechanicalPotentialEnergy, VectorAffordances::forces, MatrixAffordances::stiffness};

  template <typename SolutionVectorType = Eigen::VectorXd, typename ParameterType = double>
  class FErequirementsBuilder;

  template <typename SolutionVectorType = Eigen::VectorXd, typename ParameterType = double>
  struct FErequirements {
    friend FErequirementsBuilder<SolutionVectorType, ParameterType>;

  public:
    const SolutionVectorType &getSolution(FESolutions &&key) const {
      try {
        return sols.at(key).get();
      } catch (std::out_of_range &oor) {
        DUNE_THROW(Dune::RangeError, std::string("Out of Range error: ") + std::string(oor.what()) + " in getSolution");
        abort();
      }
    }

    const ParameterType &getParameter(FEParameter &&key) const { return parameter.at(key).get(); }

    template <FEAffordance Affordance>
    bool hasAffordance(Affordance &&affordance) const {
      if constexpr (std::is_same_v<Affordance, ScalarAffordances>)
        return affordances.scalarAffordances == affordance;
      else if constexpr (std::is_same_v<Affordance, VectorAffordances>)
        return affordances.vectorAffordances == affordance;
      else if constexpr (std::is_same_v<Affordance, MatrixAffordances>)
        return affordances.matrixAffordances == affordance;
      else if constexpr (std::is_same_v<AffordanceCollection, MatrixAffordances>)
        return affordances == affordance;
    }

  private:
    std::map<FESolutions, std::reference_wrapper<const SolutionVectorType>> sols;
    std::map<FEParameter, std::reference_wrapper<const ParameterType>> parameter;
    AffordanceCollection affordances;
  };

  template <typename SolutionVectorType, typename ParameterType>
  class FErequirementsBuilder {
  public:
    template <FEAffordance Affordance>
    FErequirementsBuilder &addAffordance(Affordance &&affordance) {
      if constexpr (std::is_same_v<Affordance, ScalarAffordances>)
        affordances.scalarAffordances = affordance;
      else if constexpr (std::is_same_v<Affordance, VectorAffordances>)
        affordances.vectorAffordances = affordance;
      else if constexpr (std::is_same_v<Affordance, MatrixAffordances>)
        affordances.matrixAffordances = affordance;
      else if constexpr (std::is_same_v<AffordanceCollection, MatrixAffordances>)
        affordances = affordance;
      return *this;
    }

    FErequirementsBuilder &insertParameter(FEParameter &&key, const ParameterType &val) {
      parameter.insert({key, val});
      return *this;
    }

    FErequirementsBuilder &insertGlobalSolution(FESolutions &&key, const SolutionVectorType &sol) {
      sols.insert({key, sol});
      return *this;
    }

    FErequirements<SolutionVectorType, ParameterType> build() {
      FErequirements<SolutionVectorType, ParameterType> req;
      req.sols        = sols;
      req.parameter   = parameter;
      req.affordances = affordances;
      return req;
    }

  private:
    std::map<FESolutions, std::reference_wrapper<const SolutionVectorType>> sols;
    std::map<FEParameter, std::reference_wrapper<const ParameterType>> parameter;
    AffordanceCollection affordances;
  };

  template <typename ParameterType = double>
  class ResultTypeMap {
  public:
    using ResultArray = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, 3, 3>;
    void insertOrAssignResult(ResultType &&resultType, const ResultArray &resultArray) {
      results.insert_or_assign(resultType, resultArray);
    }

    ResultArray &getResult(ResultType &&resultType) { return results.at(resultType); }

  private:
    std::map<ResultType, ResultArray> results;
  };

  template <typename SolutionVectorType = Eigen::VectorXd, typename ParameterType_ = double>
  class ResultRequirements : public FErequirements<SolutionVectorType, ParameterType_> {
  public:
    using ParameterType = ParameterType_;

    ResultRequirements(FErequirements<SolutionVectorType, ParameterType> &&req, std::set<ResultType> &&p_resType)
        : FErequirements<SolutionVectorType, ParameterType>(std::move(req)), resType(std::move(p_resType)) {}
    bool isResultRequested(ResultType &&key) const { return resType.contains(key); }

  private:
    std::set<ResultType> resType;
  };

  template <typename Type>
  concept ResultTypeConcept = std::is_same_v<Type, ResultType>;

  template <typename SolutionVectorType = Eigen::VectorXd, typename ParameterType = double>
  class ResultRequirementsBuilder {
  public:
    template <FEAffordance Affordance>
    ResultRequirementsBuilder &addAffordance(Affordance &&affordance) {
      reqB.addAffordance(std::forward<Affordance>(affordance));
      return *this;
    }

    ResultRequirementsBuilder &insertParameter(FEParameter &&key, const ParameterType &val) {
      reqB.insertParameter(std::forward<FEParameter>(key), val);
      return *this;
    }

    ResultRequirementsBuilder &insertGlobalSolution(FESolutions &&key, const SolutionVectorType &sol) {
      reqB.insertGlobalSolution(std::forward<FESolutions>(key), sol);
      return *this;
    }

    template <ResultTypeConcept... ResultTypes>
    ResultRequirementsBuilder &addResultRequest(ResultTypes &&...keys) {
      resType.insert({std::move(keys)...});
      return *this;
    }

    ResultRequirements<SolutionVectorType, ParameterType> build() {
      ResultRequirements<SolutionVectorType, ParameterType> resReq(reqB.build(), std::move(resType));
      return resReq;
    }

  private:
    std::set<ResultType> resType;
    FErequirementsBuilder<SolutionVectorType, ParameterType> reqB;
  };

}  // namespace Ikarus