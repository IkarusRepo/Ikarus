//
// Created by Alex on 16.06.2021.
//

#pragma once

//#include <ikarus/FiniteElements/InterfaceFiniteElement.h>
#include <optional>
#include <map>

#include <ikarus/FiniteElements/FEValues.h>
#include <ikarus/Variables/InterfaceVariable.h>
#include <ikarus/Variables/ParameterFactory.h>
#ifdef __clang__
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif

namespace Ikarus::FiniteElements {

  struct DofAtEntity {
    int entityID;
    std::optional<EntityType> entityType;
    std::vector<Variable::VariableTags> variableVector;
  };

  enum class VectorAffordances { noAffordance, forces };

  enum class MatrixAffordances {
    noAffordance,
    stiffness,
    materialstiffness,
    geometricstiffness,
    stiffnessdiffBucklingVector,
    mass
  };

  enum class ScalarAffordances { noAffordance, potentialEnergy };

  inline constexpr VectorAffordances forces = VectorAffordances::forces;

  inline constexpr MatrixAffordances stiffness                   = MatrixAffordances::stiffness;
  inline constexpr MatrixAffordances stiffnessdiffBucklingVector = MatrixAffordances::stiffnessdiffBucklingVector;
  inline constexpr MatrixAffordances mass                        = MatrixAffordances::mass;

  inline constexpr ScalarAffordances potentialEnergy = ScalarAffordances::potentialEnergy;

  struct FErequirements {
    using VariableType = FEValues;
    using DataType     = typename std::optional<std::reference_wrapper<FEValues>>;
    std::optional<std::reference_wrapper<VariableType>> variables;
    DataType data{std::nullopt};
    std::map<decltype(FEParameterValuePair::type), decltype(FEParameterValuePair::value)> parameter;
    ScalarAffordances scalarAffordances{ScalarAffordances::noAffordance};
    VectorAffordances vectorAffordances{VectorAffordances::noAffordance};
    MatrixAffordances matrixAffordances{MatrixAffordances::noAffordance};
  };

}  // namespace Ikarus::FiniteElements

namespace Ikarus::Concepts {

#define TRYCALLFUNCTION(Str, ...)                        \
  if constexpr (Ikarus::Concepts::Has##Str<FE>)          \
    return fe.Str(__VA_ARGS__);                          \
  else if constexpr (Ikarus::Concepts::HasFree##Str<FE>) \
    return Str(fe, ##__VA_ARGS__);                       \
  else                                                   \
    DUNE_THROW(Dune::InvalidStateException,              \
               "The member/free function \"" << #Str << "\" is not implemented by this element");

#define TRYCALLFUNCTIONDONTTHROW(Str)                    \
  if constexpr (Ikarus::Concepts::Has##Str<FE>)          \
    return fe.Str();                                     \
  else if constexpr (Ikarus::Concepts::HasFree##Str<FE>) \
    return Str(fe);                                      \
  else                                                   \
    return;

  template <typename FiniteElement>
  concept HascalculateMatrix = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateMatrix(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateMatrix = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateMatrix(fe, req);
  };

  template <typename FiniteElement>
  concept HascalculateMatrixWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateMatrix(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateMatrixWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateMatrix(fe, req);
  };

  template <typename FiniteElement>
  concept HascalculateScalar = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateScalar(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateScalar = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateScalar(fe, req);
  };

  template <typename FiniteElement>
  concept HascalculateScalarWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateScalar(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateScalarWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateScalar(fe, req);
  };

  template <typename FiniteElement>
  concept HascalculateVector = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateVector(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateVector = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateVector(fe, req);
  };

  template <typename FiniteElement>
  concept HascalculateVectorWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateVector(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateVectorWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateVector(fe, req);
  };

  template <typename FiniteElement>
  concept HasgetEntityVariableTuple = requires(FiniteElement fe) {
    fe.getEntityVariableTuple();
  };

  template <typename FiniteElement>
  concept HasFreegetEntityVariableTuple = requires(FiniteElement fe) {
    getEntityVariableTuple(fe);
  };

  template <typename FiniteElement>
  concept HasSomegetEntityVariablePairs
      = HasgetEntityVariableTuple<FiniteElement> || HasFreegetEntityVariableTuple<FiniteElement>;

  template <typename FiniteElement>
  concept HasdofSize = requires(FiniteElement fe) {
    fe.dofSize();
  };

  template <typename FiniteElement>
  concept HasFreedofSize = requires(FiniteElement fe) {
    dofSize(fe);
  };

  template <typename FiniteElement>
  concept HasgetEntityID = requires(FiniteElement fe) {
    fe.getEntityID();
  };

  template <typename FiniteElement>
  concept HasFreegetEntityID = requires(FiniteElement fe) {
    getEntityID(fe);
  };

  template <typename FiniteElement>
  concept HascalculateLocalSystem = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateLocalSystem(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateLocalSystem = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateLocalSystem(fe, req);
  };

  template <typename FiniteElement>
  concept HascalculateLocalSystemWithOutData = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    fe.calculateLocalSystem(req);
  };

  template <typename FiniteElement>
  concept HasFreecalculateLocalSystemWithOutData
      = requires(FiniteElement fe, Ikarus::FiniteElements::FErequirements req) {
    calculateLocalSystem(fe, req);
  };

  template <typename FiniteElement>
  concept Hasinitialize = requires(FiniteElement fe) {
    fe.initialize();
  };

  template <typename FiniteElement>
  concept HasFreeinitialize = requires(FiniteElement fe) {
    initialize(fe);
  };

  template <typename FiniteElement>
  concept HassubEntities = requires(FiniteElement fe, unsigned int codim) {
    fe.subEntities(codim);
  };

  template <typename FiniteElement>
  concept HasFreesubEntities = requires(FiniteElement fe, unsigned int codim) {
    subEntities(fe, codim);
  };

  template <typename FiniteElement>
  concept Hasdimension = requires(FiniteElement fe) {
    fe.dimension();
  };

  template <typename FiniteElement>
  concept HasFreedimension = requires(FiniteElement fe) {
    dimension(fe);
  };

  template <typename FiniteElement>
  concept HassubIndex = requires(FiniteElement fe, int i, unsigned int codim) {
    fe.subIndex(i, codim);
  };

  template <typename FiniteElement>
  concept HasFreesubIndex = requires(FiniteElement fe, int i, unsigned int codim) {
    subIndex(fe, i, codim);
  };

  template <typename FiniteElement>
  concept MinimalFiniteElementLinearAlgebraAffordances
      = HasFreecalculateScalar<FiniteElement> || HasFreecalculateVector<FiniteElement> || HasFreecalculateMatrix<FiniteElement> || HascalculateScalar<FiniteElement> || HascalculateVector<FiniteElement> || HascalculateMatrix<FiniteElement> || HasFreecalculateScalarWithOutData<
          FiniteElement> || HasFreecalculateVectorWithOutData<FiniteElement> || HasFreecalculateMatrixWithOutData<FiniteElement> || HascalculateScalarWithOutData<FiniteElement> || HascalculateVectorWithOutData<FiniteElement> || HascalculateMatrixWithOutData<FiniteElement>;

}  // namespace Ikarus::Concepts
