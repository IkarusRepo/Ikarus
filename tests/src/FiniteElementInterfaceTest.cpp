//
// Created by Alex on 21.04.2021.
//

//#include <gmock/gmock.h>
//#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include <dune/geometry/type.hh>

#include <Eigen/Core>

#include <ikarus/FiniteElements/ElasticityFE.h>
#include <ikarus/FiniteElements/InterfaceFiniteElement.h>
#include <ikarus/Geometries/GeometryType.h>
#include <ikarus/Grids/SimpleGrid/SimpleGrid.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using namespace Catch;
using namespace Ikarus;

class TestFE {
public:
  static void initialize() {}
  [[nodiscard]] static Ikarus::FiniteElements::IFiniteElement::DofPairVectorType getEntityVariablePairs() {
    return Ikarus::FiniteElements::IFiniteElement::DofPairVectorType{};
  }

  static double calculateScalar(const Ikarus::FiniteElements::FErequirements& par) {
    if (par.data)
      if (isType(par.data->get().get(Ikarus::EntityType::vertex)[0], Ikarus::Variable::VariableTags::displacement2d))
        return getValue(par.data->get().get(Ikarus::EntityType::vertex)[0])[0]
               * getValue(par.data->get().get(Ikarus::EntityType::vertex)[0])[1];
    return 5;
  }
};

class TestFE2 {
public:
  static void initialize() {}
  [[nodiscard]] static Ikarus::FiniteElements::IFiniteElement::DofPairVectorType getEntityVariablePairs() {
    return Ikarus::FiniteElements::IFiniteElement::DofPairVectorType{};
  }

  static double calculateScalar([[maybe_unused]] const Ikarus::FiniteElements::FErequirements& par) { return 5; }
};

TEST_CASE("FiniteElementInterfaceTest: createGenericFEList", "[1]") {
  using namespace Ikarus::Grid;
  using namespace Ikarus::FiniteElements;

  using Grid = SimpleGrid<2, 2>;
  SimpleGridFactory<2, 2> gridFactory;
  //  using vertexType = Eigen::Vector2d;
  std::vector<Eigen::Vector2d> verticesVec;
  verticesVec.emplace_back(0.0, 0.0);  // 0
  verticesVec.emplace_back(2.0, 0.0);  // 1
  verticesVec.emplace_back(0.0, 2.0);  // 2
  verticesVec.emplace_back(2.0, 2.0);  // 3
  verticesVec.emplace_back(4.0, 0.0);  // 4
  verticesVec.emplace_back(4.0, 2.0);  // 5

  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);

  std::vector<size_t> elementIndices;
  elementIndices.resize(4);
  elementIndices = {0, 1, 2, 3};
  gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
  elementIndices = {1, 4, 3, 5};
  gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);

  Grid grid = gridFactory.createGrid();

  auto gridView = grid.leafGridView();
  std::vector<Ikarus::FiniteElements::IFiniteElement> fes;

  for (auto&& element : surfaces(gridView))
    fes.emplace_back(Ikarus::FiniteElements::ElasticityFE(element, gridView.indexSet(), 1000, 0.3));

  Eigen::VectorXd fint{};
  Eigen::MatrixXd K{};
  fint.setZero(8);
  K.setZero(8, 8);
  for (auto&& fe : fes) {
    using namespace Ikarus::Variable;
    std::vector<IVariable> vars;
    vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
    vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
    vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
    vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));

    FiniteElements::FEValues feValues;
    feValues.add(Ikarus::EntityType::vertex, vars);
    feValues.add(Ikarus::EntityType::edge, vars);
    feValues.add(Ikarus::EntityType::surface, vars);
    feValues.add(Ikarus::EntityType::volume, vars);

    // test FE withoutData
    {
      FErequirements feParameter;
      feParameter.scalarAffordances = potentialEnergy;
      feParameter.vectorAffordances = forces;
      feParameter.matrixAffordances = stiffness;
      feParameter.variables         = feValues;
      const auto [KEle, fintEle]    = calculateLocalSystem(fe, feParameter);
      CHECK (8 == dofSize(fe));
      CHECK (8 == calculateVector(fe, feParameter).size());
      CHECK (0.0 == Approx (calculateScalar(fe, feParameter)));
      CHECK (8 == calculateMatrix(fe, feParameter).cols());
      CHECK (8 == calculateMatrix(fe, feParameter).rows());
      feParameter.matrixAffordances = mass;
      CHECK_THROWS_AS (calculateMatrix(fe, feParameter), std::logic_error);
      CHECK_THROWS_AS (calculateLocalSystem(fe, feParameter), std::logic_error);
      CHECK (8 == KEle.rows());
      CHECK (8 == KEle.cols());
      CHECK (8 == fintEle.size());
    }
    feValues.add(Ikarus::EntityType::vertex, vars[0]);
    feValues.add(Ikarus::EntityType::edge, vars[0]);
    feValues.add(Ikarus::EntityType::surface, vars[0]);
    feValues.add(Ikarus::EntityType::volume, vars[0]);

    FiniteElements::FEValues dataFeValues;
    dataFeValues.add(Ikarus::EntityType::vertex, vars);
    dataFeValues.add(Ikarus::EntityType::edge, vars);
    dataFeValues.add(Ikarus::EntityType::surface, vars);
    dataFeValues.add(Ikarus::EntityType::volume, vars);

    {
      FErequirements feParameter;
      feParameter.scalarAffordances = potentialEnergy;
      feParameter.vectorAffordances = forces;
      feParameter.matrixAffordances = stiffness;
      feParameter.variables         = feValues;
      feParameter.data              = dataFeValues;
      const auto [KEle, fintEle]    = calculateLocalSystem(fe, feParameter);
      CHECK (8 == dofSize(fe));
      CHECK (8 == calculateVector(fe, feParameter).size());
      CHECK (0.0 == Approx (calculateScalar(fe, feParameter)));
      CHECK (8 == calculateMatrix(fe, feParameter).cols());
      CHECK (8 == calculateMatrix(fe, feParameter).rows());
      feParameter.matrixAffordances = mass;
      CHECK_THROWS_AS (calculateMatrix(fe, feParameter), std::logic_error);
      CHECK_THROWS_AS (calculateLocalSystem(fe, feParameter), std::logic_error);
      CHECK (8 == KEle.rows());
      CHECK (8 == KEle.cols());
      CHECK (8 == fintEle.size());
    }
  }

  const auto entityIDDofPair = getEntityVariableTuple(fes[0]);
  using namespace Ikarus::Variable;
  std::vector<std::pair<int, Ikarus::Variable::VariableTags>> idtagExpected;
  idtagExpected.emplace_back(0, VariableTags::displacement2d);
  idtagExpected.emplace_back(1, VariableTags::displacement2d);
  idtagExpected.emplace_back(2, VariableTags::displacement2d);
  idtagExpected.emplace_back(3, VariableTags::displacement2d);
  for (int i = 0; auto&& [entityID, entityType, varVec] : entityIDDofPair) {
    CHECK (idtagExpected[i].first == entityID);
    CHECK (1 == varVec.size());
    CHECK (Ikarus::EntityType::vertex == entityType);
    CHECK (idtagExpected[i].second == varVec[0]);
    ++i;
  }

  auto feT{fes[0]};  // test copy assignment

  using namespace Ikarus::Variable;
  std::vector<IVariable> vars;
  vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
  vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
  vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
  vars.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));

  FiniteElements::FEValues feValues;
  feValues.add(Ikarus::EntityType::vertex, vars);

  std::vector<IVariable> datas;
  datas.emplace_back(VariableFactory::createVariable(VariableTags::displacement2d));
  datas[0] += Eigen::Vector2d(15, 2);

  FiniteElements::FEValues feDataValues;
  feDataValues.add(Ikarus::EntityType::vertex, datas);

  CHECK(feDataValues.get(Ikarus::EntityType::vertex).size()== 1);
  CHECK(feDataValues.get(Ikarus::EntityType::edge).size()== 0);
  CHECK(feDataValues.get(Ikarus::EntityType::surface).size()== 0);
  CHECK(feDataValues.get(Ikarus::EntityType::volume).size()== 0);

  Ikarus::FiniteElements::IFiniteElement fe((TestFE()));
  // check behaviour of dummy fe calculate scalar function before adding data and after
  FErequirements feParameter;
  feParameter.scalarAffordances = potentialEnergy;
  feParameter.vectorAffordances = forces;
  feParameter.matrixAffordances = stiffness;
  feParameter.variables         = feValues;
  feParameter.data              = feDataValues;
  CHECK (30.0 == Approx (calculateScalar(fe, feParameter)));
  datas[0] = VariableFactory::createVariable(VariableTags::displacement1d);
  CHECK (5.0 == Approx (calculateScalar(fe, feParameter)));

  Ikarus::FiniteElements::IFiniteElement fe2((TestFE2()));  // check if element without optional data is accepted

  feParameter.data = std::nullopt;
  CHECK (5.0 == Approx (calculateScalar(fe2, feParameter)));
}