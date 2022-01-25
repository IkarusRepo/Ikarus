//
// Created by Alex on 21.07.2021.
//
//#include <gmock/gmock.h>
//#include <gtest/gtest.h>

//#include "testHelpers.h"

#include "spdlog/spdlog.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include "ikarus/Assembler/SimpleAssemblers.h"
#include "ikarus/Controlroutines/LoadControl.h"
#include "ikarus/FEManager/DefaultFEManager.h"
#include "ikarus/FiniteElements/ElasticityFE.h"
#include "ikarus/FiniteElements/NonLinearElasticityFE.h"
#include "ikarus/LinearAlgebra/DirichletConditionManager.h"
#include "ikarus/utils/Observer/controlLogger.h"
#include "ikarus/utils/Observer/gridDrawerObserver.h"
#include "ikarus/utils/Observer/nonLinearSolverLogger.h"
#include <ikarus/FiniteElements/ForceLoad.h>
#include <ikarus/Grids/GridHelper/griddrawer.h>
#include <ikarus/Grids/SimpleGrid/SimpleGrid.h>
#include <ikarus/LinearAlgebra/NonLinearOperator.h>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("LoadControlTest: GridLoadControlTest", "[1]") {
  using namespace Ikarus::Grid;
  using Grid = SimpleGrid<2, 2>;
  SimpleGridFactory<2, 2> gridFactory;
  using vertexType = Eigen::Vector2d;
  std::vector<vertexType> verticesVec;
  const double L    = 10;
  const double h    = 1;
  const size_t elex = 10;
  const size_t eley = 10;
  for (size_t j = 0; j < eley + 1; ++j) {
    for (size_t i = 0; i < elex + 1; ++i)
      verticesVec.emplace_back(vertexType{i * L / (elex), j * h / (eley)});
  }
  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);
  for (size_t i = 0; i < elex; ++i) {
    for (size_t j = 0; j < eley; ++j) {
      std::vector<size_t> elementIndices;
      elementIndices.resize(4);
      elementIndices
          = {i + j * (elex + 1), i + j * (elex + 1) + 1, i + (j + 1) * (elex + 1), i + (j + 1) * (elex + 1) + 1};
      gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
    }
  }

  Grid grid     = gridFactory.createGrid();
  auto gridView = grid.leafGridView();

  //  draw(gridView);

  std::vector<Ikarus::FiniteElements::IFiniteElement> feContainer;

  //  for (auto&& ge : rootEntities(gridView))
  //    feContainer.emplace_back(Ikarus::FiniteElements::ElasticityFE(ge, gridView.indexSet(), 1000, 0.0));
  for (auto&& ge : rootEntities(gridView))
    feContainer.emplace_back(Ikarus::FiniteElements::NonLinearElasticityFE(ge, gridView.indexSet(), 1000, 0.3));

  auto spaceFunction = [](const Eigen::Vector2d&) -> Eigen::Vector2d {
    Eigen::Vector2d f{};
    f[1] = 1;
    return f;
  };

  for (auto& edge : edges(gridView)) {
    if (std::abs(edge.geometry().center()[1]) > h - 0.01) {
      //      std::cout << "edge.geometry().center() " << edge.geometry().center() << std::endl;
      feContainer.emplace_back(Ikarus::FiniteElements::ForceLoad(edge, gridView.indexSet(), spaceFunction));
    }
  }

  auto feManager = Ikarus::FEManager::DefaultFEManager(feContainer, gridView);

  Ikarus::DirichletConditionManager dirichletConditionManager(feManager);
  for (auto& vertex : vertices(gridView)) {
    if (std::abs(vertex.geometry().corner(0)[1]) < 1e-8) {
      //      std::cout << "AddConstraint at " << vertex.geometry().corner(0) << std::endl;
      dirichletConditionManager.addConstraint(vertex, 0);
      dirichletConditionManager.addConstraint(vertex, 1);
    }
  }
  dirichletConditionManager.finalize();

  auto vectorAssembler = Ikarus::Assembler::VectorAssembler(feManager, dirichletConditionManager);

  auto denseMatrixAssembler  = Ikarus::Assembler::DenseMatrixAssembler(feManager, dirichletConditionManager);
  auto sparseMatrixAssembler = Ikarus::Assembler::SparseMatrixAssembler(feManager, dirichletConditionManager);

  [[maybe_unused]] auto fintFunction = [&](auto&& lambda) -> auto& {
    return vectorAssembler.getReducedVector(Ikarus::FiniteElements::forces, lambda);
  };
  [[maybe_unused]] auto KFunction = [&](auto&& lambda) -> auto& {
    return denseMatrixAssembler.getReducedMatrix(Ikarus::FiniteElements::stiffness, lambda);
  };
  [[maybe_unused]] auto KFunctionSparse = [&](auto&& lambda) -> auto& {
    return sparseMatrixAssembler.getReducedMatrix(Ikarus::FiniteElements::stiffness, lambda);
  };

  auto controlObserver         = std::make_shared<ControlLogger>();
  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();
  auto gridDrawerObserver
      = std::make_shared<GridDrawerObserver<decltype(gridView), decltype(feManager)>>(gridView, feManager);
  auto time          = Ikarus::FEParameterFactory::createParameter(Ikarus::FEParameter::time, 1);
  time.value[0]      = 0.0;
  auto& Fint         = fintFunction(time);
  const int dofsFree = ((elex + 1) * (eley + 1) - (elex + 1)) * 2;
  CHECK (dofsFree == Fint.size());
  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::SparseLU);
  auto nonLinOp  = Ikarus::NonLinearOperator(linearAlgebraFunctions(fintFunction, KFunctionSparse), parameter(time));
  auto nr        = Ikarus::NewtonRaphson(
      nonLinOp, std::move(linSolver),
      [&dirichletConditionManager](decltype(feManager.getVariables())& x, const Eigen::VectorX<double>& D) {
        x += dirichletConditionManager.viewAsFullVector(D);
      });
  nr.subscribeAll(nonLinearSolverObserver);

  auto lc = Ikarus::LoadControl(feManager, dirichletConditionManager, std::move(nr), 10, {0, 5000});
  //  auto lc = makeLoadControl<Ikarus::NewtonRaphson>(
  //      feManager, dirichletConditionManager, linearAlgebraFunctions(fintFunction, KFunctionSparse),
  //      Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::SparseLU), 10, {0, 1});
  lc.subscribeAll(controlObserver);
  //  lc.subscribeToNonLinearSolver(nonLinearSolverObserver);
  //  lc.subscribe(ControlMessages::SOLUTION_CHANGED, gridDrawerObserver);
  lc.run();
}