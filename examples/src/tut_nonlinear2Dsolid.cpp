//
// Created by Alex on 21.07.2021.
//

#include "../../config.h"
#include <dune/alugrid/grid.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/iga/nurbsgrid.hh>

#include "spdlog/spdlog.h"

#include <Eigen/Core>

#include "ikarus/Controlroutines/LoadControl.h"
#include "ikarus/FiniteElements/NonLinearElasticityFEwithBasis.h"
#include "ikarus/Solver/NonLinearSolver/NewtonRaphson.hpp"
#include "ikarus/basis/basishelper.h"
#include "ikarus/utils/Observer/controlVTKWriter.h"
#include "ikarus/utils/Observer/nonLinearSolverLogger.h"
#include <ikarus/Assembler/SimpleAssemblers.h>
#include <ikarus/Grids/GridHelper/griddrawer.h>
#include <ikarus/LinearAlgebra/NonLinearOperator.h>
#include <ikarus/utils/utils/algorithms.h>

int main(int argc, char **argv) {
  Dune::MPIHelper::instance(argc,argv);
  using namespace Ikarus;
  constexpr int gridDim = 2;
//  /// ALUGrid Example
      using Grid = Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>;
      auto grid  = Dune::GmshReader<Grid>::read("../../tests/src/testFiles/unstructuredTrianglesfine.msh", false);
grid->globalRefine(1);
  /// IGA Grid Example
//  constexpr auto dimworld              = 2;
//  const std::array<int, gridDim> order = {2, 2};
//
//  const std::array<std::vector<double>, gridDim> knotSpans = {{{0, 0, 0, 1, 1, 1}, {0, 0, 0, 1, 1, 1}}};
//
//  using ControlPoint = Dune::IGA::NURBSPatchData<gridDim, dimworld>::ControlPointType;
//
//  const std::vector<std::vector<ControlPoint>> controlPoints
//      = {{{.p = {0, 0}, .w = 5}, {.p = {0.5, 0}, .w = 1}, {.p = {1, 0}, .w = 1}},
//         {{.p = {0, 0.5}, .w = 1}, {.p = {0.5, 0.5}, .w = 10}, {.p = {1, 0.5}, .w = 1}},
//         {{.p = {0, 1}, .w = 1}, {.p = {0.5, 1}, .w = 1}, {.p = {1, 1}, .w = 1}}};
//
//  std::array<int, gridDim> dimsize = {(int)(controlPoints.size()), (int)(controlPoints[0].size())};
//
//  auto controlNet = Dune::IGA::NURBSPatchData<gridDim, dimworld>::ControlPointNetType(dimsize, controlPoints);
//  using Grid      = Dune::IGA::NURBSGrid<gridDim, dimworld>;
//
//  Dune::IGA::NURBSPatchData<gridDim, dimworld> patchData;
//  patchData.knotSpans     = knotSpans;
//  patchData.degree        = order;
//  patchData.controlPoints = controlNet;
//  auto grid               = std::make_shared<Grid>(patchData);
//  grid->globalRefine(3);

  /// YaspGrid Example
//    using Grid        = Dune::YaspGrid<gridDim>;
//    const double L    = 1;
//    const double h    = 1;
//    const size_t elex = 10;
//    const size_t eley = 10;
//
//    Dune::FieldVector<double, 2> bbox = {L, h};
//    std::array<int, 2> eles           = {elex, eley};
//    auto grid                         = std::make_shared<Grid>(bbox, eles);

  auto gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;
    auto basis = makeBasis(gridView, power<gridDim>(gridView.getPreBasis(), FlatInterleaved()));
//  auto basis = makeBasis(gridView, power<gridDim>(lagrange<1>(), FlatInterleaved()));
  std::cout << "This gridview cotains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.size() << " Dofs" << std::endl;

  draw(gridView);
  auto localView = basis.localView();
  std::vector<Ikarus::FiniteElements::NonLinearElasticityFEWithLocalBasis<decltype(basis)>> fes;
  for (auto& element : elements(gridView))
    fes.emplace_back(basis, element, 1000, 0.3);

  std::vector<bool> dirichletFlags(basis.size(), false);

  Ikarus::markDirichletBoundaryDofs(basis, dirichletFlags,
                                    [](auto&& centerCoord) { return (std::abs(centerCoord[1]) < 1e-8); });
  auto denseAssembler = DenseFlatSimpleAssembler(basis, fes, dirichletFlags);

  auto sparseAssembler = SparseFlatAssembler(basis, fes, dirichletFlags);

  Eigen::VectorXd d;
  d.setZero(basis.size());
  double lambda = 0.0;

  auto residualFunction = [&](auto&& lambdaLocal, auto&& disp) -> auto& {
    return denseAssembler.getVector(forces, disp, lambdaLocal);
  };


//  auto KFunction = [&](auto&& lambdaLocal, auto&& disp) -> auto& {
//    return denseAssembler.getMatrix(stiffness, disp, lambdaLocal);
//  };

  auto KFunction = [&](auto&& lambdaLocal, auto&& disp) -> auto& {
    Ikarus::FErequirements req;
    req.sols.emplace_back(disp);
    req.parameter.insert({Ikarus::FEParameter::loadfactor, lambdaLocal});
    req.matrixAffordances = Ikarus::MatrixAffordances::stiffness;
    return sparseAssembler.getMatrix( req);
  };


  auto energyFunction = [&](auto&& lambdaLocal, auto&& disp) -> auto {
    return denseAssembler.getScalar(potentialEnergy, disp, lambdaLocal);
  };

  auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energyFunction, residualFunction, KFunction),
                                            parameter(lambda, d));

  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::s_SimplicialLDLT);

  auto nr                      = Ikarus::NewtonRaphson(nonLinOp.subOperator<1, 2>(), std::move(linSolver));
  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();

  auto vtkWriter = std::make_shared<ControlSubsamplingVertexVTKWriter<decltype(basis)>>(basis, d, 2);
  vtkWriter->setFileNamePrefix("Test2Dsolid");
  vtkWriter->setFieldInfo("Displacement", Dune::VTK::FieldInfo::Type::vector, 2);
  nr.subscribeAll(nonLinearSolverObserver);

  auto lc = Ikarus::LoadControl(std::move(nr), 20, {0, 2000});

  lc.subscribeAll(vtkWriter);
  std::cout << "Energy before: " << nonLinOp.value() << std::endl;
  lc.run();
  nonLinOp.update<0>();
  std::cout << "Energy after: " << nonLinOp.value() << std::endl;
}