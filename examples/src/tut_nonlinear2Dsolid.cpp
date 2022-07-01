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

#include <config.h>

#include <dune/alugrid/grid.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/iga/nurbsgrid.hh>

#include "spdlog/spdlog.h"

#include <Eigen/Core>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/feRequirements.hh>
#include <ikarus/finiteElements/mechanics/nonLinearElasticityFE.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/manifolds/realTuple.hh>
#include <ikarus/solver/nonLinearSolver/newtonRaphson.hh>
#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/utils/observer/nonLinearSolverLogger.hh>
#include <dune/fufem/functiontools/boundarydofs.hh>
#include <dune/fufem/boundarypatch.hh>
#include <dune/fufem/dunepython.hh>


int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  using namespace Ikarus;
  constexpr int gridDim = 2;
  //  //  /// ALUGrid Example
  //  using Grid = Dune::ALUGrid<gridDim, 2, Dune::simplex, Dune::conforming>;
  //  auto grid  = Dune::GmshReader<Grid>::read("../../tests/src/testFiles/unstructuredTrianglesfine.msh", false);
  //  grid->globalRefine(1);
  /// IGA Grid Example
  constexpr auto dimworld              = 2;
//  const std::array<int, gridDim> order = {2, 2};

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
//  grid->globalRefine(1);



  /// YaspGrid Example
        using Grid        = Dune::YaspGrid<gridDim>;
        const double L    = 1;
        const double h    = 1;
        const size_t elex = 10;
        const size_t eley = 10;

        Dune::FieldVector<double, 2> bbox = {L, h};
        std::array<int, 2> eles           = {elex, eley};
        auto grid                         = std::make_shared<Grid>(bbox, eles);

  auto gridView = grid->leafGridView();
  const auto& indexSet = gridView.indexSet();

  Dune::BitSetVector<1> neumannVertices(gridView.size(2), false);

  std::string lambdaNeumannVertices = std::string("lambda x: ( x[0]>0.999 )");
  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  Python::runStream()
      << std::endl << "import sys"
      << std::endl << "import os"
      << std::endl;

  auto pythonNeumannVertices = Python::make_function<bool>(Python::evaluate(lambdaNeumannVertices));

  for (auto &&vertex: vertices(gridView))
  {
    bool isNeumann = pythonNeumannVertices(vertex.geometry().corner(0));
    neumannVertices[indexSet.index(vertex)] = isNeumann;
  }

  BoundaryPatch<decltype(gridView)> neumannBoundary(gridView, neumannVertices);

  using namespace Dune::Functions::BasisFactory;
//  auto basis = makeBasis(gridView, power<gridDim>(gridView.getPreBasis(), FlatInterleaved()));
    auto basis = makeBasis(gridView, power<gridDim>(lagrange<1>(), FlatInterleaved()));
  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.size() << " Dofs" << std::endl;

  draw(gridView);
  auto localView = basis.localView();
  std::vector<Ikarus::NonLinearElasticityFE<decltype(basis)>> fes;
  auto volumeLoad = [](auto& globalCoord, auto& lamb) {
    Eigen::Vector2d fext;
    fext.setZero();
    fext[1] = 2 * lamb*0;
    fext[0] = lamb*0;
    return fext;
  };

  auto neumannBoundaryLoad = [](auto& globalCoord, auto& lamb) {
    Eigen::Vector2d fext;
    fext.setZero();
    fext[1] = lamb/40;
    fext[0] = 0;
    return fext;
  };
  for (auto& element : elements(gridView))
    fes.emplace_back(basis, element, 1000, 0.3, &neumannBoundary,neumannBoundaryLoad, volumeLoad);

  std::vector<bool> dirichletFlags(basis.size(), false);

  Dune::Functions::forEachBoundaryDOF(basis, [&](auto&& localIndex, auto&& localView, auto&& intersection) {
    if (std::abs(intersection.geometry().center()[0]) < 1e-8) {
      dirichletFlags[localView.index(localIndex)[0]] = true;
    }
  });

  auto sparseAssembler = SparseFlatAssembler(basis, fes, dirichletFlags);

  Eigen::VectorXd d;
  d.setZero(basis.size());
  double lambda = 0.0;

  auto residualFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements req = FErequirementsBuilder()
                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::VectorAffordances::forces)
                                     .build();
    return sparseAssembler.getVector(req);
  };

  auto KFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements req = FErequirementsBuilder()
                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::MatrixAffordances::stiffness)
                                     .build();
    return sparseAssembler.getMatrix(req);
  };

  auto energyFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements req = FErequirementsBuilder()
                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::ScalarAffordances::mechanicalPotentialEnergy)
                                     .build();
    return sparseAssembler.getScalar(req);
  };

  auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energyFunction, residualFunction, KFunction),
                                            parameter(d, lambda));

  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::sd_UmfPackLU);

  auto nr = Ikarus::makeNewtonRaphson(nonLinOp.subOperator<1, 2>(), std::move(linSolver));
  //  auto nr = Ikarus::makeTrustRegion(nonLinOp);
  //  nr->setup({.verbosity = 1,
  //             .maxiter   = 30,
  //             .grad_tol  = 1e-8,
  //             .corr_tol  = 1e-8,
  //             .useRand   = false,
  //             .rho_reg   = 1e6,
  //             .Delta0    = 1});

  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();

  auto vtkWriter = std::make_shared<ControlSubsamplingVertexVTKWriter<decltype(basis)>>(basis, d, 2);
  vtkWriter->setFileNamePrefix("Test2Dsolid");
  vtkWriter->setFieldInfo("Displacement", Dune::VTK::FieldInfo::Type::vector, 2);
  nr->subscribeAll(nonLinearSolverObserver);

  auto lc = Ikarus::LoadControl(nr, 20, {0, 2000});

  lc.subscribeAll(vtkWriter);
  std::cout << "Energy before: " << nonLinOp.value() << std::endl;
  lc.run();
  nonLinOp.update<0>();
  std::cout << "Energy after: " << nonLinOp.value() << std::endl;
}