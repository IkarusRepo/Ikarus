//
// Created by Alex on 21.07.2021.
//

#include <config.h>

#include <dune/alugrid/grid.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
//#include <dune/grid/yaspgrid.hh>
//#include <dune/iga/nurbsgrid.hh>
#include <dune/grid/uggrid.hh>

#include "spdlog/spdlog.h"

#include <Eigen/Core>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/feRequirements.hh>
#include <ikarus/finiteElements/mechanics/nonLinearElasticityFE.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/manifolds/realTuple.hh>
#include <ikarus/solver/linearSolver/geometricMultigrid/gridTransfer.h>
#include <ikarus/solver/nonLinearSolver/newtonRaphson.hh>
#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/utils/observer/nonLinearSolverLogger.hh>

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  using namespace Ikarus;
  constexpr int gridDim = 2;
  //  //  /// ALUGrid Example
    using Grid = Dune::UGGrid<gridDim>;
    Dune::GridFactory<Grid> gridFactory;

    const double L = 1.0;

    gridFactory.insertVertex({0, 0});
    gridFactory.insertVertex({L/2, 0});
    gridFactory.insertVertex({0, L/2});
    gridFactory.insertVertex({L/2, L/2});
    gridFactory.insertVertex({0, L});
    gridFactory.insertVertex({L/2, L});
    gridFactory.insertVertex({L, L});
    gridFactory.insertVertex({L, L/2});

    gridFactory.insertElement(Dune::GeometryTypes::quadrilateral, {0, 1, 2, 3});
    gridFactory.insertElement(Dune::GeometryTypes::quadrilateral, {2, 3,4,5});
    gridFactory.insertElement(Dune::GeometryTypes::quadrilateral, {3,7,5,6});

    auto grid     = gridFactory.createGrid();
    grid->globalRefine(2);
    auto leafGridView = grid->leafGridView();

    draw(leafGridView);
    draw(grid->levelGridView(1));
    draw(grid->levelGridView(0));

//
  using namespace Dune::Functions::BasisFactory;

    auto preBasisFactory = power<gridDim>(lagrange<1>(), FlatInterleaved());

    Ikarus::GridTransfer transfer(grid);

    transfer.createOperators(preBasisFactory);


    MultiGridSolver solver(grid,basis,feVectorCoarse);

//  std::cout << "This gridview contains: " << std::endl;
//  std::cout << gridView.size(2) << " vertices" << std::endl;
//  std::cout << gridView.size(1) << " edges" << std::endl;
//  std::cout << gridView.size(0) << " elements" << std::endl;
//  std::cout << basis.size() << " Dofs" << std::endl;
//
//  draw(gridView);
//  auto localView = basis.localView();
//  std::vector<Ikarus::NonLinearElasticityFE<decltype(basis)>> fes;
//  auto volumeLoad = [](auto& globalCoord, auto& lamb) {
//    Eigen::Vector2d fext;
//    fext.setZero();
//    fext[1] = 2 * lamb;
//    fext[0] = lamb;
//    return fext;
//  };
//  for (auto& element : elements(gridView))
//    fes.emplace_back(basis, element, 1000, 0.3, volumeLoad);
//
//  std::vector<bool> dirichletFlags(basis.size(), false);
//
//  Dune::Functions::forEachBoundaryDOF(basis, [&](auto&& localIndex, auto&& localView, auto&& intersection) {
//    if (std::abs(intersection.geometry().center()[1]) < 1e-8) {
//      dirichletFlags[localView.index(localIndex)[0]] = true;
//    }
//  });
//
//  auto sparseAssembler = SparseFlatAssembler(basis, fes, dirichletFlags);
//
//  Eigen::VectorXd d;
//  d.setZero(basis.size());
//  double lambda = 0.0;
//
//  auto residualFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto& {
//    Ikarus::FErequirements req = FErequirementsBuilder()
//                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
//                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
//                                     .addAffordance(Ikarus::VectorAffordances::forces)
//                                     .build();
//    return sparseAssembler.getVector(req);
//  };
//
//  auto KFunction = [&](auto&& disp, auto&& lambdaLocal) -> auto& {
//    Ikarus::FErequirements req = FErequirementsBuilder()
//                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp)
//                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
//                                     .addAffordance(Ikarus::MatrixAffordances::stiffness)
//                                     .build();
//    return sparseAssembler.getMatrix(req);
//  };
//
//  auto energyFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
//    Ikarus::FErequirements req = FErequirementsBuilder()
//                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
//                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
//                                     .addAffordance(Ikarus::ScalarAffordances::mechanicalPotentialEnergy)
//                                     .build();
//    return sparseAssembler.getScalar(req);
//  };
//
//  auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energyFunction, residualFunction, KFunction),
//                                            parameter(d, lambda));
//
//  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::sd_UmfPackLU);
//
//  auto nr = Ikarus::makeNewtonRaphson(nonLinOp.subOperator<1, 2>(), std::move(linSolver));
//  //  auto nr = Ikarus::makeTrustRegion(nonLinOp);
//  //  nr->setup({.verbosity = 1,
//  //             .maxiter   = 30,
//  //             .grad_tol  = 1e-8,
//  //             .corr_tol  = 1e-8,
//  //             .useRand   = false,
//  //             .rho_reg   = 1e6,
//  //             .Delta0    = 1});
//
//  auto nonLinearSolverObserver = std::make_shared<NonLinearSolverLogger>();
//
//  auto vtkWriter = std::make_shared<ControlSubsamplingVertexVTKWriter<decltype(basis)>>(basis, d, 2);
//  vtkWriter->setFileNamePrefix("Test2Dsolid");
//  vtkWriter->setFieldInfo("Displacement", Dune::VTK::FieldInfo::Type::vector, 2);
//  nr->subscribeAll(nonLinearSolverObserver);
//
//  auto lc = Ikarus::LoadControl(nr, 20, {0, 2000});
//
//  lc.subscribeAll(vtkWriter);
//  std::cout << "Energy before: " << nonLinOp.value() << std::endl;
//  lc.run();
//  nonLinOp.update<0>();
//  std::cout << "Energy after: " << nonLinOp.value() << std::endl;
}