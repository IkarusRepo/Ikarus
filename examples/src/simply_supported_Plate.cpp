//
// Created by ac136645 on 7/1/2022.
//

#include <config.h>

#include "src/include/ikarus/finiteElements/feTraits.hh"

#include <matplot/matplot.h>
#include <numbers>

#include <dune/common/indices.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/iga/igaalgorithms.hh>
#include <dune/iga/nurbsgrid.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/feBases/autodiffFE.hh>
#include <ikarus/finiteElements/feBases/scalarFE.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/localBasis/localBasis.hh>
#include <ikarus/solver/nonLinearSolver/newtonRaphson.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/concepts.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/eigenDuneTransformations.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/utils/observer/loadControlObserver.hh>
#include <ikarus/utils/observer/nonLinearSolverLogger.hh>
#include <ikarus/finiteElements/mechanics/kirchhoffPlate.hh>

// Reference: Echter, R., Oesterle, B., Bischoff, M., 2013. A hierarchic family of isogeometric shell finite elements.
// Computer Methods in Applied Mechanics and Engineering 254, 170–180. https://doi.org/10.1016/j.cma.2012.10.018
// Section 5.1

int main() {
  /// Create 2D nurbs grid
  using namespace Ikarus;
  constexpr int griddim                                    = 2;
  constexpr int dimworld                                   = 2;
  const std::array<std::vector<double>, griddim> knotSpans = {{{0, 0, 1, 1}, {0, 0, 1, 1}}};

  using ControlPoint = Dune::IGA::NURBSPatchData<griddim, dimworld>::ControlPointType;

  const double L = 10.0;
  const std::vector<std::vector<ControlPoint>> controlPoints
      = {{{.p = {0, 0}, .w = 1}, {.p = {0, L}, .w = 1}}, {{.p = {L, 0}, .w = 1}, {.p = {L, L}, .w = 1}}};

  std::array<int, griddim> dimsize = {2, 2};

  std::vector<double> dofsVec;
  std::vector<double> l2Evector;
  auto controlNet = Dune::IGA::NURBSPatchData<griddim, dimworld>::ControlPointNetType(dimsize, controlPoints);
  using Grid      = Dune::IGA::NURBSGrid<griddim, dimworld>;

  Dune::IGA::NURBSPatchData<griddim, dimworld> patchData;
  patchData.knotSpans     = knotSpans;
  patchData.degree        = {1, 1};
  patchData.controlPoints = controlNet;
  /// Increase polynomial degree in each direction
  patchData = Dune::IGA::degreeElevate(patchData, 0, 1);
  patchData = Dune::IGA::degreeElevate(patchData, 1, 1);
  Grid grid(patchData);

  grid.globalRefine(2);

  auto gridView = grid.leafGridView();
  //    draw(gridView);
  using namespace Dune::Functions::BasisFactory;
  /// Create nurbs basis with extracted preBase from grid
  auto basis = makeBasis(gridView, gridView.getPreBasis());
  /// Fix complete boundary (simply supported plate)
  std::vector<bool> dirichletFlags(basis.size(), false);
  Dune::Functions::forEachBoundaryDOF(basis, [&](auto&& index) { dirichletFlags[index] = true; });

  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.size() << " Dofs" << std::endl;

  /// Create finite elements
  auto localView         = basis.localView();
  const double Emod      = 1000;
  const double nu        = 0.3;
  const double thickness = 0.001;
  std::vector<KirchhoffPlate<decltype(basis)>> fes;
  for (auto& ele : elements(gridView))
    fes.emplace_back(basis, ele, Emod, nu, thickness);

  /// Create assembler
  auto denseAssembler = DenseFlatAssembler(basis, fes, dirichletFlags);

  /// Create non-linear operator with potential energy
  Eigen::VectorXd w;
  w.setZero(basis.size());

  const double qz = 1.0*thickness*thickness*thickness;

  auto kFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements req = FErequirementsBuilder()
                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::MatrixAffordances::stiffness)
                                     .build();
    return denseAssembler.getMatrix(req);
  };

  auto rFunction = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
    Ikarus::FErequirements req = FErequirementsBuilder()
                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::VectorAffordances::forces)
                                     .build();
    return denseAssembler.getVector(req);
  };

  const auto& K = kFunction(w, qz);
  const auto& R = rFunction(w, qz);
  Eigen::LDLT<Eigen::MatrixXd> solver;
  solver.compute(K);
  w -= solver.solve(R);

  // Output solution to vtk
  auto wGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(basis, w);
  Dune::SubsamplingVTKWriter vtkWriter(gridView, Dune::refinementLevels(0));
  vtkWriter.addVertexData(wGlobalFunc, Dune::VTK::FieldInfo("w", Dune::VTK::FieldInfo::Type::scalar, 1));
  vtkWriter.write("Simply_Supported_Kirchhoff_Plate");

  /// Create analytical solution function for the simply supported case
  const double pi         = std::numbers::pi;
  const double factor_ana = (qz * Dune::power(L,4) * 12.0 * (1.0 - nu*nu))/(Emod * Dune::power(thickness,3));
  const double w_max_ana = ((5.0/384.0) - (4.0/Dune::power(pi,5)) * (0.68562 + 0.00025)) * factor_ana;

  auto wGlobalFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 1>>(basis, w);
  auto localw          = localFunction(wGlobalFunction);

  double w_fe = 0.0;

  for (auto& ele : elements(gridView)) {
    localView.bind(ele);
    localw.bind(ele);
    const auto geo   = localView.element().geometry();
//    std::cout<<"Center:  "<<geo.center()<<std::endl;
//    w_fe = localw(geo.center());
  }

  std::cout << "w_max_ana: " << w_max_ana << std::endl<<" w_fe: " << w_fe << std::endl;

}
