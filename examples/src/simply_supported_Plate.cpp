// /*
//  *  This file is part of the Ikarus distribution (https://github.com/rath3t/Ikarus).
//  *  Copyright (c) 2021 Alexander Müller.
//  *  Institut fuer Baustatik und Baudynamik
//  *  Universität Stuttgart
//  *
//  *  This library is free software; you can redistribute it and/or
//  *   modify it under the terms of the GNU Lesser General Public
//  *   License as published by the Free Software Foundation; either
//  *   version 2.1 of the License, or (at your option) any later version.
//
// *   This library is distributed in the hope that it will be useful,
// *   but WITHOUT ANY WARRANTY; without even the implied warranty of
// *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// *   Lesser General Public License for more details.
//
// *   You should have received a copy of the GNU Lesser General Public
// *   License along with this library; if not, write to the Free Software
// *   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
// *  USA
// *

#include <config.h>

#include <vector>

#include <dune/common/parametertreeparser.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/iga/igaalgorithms.hh>
#include <dune/iga/nurbsgrid.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/finiteElements/mechanics/ReissnerMindlinPlate.hh>
#include <ikarus/finiteElements/mechanics/kirchhoffPlate.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/localBasis/localBasis.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/eigenDuneTransformations.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/solver/linearSolver/linearSolver.hh>


// References:
// [1] Echter, R., Oesterle, B., Bischoff, M., 2013. A hierarchic family of isogeometric shell finite elements.
// Computer Methods in Applied Mechanics and Engineering 254, 170–180. https://doi.org/10.1016/j.cma.2012.10.018
// Section 5.1

// [2] Oesterle, B., Ramm, E., Bischoff, M., 2016. A shear deformable, rotation-free isogeometric shell formulation.
// Computer Methods in Applied Mechanics and Engineering 307, 235–255. https://doi.org/10.1016/j.cma.2016.04.015
// Section 5.1

// Element types (eletype):
// 0 = Kirchhoff Plate Element (w)
// 1 = Reissner-Mindlin Plate Element (w, thetax, thetay)

#define eletype 1

using namespace Ikarus;
using namespace Dune::Indices;

int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  constexpr int griddim  = 2;
  constexpr int dimworld = 2;
  constexpr int order    = 1;

  /// read in parameters
  Dune::ParameterTree parameterSet;
  Dune::ParameterTreeParser::readINITree(argv[1], parameterSet);

  const Dune::ParameterTree& gridParameters     = parameterSet.sub("GridParameters");
  const Dune::ParameterTree& controlParameters  = parameterSet.sub("ControlParameters");
  const Dune::ParameterTree& materialParameters = parameterSet.sub("MaterialParameters");

  const double L             = gridParameters.get<double>("L");
  const double Emod          = materialParameters.get<double>("E");
  const double nu            = materialParameters.get<double>("nu");
  const double thickness     = materialParameters.get<double>("thickness");
  const int refinement_level = gridParameters.get<int>("refinement");

#if eletype == 0
  /// Create 2D nurbs grid
  const std::array<std::vector<double>, griddim> knotSpans = {{{0, 0, 1, 1}, {0, 0, 1, 1}}};

  using ControlPoint = Dune::IGA::NURBSPatchData<griddim, dimworld>::ControlPointType;

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
  grid.globalRefine(refinement_level);
  auto gridView = grid.leafGridView();
  using namespace Dune::Functions::BasisFactory;

  /// Create nurbs basis with extracted preBase from grid
  auto basis = makeBasis(gridView, gridView.getPreBasis());

  /// Fix complete boundary (simply supported plate)
  std::vector<bool> dirichletFlags(basis.size(), false);
  Dune::Functions::forEachBoundaryDOF(basis, [&](auto&& index) { dirichletFlags[index] = true; });

  /// Create finite elements
  std::vector<KirchhoffPlate<decltype(basis)>> fes;
  for (auto& ele : elements(gridView)) {
    fes.emplace_back(basis, ele, Emod, nu, thickness);
  }

  std::string output_file = "Simply_Supported_Kirchhoff_Plate";
#endif

#if eletype == 1
  /// Creating YaspGrid
  using Grid        = Dune::YaspGrid<griddim>;
  const size_t elex = 1;

  Dune::FieldVector<double, 2> bbox = {L, L};
  std::array<int, 2> eles           = {elex, elex};
  auto grid                         = std::make_shared<Grid>(bbox, eles);
  grid->globalRefine(refinement_level);
  auto gridView = grid->leafGridView();

  using namespace Dune::Functions::BasisFactory;

  /// Create power basis
  auto basis = makeBasis(gridView, power<3>(lagrange<order>(), FlatInterleaved()));

  /// Fix complete boundary (simply supported plate) - Soft Support
  std::vector<bool> dirichletFlags(basis.size(), false);
  Dune::Functions::forEachBoundaryDOF(Dune::Functions::subspaceBasis(basis, _0),
                                      [&](auto&& index) { dirichletFlags[index] = true; });

  //  /// Fix complete boundary (simply supported plate) - Hard Support
  //  std::vector<bool> dirichletFlags(basis.size(), false);
  //  Dune::Functions::forEachBoundaryDOF(Dune::Functions::subspaceBasis(basis, _0), [&](auto&& index) {
  //  dirichletFlags[index] = true; }); Dune::Functions::forEachBoundaryDOF(Dune::Functions::subspaceBasis(basis, _1),
  //  [&](auto &&localIndex, auto &&localView, auto &&intersection) {
  //    if ((std::abs(intersection.geometry().center()[0]) < 1e-8) or (std::abs(intersection.geometry().center()[0]-L) <
  //    1e-8))
  //      dirichletFlags[localView.index(localIndex)] = true;
  //  });
  //  Dune::Functions::forEachBoundaryDOF(Dune::Functions::subspaceBasis(basis, _2), [&](auto &&localIndex, auto
  //  &&localView, auto &&intersection) {
  //    if ((std::abs(intersection.geometry().center()[1]) < 1e-8) or (std::abs(intersection.geometry().center()[1]-L) <
  //    1e-8))
  //      dirichletFlags[localView.index(localIndex)] = true;
  //  });

  // Function for distributed load
  auto volumeLoad = [](auto& lamb) {
    Eigen::Vector<double, 3> fext;
    fext.setZero();
    fext[0] = lamb;
    return fext;
  };

  /// Create finite elements
  std::vector<ReissnerMindlinPlate<decltype(basis)>> fes;
  for (auto& ele : elements(gridView)) {
    fes.emplace_back(basis, ele, Emod, nu, thickness, volumeLoad);
  }

  std::string output_file = "Simply_Supported_Reissner_Mindlin_Plate";
#endif

  std::cout << "#################################################" << std::endl;
  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.size() << " Dofs" << std::endl;

  /// Create assembler
  auto denseAssembler = SparseFlatAssembler(basis, fes, dirichletFlags);

  Eigen::VectorXd w;
  w.setZero(basis.size());

  const double qz = 1.0 * thickness * thickness * thickness;

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

//  Eigen::LDLT<Eigen::MatrixXd> solver;
//  solver.compute(K);
//  w -= solver.solve(R);

  auto linSolver = Ikarus::ILinearSolver<double>(Ikarus::SolverTypeTag::sd_SimplicialLDLT);
  linSolver.compute(K);
  linSolver.solve(w, -R);


#if eletype == 0
  auto wGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(basis, w);
#endif

#if eletype == 1
  auto wGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(subspaceBasis(basis, _0), w);

  auto resultRequirements = Ikarus::ResultRequirementsBuilder<Eigen::VectorXd>()
                                .insertGlobalSolution(Ikarus::FESolutions::displacement, w)
                                .insertParameter(Ikarus::FEParameter::loadfactor, qz)
                                .addResultRequest(ResultType::stressResultant)
                                .build();

  ResultTypeMap<double> result;
  auto scalarBasis     = makeBasis(gridView, lagrangeDG<order>());
  auto localScalarview = scalarBasis.localView();
  std::vector<Dune::FieldMatrix<double, 3, 3>> stressRes(scalarBasis.size());
  std::vector<Dune::FieldVector<double, 3>> stressResVector(scalarBasis.size());
  auto ele = elements(gridView).begin();

  for (auto& fe : fes) {
    localScalarview.bind(*ele);
    const auto& fe2              = localScalarview.tree().finiteElement();
    const auto& referenceElement = Dune::ReferenceElements<double, griddim>::general(ele->type());
    for (auto c = 0UL; c < fe2.size(); ++c) {
      const auto fineKey                        = fe2.localCoefficients().localKey(c);
      const auto nodalPositionInChildCoordinate = referenceElement.position(fineKey.subEntity(), fineKey.codim());
      auto coord                                = toEigenVector(nodalPositionInChildCoordinate);
      fe.calculateAt(resultRequirements, coord, result);
      auto resVector = toFieldMatrix(result.getResult(ResultType::stressResultant));
      //      stressRes[localScalarview.index(localScalarview.tree().localIndex(c))[0]] =
      //      toFieldMatrix(result.getResult(ResultType::stressResultant));
      stressResVector[localScalarview.index(localScalarview.tree().localIndex(c))[0]][0] = resVector[0][1];
      stressResVector[localScalarview.index(localScalarview.tree().localIndex(c))[0]][1] = resVector[0][2];
      stressResVector[localScalarview.index(localScalarview.tree().localIndex(c))[0]][2] = resVector[1][2];
    }
    ++ele;
  }
  auto stressResGlobalFunc
      = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 3>>(scalarBasis, stressResVector);
#endif

  // Output solution to vtk
  Dune::SubsamplingVTKWriter vtkWriter(gridView, Dune::refinementLevels(0), Dune::VTK::nonconforming);
  vtkWriter.addVertexData(wGlobalFunc, Dune::VTK::FieldInfo("w", Dune::VTK::FieldInfo::Type::scalar, 1));
  vtkWriter.addVertexData(stressResGlobalFunc,
                          Dune::VTK::FieldInfo("stressRes", Dune::VTK::FieldInfo::Type::vector, 3));
  vtkWriter.write(output_file);

  /// Create analytical solution function for the simply supported case
  const double pi         = std::numbers::pi;
  const double factor_ana = (qz * Dune::power(L, 4) * 12.0 * (1.0 - nu * nu)) / (Emod * Dune::power(thickness, 3));
  const double w_max_ana  = ((5.0 / 384.0) - (4.0 / Dune::power(pi, 5)) * (0.68562 + 0.00025)) * factor_ana;

  /// Find displacement w at the center of the plate (x=y=5.0=Lmid)
  //  auto wGlobalFunction = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double,
  //  1>>(subspaceBasis(basis, _0), w);
  auto localView    = basis.localView();
  auto localw       = localFunction(wGlobalFunc);
  double w_fe       = 0.0;
  const double Lmid = L / 2.0;
  Eigen::Vector2d c_pos;
  c_pos[0] = c_pos[1] = Lmid;

  for (auto& ele : elements(gridView)) {
    localView.bind(ele);
    localw.bind(ele);
    const auto geo = localView.element().geometry();
    if (((geo.corner(0)[0] <= Lmid) and (Lmid <= geo.corner(1)[0]))
        and ((geo.corner(0)[1] <= Lmid) and (Lmid <= geo.corner(2)[1]))) {
      const auto local_pos = geo.local(toFieldVector(c_pos));
      w_fe                 = localw(local_pos);
    }
  }
  std::cout << "#################################################";
  std::cout << std::endl << "w_max_ana: " << w_max_ana << std::endl << "w_fe     : " << w_fe << std::endl;
  std::cout << "The error is:" << sqrt(Dune::power((w_max_ana - w_fe), 2)) << std::endl;
  std::cout << "#################################################";
}

//  ################################################################
//  ###### With Automatic Differentiation for Kirchhoff Plate ######
//  ################################################################
//  std::vector<KirchhoffPlateAD<decltype(basis)>> fesAD;
//  for (auto& ele : elements(gridView)) {
//    fesAD.emplace_back(basis, ele, Emod, nu, thickness);
//  }
//  auto denseAssemblerAD = DenseFlatAssembler(basis, fesAD, dirichletFlags);
//  auto kFunctionAD = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
//         Ikarus::FErequirements req = FErequirementsBuilder()
//                                          .insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
//                                          .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
//                                          .addAffordance(Ikarus::MatrixAffordances::stiffness)
//                                          .build();
//         return denseAssemblerAD.getMatrix(req);
//  };
//
//  auto rFunctionAD = [&](auto&& disp_, auto&& lambdaLocal) -> auto& {
//    Ikarus::FErequirements req = FErequirementsBuilder()
//                                     .insertGlobalSolution(Ikarus::FESolutions::displacement, disp_)
//                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
//                                     .addAffordance(Ikarus::VectorAffordances::forces)
//                                     .build();
//    return denseAssemblerAD.getVector(req);
//  };
//  const auto& KAD = kFunctionAD(w, qz);
//  const auto& RAD = rFunctionAD(w, qz);
//  if(RAD.isApprox(R))
//    std::cout<<std::endl<<"Coinciding external forces :)"<<std::endl<<std::endl;
//
//  if(KAD.isApprox(K))
//    std::cout<<std::endl<<"Coinciding stiffness :)"<<std::endl<<std::endl;