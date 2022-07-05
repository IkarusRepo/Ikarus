//
// Created by Alex on 21.07.2021.
//

#include <config.h>

#include "src/include/ikarus/utils/algorithms.hh"

#include <dune/alugrid/grid.hh>
#include <dune/common/parametertreeparser.hh>
#include <dune/fufem/dunepython.hh>
#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/compositebasis.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>
#include <dune/grid/yaspgrid.hh>

#include <spdlog/spdlog.h>

#include <Eigen/Core>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/micromagnetics/microMangeticsWithVectorPotential.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/solver/nonLinearSolver/trustRegion.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/functionSanityChecks.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/utils/observer/genericControlObserver.hh>

constexpr int magnetizationOrder    = 1;
constexpr int vectorPotOrder        = 1;
constexpr int gridDim               = 3;
constexpr int directorDim           = 3;
constexpr int vectorPotDim          = gridDim == 2 ? 1 : 3;
constexpr int directorCorrectionDim = directorDim - 1;

using DirectorVector  = Dune::BlockVector<Ikarus::UnitVector<double, directorDim>>;
using VectorPotVector = Dune::BlockVector<Ikarus::RealTuple<double, vectorPotDim>>;
using MultiTypeVector = Dune::MultiTypeBlockVector<DirectorVector, VectorPotVector>;

int main(int argc, char **argv) {
  Dune::MPIHelper::instance(argc, argv);
  Dune::ParameterTree parameterSet;
  Dune::ParameterTreeParser::readINITree(argv[1], parameterSet);

  const Dune::ParameterTree &gridParameters     = parameterSet.sub("GridParameters");
  const Dune::ParameterTree &controlParameters  = parameterSet.sub("ControlParameters");
  const Dune::ParameterTree &materialParameters = parameterSet.sub("MaterialParameters");

  const auto refinement      = gridParameters.get<int>("refinement");
  const auto innerRadius     = gridParameters.get<double>("innerRadius");
  const auto mshfilepath     = gridParameters.get<std::string>("mshfilepath");
  const auto loadSteps       = controlParameters.get<int>("loadSteps");
  const auto loadFactorRange = controlParameters.get<std::array<double, 2>>("loadFactorRange");

  const auto A  = materialParameters.get<double>("A");
  const auto K  = materialParameters.get<double>("K");
  const auto ms = materialParameters.get<double>("ms");

  Python::start();
  Python::Reference main = Python::import("__main__");
  Python::run("import math");

  Python::runStream() << std::endl << "import sys" << std::endl << "import os" << std::endl;

  std::string dirichletVerticesPredicateText
      = std::string("lambda x: (") + gridParameters.get<std::string>("innerDomainPredicate") + std::string(")");
  auto isInsidePredicate = Python::make_function<bool>(Python::evaluate(dirichletVerticesPredicateText));

  using namespace Ikarus;
  Ikarus::MagneticMaterial mat({.A = A, .K = K, .ms = ms});
  //  Ikarus::FiniteElements::MagneticMaterial mat({.A = 2.0e-11, .K = 1e-3, .ms = 8e2});
  const double lx = sqrt(2 * mat.A / (mat.mu0 * mat.ms * mat.ms));
  //  const double lengthUnit      = 1e-9;
  //  const double sizedom1InMeter = 30 * lengthUnit;
  //  const double sizedom1        = sizedom1InMeter / lx;
  //  const double sizedom2        = sizedom1 / 2;
  //  const double freeSpaceX      = sizedom1 * 8;
  //  const double freeSpaceY      = sizedom1 * 4;

  //  const double a = 100*1e-4/ lx;
  //  const double sizedom1        = 2*a ;
  //  const double sizedom2        = sizedom1 / 2;
  //  const double freeSpaceX        = 10*a;
  //  const double freeSpaceY        = 5*a;


  std::cout << "InnerRadius is " << innerRadius << std::endl;

  //  using Grid        = Dune::YaspGrid<gridDim>;
  //  const size_t elex = 60;
  //  const size_t eley = elex / 2;
  //  const size_t elez = 1;
  //  const double Lx   = freeSpaceX;
  //  const double Ly   = freeSpaceY;
  //  const double Lz   = freeSpaceY;
  //
  //  Dune::FieldVector<double, gridDim> bbox;
  //  std::array<int, gridDim> eles{};
  //  if constexpr (gridDim == 2) {
  //    bbox = {Lx, Ly};
  //    eles = {elex, eley};
  //  } else if constexpr (gridDim == 3) {
  //  }
  //
  //  auto grid = std::make_shared<Grid>(bbox, eles);

  using Grid = Dune::ALUGrid<gridDim, gridDim, Dune::simplex, Dune::conforming>;
  auto grid  = Dune::GmshReader<Grid>::read(mshfilepath, false, false);



  auto gridView = grid->leafGridView();


//  for (int i = 0; i < refinement; ++i) {
//    for (auto &element: elements(gridView)) {
//      auto geoCoord = element.geometry().center();
//      if (geoCoord.two_norm()<2.5 )
//        grid->mark(1, element);
//    }
//    grid->preAdapt();
//    grid->adapt();
//    grid->postAdapt();
//  }

  grid->globalRefine(refinement);
  gridView = grid->leafGridView();

  //  draw(gridView);
  spdlog::info("The exchange length is {}.", lx);
  //  spdlog::info("The domain has a length of {}.", sizedom1);

  using namespace Dune::Functions::BasisFactory;
  //  auto basisEmbedded = makeBasis(gridView, power<directorDim>(lagrange<magnetizationOrder>(),
  //  BlockedInterleaved()));
  auto basisEmbeddedC
      = makeBasis(gridView, composite(power<directorDim>(lagrange<magnetizationOrder>(), BlockedInterleaved()),
                                      power<vectorPotDim>(lagrange<vectorPotOrder>(), BlockedInterleaved()),
                                      BlockedLexicographic{}));

  auto basisRieC = makeBasis(
      gridView, composite(power<directorCorrectionDim>(lagrange<magnetizationOrder>(), FlatInterleaved()),
                          power<vectorPotDim>(lagrange<vectorPotOrder>(), FlatInterleaved()), FlatLexicographic{}));
  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basisRieC.size() << " Dofs" << std::endl;

  //  draw(gridView);

  std::vector<Ikarus::MicroMagneticsWithVectorPotential<decltype(basisEmbeddedC), decltype(basisRieC)>> fes;
  auto volumeLoad = [](auto &globalCoord, auto &lamb) {
    Eigen::Vector<double, directorDim> fext;
    fext.setZero();
    fext[0] = 0;
    fext[1] = 0;
    return fext;
  };

  int insideCounter = 0;
  for (auto &element : elements(gridView)) {
    auto geoCoord       = element.geometry().center();
    const bool isInside = isInsidePredicate(geoCoord);
    fes.emplace_back(basisEmbeddedC, basisRieC, element, mat, volumeLoad, isInside);
    if (isInside) ++insideCounter;
  }

  std::cout << "There are " << insideCounter << " Elements inside" << std::endl;

  DirectorVector mBlocked(basisEmbeddedC.size({Dune::Indices::_0}));
  for (auto &msingle : mBlocked) {
    if constexpr (directorDim == 3)
      msingle.setValue( Eigen::Vector<double, directorDim>::UnitX());
    else
      msingle.setValue(0.1 * Eigen::Vector<double, directorDim>::Random()
                       + Eigen::Vector<double, directorDim>::UnitX());
  }

  VectorPotVector aBlocked(basisEmbeddedC.size({Dune::Indices::_1}));
  for (auto &asingle : aBlocked) {
    asingle.setValue(Eigen::Vector<double, vectorPotDim>::Random());
  }

  MultiTypeVector mAndABlocked(mBlocked, aBlocked);

  std::vector<bool> dirichletFlags(basisRieC.size(), false);
  // Fix vector potential on the whole boundary
  Dune::Functions::forEachBoundaryDOF(Dune::Functions::subspaceBasis(basisRieC, Dune::Indices::_1),
                                      [&](auto &&globalIndex) { dirichletFlags[globalIndex[0]] = true; });

  // Set vector potential on the whole boundary to zero
  Dune::Functions::forEachBoundaryDOF(
      Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_1), [&](auto &&globalIndex) {
        mAndABlocked[Dune::Indices::_1][globalIndex[1]].setValue(Eigen::Vector<double, vectorPotDim>::Zero());
      });

  auto magnetBasis         = Dune::Functions::subspaceBasis(basisRieC, Dune::Indices::_0);
  auto localView           = magnetBasis.localView();
  auto seDOFs              = subEntityDOFs(magnetBasis);
  const auto &gridViewMagn = magnetBasis.gridView();
  for (auto &&element : elements(gridViewMagn)) {
    localView.bind(element);
    for (const auto &intersection : intersections(gridViewMagn, element)) {
      bool isIntersectionInside = false;

      for (int i = 0; i < intersection.geometry().corners(); ++i) {
        if (isInsidePredicate(intersection.geometry().corner(i))) {
          isIntersectionInside = true;
          break;
        } else
          isIntersectionInside = false;
      }

//      if (!isIntersectionInside) {
        for (auto localIndex : seDOFs.bind(localView, intersection))
          dirichletFlags[localView.index(localIndex)[0]] = true;
//      }
    }
  }

  auto magnetBasisBlocked   = Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_0);
  auto localView2           = magnetBasisBlocked.localView();
  auto seDOFs2              = subEntityDOFs(magnetBasisBlocked);
  const auto &gridViewMagn2 = magnetBasisBlocked.gridView();
  for (auto &&element : elements(gridViewMagn2)) {
    localView2.bind(element);
    for (const auto &intersection : intersections(gridViewMagn2, element))
      for (auto localIndex : seDOFs2.bind(localView2, intersection)) {
        bool isIntersectionInside = false;

        for (int i = 0; i < intersection.geometry().corners(); ++i) {
          if (isInsidePredicate(intersection.geometry().corner(i))) {
            isIntersectionInside = true;
            break;
          } else
            isIntersectionInside = false;
        }

        if (!isIntersectionInside) {
          auto b = mAndABlocked[Dune::Indices::_0][localView2.index(localIndex)[1]].begin();
          auto e = mAndABlocked[Dune::Indices::_0][localView2.index(localIndex)[1]].end();
          std::fill(b, e, 0.0);
        }
      }
  }

  auto denseAssembler  = DenseFlatAssembler(basisRieC, fes, dirichletFlags);
  auto sparseAssembler = SparseFlatAssembler(basisRieC, fes, dirichletFlags);
  double lambda        = 0.0;

  auto residualFunction = [&](auto &&disp, auto &&lambdaLocal) -> auto & {
    Ikarus::FErequirements req = FErequirementsBuilder<MultiTypeVector>()
                                     .insertGlobalSolution(Ikarus::FESolutions::magnetizationAndVectorPotential, disp)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::VectorAffordances::microMagneticForces)
                                     .build();
    return denseAssembler.getReducedVector(req);
  };

  auto hessianFunction = [&](auto &&disp, auto &&lambdaLocal) -> auto & {
    Ikarus::FErequirements req = FErequirementsBuilder<MultiTypeVector>()
                                     .insertGlobalSolution(Ikarus::FESolutions::magnetizationAndVectorPotential, disp)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::MatrixAffordances::microMagneticHessian)
                                     .build();
    return sparseAssembler.getReducedMatrix(req);
  };

  auto energyFunction = [&](auto &&disp, auto &&lambdaLocal) -> auto {
    Ikarus::FErequirements req = FErequirementsBuilder<MultiTypeVector>()
                                     .insertGlobalSolution(Ikarus::FESolutions::magnetizationAndVectorPotential, disp)
                                     .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
                                     .addAffordance(Ikarus::ScalarAffordances::microMagneticPotentialEnergy)
                                     .build();

    return denseAssembler.getScalar(req);
  };

  auto nonLinOp = Ikarus::NonLinearOperator(linearAlgebraFunctions(energyFunction, residualFunction, hessianFunction),
                                            parameter(mAndABlocked, lambda));

  const auto& K = nonLinOp.secondDerivative();
  std::cout<<"Percentage of non-zeros in matrix is: "<<static_cast<double>(K.nonZeros())/static_cast<double>(Dune::power(K.rows,2))<<std::endl;
  auto updateFunction = std::function([&](MultiTypeVector &multiTypeVector, const Eigen::VectorXd &d) {
    auto dFull = denseAssembler.createFullVector(d);
    multiTypeVector += dFull;
  });

//  checkGradient(nonLinOp, {.draw = false, .writeSlopeStatement = true}, updateFunction);
//  checkHessian(nonLinOp, {.draw = false, .writeSlopeStatement = true}, updateFunction);

  auto nr = Ikarus::makeTrustRegion(nonLinOp, updateFunction);
  //  auto nr = Ikarus::makeTrustRegion< decltype(nonLinOp),PreConditioner::DiagonalPreconditioner>(nonLinOp,
  //  updateFunction);
  nr->setup({.verbosity = 1,
             .maxiter   = 100000,
             .grad_tol  = 1e-8,
             .corr_tol  = 1e-16,
             .useRand   = false,
             .rho_reg   = 1e6,
             .Delta0    = 1000});

  auto lc = Ikarus::LoadControl(nr, loadSteps, loadFactorRange);

  auto scalarMagnBasis          = makeBasis(gridView, lagrangeDG<magnetizationOrder>());
  auto localViewScalarMagnBasis = scalarMagnBasis.localView();

  std::vector<double> gradMNodalRes(scalarMagnBasis.size());
  std::vector<double> divAfieldNodalRes(scalarMagnBasis.size());
  std::vector<Dune::FieldVector<double, 3>> BfieldNodalRes(scalarMagnBasis.size());
  std::vector<Dune::FieldVector<double, 3>> HfieldNodalRes(scalarMagnBasis.size());

  auto writerObserver = std::make_shared<Ikarus::GenericControlObserver>(ControlMessages::STEP_ENDED, [&](auto i) {
    auto mGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, directorDim>>(
        Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_0), mAndABlocked);
    auto AGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, vectorPotDim>>(
        Dune::Functions::subspaceBasis(basisEmbeddedC, Dune::Indices::_1), mAndABlocked);

    Dune::VTKWriter vtkWriter(gridView, Dune::VTK::nonconforming);
    vtkWriter.addVertexData(mGlobalFunc, Dune::VTK::FieldInfo("m", Dune::VTK::FieldInfo::Type::vector, directorDim));
    vtkWriter.addVertexData(AGlobalFunc, Dune::VTK::FieldInfo("A", Dune::VTK::FieldInfo::Type::vector, vectorPotDim));

    auto resultRequirements
        = Ikarus::ResultRequirementsBuilder<MultiTypeVector>()
              .insertGlobalSolution(Ikarus::FESolutions::magnetizationAndVectorPotential, mAndABlocked)
              .insertParameter(Ikarus::FEParameter::loadfactor, lambda)
              .addResultRequest(ResultType::gradientNormOfMagnetization, ResultType::BField, ResultType::HField,
                                ResultType::divergenceOfVectorPotential)
              .build();
    auto localmFunction = localFunction(mGlobalFunc);

    auto ele = elements(gridView).begin();
    ResultTypeMap<double> result;
    for (auto &fe : fes) {
      localViewScalarMagnBasis.bind(*ele);
      const auto &fe2              = localViewScalarMagnBasis.tree().finiteElement();
      const auto &referenceElement = Dune::ReferenceElements<double, gridDim>::general(ele->type());
      for (auto c = 0UL; c < fe2.size(); ++c) {
        const auto fineKey                        = fe2.localCoefficients().localKey(c);
        const auto nodalPositionInChildCoordinate = referenceElement.position(fineKey.subEntity(), fineKey.codim());

        auto coord = toEigenVector(nodalPositionInChildCoordinate);
        fe.calculateAt(resultRequirements, coord, result);
        gradMNodalRes[localViewScalarMagnBasis.index(localViewScalarMagnBasis.tree().localIndex(c))[0]]
            = result.getResult(ResultType::gradientNormOfMagnetization)(0, 0);
        BfieldNodalRes[localViewScalarMagnBasis.index(localViewScalarMagnBasis.tree().localIndex(c))[0]]
            = toFieldVector(result.getResult(ResultType::BField));
        HfieldNodalRes[localViewScalarMagnBasis.index(localViewScalarMagnBasis.tree().localIndex(c))[0]]
            = toFieldVector(result.getResult(ResultType::HField));
        divAfieldNodalRes[localViewScalarMagnBasis.index(localViewScalarMagnBasis.tree().localIndex(c))[0]]
            = result.getResult(ResultType::divergenceOfVectorPotential)(0, 0);
      }
      ++ele;
    }

    auto gradmGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(scalarMagnBasis, gradMNodalRes);
    auto divAGlobalFunc  = Dune::Functions::makeDiscreteGlobalBasisFunction<double>(scalarMagnBasis, divAfieldNodalRes);
    auto bFieldGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 3>>(
        scalarMagnBasis, BfieldNodalRes);
    auto hFieldGlobalFunc = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, 3>>(
        scalarMagnBasis, HfieldNodalRes);

    vtkWriter.addVertexData(gradmGlobalFunc, Dune::VTK::FieldInfo("gradMNorm", Dune::VTK::FieldInfo::Type::scalar, 1));
    vtkWriter.addVertexData(bFieldGlobalFunc, Dune::VTK::FieldInfo("B", Dune::VTK::FieldInfo::Type::vector, 3));
    vtkWriter.addVertexData(hFieldGlobalFunc, Dune::VTK::FieldInfo("H", Dune::VTK::FieldInfo::Type::vector, 3));
    vtkWriter.addVertexData(divAGlobalFunc, Dune::VTK::FieldInfo("divA", Dune::VTK::FieldInfo::Type::scalar, 1));
    auto isInsideFunc = Dune::Functions::makeAnalyticGridViewFunction(isInsidePredicate, gridView);
    vtkWriter.addCellData(isInsideFunc, Dune::VTK::FieldInfo("isInside", Dune::VTK::FieldInfo::Type::scalar, 1));
    vtkWriter.write(std::string("Magnet") + std::to_string(i));

    std::cout << "Writing " << std::string("Magnet") + std::to_string(i) << " to file. The load factor is " << lambda
              << std::endl;
  });
  lc.subscribeAll(writerObserver);
  lc.run();
  nonLinOp.update<0>();
  const double volume = std::numbers::pi * Dune::power(innerRadius, 2);
  spdlog::info("Energy: {}, Volume: {}, Energy/(0.5*V): {}", nonLinOp.value(), volume,
               nonLinOp.value() / (0.5 * volume));


  auto energyFunctionNew = [&](auto &&disp, auto &&lambdaLocal,auto&& req) -> auto {
    Ikarus::FErequirements reqs = FErequirementsBuilder<MultiTypeVector>()
        .insertGlobalSolution(Ikarus::FESolutions::magnetizationAndVectorPotential, disp)
        .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLocal)
        .addAffordance(std::move(req))
        .build();

    return denseAssembler.getScalar(reqs);
  };


  std::cout<<"Total Energy: "<< energyFunctionNew(mAndABlocked,lambda,Ikarus::ScalarAffordances::microMagneticPotentialEnergy)<<std::endl;
  std::cout<<"Total Energy: "<< energyFunction(mAndABlocked,lambda)<<std::endl;
  std::cout<<"Total EnergyFromNonOp: "<< nonLinOp.value()<<std::endl;
  std::cout<<"Exchange Energy: "<< energyFunctionNew(mAndABlocked,lambda,Ikarus::ScalarAffordances::microMagneticExchangeEnergy)<<std::endl;
  std::cout<<"Total MagnetoStatic Energy: "<< energyFunctionNew(mAndABlocked,lambda,Ikarus::ScalarAffordances::microMagneticTotalMagnetoStaticEnergy)<<std::endl;
  std::cout<<"Inside MagnetoStatic Energy: "<< energyFunctionNew(mAndABlocked,lambda,Ikarus::ScalarAffordances::microMagneticInsideMagnetoStaticEnergy)<<std::endl;
  std::cout<<"Inside HtimesM MagnetoStatic Energy: "<< energyFunctionNew(mAndABlocked,lambda,Ikarus::ScalarAffordances::microMagneticInsideFromMTimesHMagnetoStaticEnergy)<<std::endl;


}