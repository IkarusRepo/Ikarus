//
// Created by Alex on 21.07.2021.
//
#include <config.h>

#include <gmock/gmock.h>

#include "testHelpers.hh"

#include <dune/functions/functionspacebases/basistags.hh>
#include <dune/grid/yaspgrid.hh>
#include <dune/localfunctions/lagrange/pqkfactory.hh>

#include <Eigen/Core>

#include <ikarus/solver/linearSolver/geometricMultigrid/gridTransfer.h>
//#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/powerbasis.hh>

TEST(MultiGrid, TransferOperator) {
  constexpr int gridDim = 2;
  using Grid            = Dune::YaspGrid<gridDim>;
  const double L        = 2;
  const double h        = 2;
  const size_t elex     = 1;
  const size_t eley     = 1;

  Dune::FieldVector<double, gridDim> bbox = {L, h};
  std::array<int, gridDim> eles           = {elex, eley};
  auto grid                               = std::make_shared<Grid>(bbox, eles);
  auto gridView                           = grid->leafGridView();

  grid->globalRefine(1);

//  auto coarseGridView = grid->levelGridView(0);
//  auto fineGridView   = grid->levelGridView(1);
//
//  const auto& coarseIndexSet = coarseGridView.indexSet();
//  const auto& fineIndexSet   = fineGridView.indexSet();
//
//  // A factory for the shape functions
//  typedef typename Dune::PQkLocalFiniteElementCache<double, double, gridDim, 1> P1FECache;
//  typedef typename P1FECache::FiniteElementType FEType;
//  P1FECache p1FECache;

  constexpr int numDofPerNode = 2;

  using namespace Dune::Functions::BasisFactory;
  auto preBasisFactory = power<gridDim>(lagrange<1>(), FlatInterleaved());

//  Eigen::MatrixXd P(numDofPerNode * grid->size(1, gridDim), numDofPerNode * grid->size(0, gridDim));
//
//  std::vector<Dune::FieldVector<double, 1>> NcoarseEvaluated;
//
//  for (auto& coarseElement : elements(coarseGridView)) {
//    const FEType& coarseFE = p1FECache.get(coarseElement.type());
//    const int numNCoarse   = coarseFE.localBasis().size();  // Chapter 8
//    NcoarseEvaluated.resize(numNCoarse);
//
//    for (auto& childsElement : descendantElements(coarseElement, 1)) {
//      const FEType& fineFE = p1FECache.get(coarseElement.type());
//      const int numNFine   = fineFE.localBasis().size();
//
//      // CoarseIndex Set Chapter 5.6
//      const auto geoInFather = childsElement.geometryInFather();
//      const auto& fineReferenceElement
//          = Dune::ReferenceElements<double, gridDim>::general(childsElement.type());  // Chapter 5.5
//      for (int i = 0; i < numNFine; ++i) {
//        const auto fineKey                        = fineFE.localCoefficients().localKey(i);
//        const auto nodalPositionInChildCoordinate = fineReferenceElement.position(fineKey.subEntity(), fineKey.codim());
//        const auto localInFather                  = geoInFather.global(nodalPositionInChildCoordinate);
//        coarseFE.localBasis().evaluateFunction(localInFather, NcoarseEvaluated);
//        const size_t globalFine = fineIndexSet.subIndex(childsElement, fineKey.subEntity(), fineKey.codim());
//
//        for (int j = 0; j < numNCoarse; ++j) {
//          const auto coarseKey      = coarseFE.localCoefficients().localKey(j);
//          const size_t globalCoarse = coarseIndexSet.subIndex(coarseElement, coarseKey.subEntity(), coarseKey.codim());
//          P.block<numDofPerNode, numDofPerNode>(globalFine * numDofPerNode, globalCoarse * numDofPerNode)
//              = NcoarseEvaluated[j] * Eigen::Matrix<double, numDofPerNode, numDofPerNode>::Identity();
//        }
//      }
//    }
//  }

  Ikarus::GridTransfer transfer(grid);

  transfer.createOperators(preBasisFactory);



      Eigen::MatrixXd PExpected;
  PExpected.resize(numDofPerNode * grid->size(1, gridDim), numDofPerNode * grid->size(0, gridDim));
  PExpected << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.25, 0,
      0.25, 0, 0.25, 0, 0.25, 0, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0,
      0.5, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 0, 0.5, 0, 0,
      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;

  Eigen::VectorXd v = Eigen::VectorXd::Random(numDofPerNode * grid->size(0, gridDim));
  Eigen::VectorXd vfine ;
  transfer.prolongateFrom(0,v,vfine);


  EXPECT_THAT(vfine, EigenApproxEqual(PExpected*v, 1e-15));
}