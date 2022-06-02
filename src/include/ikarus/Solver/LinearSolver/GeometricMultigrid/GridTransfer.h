//
// Created by ac129893 on 21.03.2022.
//

#pragma once
#include <Eigen/Core>
#include <vector>
#include <memory>

#include <dune/localfunctions/lagrange/pqkfactory.hh>

namespace Ikarus{
template< typename Grid>
class GridTransfer {

  static constexpr int gridDim = Grid::dimension;

public:
  GridTransfer(const std::shared_ptr<Grid>& p_grid) : grid{p_grid}{}

  void prolongateFrom(int coarseID, const Eigen::VectorXd& coarse,Eigen::VectorXd& fine )
  {
    fine = transferMatrices[coarseID] * coarse;
  }

  void restrictTo(int coarseID, const Eigen::VectorXd& fine,Eigen::VectorXd& coarse )
  {
    coarse = transferMatrices[coarseID].transpose() * fine;
  }

  template<typename PreBasisFactory>
  void createOperators(const PreBasisFactory& preBasisFactory)
  {
    transferMatrices.resize(grid->maxLevel());

    for (int level = 0; level < grid->maxLevel(); ++level) {

      auto coarseGridView = grid->levelGridView(level);
      auto fineGridView   = grid->levelGridView(level+1);

      const auto& coarseIndexSet = coarseGridView.indexSet();
      const auto& fineIndexSet   = fineGridView.indexSet();

      auto coarseBasis = makeBasis(coarseGridView,preBasisFactory);
      constexpr int numDofPerNode = decltype(coarseBasis)::PreBasis::Node::CHILDREN;
      auto fineBasis = makeBasis(fineGridView,preBasisFactory);
      auto coarseLocalView = coarseBasis.localView();
      auto fineLocalView = fineBasis.localView();

      transferMatrices[level].resize(numDofPerNode * grid->size(level+1, gridDim), numDofPerNode * grid->size(level, gridDim));

      std::vector<Dune::FieldVector<double, 1>> NcoarseEvaluated;

      for (auto& coarseElement : elements(coarseGridView)) {

        coarseLocalView.bind(coarseElement);
        const auto& coarseFE = coarseLocalView.tree().child(0).finiteElement();
        const int numNCoarse   = coarseFE.localBasis().size();  // Chapter 8
        NcoarseEvaluated.resize(numNCoarse);

        for (auto& childsElement : descendantElements(coarseElement, 1)) {
          fineLocalView.bind(childsElement);
          const auto& fineFE = fineLocalView.tree().child(0).finiteElement();
          const int numNFine   = fineFE.localBasis().size();

          // CoarseIndex Set Chapter 5.6
          const auto geoInFather = childsElement.geometryInFather();
          const auto& fineReferenceElement
              = Dune::ReferenceElements<double, gridDim>::general(childsElement.type());  // Chapter 5.5
          for (int i = 0; i < numNFine; ++i) {
            const auto fineKey                        = fineFE.localCoefficients().localKey(i);
            const auto nodalPositionInChildCoordinate = fineReferenceElement.position(fineKey.subEntity(), fineKey.codim());
            const auto localInFather                  = geoInFather.global(nodalPositionInChildCoordinate);
            coarseFE.localBasis().evaluateFunction(localInFather, NcoarseEvaluated);
            const size_t globalFine = fineIndexSet.subIndex(childsElement, fineKey.subEntity(), fineKey.codim());

            for (int j = 0; j < numNCoarse; ++j) {
              const auto coarseKey      = coarseFE.localCoefficients().localKey(j);
              const size_t globalCoarse = coarseIndexSet.subIndex(coarseElement, coarseKey.subEntity(), coarseKey.codim());
              transferMatrices[level].block(globalFine * numDofPerNode, globalCoarse * numDofPerNode,numDofPerNode, numDofPerNode)
                  = NcoarseEvaluated[j] * Eigen::MatrixXd::Identity(numDofPerNode,numDofPerNode);
            }
          }
        }
      }
    }
  }

private:


  std::vector<Eigen::MatrixXd> transferMatrices;
  std::shared_ptr<Grid> grid;


};

}
