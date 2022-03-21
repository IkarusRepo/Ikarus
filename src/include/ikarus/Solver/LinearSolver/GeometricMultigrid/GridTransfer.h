//
// Created by ac129893 on 21.03.2022.
//

#pragma once
#include <Eigen/Core>
#include <vector>
#include <memory>

namespace Ikarus{
template< typename Grid>
class GridTransfer {


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

  void createOperators()
  {

    // ask grid how many levels exist
    //resize pVector accordingly
    //code from Test

    // write new tests in test file
  }

private:


  std::vector<Eigen::MatrixXd> transferMatrices;
  std::shared_ptr<Grid> grid;


};

}
