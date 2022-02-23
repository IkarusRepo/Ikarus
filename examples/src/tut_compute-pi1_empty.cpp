//
// Created by Alex on 21.07.2021.
//
#include <../../config.h>
#include <numbers>

#include <dune/alugrid/grid.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/grid/common/boundarysegment.hh>
//#include <dune/common/function.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/Grids/GridHelper/griddrawer.h>

/// Create function for boundarySegment


int main(int argc, char** argv) {
  Dune::MPIHelper::instance(argc, argv);
  /// Create grid from 6 triangles align in unit disc

  /// Create boundary segments which map the boundaries onto the unit circle

  /// Create grid

  /// Calculate area

  /// Recalculate area with refined grid

  /// Calculate circumference and compare to pi

}