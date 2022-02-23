//
// Created by Alex on 21.07.2021.
//
#include <../../config.h>
#include <matplot/matplot.h>
#include <numbers>

#include <dune/common/indices.hh>
#include <dune/functions/functionspacebases/boundarydofs.hh>
#include <dune/functions/functionspacebases/subspacebasis.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/iga/igaalgorithms.hh>
#include <dune/iga/nurbsgrid.hh>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "ikarus/Controlroutines/LoadControl.h"
#include "ikarus/LocalBasis/localBasis.h"
#include "ikarus/Solver/NonLinearSolver/NewtonRaphson.hpp"
#include "ikarus/utils/Observer/LoadControlObserver.h"
#include "ikarus/utils/Observer/controlVTKWriter.h"
#include "ikarus/utils/Observer/nonLinearSolverLogger.h"
#include "ikarus/utils/utils/algorithms.h"
#include <ikarus/Assembler/SimpleAssemblers.h>
#include <ikarus/FiniteElements/AutodiffFE.h>
#include <ikarus/FiniteElements/FEPolicies.h>
#include <ikarus/Grids/GridHelper/griddrawer.h>
#include <ikarus/LinearAlgebra/NonLinearOperator.h>
#include <ikarus/utils/concepts.h>


/// Formulate element with energy and Autodiff


int main() {
  /// Create 2D nurbs grid

  /// Increase polynomial degree in each direction

    /// Create nurbs basis with extracted preBase from grid

    /// Fix complete boundary (simply supported plate)

    /// Create finite elements

    /// Create assembler

    /// Create solution vector

    /// Create K and R

    /// Solve

    /// Output solution to vtk


    /// Create analytical solution function for the simply supported case

    /// Displacement at center of clamped square plate

    /// Calculate L_2 error for simply supported case

  /// Draw L_2 error over dofs count

}