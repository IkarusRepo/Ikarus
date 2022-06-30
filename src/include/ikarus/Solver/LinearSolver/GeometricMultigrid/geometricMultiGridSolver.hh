//
// Created by ac129893 on 30.06.2022.
//

#pragma once

#include "gridTransfer.h"

#include <utility>

#include <dune/functions/functionspacebases/boundarydofs.hh>

#include <Eigen/Core>

#include "ikarus/assembler/simpleAssemblers.hh"
#include <ikarus/finiteElements/feRequirements.hh>
#include <ikarus/solver/linearSolver/linearSolver.hh>

namespace Ikarus {
  template <typename Grid, typename PreBasisFactory, typename FEContainer>
  class GeometricMultiGridSolver {
    using RequirementType = typename FEContainer::value_type::FERequirementType;

    using BasisType
        = decltype(makeBasis(std::declval<typename Grid::LevelGridView>(), std::declval<PreBasisFactory>()));
  public:
    GeometricMultiGridSolver(const Grid* grid, const PreBasisFactory& preBasisFactory,
                             const FEContainer& feVectorCoarse)
        : transfer{grid}, directSolver{SolverTypeTag::sd_SimplicialLDLT} {
      fes.resize(grid->maxLevel());
      fes[0] = feVectorCoarse;
      for (int level = 0; level < grid->maxLevel(); ++level) {
        auto coarseGridView = grid->levelGridView(level);
        auto fineGridView   = grid->levelGridView(level + 1);

        auto coarseBasis = makeBasis(coarseGridView, preBasisFactory);
        auto fineBasis   = makeBasis(fineGridView, preBasisFactory);

        auto coarseElement = elements(coarseGridView).begin();
        for (auto& coarseFe : fes[0]) {
          {
            for (auto& childsElement : descendantElements(*coarseElement, 1)) {
              fes[level + 1].emplace_back(fineBasis, childsElement, coarseFe.settings());
            }
            ++coarseElement;
          }
        }
      }

      for (int level = 0; level < grid->maxLevel(); ++level) {
        auto gridView = grid->levelGridView(level);

        auto basis          = std::make_shared<BasisType>(makeBasis(gridView, preBasisFactory));
        auto dirichletFlags = std::make_shared<std::vector<bool>>((*basis).size(), false);

        Dune::Functions::forEachBoundaryDOF(*basis, [&](auto&& localIndex, auto&& localView, auto&& intersection) {
          if (std::abs(intersection.geometry().center()[1]) < 1e-8) {
            (*dirichletFlags)[localView.index(localIndex)[0]] = true;
          }
        });
        assemblers.emplace_back(basis, fes[level], dirichletFlags);
      }

      transfer.createOperators(preBasisFactory);
      iterativeSolver.setMaxIterations(100);
    }

    mutable Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> iterativeSolver;
    Ikarus::GridTransfer<Grid> transfer;
    mutable ILinearSolver<double> directSolver;
    mutable RequirementType requirementType;


    mutable std::vector<Ikarus::SparseFlatAssembler<BasisType, FEContainer>> assemblers;
    std::vector<FEContainer> fes;
    std::vector<BasisType> bases;

  private:
  public:
    template <typename ScalarType>
    void smoothing(const Eigen::VectorX<ScalarType>& dFineFull, const Eigen::VectorX<ScalarType>& RfineRed,
                   Eigen::VectorX<ScalarType>& dFineRed) const {
      assemblers[1].createReducedVector(dFineFull, dFineRed);
      dFineRed = iterativeSolver.solve(RfineRed);
    }

    void transformToFineFull(const Eigen::VectorX<double>& dFineRed,Eigen::VectorX<double>& dFineFull)
    {
      assemblers[1].createFullVector(dFineRed, dFineFull);
    }

    template <typename ScalarType>
    void solve(Eigen::VectorX<ScalarType>& dFineRed, const Eigen::VectorX<ScalarType>& RfineRed_) const {
      //    Eigen::MatrixXd KcoarseRed,KfineRed;
      Eigen::VectorXd RcoarseRed, dCoarseRed, dCoarseFull,RfineFull,RcoarseFull;
      dCoarseFull.setZero(assemblers[0].size());
      Eigen::VectorXd dFineFull;
      dFineFull.setZero(assemblers[1].size());

      double lambdaLoad = 1;

      requirementType = Ikarus::FErequirementsBuilder()
                            .insertGlobalSolution(Ikarus::FESolutions::displacement, dFineFull)
                            .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLoad)
                            .addAffordance(Ikarus::VectorAffordances::forces)
                            .build();
      Eigen::VectorXd RfineRed = -assemblers[1].getReducedVector(requirementType);
      requirementType   = Ikarus::FErequirementsBuilder()
                            .insertGlobalSolution(Ikarus::FESolutions::displacement, dCoarseFull)
                            .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLoad)
                            .addAffordance(Ikarus::VectorAffordances::forces)
                            .build();
      auto& KcoarseRed         = assemblers[0].getReducedMatrix(requirementType);

      directSolver.compute(KcoarseRed);
      assemblers[1].createFullVector(RfineRed, RfineFull);
      transfer.restrictTo(0, RfineFull, RcoarseFull);
      assemblers[0].createReducedVector(RcoarseFull, RcoarseRed);

      directSolver.solve(dCoarseRed, RcoarseRed);

      assemblers[0].createFullVector(dCoarseRed, dCoarseFull);

      transfer.prolongateFrom(0, dCoarseFull, dFineFull);
      requirementType = Ikarus::FErequirementsBuilder()
                            .insertGlobalSolution(Ikarus::FESolutions::displacement, dFineFull)
                            .insertParameter(Ikarus::FEParameter::loadfactor, lambdaLoad)
                            .addAffordance(Ikarus::VectorAffordances::forces)
                            .build();
      auto& KfineRed = assemblers[1].getReducedMatrix(requirementType);
      iterativeSolver.compute(KfineRed);
      assemblers[1].createReducedVector(dFineFull, dFineRed);
      // Pre-Smoothing
      smoothing(dFineFull, RfineRed, dFineRed);

      Eigen::VectorXd residualMGFineRed = RfineRed - KfineRed * dFineRed;
      Eigen::VectorXd residualMGFineFull;
      assemblers[1].createFullVector(residualMGFineRed, residualMGFineFull);
      Eigen::VectorXd residualMGCoarseFull;
      transfer.restrictTo(0, residualMGFineFull, residualMGCoarseFull);
      Eigen::VectorXd residualMGCoarseRed;
      assemblers[0].createReducedVector(residualMGCoarseFull, residualMGCoarseRed);

      Eigen::VectorXd eCoarseRed;
      directSolver.solve(eCoarseRed, residualMGCoarseRed);
      Eigen::VectorXd eCoarseFull;
      assemblers[0].createFullVector(eCoarseRed, eCoarseFull);
      Eigen::VectorXd eFineFull;
      transfer.prolongateFrom(0, eCoarseFull, eFineFull);

      dFineFull += eFineFull;
      assemblers[1].createReducedVector(dFineFull, dFineRed);
      // Post smoothing
      smoothing(dFineFull, RfineRed, dFineRed);

      //
      //    % VON MITTE AUF FEIN
      //        %Verffeinerung auf drittgröbstes Netz
      //            [discretization_2,P_2,R_2,BC_2] = refinement_test(discretization_1,BC_1);
      //    %Umrechnung der ersten Lösung auf das feinere Netz
      //            sol_2=P_2*solution_neu.D;
      //    %Iterative Lösung auf feinstem Gitter
      //            [solution_2,BC_2] = solve_stat_lin_iter(analysis,discretization_2,material,BC_2,sol_2);
      //    [soll_ex_fein,BC_ex_fein]=solve_stat_lin_dir(analysis,discretization_2,material,BC_2);
      //    %Zwischendurch Postprocess mit Startlösung
      //        %solution_2.D=sol_2;
      //    Postprocess(output,analysis,discretization_2,solution_2,BC_2,material);
      //    %Residuumsgleichung
      //            Residuum_2=solution_2.F_ext_red-solution_2.K_red*solution_2.D_red;
      //    Residuum_2_full=expansion_vector(Residuum_2,solution_2.dirichlet_flag,discretization_2.n_dof);
      //    %Umrechnung Residuum auf mittleres Gitter
      //            Residuum_1=R_2*Residuum_2_full;
      //    Residuum_1_red=reduktion_vector(Residuum_1,solution_1.dirichlet_flag,solution_1.dirichlet_n);
      //    %Iterative Lösung auf mittlerem Gitter
      //            [solution_1_iter] = solve_iter(analysis,solution_1.K_red,Residuum_1_red);
      //    e_full_1=expansion_vector(solution_1_iter.D_red,solution_1.dirichlet_flag,discretization_1.n_dof);
      //    %Residuumsgleichung
      //            Residuum_1=Residuum_1-solution_1.K*e_full_1;
      //    %Residuum_0_full=expansion_vector(Residuum_0,solution_2.dirichlet_flag,discretization_2.n_dof);
      //    %Umrechnung Residuum auf gröbstes Gitter
      //            Residuum_0=R_1*Residuum_1;
      //    %reduzieren von residuum
      //            Residuum_0_red=reduktion_vector(Residuum_0,solution_1.dirichlet_flag,solution_1.dirichlet_n);
      //    %Direkte Lösung auf gröbstem Gitter
      //            fprintf('Direkte Loesung...\n');
      //    tic
      //        e_red=solution.K_red\Residuum_0_red;
      //    toc
      //        %Erweiterung und Prolongation auf feineres Gitter
      //            e_full=expansion_vector(e_red,solution.dirichlet_flag,discretization.n_dof);
      //    e_1=P_1*e_full;
      //    %Addition zu iterativem Ergebnis von vorher
      //            D_1=e_full_1+e_1;
      //    solution_neu=solution_1;
      //    solution_neu.D=D_1;
      //    %Postsmoothing
      //            D_1_red_post=reduktion_vector(D_1,solution_1.dirichlet_flag,solution_1.dirichlet_n);
      //    [solution_post] = solve_iter_start(analysis,solution_1.K_red,Residuum_1_red,D_1_red_post);
      //    D_1=expansion_vector(solution_post.D_red,solution_1.dirichlet_flag,discretization_1.n_dof);
      //    % D_1=solution_post.D;
      //    %Erweiterung und Prolongation auf feineres Gitter
      //        %e_full=expansion_vector(e_red,solution.dirichlet_flag,discretization.n_dof);
      //    e_2=P_2*D_1;
      //    %Addition zu iterativem Ergebnis von vorher
      //            D_2=solution_2.D+e_2;
      //    solution_neu=solution_2;
      //    solution_neu.D=D_2;
      //    %finales Post-smoothing
      //            [solution_final,BC_final] = solve_stat_lin_iter(analysis,discretization_2,material,BC_2,D_2);
      //    %solution_final.D=2*solution_final.D;
      //    %Zwischendurch Postprocess
      //            Postprocess(output,analysis,discretization_2,solution_final,BC_2,material);
    }
  };
}  // namespace Ikarus
