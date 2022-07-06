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

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/feBases/autodiffFE.hh>
#include <ikarus/finiteElements/feBases/scalarFE.hh>
#include <ikarus/finiteElements/mechanics/RMPTFE.hh>
#include <ikarus/linearAlgebra/nonLinearOperator.hh>
#include <ikarus/localBasis/localBasis.hh>
#include <ikarus/localFunctions/impl/standardLocalFunction.hh>
#include <ikarus/manifolds/realTuple.hh>
#include <ikarus/solver/nonLinearSolver/newtonRaphson.hh>
#include <ikarus/utils/algorithms.hh>
#include <ikarus/utils/concepts.hh>
#include <ikarus/utils/drawing/griddrawer.hh>
#include <ikarus/utils/eigenDuneTransformations.hh>
#include <ikarus/utils/observer/controlVTKWriter.hh>
#include <ikarus/utils/observer/loadControlObserver.hh>
#include <ikarus/utils/observer/nonLinearSolverLogger.hh>

namespace Ikarus {

  template <typename Basis>
  class ReissnerMindlinPlate : public RMPTFE<Basis>{
  public:
    using BaseDisp = RMPTFE<Basis>;  // Handles globalIndices function
    using GlobalIndex = typename RMPTFE<Basis>::GlobalIndex;
    using FERequirementType = FErequirements<Eigen::VectorXd>;
    using LocalView         = typename Basis::LocalView;
    using GridView         = typename Basis::GridView;

    template <typename VolumeLoad>
    ReissnerMindlinPlate(Basis& basis,
                         const typename LocalView::Element& element,
                         const double p_Emodul,
                         const double p_nu,
                         const double p_thickness,
                         const VolumeLoad& p_volumeLoad)
        : BaseDisp(basis, element),
          localView_{basis.localView()},
          Emodul{p_Emodul},
          nu{p_nu},
          thickness{p_thickness},
          volumeLoad_(p_volumeLoad){
      localView_.bind(element);
      const int order = 2 * (localView_.tree().child(Dune::Indices::_0).finiteElement().localBasis().order());
      geometry_ = localView_.element().geometry();
      localBasis_      = Ikarus::LocalBasis(localView_.tree().child(Dune::Indices::_0).finiteElement().localBasis());
      localBasis_.bind(Dune::QuadratureRules<double, Traits::mydim>::rule(localView_.element().type(), order),
                       bindDerivatives(0, 1));
    }

    static Eigen::Matrix<double, 5, 5> constitutiveMatrix(double Emod, double p_nu, double p_thickness) {
      const double factor = Emod * Dune::power(p_thickness, 3) / (12.0 * (1.0 - p_nu * p_nu));
      Eigen::Matrix<double, 5, 5> D;
      D.setZero();
      D(0, 0) = D(1, 1) = 1;
      D(0, 1) = D(1, 0) = p_nu;
      D(2, 2)           = (1 - p_nu) / 2.0;
      D *= factor;
      const double shear_term = (5.0 / 6.0) * p_thickness * Emod / (2.0 * (1.0 + p_nu));
      D(3, 3) = D(4, 4) = shear_term;
      return D;
    }

    using Traits = TraitsFromLocalView<LocalView>;

  public:
    double calculateScalar(const FERequirementType& par) const {
      std::cerr<<"Returning zero energy via calculateScalar(feRequirement)"<<std::endl;
      return 0.0;
    }

    void calculateVector(const FERequirementType& par, typename Traits::VectorType& g) const {
      const auto& lambda  = par.getParameter(Ikarus::FEParameter::loadfactor);
      const auto D        = constitutiveMatrix(Emodul, nu, thickness);
      using namespace DerivativeDirections;
      auto& ele           = localView_.element();
      auto& fe            = localView_.tree().child(Dune::Indices::_0).finiteElement();

      const auto& localBasis = fe.localBasis();
      const auto geo = localView_.element().geometry();
      const auto& rule = Dune::QuadratureRules<double, 2>::rule(ele.type(), 2 * localBasis.order());

      g.template setZero(localView_.size());
      for (const auto& gp : rule){
        const auto Jinv = Ikarus::toEigenMatrix(geo.jacobianInverseTransposed(gp.position())).transpose().eval();
        const double intElement = geo.integrationElement(gp.position()) * gp.weight();
        std::vector<Dune::FieldVector<double,1>> shapeFunctionValues;
        localBasis.evaluateFunction(gp.position(), shapeFunctionValues);
        Eigen::VectorXd N;
        N.setZero(localView_.size());
        Eigen::Vector<double, 3> fext = volumeLoad_(toEigenVector(gp.position()), lambda);
        for(size_t nn=0; nn<localView_.size(); ++nn)
          N[nn] = shapeFunctionValues[nn];
        for(size_t kk=0; kk<localView_.size(); ++kk)
          g[kk] -= N[kk] * lambda * intElement;
      }
    }

    void calculateMatrix(const FERequirementType& par, typename Traits::MatrixType& h) const {
      const auto& wGlobal = par.getSolution(Ikarus::FESolutions::displacement);
      const auto& lambda  = par.getParameter(Ikarus::FEParameter::loadfactor);
      const auto D        = constitutiveMatrix(Emodul, nu, thickness);
      using namespace DerivativeDirections;
      auto& ele           = localView_.element();
      auto& fe            = localView_.tree().child(Dune::Indices::_0).finiteElement();

      const auto& localBasis = fe.localBasis();
      const auto geo = localView_.element().geometry();
      const auto& rule = Dune::QuadratureRules<double, 2>::rule(ele.type(), 2 * localBasis.order());
      h.template setZero(localView_.size(),localView_.size());

      for (const auto& gp : rule){
        const auto Jinv = Ikarus::toEigenMatrix(geo.jacobianInverseTransposed(gp.position())).transpose().eval();
        const double intElement = geo.integrationElement(gp.position()) * gp.weight();
        std::vector<Dune::FieldVector<double, 1>> dN_xixi;
        std::vector<Dune::FieldVector<double, 1>> dN_xieta;
        std::vector<Dune::FieldVector<double, 1>> dN_etaeta;

        localBasis.partial({2, 0}, gp.position(), dN_xixi);
        localBasis.partial({1, 1}, gp.position(), dN_xieta);
        localBasis.partial({0, 2}, gp.position(), dN_etaeta);

        Eigen::VectorXd dN_xx(localView_.size());
        Eigen::VectorXd dN_yy(localView_.size());
        Eigen::VectorXd dN_xy(localView_.size());
        using Dune::power;
        for (auto i = 0U; i < localView_.size(); ++i) {
          dN_xx[i] = dN_xixi[i] * power(Jinv(0, 0), 2) + dN_etaeta[i] * power(Jinv(0, 1), 2);
          dN_yy[i] = dN_xixi[i] * power(Jinv(1, 0), 2) + dN_etaeta[i] * power(Jinv(1, 1), 2);
          dN_xy[i] = 2 * dN_xieta[i] * Jinv(0, 0) * Jinv(1, 1);
        }
        Eigen::MatrixXd bop;
        bop.setZero(3,localView_.size());
        for (auto i = 0U; i < localView_.size(); ++i) {
          bop(0,i) = dN_xx(i);
          bop(1,i) = dN_yy(i);
          bop(2,i) = dN_xy(i);
        }
        h += bop.transpose() * D * bop * intElement;
      }
    }

    LocalView localView_;
    Ikarus::LocalBasis<std::remove_cvref_t<decltype(std::declval<LocalView>().tree().child(Dune::Indices::_0).finiteElement().localBasis())>>
        localBasis_;
    typename LocalView::Element::Geometry geometry_;
    double Emodul;
    double nu;
    double thickness;
    std::function<Eigen::Vector<double, 3>(const Eigen::Vector<double, 3>&,const double&)> volumeLoad_;
  };

} // namespace Ikarus