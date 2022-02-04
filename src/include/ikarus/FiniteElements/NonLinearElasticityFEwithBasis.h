
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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <concepts>
#include <iostream>

#include <dune/common/classname.hh>
#include <dune/geometry/quadraturerules.hh>
#include <dune/geometry/type.hh>

#include <spdlog/spdlog.h>

#include "ikarus/AnsatzFunctions/Lagrange.h"
#include "ikarus/utils/LinearAlgebraHelper.h"
#include <ikarus/FiniteElements/FEPolicies.h>
#include <ikarus/FiniteElements/FiniteElementFunctionConcepts.h>
#include <ikarus/FiniteElements/InterfaceFiniteElement.h>
#include <ikarus/FiniteElements/physicsHelper.h>
#include <ikarus/Geometries/GeometryType.h>
#include <ikarus/Geometries/GeometryWithExternalInput.h>
#include <ikarus/Variables/VariableDefinitions.h>
#include <ikarus/utils/LinearAlgebraTypedefs.h>

namespace Ikarus::Variable {
  class IVariable;
}

namespace Ikarus::FiniteElements {

  template <typename LocalView>
  class NonLinearElasticityFEWithLocalBasis {
  public:
    //    using Base = FEVertexDisplacement<GridElementEntityType, IndexSetType>;
    NonLinearElasticityFEWithLocalBasis(LocalView& localView, double emod, double nu)
        : localView_{&localView}, emod_{emod}, nu_{nu} {}

    using Traits = TraitsFromLocalView<LocalView>;
    template <typename ST>
    using DefoGeo = Ikarus::Geometry::GeometryWithExternalInput<ST, Traits::mydim, Traits::dimension>;

    [[nodiscard]] typename Traits::MatrixType calculateMatrix(const Eigen::VectorXd& displacements,
                                                              const double& lambda) const {
      Eigen::VectorXdual2nd dx(localView_->size());
      dx.setZero();
      auto f = [&](auto& x) { return calculateScalarImpl<autodiff::dual2nd>(displacements, lambda, x); };
      return hessian(f, wrt(dx), at(dx));
    }

    [[nodiscard]] typename Traits::VectorType calculateVector(const Eigen::VectorXd& displacements,
                                                              const double& lambda) const {
      Eigen::VectorXdual dx(localView_->size());
      dx.setZero();
      auto f = [&](auto& x) { return calculateScalarImpl<autodiff::dual>(displacements, lambda, x); };
      return gradient(f, wrt(dx), at(dx));
    }

    template <class ScalarType>
    ScalarType calculateScalarImpl([[maybe_unused]] const Eigen::VectorXd& displacements, const double& lambda,
                                   Eigen::VectorX<ScalarType>& dx) const {
      Eigen::VectorX<ScalarType> localDisp(localView_->size());
      localDisp.setZero();
      auto& first_child = localView_->tree().child(0);
      const auto& fe    = first_child.finiteElement();
      Eigen::Matrix<ScalarType, Traits::dimension, Eigen::Dynamic> disp;
      disp.setZero(Eigen::NoChange, fe.size());
      for (auto i = 0U; i < fe.size(); ++i)
        for (auto k2 = 0U; k2 < Traits::mydim; ++k2)
          disp.col(i)(k2) = dx[localView_->tree().child(k2).localIndex(i)]
                            + displacements[localView_->index(localView_->tree().child(k2).localIndex(i))[0]];
      ScalarType energy = 0.0;

      const int order = 2 * (fe.localBasis().order());
      const auto rule = Dune::QuadratureRules<double, Traits::mydim>::rule(localView_->element().type(), order);
      Eigen::Matrix3<ScalarType> C;
      C.setZero();
      C(0, 0) = C(1, 1) = 1;
      C(0, 1) = C(1, 0) = nu_;
      C(2, 2)           = (1 - nu_) / 2;
      C *= emod_ / (1 - nu_ * nu_);
      const auto geo = localView_->element().geometry();
      for (auto& gp : rule) {
        const auto J = toEigenMatrix(geo.jacobianTransposed(gp.position()));
        std::vector<Dune::FieldMatrix<double, 1, Traits::mydim>> dNM;
        fe.localBasis().evaluateJacobian(gp.position(), dNM);
        std::vector<Dune::FieldVector<double, 1>> NM;
        fe.localBasis().evaluateFunction(gp.position(), NM);
        Eigen::Vector<double, Traits::worlddim> X     = toEigenVector(geo.global(gp.position()));
        Eigen::Vector<ScalarType, Traits::worlddim> x = X;
        for (size_t i = 0; i < NM.size(); ++i)
          x += disp.col(i) * NM[i];
        Eigen::Matrix<double, Eigen::Dynamic, Traits::mydim> dN;
        dN.setZero(fe.size(), Eigen::NoChange);
        for (auto i = 0U; i < fe.size(); ++i)
          for (int j = 0; j < Traits::mydim; ++j)
            dN(i, j) = dNM[i][0][j];

        dN *= J.inverse();
        const auto H      = DefoGeo<ScalarType>::jacobianTransposed(dN, disp).transpose().eval();
        const auto E      = (0.5 * (H.transpose() + H + H.transpose() * H)).eval();
        const auto EVoigt = toVoigt(E);

        Eigen::Vector<double, Traits::worlddim> fext;
        fext.setZero();
        fext[1] = lambda;
        energy += (EVoigt.dot(C * EVoigt) - x.dot(fext)) * geo.integrationElement(gp.position()) * gp.weight();
      }
      return energy;
    }

    LocalView const* localView_;
    double emod_;
    double nu_;
  };

}  // namespace Ikarus::FiniteElements