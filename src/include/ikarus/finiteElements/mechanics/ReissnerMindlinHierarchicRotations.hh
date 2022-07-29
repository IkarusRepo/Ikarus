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

#include <ikarus/finiteElements/feBases/powerBasisFE.hh>
#include <ikarus/assembler/simpleAssemblers.hh>
#include <ikarus/controlRoutines/loadControl.hh>
#include <ikarus/finiteElements/feBases/autodiffFE.hh>
#include <ikarus/finiteElements/feBases/scalarFE.hh>
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
  class ReissnerMindlinPlateHierarchicRotations : public PowerBasisFE<Basis>{
  public:
    using BaseDisp = PowerBasisFE<Basis>;  // Handles globalIndices function
    using GlobalIndex = typename PowerBasisFE<Basis>::GlobalIndex;
    using FERequirementType = FErequirements<Eigen::VectorXd>;
    using ResultRequirementsType = ResultRequirements<Eigen::VectorXd>;
    using LocalView         = typename Basis::LocalView;
    using GridView         = typename Basis::GridView;

    template <typename VolumeLoad>
    ReissnerMindlinPlateHierarchicRotations(Basis& basis,
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
      const int order = 2 * (localView_.tree().child(0).finiteElement().localBasis().order());
      localBasis_      = Ikarus::LocalBasis(localView_.tree().child(0).finiteElement().localBasis());
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
      auto& fe            = localView_.tree().child(0).finiteElement();

      const auto& localBasis = fe.localBasis();
      const auto geo = localView_.element().geometry();
      const auto& rule = Dune::QuadratureRules<double, 2>::rule(ele.type(), 2 * localBasis.order());
      Eigen::Vector<double, 3> fext = volumeLoad_(lambda);
      g.template setZero(localView_.size());
      for (const auto& gp : rule){
        const auto JinvT = geo.jacobianInverseTransposed(gp.position());
        const double intElement = geo.integrationElement(gp.position()) * gp.weight();
        std::vector<Dune::FieldMatrix<double,1,2>> referenceGradients;
        localBasis.evaluateJacobian(gp.position(),referenceGradients);
        std::vector<Dune::FieldVector<double,2>> gradients(referenceGradients.size());
        for (size_t i=0; i<gradients.size(); i++)
          JinvT.mv(referenceGradients[i][0],gradients[i]);
        std::vector<Dune::FieldVector<double,1>> shapeFunctionValues;
        localBasis.evaluateFunction(gp.position(), shapeFunctionValues);
        Eigen::VectorXd dNdx = Eigen::VectorXd::Zero(shapeFunctionValues.size());
        Eigen::VectorXd dNdy = Eigen::VectorXd::Zero(shapeFunctionValues.size());
        for(size_t i=0;i<shapeFunctionValues.size();i++)
        {
          dNdx[i] = gradients[i][0];
          dNdy[i] = gradients[i][1];
        }
        Eigen::MatrixXd N;
        N.setZero(3, localView_.size());
        for(size_t nn=0; nn<shapeFunctionValues.size(); ++nn){
          N(0,3*nn) = shapeFunctionValues[nn];
          N(1,3*nn) = -dNdx[nn];
          N(1,3*nn+1) = shapeFunctionValues[nn];
          N(2,3*nn) = -dNdy[nn];
          N(2,3*nn+2) = shapeFunctionValues[nn];
        }
        g -= N.transpose() * fext * intElement;
      }
    }

    void calculateMatrix(const FERequirementType& par, typename Traits::MatrixType& h) const {
      const auto& lambda  = par.getParameter(Ikarus::FEParameter::loadfactor);
      const auto D        = constitutiveMatrix(Emodul, nu, thickness);
      using namespace DerivativeDirections;
      auto& ele           = localView_.element();
      auto& fe            = localView_.tree().child(0).finiteElement();

      const auto& localBasis = fe.localBasis();
      const auto geo = localView_.element().geometry();
      const auto& rule = Dune::QuadratureRules<double, 2>::rule(ele.type(), 2 * localBasis.order());
      h.template setZero(localView_.size(),localView_.size());

      for (const auto& gp : rule){
        const auto JinvT = geo.jacobianInverseTransposed(gp.position());
        const double intElement = geo.integrationElement(gp.position()) * gp.weight();
        std::vector<Dune::FieldMatrix<double,1,2>> referenceGradients;
        localBasis.evaluateJacobian(gp.position(),referenceGradients);
        std::vector<Dune::FieldVector<double,2>> gradients(referenceGradients.size());

        for (size_t i=0; i<gradients.size(); i++)
          JinvT.mv(referenceGradients[i][0],gradients[i]);

        std::vector<Dune::FieldVector<double,1>> shapeFunctionValues;
        localBasis.evaluateFunction(gp.position(), shapeFunctionValues);

        Eigen::VectorXd dNdx = Eigen::VectorXd::Zero(shapeFunctionValues.size());
        Eigen::VectorXd dNdy = Eigen::VectorXd::Zero(shapeFunctionValues.size());
        for(size_t i=0;i<shapeFunctionValues.size();i++)
        {
          dNdx[i] = gradients[i][0];
          dNdy[i] = gradients[i][1];
        }

        const auto Jinv = Ikarus::toEigenMatrix(JinvT).transpose().eval();
        std::vector<Dune::FieldVector<double, 1>> dN_xixi;
        std::vector<Dune::FieldVector<double, 1>> dN_xieta;
        std::vector<Dune::FieldVector<double, 1>> dN_etaeta;

        localBasis.partial({2, 0}, gp.position(), dN_xixi);
        localBasis.partial({1, 1}, gp.position(), dN_xieta);
        localBasis.partial({0, 2}, gp.position(), dN_etaeta);

        Eigen::VectorXd dN_xx(fe.size());
        Eigen::VectorXd dN_yy(fe.size());
        Eigen::VectorXd dN_xy(fe.size());

        using Dune::power;
        for (auto i = 0U; i < fe.size(); ++i) {
          dN_xx[i] = dN_xixi[i] * power(Jinv(0, 0), 2) + dN_etaeta[i] * power(Jinv(0, 1), 2);
          dN_yy[i] = dN_xixi[i] * power(Jinv(1, 0), 2) + dN_etaeta[i] * power(Jinv(1, 1), 2);
          dN_xy[i] = 2 * dN_xieta[i] * Jinv(0, 0) * Jinv(1, 1);
        }

        Eigen::MatrixXd bop;
        bop.setZero(5,localView_.size());
        for (auto i = 0U; i < shapeFunctionValues.size(); ++i) {
          bop(0, 3*i) = -dN_xx(i);
          bop(0, 3*i+1) = dNdx(i);

          bop(1, 3*i) = -dN_yy(i);
          bop(1, 3*i+2) = dNdy(i);

          bop(2, 3*i) = -dN_xy(i);
          bop(2, 3*i+1) = dNdy(i);
          bop(2, 3*i+2) = dNdx(i);

          bop(3, 3*i+1) = shapeFunctionValues[i];
          bop(4, 3*i+2) = shapeFunctionValues[i];
        }
        h += bop.transpose() * D * bop * intElement;
      }
    }

    void calculateAt(const ResultRequirementsType& req, const Eigen::Vector<double, Traits::mydim>& local,
                     ResultTypeMap<double>& result) const {
      using namespace Dune::Indices;
      const auto& disp = req.getSolution(Ikarus::FESolutions::displacement);
      const auto D     = constitutiveMatrix(Emodul, nu, thickness);
      auto& fe         = localView_.tree().child(0).finiteElement();
      const auto& localBasis = fe.localBasis();
      const auto geo   = localView_.element().geometry();
      auto gp          = toFieldVector(local);
      Eigen::VectorXd local_disp;
      local_disp.setZero(localView_.size());

      int disp_counter = 0;
      for (size_t i=0; i < fe.size(); ++i)
        for (size_t j=0; j < 3; ++j){
          auto globalIndex = localView_.index(localView_.tree().child(j).localIndex(i));
          local_disp[disp_counter] = disp[globalIndex];
          disp_counter++;
        }

      const auto JinvT = geo.jacobianInverseTransposed(gp);
      std::vector<Dune::FieldMatrix<double,1,2>> referenceGradients;
      localBasis.evaluateJacobian(gp,referenceGradients);
      std::vector<Dune::FieldVector<double,2>> gradients(referenceGradients.size());

      for (size_t i=0; i<gradients.size(); i++)
        JinvT.mv(referenceGradients[i][0],gradients[i]);

      std::vector<Dune::FieldVector<double,1>> shapeFunctionValues;
      localBasis.evaluateFunction(gp, shapeFunctionValues);

      Eigen::VectorXd dNdx = Eigen::VectorXd::Zero(shapeFunctionValues.size());
      Eigen::VectorXd dNdy = Eigen::VectorXd::Zero(shapeFunctionValues.size());
      for(size_t i=0;i<shapeFunctionValues.size();i++)
      {
        dNdx[i] = gradients[i][0];
        dNdy[i] = gradients[i][1];
      }

      const auto Jinv = Ikarus::toEigenMatrix(JinvT).transpose().eval();
      std::vector<Dune::FieldVector<double, 1>> dN_xixi;
      std::vector<Dune::FieldVector<double, 1>> dN_xieta;
      std::vector<Dune::FieldVector<double, 1>> dN_etaeta;

      localBasis.partial({2, 0}, gp, dN_xixi);
      localBasis.partial({1, 1}, gp, dN_xieta);
      localBasis.partial({0, 2}, gp, dN_etaeta);

      Eigen::VectorXd dN_xx(fe.size());
      Eigen::VectorXd dN_yy(fe.size());
      Eigen::VectorXd dN_xy(fe.size());

      using Dune::power;
      for (auto i = 0U; i < fe.size(); ++i) {
        dN_xx[i] = dN_xixi[i] * power(Jinv(0, 0), 2) + dN_etaeta[i] * power(Jinv(0, 1), 2);
        dN_yy[i] = dN_xixi[i] * power(Jinv(1, 0), 2) + dN_etaeta[i] * power(Jinv(1, 1), 2);
        dN_xy[i] = 2 * dN_xieta[i] * Jinv(0, 0) * Jinv(1, 1);
      }

      Eigen::MatrixXd bop;
      bop.setZero(5,localView_.size());
      for (auto i = 0U; i < shapeFunctionValues.size(); ++i) {
        bop(0, 3*i) = -dN_xx(i);
        bop(0, 3*i+1) = dNdx(i);

        bop(1, 3*i) = -dN_yy(i);
        bop(1, 3*i+2) = dNdy(i);

        bop(2, 3*i) = -dN_xy(i);
        bop(2, 3*i+1) = dNdy(i);
        bop(2, 3*i+2) = dNdx(i);

        bop(3, 3*i+1) = shapeFunctionValues[i];
        bop(4, 3*i+2) = shapeFunctionValues[i];
      }

      Eigen::Vector<double, 5> req_res;
      req_res.setZero();
      req_res = D * bop * local_disp;

      Eigen::Matrix<double, 3, 3> sent_res;
      sent_res(0, 0) = req_res[0];
      sent_res(1, 1) = req_res[1];
      sent_res(0, 1) = sent_res(1, 0) = req_res[2];
      sent_res(0, 2) = sent_res(2, 0) = req_res[3];
      sent_res(1, 2) = sent_res(2, 1) = req_res[4];

      typename ResultTypeMap<double>::ResultArray resv;
      if (req.isResultRequested(ResultType::stressResultant)) {
        resv.resize(3, 3);
        resv = sent_res;
        result.insertOrAssignResult(ResultType::stressResultant, resv);
      }
    }

    LocalView localView_;
    Ikarus::LocalBasis<std::remove_cvref_t<decltype(std::declval<LocalView>().tree().child(0).finiteElement().localBasis())>>
        localBasis_;
    double Emodul;
    double nu;
    double thickness;
//    std::function<Eigen::Vector<double, 3>(const Eigen::Vector<double, 3>&,const double&)> volumeLoad_;
    std::function<Eigen::Vector<double, 3>(const double&)> volumeLoad_;
  };

} // namespace Ikarus