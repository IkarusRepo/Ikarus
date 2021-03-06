/*
 * This file is part of the Ikarus distribution (https://github.com/IkarusRepo/Ikarus).
 * Copyright (c) 2022. The Ikarus developers.
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#pragma once
#include <string>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
#include <dune/grid/io/file/vtk/subsamplingvtkwriter.hh>

#include <spdlog/spdlog.h>

#include <ikarus/utils/observer/observer.hh>
#include <ikarus/utils/observer/observerMessages.hh>

template <typename Basis>  // Check basis
class ControlSubsamplingVertexVTKWriter : public IObserver<ControlMessages> {
  static constexpr int components = Basis::LocalView::Tree::CHILDREN == 0 ? 1 : Basis::LocalView::Tree::CHILDREN;

public:
  ControlSubsamplingVertexVTKWriter(const Basis& p_basis, const Eigen::VectorXd& sol, int refinementLevels = 0)
      : basis{&p_basis}, vtkWriter(p_basis.gridView(), Dune::refinementLevels(refinementLevels)), solution{&sol} {}

  auto setFieldInfo(std::string&& name, Dune::VTK::FieldInfo::Type type, std::size_t size,
                    Dune::VTK::Precision prec = Dune::VTK::Precision::float32) {
    fieldInfo      = Dune::VTK::FieldInfo(std::move(name), type, size, prec);
    isFieldInfoSet = true;
  }

  auto setFileNamePrefix(std::string&& p_name) { prefixString = std::move(p_name); }

  void updateImpl(ControlMessages message) override {
    assert(isFieldInfoSet && "You need to call setFieldInfo first!");
    switch (message) {
      case ControlMessages::SOLUTION_CHANGED: {
        auto disp = Dune::Functions::makeDiscreteGlobalBasisFunction<Dune::FieldVector<double, components>>(*basis,
                                                                                                            *solution);
        vtkWriter.addVertexData(disp, fieldInfo);
        vtkWriter.write(prefixString + std::to_string(step++));
      } break;
      default:
        break;  //   default: do nothing when notified
    }
  }

  void updateImpl(ControlMessages, double) override {}
  void updateImpl(ControlMessages, const Eigen::VectorXd&) override {}

private:
  Basis const* basis;
  Dune::SubsamplingVTKWriter<typename Basis::GridView> vtkWriter;
  Eigen::VectorXd const* solution;
  int step{0};
  Dune::VTK::FieldInfo fieldInfo{"Default", Dune::VTK::FieldInfo::Type::scalar, 1};
  std::string prefixString{};
  bool isFieldInfoSet{false};
};
