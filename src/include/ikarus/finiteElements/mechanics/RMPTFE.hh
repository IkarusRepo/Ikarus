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

#include "src/include/ikarus/finiteElements/feTraits.hh"
namespace Ikarus {
 template <typename Basis>
 class RMPTFE {
 public:
   using RootBasis   = Basis;
   using LocalView   = typename Basis::LocalView;
   using GlobalIndex = typename LocalView::MultiIndex;
   explicit RMPTFE(const Basis& p_basis, const typename LocalView::Element& element)
       : localView{p_basis.localView()} {
     static_assert(Ikarus::Concepts::PowerBasis<RootBasis>,
                   "You didn't pass a localview of a power basis to this method");
     static_assert(RootBasis::PreBasis::Node::CHILDREN == 3,
                   "The power basis children number should be three");
     static_assert(
         Ikarus::Concepts::FlatIndexBasis<RootBasis>,
         "The Index merging strategy of the basis you passed has to be FlatLexicographic or FlatInterLeaved");
     localView.bind(element);
   }

   /** \brief Type of the Pairs of gridEntities and variable tags */
   using GridElementEntityType = typename LocalView::Element;
   using Traits                = FETraits<GridElementEntityType>;

   /** \brief Dimension of the world space */
   static constexpr int worlddim = Traits::worlddim;

   [[nodiscard]] constexpr int size() const { return localView.size(); }

   void globalIndices(std::vector<GlobalIndex>& globalIndices) const {
     const auto& fe = localView.tree().child(Dune::Indices::_0,0).finiteElement();
     for (size_t i = 0; i < fe.size(); ++i) {
       for (size_t j = 0; j < 3; ++j) {
         globalIndices.push_back(localView.index((localView.tree().child(Dune::Indices::_0,j).localIndex(i))));
       }
     }
   }
   const GridElementEntityType& getEntity() { return localView.element(); }

 private:
   LocalView localView;
 };
}  // namespace Ikarus