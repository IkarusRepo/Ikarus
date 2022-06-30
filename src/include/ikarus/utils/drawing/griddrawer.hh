//
// Created by alex on 24.06.21.
//

#pragma once

#include <matplot/matplot.h>
#include <ranges>
#include <set>

#include <dune/geometry/dimension.hh>

template <typename GridView>
void draw(const GridView& gridView) {
  using namespace matplot;
  auto f  = figure(true);
  auto ax = gca();
  hold(ax, true);
  constexpr int edgeCodim = GridView::dimension-1;
    for (auto&& element : elements(gridView)) {
      std::array<std::array<double, 2>,GridView::dimensionworld> edgeCoords{};
      for (int edgeIndex = 0; edgeIndex < element.subEntities(edgeCodim); ++edgeIndex) {
        auto edge = element.template subEntity<edgeCodim>(edgeIndex);
        for (int i = 0; i < 2; ++i) {
          const auto vertCoords = edge.geometry().corner(i);
          for (int j = 0; j < GridView::dimensionworld; ++j)
            edgeCoords[j][i] = vertCoords[j];

        }
        if constexpr (GridView::dimensionworld==3 ) {
          auto l = ax->plot3(edgeCoords[0], edgeCoords[1], edgeCoords[2], "-o");
          l->line_width(2);
          l->color("black");
          l->marker_size(10);
          l->marker_face_color("red");
        } else {
          auto l = ax->plot(edgeCoords[0], edgeCoords[1], "-o");
          l->line_width(2);
          l->color("black");
          l->marker_size(10);
          l->marker_face_color("red");
        }
      }

    }


  f->show();
  f.reset();
}
