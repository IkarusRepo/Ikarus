//
// Created by Alex on 25.05.2021.
//
//#include <gmock/gmock.h>
//#include <gtest/gtest.h>

#include "testHelpers.h"

#include <dune/geometry/type.hh>

#include <ikarus/Geometries/GeometryType.h>
#include <ikarus/Grids/GridHelper/griddrawer.h>
#include <ikarus/Grids/SimpleGrid/SimpleGrid.h>
#include <catch2/catch_test_macros.hpp>
/** @addtogroup Tests
 *  This module includes all tests
 *  @{
 */

/**
 * \addtogroup GridTests
 * \test This test checks the insertion of vertices and elements in a SimpleGrid<2,2>
 *
 * It also checks the correct obtain unique identifiers from getID().
 *
 * The tested grid looks as follows
 *        13           16
 *   2------------3------------5                \n
 *   |            |            |  \             \n
 *   |    7       |   8        |     \ 16       \n
 * 10|   quad   12|    quad  15|       \        \n
 *   |            |            |    9    \      \n
 *   |            |            |triangle  \     \n
 *   0------------1------------4------------6   \n
 *      11              14           17         \n
 */
TEST_CASE("GridTest: GridViewTest", "[1]") {
  using namespace Ikarus::Grid;
  using Grid = SimpleGrid<2, 2>;
  SimpleGridFactory<2, 2> gridFactory;
  using vertexType = Eigen::Vector2d;
  std::vector<vertexType> verticesVec;
  verticesVec.emplace_back(0.0, 0.0);  // 0
  verticesVec.emplace_back(2.0, 0.0);  // 1
  verticesVec.emplace_back(0.0, 2.0);  // 2
  verticesVec.emplace_back(2.0, 2.0);  // 3
  verticesVec.emplace_back(4.0, 0.0);  // 4
  verticesVec.emplace_back(4.0, 2.0);  // 5
  verticesVec.emplace_back(6.0, 0.0);  // 6

  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);

  std::vector<size_t> elementIndices;
  elementIndices.resize(4);
  elementIndices = {0, 1, 2, 3};
  gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
  elementIndices = {1, 4, 3, 5};
  gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
  elementIndices.resize(3);
  elementIndices = {4, 6, 5};
  gridFactory.insertElement(Ikarus::GeometryType::linearTriangle, elementIndices);

  Grid grid = gridFactory.createGrid();

  auto gridView = grid.leafGridView();
  CHECK (9 == edges(gridView).size());
  CHECK (3 == surfaces(gridView).size());
  CHECK (7 == vertices(gridView).size());

  std::size_t expectedEdgeId = 0;
  std::vector<std::array<size_t, 2>> expectedEdgeVertexId;
  expectedEdgeVertexId.push_back({0, 2});
  expectedEdgeVertexId.push_back({0, 1});
  expectedEdgeVertexId.push_back({1, 3});
  expectedEdgeVertexId.push_back({2, 3});
  expectedEdgeVertexId.push_back({1, 4});
  expectedEdgeVertexId.push_back({4, 5});
  expectedEdgeVertexId.push_back({3, 5});
  expectedEdgeVertexId.push_back({4, 6});
  expectedEdgeVertexId.push_back({5, 6});

  const auto indexSet = gridView.indexSet();

  int elementCounter = 0;
  for (auto& edge : edges(gridView)) {
    CHECK (expectedEdgeId++ == indexSet.index(edge));

    int vertexCounter = 0;
    for (auto& vertex : vertices(edge)) {
      CHECK (Ikarus::GeometryType::vertex == vertex.type());
      CHECK (expectedEdgeVertexId[elementCounter][vertexCounter] == indexSet.index(vertex));
      ++vertexCounter;
    }
    ++elementCounter;
  }

  std::vector<std::vector<int>> expectedElementEdgeIds;
  expectedElementEdgeIds.push_back({10, 11, 12, 13});
  expectedElementEdgeIds.push_back({12, 14, 15, 16});
  expectedElementEdgeIds.push_back({17, 15, 18});

  int eleCounter = 0;
  for (auto& singleElement : surfaces(gridView)) {
    int edgeCounter = 0;
    for (auto& edge : edges(singleElement)) {
      CHECK (Ikarus::GeometryType::linearLine == edge.type());
      ++edgeCounter;
    }
    ++eleCounter;
  }
  auto ele1 = surfaces(gridView).begin();
  CHECK_THROWS_AS (ele1->subEntities(3), std::logic_error);
  CHECK (4 == ele1->subEntities(2));
  CHECK (4 == ele1->subEntities(1));
  ++ele1;
  CHECK_THROWS_AS (ele1->subEntities(3), std::logic_error);
  CHECK (4 == ele1->subEntities(2));
  CHECK (4 == ele1->subEntities(1));
  ++ele1;
  CHECK_THROWS_AS (ele1->subEntities(3), std::logic_error);
  CHECK (3 == ele1->subEntities(2));
  CHECK (3 == ele1->subEntities(1));
}

/**
 * \addtogroup GridTests
 * \test This test checks the insertion of vertices and elements in a SimpleGrid<2,3>
 *
 */
TEST_CASE("GridTest: GridView3DSurfaceTest", "[1]") {
  using namespace Ikarus::Grid;
  using Grid = SimpleGrid<2, 3>;
  SimpleGridFactory<2, 3> gridFactory;
  using vertexType = Eigen::Vector3d;
  std::vector<vertexType> verticesVec;
  verticesVec.emplace_back(0.0, 0.0, -3.0);  // 0
  verticesVec.emplace_back(2.0, 0.0, -3.0);  // 1
  verticesVec.emplace_back(0.0, 2.0, +3.0);  // 2
  verticesVec.emplace_back(2.0, 2.0, -3.0);  // 3
  verticesVec.emplace_back(4.0, 0.0, -3.0);  // 4
  verticesVec.emplace_back(4.0, 2.0, +3.0);  // 5
  verticesVec.emplace_back(6.0, 0.0, -3.0);  // 6

  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);

  std::vector<size_t> elementIndices;
  elementIndices.resize(4);
  elementIndices = {0, 1, 2, 3};
  gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
  elementIndices = {1, 4, 3, 5};
  gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices);
  elementIndices.resize(3);
  elementIndices = {4, 6, 5};
  gridFactory.insertElement(Ikarus::GeometryType::linearTriangle, elementIndices);

  Grid actualGrid = gridFactory.createGrid();
  auto gridView   = actualGrid.leafGridView();
  CHECK (9 == edges(gridView).size());
  CHECK (3 == surfaces(gridView).size());
  CHECK (7 == vertices(gridView).size());

  for (int i = 0; auto&& vertex : vertices(gridView)) {
    CHECK (Ikarus::GeometryType::vertex == vertex.type());
    CHECK (verticesVec[i] == vertex.getPosition());
    ++i;
  }

  auto&& eleIterator = surfaces(gridView).begin();
  CHECK (Ikarus::GeometryType::linearQuadrilateral == eleIterator->type());
  ++eleIterator;
  CHECK (Ikarus::GeometryType::linearQuadrilateral == eleIterator->type());
  ++eleIterator;
  CHECK (Ikarus::GeometryType::linearTriangle == eleIterator->type());

  std::vector<std::vector<size_t>> expectedElementEdgeIds;
  expectedElementEdgeIds.push_back({0, 1, 2, 3});
  expectedElementEdgeIds.push_back({2, 4, 5, 6});
  expectedElementEdgeIds.push_back({7, 5, 8});

  const auto indexSet = gridView.indexSet();

  int eleCounter = 0;
  for (auto& singleElement : surfaces(gridView)) {
    int edgeCounter = 0;
    for (auto& edge : edges(singleElement)) {
      CHECK (Ikarus::GeometryType::linearLine == edge.type());
      CHECK (expectedElementEdgeIds[eleCounter][edgeCounter] == indexSet.index(edge));
      ++edgeCounter;
    }
    ++eleCounter;
  }
  auto ele1 = surfaces(gridView).begin();
  CHECK_THROWS_AS (ele1->subEntities(3), std::logic_error);
  CHECK (4 == ele1->subEntities(2));
  CHECK (4 == ele1->subEntities(1));
  ++ele1;
  CHECK_THROWS_AS ( ele1->subEntities(3), std::logic_error);
  CHECK (4 == ele1->subEntities(2));
  CHECK (4 == ele1->subEntities(1));
  ++ele1;
  CHECK_THROWS_AS ( ele1->subEntities(3), std::logic_error);
  CHECK (3 == ele1->subEntities(2));
  CHECK (3 == ele1->subEntities(1));
}

/**
 * \addtogroup GridTests
 * \test This test checks the insertion of vertices and elements in a SimpleGrid<2,3>
 *
 */
TEST_CASE("GridTest: GridView3DSolidTest", "[1]") {
  using namespace Ikarus::Grid;
  using Grid = SimpleGrid<3, 3>;
  SimpleGridFactory<3, 3> gridFactory;
  using vertexType = Eigen::Vector3d;
  std::vector<vertexType> verticesVec;
  verticesVec.emplace_back(0.0, 0.0, -3.0);  // 0
  verticesVec.emplace_back(2.0, 0.0, -3.0);  // 1
  verticesVec.emplace_back(0.0, 2.0, -3.0);  // 2
  verticesVec.emplace_back(2.0, 2.0, -3.0);  // 3
  verticesVec.emplace_back(0.0, 0.0, +3.0);  // 4
  verticesVec.emplace_back(2.0, 0.0, +3.0);  // 5
  verticesVec.emplace_back(0.0, 2.0, +3.0);  // 6
  verticesVec.emplace_back(2.0, 2.0, +3.0);  // 7
  verticesVec.emplace_back(4.0, 0.0, +3.0);  // 8

  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);

  std::vector<size_t> elementIndices;
  elementIndices.resize(8);
  elementIndices = {0, 1, 2, 3, 4, 5, 6, 7};
  gridFactory.insertElement(Ikarus::GeometryType::linearHexahedron, elementIndices);
  elementIndices.resize(4);
  elementIndices = {1, 8, 3, 5};
  gridFactory.insertElement(Ikarus::GeometryType::linearTetrahedron, elementIndices);

  Grid actualGrid = gridFactory.createGrid();

  auto gridView = actualGrid.leafGridView();
  // Element   Edge        VertexIDs
  std::vector<std::vector<std::array<size_t, 2>>> expectedElementEdgeVertexId;
  expectedElementEdgeVertexId.emplace_back();
  expectedElementEdgeVertexId[0].push_back({0, 4});  // 0
  expectedElementEdgeVertexId[0].push_back({1, 5});  // 1
  expectedElementEdgeVertexId[0].push_back({2, 6});  // 2
  expectedElementEdgeVertexId[0].push_back({3, 7});  // 3
  expectedElementEdgeVertexId[0].push_back({0, 2});  // 4
  expectedElementEdgeVertexId[0].push_back({1, 3});  // 5
  expectedElementEdgeVertexId[0].push_back({0, 1});  // 6
  expectedElementEdgeVertexId[0].push_back({2, 3});  // 7
  expectedElementEdgeVertexId[0].push_back({4, 6});  // 8
  expectedElementEdgeVertexId[0].push_back({5, 7});  // 9
  expectedElementEdgeVertexId[0].push_back({4, 5});  // 10
  expectedElementEdgeVertexId[0].push_back({6, 7});  // 11
  expectedElementEdgeVertexId.emplace_back();
  expectedElementEdgeVertexId[1].push_back({1, 8});  // 0
  expectedElementEdgeVertexId[1].push_back({1, 3});  // 1
  expectedElementEdgeVertexId[1].push_back({3, 8});  // 2
  expectedElementEdgeVertexId[1].push_back({1, 5});  // 3
  expectedElementEdgeVertexId[1].push_back({5, 8});  // 4
  expectedElementEdgeVertexId[1].push_back({3, 5});  // 4
  CHECK (!edges(gridView).empty());
  CHECK (!volumes(gridView).empty());

  const auto indexSet = gridView.indexSet();

  for (int EleIter = 0; auto& ele : volumes(gridView)) {
    CHECK (!edges(ele).empty());
    for (int edgeIter = 0; auto& edge : edges(ele)) {
      CHECK (!vertices(edge).empty());
      for (int i = 0; auto&& verticesOfEdge : vertices(edge)) {
        CHECK (expectedElementEdgeVertexId[EleIter][edgeIter][i] == indexSet.index(verticesOfEdge));
        CHECK_THAT(verticesOfEdge.getPosition(),
                    EigenApproxEqual(verticesVec[expectedElementEdgeVertexId[EleIter][edgeIter][i]], 1e-15));
        ++i;
      }
      ++edgeIter;
    }
    ++EleIter;
  }

  auto&& eleIterator = volumes(gridView).begin();
  CHECK (Ikarus::GeometryType::linearHexahedron == eleIterator->type());
  ++eleIterator;
  CHECK (Ikarus::GeometryType::linearTetrahedron == eleIterator->type());

  std::vector<size_t> expectedEdgesAtVertex{3, 4, 3, 5, 3, 5, 3, 3, 3};
  for (int i = 0; auto&& vertex : vertices(gridView))
    CHECK (expectedEdgesAtVertex[i++] == edges(vertex).size());

  auto ele1 = volumes(gridView).begin();
  CHECK (8 == ele1->subEntities(3));
  CHECK (12 == ele1->subEntities(2));
  CHECK (6 == ele1->subEntities(1));

  // surface tests
  std::vector<std::vector<std::vector<size_t>>> expectedElementSurfaceVertexId;
  expectedElementSurfaceVertexId.emplace_back();
  expectedElementSurfaceVertexId[0].push_back({0, 2, 4, 6});  // 0
  expectedElementSurfaceVertexId[0].push_back({1, 3, 5, 7});  // 1
  expectedElementSurfaceVertexId[0].push_back({0, 1, 4, 5});  // 2
  expectedElementSurfaceVertexId[0].push_back({2, 3, 6, 7});  // 3
  expectedElementSurfaceVertexId[0].push_back({0, 1, 2, 3});  // 4
  expectedElementSurfaceVertexId[0].push_back({4, 5, 6, 7});  // 5

  expectedElementSurfaceVertexId.emplace_back();
  expectedElementSurfaceVertexId[1].push_back({1, 3, 8});  // 0
  expectedElementSurfaceVertexId[1].push_back({1, 5, 8});  // 1
  expectedElementSurfaceVertexId[1].push_back({1, 3, 5});  // 2
  expectedElementSurfaceVertexId[1].push_back({3, 5, 8});  // 3

  std::vector<std::vector<std::vector<size_t>>> expectedElementSurfaceEdgeId;
  expectedElementSurfaceEdgeId.emplace_back();
  expectedElementSurfaceEdgeId[0].push_back({0, 2, 4, 8});    // 0
  expectedElementSurfaceEdgeId[0].push_back({1, 3, 5, 9});    // 1
  expectedElementSurfaceEdgeId[0].push_back({0, 1, 6, 10});   // 2
  expectedElementSurfaceEdgeId[0].push_back({2, 3, 7, 11});   // 3
  expectedElementSurfaceEdgeId[0].push_back({4, 5, 6, 7});    // 4
  expectedElementSurfaceEdgeId[0].push_back({8, 9, 10, 11});  // 5
  expectedElementSurfaceEdgeId.emplace_back();
  expectedElementSurfaceEdgeId[1].push_back({12, 5, 13});   // 0
  expectedElementSurfaceEdgeId[1].push_back({12, 1, 14});   // 1
  expectedElementSurfaceEdgeId[1].push_back({5, 1, 15});    // 2
  expectedElementSurfaceEdgeId[1].push_back({13, 14, 15});  // 3

  for (int EleIter = 0; auto& ele : volumes(gridView)) {
    CHECK (!edges(ele).empty());
    for (int surfIter = 0; auto& surf : surfaces(ele)) {
      CHECK (!vertices(surf).empty());
      for (int i = 0; auto& verticesOfSurface : vertices(surf)) {
        CHECK (expectedElementSurfaceVertexId[EleIter][surfIter][i] == indexSet.index(verticesOfSurface));
        CHECK_THAT(verticesOfSurface.getPosition(),
                    EigenApproxEqual(verticesVec[expectedElementSurfaceVertexId[EleIter][surfIter][i]], 1e-15));
        ++i;
      }
      for (int i = 0; auto&& edgesOfSurface : edges(surf))
        CHECK (expectedElementSurfaceEdgeId[EleIter][surfIter][i++] == indexSet.index(edgesOfSurface));

      ++surfIter;
    }
    ++EleIter;
  }
  std::vector<size_t> expectedSurfacesAtVertex{3, 6, 3, 6, 3, 6, 3, 3, 3};
  for (int i = 0; auto& vertex : vertices(gridView))
    CHECK (expectedSurfacesAtVertex[i++] == surfaces(vertex).size());
}

TEST_CASE("GridTest: GridInsertionException", "[1]") {
  using namespace Ikarus::Grid;
  SimpleGridFactory<2, 2> gridFactory;
  using vertexType = Eigen::Vector2d;
  std::vector<vertexType> verticesVec;
  verticesVec.emplace_back(0.0, 0.0);  // 0
  verticesVec.emplace_back(2.0, 0.0);  // 1
  verticesVec.emplace_back(0.0, 2.0);  // 2
  verticesVec.emplace_back(2.0, 2.0);  // 3

  for (auto&& vert : verticesVec)
    gridFactory.insertVertex(vert);

  std::vector<size_t> elementIndices;
  elementIndices.resize(4);
  elementIndices = {0, 1, 2, 3};
  CHECK_THROWS_AS (gridFactory.insertElement(Ikarus::GeometryType::linearTriangle, elementIndices), Dune::GridError);

  CHECK_THROWS_AS (gridFactory.insertElement(Ikarus::GeometryType::linearHexahedron, elementIndices), Dune::GridError);

  CHECK_THROWS_AS (gridFactory.insertElement(Ikarus::GeometryType::linearLine, elementIndices), Dune::GridError);
  elementIndices.resize(3);
  elementIndices = {0, 1, 2};

  CHECK_THROWS_AS (gridFactory.insertElement(Ikarus::GeometryType::linearQuadrilateral, elementIndices), Dune::GridError);
}

TEST_CASE("GridTest: GridEmptyGridCreation", "[1]") {
  using namespace Ikarus::Grid;
  SimpleGridFactory<2, 2> gridFactory;
  CHECK_THROWS_AS (gridFactory.createGrid(), Dune::GridError);
  gridFactory.insertVertex({2.0, 1.0});
  CHECK_THROWS_AS (gridFactory.createGrid(), Dune::GridError);
}
/*\@}*/
