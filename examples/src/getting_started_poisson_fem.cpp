//
// Created by ac136645 on 6/14/2022.
//
#include <config.h>
#include <vector>

#include <dune/grid/uggrid.hh>
#include <dune/alugrid/grid.hh>
#include <dune/grid/io/file/gmshreader.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/istl/matrix.hh>
#include <dune/istl/bccsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/matrixmarket.hh>

#include <dune/functions/functionspacebases/lagrangebasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>

#include <ikarus/utils/drawing/griddrawer.hh>

using namespace Dune;

template <class LocalView, class Matrix>
void assembleElementStiffnessMatrix(const LocalView& localView,Matrix& elementMatrix)
{
  using Element = typename LocalView::Element;
  constexpr int dim = Element::dimension;
  auto element = localView.element();
  auto geometry = element.geometry();

  const auto& localFiniteElement = localView.tree().finiteElement();

  elementMatrix.setSize(localView.size(),localView.size());
  elementMatrix = 0;

  int order = 2 * (localFiniteElement.localBasis().order()-1);
  const auto& quadRule = QuadratureRules<double, dim>::rule(element.type(),order);

  for (const auto& quadPoint : quadRule)
  {
    const auto quadPos = quadPoint.position();
    const auto jacobian = geometry.jacobianInverseTransposed(quadPos); //J^{-1}.Transpose()
    const auto integrationElement = geometry.integrationElement(quadPos); //determinant(J)

    std::vector<FieldMatrix<double,1,dim>> referenceGradients;
    localFiniteElement.localBasis().evaluateJacobian(quadPos,referenceGradients);
    std::vector<FieldVector<double,dim>> gradients(referenceGradients.size());

    for (size_t i=0; i<gradients.size(); i++)
      jacobian.mv(referenceGradients[i][0],gradients[i]);
    std::cout<<std::endl<<"Grad"<<referenceGradients[0]<<std::endl;
    for (size_t p=0; p<elementMatrix.N(); p++)
    {
      auto localRow = localView.tree().localIndex(p);
      for (size_t q=0; q<elementMatrix.M(); q++)
      {
        auto localCol = localView.tree().localIndex(q);
        elementMatrix[localRow][localCol] += (gradients[p] * gradients[q]) * quadPoint.weight() * integrationElement;
      }
    }
  }
}

template <class LocalView>
void assembleElementVolumeTerm(const LocalView& localView,
                               BlockVector<double>& localB,
                               const std::function<double(FieldVector<double, LocalView::Element::dimension>)> volumeTerm)
{
  using Element = typename LocalView::Element;
  auto element = localView.element();
  constexpr int dim = Element::dimension;

  const auto& localFiniteElement = localView.tree().finiteElement();
  localB.resize(localFiniteElement.size());
  localB = 0;

  int order = dim;
  const auto& quadRule = QuadratureRules<double,dim>::rule(element.type(),order);

  for (const auto& quadPoint : quadRule)
  {
    const FieldVector<double,dim>& quadPos = quadPoint.position();
    const double integrationElement = element.geometry().integrationElement(quadPos);
    double functionValue = volumeTerm(element.geometry().global(quadPos));

    std::vector<FieldVector<double,1>> shapeFunctionValues;
    localFiniteElement.localBasis().evaluateFunction(quadPos, shapeFunctionValues);

    for (size_t p=0; p<localB.size(); p++)
    {
      auto localIndex = localView.tree().localIndex(p);
      localB[localIndex] += shapeFunctionValues[p] * functionValue * quadPoint.weight() * integrationElement;
    }
  }
}

template<class Basis>
void getOccupationPattern(const Basis& basis, MatrixIndexSet& nb)
{
  nb.resize(basis.size(),basis.size());
  auto gridView = basis.gridView();
  auto localView = basis.localView();

  for (const auto& element : elements(gridView))
  {
    localView.bind(element);

    for (size_t i=0; i<localView.size(); i++)
    {
      auto row = localView.index(i);
      for (size_t j=0; j<localView.size(); j++)
      {
        auto col = localView.index(j);
        nb.add(row,col);
      }
    }
  }
}

template<class Basis>
void assemblePoissonProblem(const Basis& basis,
                            BCRSMatrix<double>& matrix,
                            BlockVector<double>& b,
                            const std::function<double(FieldVector<double,Basis::GridView::dimension>)> volumeTerm)
{
  auto gridView = basis.gridView();
  MatrixIndexSet occupationPattern;
  getOccupationPattern(basis, occupationPattern);
  occupationPattern.exportIdx(matrix);

  matrix = 0;

  b.resize(basis.dimension());
  b = 0;

  auto localView = basis.localView();

  for (const auto& element : elements(gridView))
  {
    localView.bind(element);
    Matrix<double> elementMatrix;
    assembleElementStiffnessMatrix(localView, elementMatrix);
    for (size_t p=0; p<elementMatrix.N(); p++)
    {
      auto row = localView.index(p);
      for (size_t q=0; q<elementMatrix.M(); q++)
      {
        auto col = localView.index(q);
        matrix[row][col] += elementMatrix[p][q];
      }
    }
    BlockVector<double> localB;
    assembleElementVolumeTerm(localView, localB, volumeTerm);
    for (size_t p=0; p<localB.size(); p++)
    {
      auto row = localView.index(p);
      b[row] += localB[p];
    }
  }
}

int main(int argc, char **argv) {
  MPIHelper::instance(argc,argv);

  constexpr int dim = 2;
  using Grid = UGGrid<dim>;
  std :: shared_ptr<Grid> grid = GmshReader<Grid>::read("../../tests/src/testFiles/lshape.msh");

  // using Grid = Dune::ALUGrid<dim, 2, Dune::simplex, Dune::conforming>;
  // auto grid  = Dune::GmshReader<Grid>::read("../../tests/src/testFiles/lshape.msh", false);
  // ALUGrid works only for simple rectangular meshes

  grid->globalRefine(2);

  using GridView = Grid::LeafGridView;
  GridView gridView = grid->leafGridView();

  using Matrix = BCRSMatrix<double>;
  using Vector = BlockVector<double>;

  Matrix stiffnessMatrix;
  Vector b;
  // draw(gridView);
  Functions::LagrangeBasis<GridView,1> basis(gridView);

  std::cout << "This gridview contains: " << std::endl;
  std::cout << gridView.size(2) << " vertices" << std::endl;
  std::cout << gridView.size(1) << " edges" << std::endl;
  std::cout << gridView.size(0) << " elements" << std::endl;
  std::cout << basis.size() << " Dofs" << std::endl;

  auto sourceTerm = [](const FieldVector<double,dim>& x){return -5.0;};

  assemblePoissonProblem(basis, stiffnessMatrix, b, sourceTerm);

  auto predicate = [](auto x)
  {
    return x[0] < 1e-8
    || x[1] < 1e-8
    || (x[0]>0.4999 && x[1] > 0.4999);
  };
  std::vector<bool> dirichletNodes;
  Functions::interpolate(basis, dirichletNodes, predicate);

  for (size_t i=0; i<stiffnessMatrix.N(); i++)
  {
    if (dirichletNodes[i])
    {
      auto cIt = stiffnessMatrix[i].begin();
      auto cEndIt = stiffnessMatrix[i].end();
      for (; cIt!=cEndIt; ++cIt)
        *cIt = (cIt.index()==i) ? 1.0 : 0.0;
    }
  }
  auto dirichletValues = [](auto x)
  {
    return (x[0]< 1e-8 || x[1]< 1e-8) ? 0: 0.5;
  };
  Functions::interpolate(basis,b,dirichletValues,dirichletNodes);
  std::cout<<"Basis Size: "<<basis.size()<<std::endl;
  std::cout<<"Dirichlet Nodes: "<<dirichletNodes[0]<<std::endl;


  std::string baseName = "getting-started-poisson-fem-"+std::to_string(grid->maxLevel())+"-refinements";
  storeMatrixMarket(stiffnessMatrix,baseName + "-matrix.mtx");
  storeMatrixMarket(b, baseName + "-rhs.mtx");

  Vector x(basis.size());
  x = b;

  MatrixAdapter<Matrix,Vector,Vector> linearOperator(stiffnessMatrix);
  SeqILU<Matrix,Vector,Vector> preconditioner(stiffnessMatrix, 1.0);

  CGSolver<Vector> cg(linearOperator,
                      preconditioner,
                      1e-5,
                      50,
                      2);

  InverseOperatorResult statistics;
  cg.apply(x,b, statistics);
  std::cout<<std::endl<<"x size: "<<x.size()<<std::endl;
  VTKWriter<GridView> vtkWriter(gridView);
  vtkWriter.addVertexData(x, "displacement");
  vtkWriter.write("getting-started-poisson-fem-result");
}
