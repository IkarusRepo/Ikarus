

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
#include <autodiff/forward/dual/dual.hpp>
#include <iosfwd>

#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockvector.hh>

#include <Eigen/Core>

#include <ikarus/localFunctions/meta.hh>
#include <ikarus/manifolds/manifoldInterface.hh>
#include <ikarus/utils/concepts.hh>

namespace Ikarus {

  /** \brief Orthonormalizes all Matrix columns */
  template <typename Derived>
  auto orthonormalizeMatrixColumns(const Eigen::MatrixBase<Derived>& A) {
    // Gram Schmidt Ortho
    auto Q = A.eval();

    Q.col(0).normalize();

    for (int colIndex = 1; colIndex < Q.cols(); colIndex++) {
      Q.col(colIndex) -= Q.leftCols(colIndex) * (Q.leftCols(colIndex).transpose() * A.col(colIndex));
      Q.col(colIndex).normalize();
    }

    return Q;
  }

  /** \brief View Dune::BlockVector as a Eigen::Vector */
  template <typename ValueType>
  auto viewAsFlatEigenVector(Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<Eigen::VectorX<typename ValueType::field_type>> vec(&blockedVector.begin()->begin().operator*(),
                                                                   blockedVector.size() * blockedVector[0].size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Matrix with dynamic rows and fixed columns depending on the size of  the
   * ValueType*/
  template <typename ValueType>
  auto viewAsEigenMatrixAsDynFixed(Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<Eigen::Matrix<typename ValueType::field_type, Eigen::Dynamic, ValueType::valueSize, Eigen::RowMajor>>
        vec(&blockedVector.begin()->begin().operator*(), blockedVector.size(), blockedVector[0].size());

    return vec;
  }

  /** \brief Const view Dune::BlockVector as a Eigen::Matrix with dynamic rows and fixed columns depending on the size
   * of  the ValueType */
  template <typename ValueType>
  auto viewAsEigenMatrixAsDynFixed(const Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<
        const Eigen::Matrix<typename ValueType::field_type, Eigen::Dynamic, ValueType::valueSize, Eigen::RowMajor>>
        vec(&blockedVector.begin()->begin().operator*(), blockedVector.size(), blockedVector[0].size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Matrix with fixed rows depending on the size of  the ValueType and
   * dynamics columns */
  template <typename ValueType>
  auto viewAsEigenMatrixFixedDyn(Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<Eigen::Matrix<typename ValueType::field_type, ValueType::valueSize, Eigen::Dynamic>> vec(
        &blockedVector.begin()->begin().operator*(), blockedVector[0].size(), blockedVector.size());

    return vec;
  }

  /** \brief Const view Dune::BlockVector as a Eigen::Matrix with fixed rows depending on the size of  the ValueType and
   * dynamics columns */
  template <typename ValueType>
  auto viewAsEigenMatrixFixedDyn(const Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<const Eigen::Matrix<typename ValueType::field_type, ValueType::valueSize, Eigen::Dynamic>> vec(
        &blockedVector.begin()->begin().operator*(), blockedVector[0].size(), blockedVector.size());

    return vec;
  }

  /** \brief View Dune::BlockVector as a Eigen::Vector */
  template <typename ValueType>
  auto viewAsFlatEigenVector(const Dune::BlockVector<ValueType>& blockedVector) {
    Eigen::Map<const Eigen::VectorX<typename ValueType::field_type>> vec(
        &blockedVector.begin()->begin().operator*(), blockedVector.size() * blockedVector[0].size());

    return vec;
  }

  /* Returns the total correction size of a block vector with a Manifold as type */
  template <typename Type>
  size_t correctionSize(const Dune::BlockVector<Type>& a) requires requires {
    Type::correctionSize;
  }
  { return a.size() * Type::correctionSize; }

  /* Enables the += operator for Dune::BlockVector += Eigen::Vector */
  template <typename Type, typename Derived>
  Dune::BlockVector<Type>& operator+=(Dune::BlockVector<Type>& a, const Eigen::MatrixBase<Derived>& b) requires(
      Ikarus::Concepts::AddAssignAble<Type, decltype(b.template segment<Type::correctionSize>(0))>and requires() {
        Type::correctionSize;
      }) {
    for (auto i = 0U; i < a.size(); ++i)
      a[i] += b.template segment<Type::correctionSize>(i * Type::correctionSize);
    return a;
  }

  /* Enables the -= operator for Dune::BlockVector += Eigen::Vector */
  template <typename Type, typename Derived>
  Dune::BlockVector<Type>& operator-=(Dune::BlockVector<Type>& a, const Eigen::MatrixBase<Derived>& b) requires(
      Ikarus::Concepts::AddAssignAble<Type, decltype(b.template segment<Type::correctionSize>(0))>and requires() {
        Type::correctionSize;
      }) {
    return a += (-b);
  }

  /* Enables the += operator for Dune::MultiTypeBlockVector += Eigen::Vector */
  template <typename... Types, typename Derived>
  Dune::MultiTypeBlockVector<Types...>& operator+=(Dune::MultiTypeBlockVector<Types...>& a,
                                                   const Eigen::MatrixBase<Derived>& b) {
    using namespace Dune::Indices;
    size_t posStart = 0;
    Dune::Hybrid::forEach(Dune::Hybrid::integralRange(Dune::index_constant<a.size()>()), [&](const auto i) {
      const size_t size = correctionSize(a[i]);
      a[i] += b(Eigen::seqN(posStart, size));
      posStart += size;
    });

    return a;
  }

  /** \brief Adding free norm function to Eigen types */
  template <typename Derived>
  requires(!std::floating_point<Derived>) auto norm(const Eigen::MatrixBase<Derived>& v) { return v.norm(); }

  /** \brief Helper Free Function to have the same interface as for Eigen Vector Types */
  auto norm(const std::floating_point auto& v) { return std::abs(v); }

  /** \brief Eigen::DiagonalMatrix Product Missing in Eigen*/
  template <typename Scalar, int size>
  auto operator*(const Eigen::DiagonalMatrix<Scalar, size>& a, const Eigen::DiagonalMatrix<Scalar, size>& b) {
    return (a.diagonal().cwiseProduct(b.diagonal())).asDiagonal();
  }

  template <typename Scalar, int size>
  auto operator+=(Eigen::DiagonalMatrix<Scalar, size>& a, const Eigen::DiagonalMatrix<Scalar, size>& b) {
    a.diagonal() += b.diagonal();
    return a;
  }

  /** \brief Eigen::Matrix + Eigen::DiagonalMatrix addition missing in Eigen*/
  template <typename Derived, typename Scalar, int size>
  auto operator+(const Eigen::MatrixBase<Derived>& a, const Eigen::DiagonalMatrix<Scalar, size>& b) {
    auto c = a.derived().eval();
    c.diagonal() += b.diagonal();
    return c;
  }

  /** \brief Eigen::DiagonalMatrix + Eigen::Matrix addition missing in Eigen*/
  template <typename Derived, typename Scalar, int size>
  auto operator+(const Eigen::DiagonalMatrix<Scalar, size>& a, const Eigen::MatrixBase<Derived>& b) {
    return b + a;
  }

  template <typename Scalar, int size>
  auto operator-(const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return (-a.diagonal()).asDiagonal();
  }

  template <typename Scalar, int size>
  auto operator+(const Eigen::DiagonalMatrix<Scalar, size>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return a.derived();
  }

  template <typename Scalar, int size>
  auto operator+(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return a.derived();
  }

  template <typename Scalar, int size>
  auto operator*(const Eigen::DiagonalMatrix<Scalar, size>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return Ikarus::DerivativeDirections::DerivativeNoOp();
  }

  template <typename Scalar, int size>
  auto operator*(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return Ikarus::DerivativeDirections::DerivativeNoOp();
  }

  template <typename Scalar, int size>
  auto operator-(const Eigen::DiagonalMatrix<Scalar, size>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return a.derived();
  }

  template <typename Scalar, int size>
  auto operator-(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::DiagonalMatrix<Scalar, size>& a) {
    return a.derived();
  }

  template <typename Derived, typename Derived2>
  auto operator+(const Eigen::MatrixBase<Derived>& a, const Eigen::DiagonalWrapper<Derived2>& b) {
    auto c = a.derived().eval();
    c.diagonal() += b.diagonal();
    return c;
  }

  template <typename Derived>
  auto operator+(const Eigen::MatrixBase<Derived>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return a.derived();
  }

  template <typename Derived>
  auto operator+(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::MatrixBase<Derived>& a) {
    return a.derived();
  }

  Ikarus::DerivativeDirections::DerivativeNoOp operator+(Ikarus::DerivativeDirections::DerivativeNoOp,
                                                         Ikarus::DerivativeDirections::DerivativeNoOp);

  Ikarus::DerivativeDirections::DerivativeNoOp operator-(Ikarus::DerivativeDirections::DerivativeNoOp,
                                                         Ikarus::DerivativeDirections::DerivativeNoOp);

  template <typename Derived>
  auto operator*(const Eigen::MatrixBase<Derived>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return Ikarus::DerivativeDirections::DerivativeNoOp();
  }

  template <std::floating_point T>
  auto operator*(const T&, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return Ikarus::DerivativeDirections::DerivativeNoOp();
  }

  template <std::floating_point T>
  auto operator*(Ikarus::DerivativeDirections::DerivativeNoOp b, const T& a) {
    return a * b;
  }

  template <typename Derived>
  auto operator*(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::MatrixBase<Derived>& a) {
    return Ikarus::DerivativeDirections::DerivativeNoOp();
  }

  template <typename Derived>
  auto operator-(const Eigen::MatrixBase<Derived>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return a.derived();
  }

  template <typename Derived>
  auto operator-(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::MatrixBase<Derived>& a) {
    return a.derived();
  }

  template <typename Derived, typename Derived2>
  auto operator+(const Eigen::DiagonalWrapper<Derived>& a, const Eigen::MatrixBase<Derived2>& b) {
    return b + a;
  }

  template <typename Derived>
  auto operator+(const Eigen::DiagonalWrapper<Derived>& a, Ikarus::DerivativeDirections::DerivativeNoOp) {
    return a;
  }

  Ikarus::DerivativeDirections::DerivativeNoOp operator-(Ikarus::DerivativeDirections::DerivativeNoOp);

  template <typename Derived>
  auto operator+(Ikarus::DerivativeDirections::DerivativeNoOp, const Eigen::DiagonalWrapper<Derived>& a) {
    return a;
  }

  template <typename Scalar, int size>
  std::ostream& operator<<(std::ostream& os, const Eigen::DiagonalMatrix<Scalar, size>& a) {
    os << Eigen::Matrix<Scalar, size, size>(a);
    return os;
  }

  /** \brief Returns the symmetric part of a matrix*/
  template <typename Derived>
  Derived sym(const Eigen::MatrixBase<Derived>& A) {
    return 0.5 * (A + A.transpose());
  }

  /** \brief Returns the skew part of a matrix*/
  template <typename Derived>
  Derived skew(const Eigen::MatrixBase<Derived>& A) {
    return 0.5 * (A - A.transpose());
  }

  /** \brief Evaluates Eigen expressions */
  template <typename Derived>
  auto eval(const Eigen::EigenBase<Derived>& A) {
    if constexpr (static_cast<bool>(
                      Eigen::internal::is_diagonal<Derived>::ret))  // workaround needed since Eigen::DiagonalWrapper
                                                                    // does not has a eval function
    {
      using Scalar = typename Derived::Scalar;
      using namespace Eigen;
      constexpr int diag_size = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, Derived::ColsAtCompileTime);
      constexpr int max_diag_size
          = EIGEN_SIZE_MIN_PREFER_FIXED(Derived::MaxRowsAtCompileTime, Derived::MaxColsAtCompileTime);

      return Eigen::DiagonalMatrix<Scalar, diag_size, max_diag_size>(A.derived().diagonal());
    } else
      return A.derived().eval();
  }

  /** \brief Does nothing if type is not an Eigen type but our manifolds type or floatingin point instead*/
  template <typename Type>
  requires Ikarus::Concepts::Manifold<Type> or std::floating_point<Type>
  auto eval(const Type& A) { return A; }

  /** \brief  eval overload for autodiff scalars */
  template <typename T>
  requires autodiff::detail::isDual<T> || autodiff::detail::isExpr<T> || autodiff::detail::isArithmetic<T>
  auto eval(T&& t) { return autodiff::detail::eval(t); }

  /** \brief  Transpose for scalars and autodiff types */
  template <typename T>
  requires autodiff::detail::isDual<T> || autodiff::detail::isExpr<T> || autodiff::detail::isArithmetic<T>
  auto transpose(T&& t) { return t; }

  /** \brief  eval overload for std::array  */
  template <typename Type, std::size_t d>
  auto eval(std::array<Type, d>&& t) {
    return t;
  }

  Ikarus::DerivativeDirections::DerivativeNoOp transpose(const Ikarus::DerivativeDirections::DerivativeNoOp&);
  Ikarus::DerivativeDirections::DerivativeNoOp eval(const Ikarus::DerivativeDirections::DerivativeNoOp&);

  template <typename Derived>
  auto transpose(const Eigen::EigenBase<Derived>& A) {
    if constexpr (requires {
                    A.derived().transpose();
                  })  // workaround needed since Eigen::Diagonalmatrix has no transpose function
      return A.derived().transpose();
    else if constexpr (Ikarus::Std::IsSpecializationTypeAndNonTypes<Eigen::DiagonalMatrix, Derived>::value
                       or Std::isSpecialization<Eigen::DiagonalWrapper, Derived>::value)
      return A.derived();
    else
      static_assert((requires { A.derived().transpose(); })
                    or Ikarus::Std::IsSpecializationTypeAndNonTypes<Eigen::DiagonalMatrix, Derived>::value
                    or Std::isSpecialization<Eigen::DiagonalWrapper, Derived>::value);
  }

  template <typename To, typename From>
  requires std::convertible_to<typename From::ctype, To>
  auto convertUnderlying(const Dune::BlockVector<From>& from) {
    Dune::BlockVector<typename From::template Rebind<To>::other> to;
    to.resize(from.size());
    for (std::size_t i = 0; i < to.size(); ++i)
      to[i] = from[i];

    return to;
  }

  /* Enables the += operator for std::array if the underlying objects are addable  */
  template <typename Type, typename Type2, std::size_t d>
  std::array<Type, d> operator+(const std::array<Type, d>& a,
                                const std::array<Type2, d>& b) requires Concepts::AddAble<Type, Type2> {
    std::array<Type, d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = a[i] + b[i];
    return res;
  }

  /* Enables the - operator for std::array if the underlying objects are negate able  */
  template <std::size_t d, typename Type>
  std::array<Type, d> operator-(const std::array<Type, d>& a) requires Concepts::NegateAble<Type> {
    std::array<Type, d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = -a[i];
    return res;
  }

  /* Enables the transposition for std::array if the underlying objects are transposable able  */
  template <std::size_t d, typename Type>
  auto transpose(const std::array<Type, d>& a) requires Concepts::TransposeAble<Type> {
    std::array<decltype(transpose(a[0])), d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = transpose(a[i]);
    return res;
  }

  template <std::size_t d, typename Scalar, typename Type>
  requires Concepts::MultiplyAble<Scalar, Type>
  auto operator*(Scalar b, const std::array<Type, d>& a) {
    std::array<decltype(eval(b * a[0])), d> res;
    for (size_t i = 0U; i < d; ++i)
      res[i] = b * a[i];
    return res;
  }

  /* Method to print to cout the matrix in a format that can directly be copied to maple*/
  template <typename Derived>
  void printForMaple(const Eigen::EigenBase<Derived>& A) {
    Eigen::IOFormat mapleFmt(Eigen::FullPrecision, 0, ", ", "|\n", "<", ">", "<", ">");
    if constexpr (std::convertible_to<Derived, const Eigen::MatrixBase<Derived>&>) {
      std::cout << "\n" << A.derived().format(mapleFmt) << std::endl;
    } else {  // branch for Eigen::DiagonalMatrix
      using Scalar = typename Derived::Scalar;
      using namespace Eigen;
      constexpr int diag_size = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, Derived::ColsAtCompileTime);
      std::cout << "\n"
                << Eigen::Matrix<Scalar, diag_size, diag_size>(A.derived().diagonal().asDiagonal()).format(mapleFmt)
                << std::endl;
    }
  }

}  // namespace Ikarus