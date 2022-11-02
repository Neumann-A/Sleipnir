// Copyright (c) Sleipnir contributors

#pragma once

#include <algorithm>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

#include <Eigen/SparseCore>
#include <Eigen/src/Core/PermutationMatrix.h>
#include <Eigen/src/SparseCore/SparseSolverBase.h>

namespace Eigen {

template <typename _MatrixType>
class BlockedLBLT;

namespace internal {

template <typename _MatrixType>
struct traits<BlockedLBLT<_MatrixType>> {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using MatrixL = TriangularView<const MatrixType, Eigen::UnitLower>;
  using MatrixU = TriangularView<const typename MatrixType::AdjointReturnType,
                                 Eigen::UnitUpper>;

  static inline MatrixL getL(const MatrixType& m) { return MatrixL(m); }
  static inline MatrixU getU(const MatrixType& m) {
    return MatrixU(m.adjoint());
  }
};

}  // namespace internal

/**
 * \class BlockedLBLT \brief A direct sparse LBLᵀ factorization with rook
 * pivoting.
 *
 * Solves a linear system using the PᵀAP = LBLᵀ factorization. The factorization
 * allows for solving Ax = b where X and B can be either dense or sparse.
 *
 * \tparam MatrixType_ the type of the sparse matrix A, it must be a
 * SparseMatrix<>.
 */
template <typename _MatrixType>
class BlockedLBLT : public SparseSolverBase<BlockedLBLT<_MatrixType>> {
 public:
  using MatrixType = _MatrixType;
  using PermutationType = PermutationMatrix<Dynamic>;
  using Base = SparseSolverBase<BlockedLBLT<MatrixType>>;
  using Base::m_isInitialized;

  using StorageIndex = typename MatrixType::StorageIndex;
  using Traits = internal::traits<BlockedLBLT>;
  using MatrixL = typename Traits::MatrixL;
  using MatrixU = typename Traits::MatrixU;

  enum {
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
  };

  /**
   * Default constructor.
   */
  BlockedLBLT() = default;

  /**
   * Constructs and performs the block LBLT factorization of \a matrix.
   */
  explicit BlockedLBLT(const MatrixType& matrix) { compute(matrix); }

  /**
   * \returns the diagonal B.
   */
  inline MatrixType matrixB() const {
    std::vector<Triplet<double>> triplets;

    int row = 0;
    while (row < m_D.rows()) {
      if (m_D(row, 1) == 0.0) {
        triplets.emplace_back(row, row, m_D(row, 0));
        ++row;
      } else {
        for (int i = 0; i < 2; ++i) {
          for (int j = 0; j < 2; ++j) {
            triplets.emplace_back(row + i, row + j, m_D(row + i, j));
          }
        }
        row += 2;
      }
    }

    Eigen::SparseMatrix<double> B{m_D.rows(), m_D.rows()};
    B.setFromTriplets(triplets.begin(), triplets.end());
    return B;
  }

  /**
   * \returns an expression of the factor L.
   */
  inline const MatrixL matrixL() const { return Traits::getL(m_L); }

  /**
   * \returns an expression of the factor U (= L*).
   */
  inline const MatrixU matrixU() const { return Traits::getU(m_L); }

  /**
   * \returns the permutation matrix P.
   */
  inline const PermutationType& permutationP() const { return m_P; }

  /**
   * \returns the number of rows in the factorization.
   */
  inline Index rows() const { return m_L.rows(); }

  /**
   * \returns the number of columns in the factorization.
   */
  inline Index cols() const { return m_L.cols(); }

  /**
   * Computes the sparse Cholesky decomposition of \a matrix.
   */
  BlockedLBLT& compute(const MatrixType& matrix) {
    MatrixType B = matrix;

    // Create identity permutation matrix for L
    VectorXi m_p{matrix.rows()};
    for (int row = 0; row < matrix.rows(); ++row) {
      m_p(row) = row;
    }
    m_P = PermutationMatrix<Dynamic>{m_p};
    m_L = sparseIdentity(matrix.rows());
    m_D = MatrixXd::Zero(matrix.rows(), 2);

    int k = 0;
    int blockSize;

    while (k < matrix.cols()) {
      // Max value on diagonal ξ_dia (row, col, value)
      std::tuple<int, int, double> maxDia;

      // Max value off diagonal ξ_off (row, col, value)
      std::tuple<int, int, double> maxOff;

      // Max error growth μ
      double mu = 10;

      // Identify max elements in the matrix
      for (int row = 0; row < B.outerSize(); ++row) {
        for (SparseMatrix<double>::InnerIterator it(B, row); it; ++it) {
          int row = it.row();
          int col = it.col();
          double val = std::abs(it.value());
          if (row == col && val > std::get<2>(maxDia)) {
            maxDia = {row, col, val};
          } else if (row != col && val > std::get<2>(maxOff)) {
            maxOff = {row, col, val};
          }
        }
      }

      // Create identity permutation matrix
      VectorXi p{B.rows()};
      for (int row = 0; row < p.rows(); ++row) {
        p(row) = row;
      }
      // If ξ_off < μξ_dia pivot
      if (std::get<2>(maxOff) < mu * std::get<2>(maxDia)) {
        p.row(0).swap(p.row(std::get<0>(maxDia)));
        m_p.row(k).swap(m_p.row(std::get<0>(maxDia) + k));
        blockSize = 1;
      } else {
        p.row(0).swap(
            p.row(std::min(std::get<0>(maxOff), std::get<1>(maxOff))));
        p.row(1).swap(
            p.row(std::max(std::get<0>(maxOff), std::get<1>(maxOff))));
        m_p.row(k).swap(
            m_p.row(k + std::min(std::get<0>(maxOff), std::get<1>(maxOff))));
        m_p.row(k + 1).swap(
            m_p.row(k + std::max(std::get<0>(maxOff), std::get<1>(maxOff))));
        blockSize = 2;
      }

      PermutationMatrix<Dynamic> P{p};
      B = P * B * P.transpose();

      // [E Cᵀ]
      // [C H ]
      auto E = B.block(0, 0, blockSize, blockSize);
      auto opt = blockInverse(E.toDense());
      if (!opt) {
        m_info = NumericalIssue;
        return *this;
      }
      auto inverseE = opt.value().sparseView().eval();
      auto C = B.block(blockSize, 0, B.rows() - blockSize, blockSize);

      // [ I   0][E     0      ][I  E⁻¹Cᵀ]
      // [CE⁻¹ I][0  H − CE⁻¹Cᵀ][0    I  ]
      MatrixType bottomLeft = C * inverseE;
      MatrixType L_k1 = sparseIdentity(m_L.cols());
      for (int row = 0; row < bottomLeft.outerSize(); ++row) {
        for (SparseMatrix<double>::InnerIterator it(bottomLeft, row); it;
             ++it) {
          L_k1.insert(it.row() + k + blockSize, it.col() + k) = it.value();
        }
      }
      PermutationMatrix<Dynamic> P1{m_p};
      m_L = m_L * P1 * L_k1;
      m_P = m_P * P1;
      m_D.block(k, 0, blockSize, blockSize) = E;

      // Revert to identity matrix for next iteration.
      m_p = P1 * m_p;

      // B = B − CE⁻¹Cᵀ
      B = B.block(blockSize, blockSize, B.rows() - blockSize,
                  B.cols() - blockSize) -
          C * inverseE * MatrixType(C.transpose());

      k += blockSize;
    }

    // Unpermute L
    m_L = m_P.transpose() * m_L;

    m_isInitialized = true;
    m_info = Success;

    return *this;
  }

  /** \brief Reports whether previous computation was successful.
   *
   * \returns \c Success if computation was successful,
   *          \c NumericalIssue if the factorization failed because of a zero
   * pivot.
   */
  ComputationInfo info() const {
    eigen_assert(m_isInitialized && "BlockedLBLT is not initialized.");
    return m_info;
  }

  /**
   * \internal.
   */
  template <typename Rhs, typename Dest>
  void _solve_impl(const MatrixBase<Rhs>& b, MatrixBase<Dest>& dest) const {
    eigen_assert(m_isInitialized && "BlockedLBLT is not initialized.");

    // Repeatedly backsolve to yield the solution.
    //
    // Ly = PbPᵀ
    // Bz = y
    // Lᵀx = z

    dest = m_P.transpose() * b;

    TriangularView<const MatrixType, Lower> L{m_L};
    L.solveInPlace(dest);

    int index = 0;
    int blockSize;
    while (index < m_D.rows()) {
      // Check if block is 1x1 or 2x2, 1x1 blocks will have a value of zero in
      // the top right corner.
      if (m_D(index, 1) == 0) {
        blockSize = 1;
      } else {
        blockSize = 2;
      }

      MatrixXd block = m_D.block(index, 0, blockSize, blockSize);
      dest.segment(index, block.rows()) =
          blockInverse(block).value() * dest.segment(index, block.rows());

      index += block.rows();
    }

    L.adjoint().solveInPlace(dest);

    dest = m_P * dest;
  }

  /**
   * \internal.
   */
  template <typename Rhs, typename Dest>
  void _solve_impl(const SparseMatrixBase<Rhs>& b,
                   SparseMatrixBase<Dest>& dest) const {
    // Repeatedly backsolve to yield the solution.
    //
    // Ly = PbPᵀ
    // Bz = y
    // Lᵀx = z

    // Eigen::VectorXd x = m_P * b * m_P.transpose();
    dest = m_P.transpose() * b;

    TriangularView<const MatrixType, Lower> L{m_L};
    L.solveInPlace(dest);

    int index = 0;
    int blockSize;
    while (index < m_D.rows()) {
      // Check if block is 1x1 or 2x2. 1x1 blocks will have a value of zero in
      // the top right corner.
      if (m_D(index, 1) == 0) {
        blockSize = 1;
      } else {
        blockSize = 2;
      }

      MatrixXd block = m_D.block(index, 0, blockSize, blockSize);
      dest.segment(index, block.rows()) =
          blockInverse(block) * dest.segment(index, block.rows());

      index += block.rows();
    }

    L.adjoint().solveInPlace(dest);

    dest = m_P * dest;
  }

 private:
  // the permutation
  PermutationMatrix<Dynamic> m_P;

  // the lower triangular factor
  MatrixType m_L;

  // Flattened diagonal matrix into a Nx2 matrix. 1x1 blocks are placed in the
  // top-left corner of their block. The matrix:
  //
  // [ 4  12  0]
  // [12  37  0]
  // [ 0   0  9]
  //
  // Would be flattened into:
  //
  // [ 4  12]
  // [12  37]
  // [ 9   0]
  MatrixXd m_D;

  ComputationInfo m_info;

  /**
   * Returns a sparse identity matrix of the given size.
   */
  static MatrixType sparseIdentity(int rows) {
    std::vector<Triplet<double>> triplets;
    triplets.reserve(rows);

    for (int row = 0; row < rows; ++row) {
      triplets.emplace_back(row, row, 1.0);
    }

    MatrixType result{rows, rows};
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
  }

  [[noreturn]] inline static void unreachable() {
    // Uses compiler specific extensions if possible. Even if no extension is
    // used, undefined behavior is still raised by an empty function body and
    // the noreturn attribute.
#ifdef __GNUC__  // GCC, Clang, ICC
    __builtin_unreachable();
#elif defined(_MSC_VER)  // MSVC
    __assume(false);
#endif
  }

  /**
   * Returns the inverse of a 1x1 or 2x2 matrix.
   */
  static std::optional<MatrixXd> blockInverse(const MatrixXd& M) {
    if (M.rows() == 1 && M.cols() == 1) {
      if (M(0, 0) != 0.0) {
        return Eigen::MatrixXd{{1.0 / M(0, 0)}};
      } else {
        return std::nullopt;
      }
    } else if (M.rows() == 2 && M.cols() == 2) {
      double det = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
      if (det != 0.0) {
        Eigen::MatrixXd ret{2, 2};
        ret << M(1, 1) / det, -M(0, 1) / det, -M(1, 0) / det, M(0, 0) / det;
        return ret;
      } else {
        return std::nullopt;
      }
    } else {
      unreachable();
    }
  }
};

}  // namespace Eigen
