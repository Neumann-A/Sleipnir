// Copyright (c) Sleipnir contributors

#include <vector>

#include <Eigen/BlockedLBLT.h>
#include <Eigen/SparseCore>
#include <gtest/gtest.h>

// TODO: Add matrices whose B has a 2x2 pivot for coverage

TEST(BlockedLBLTTest, Matrix3x3) {
  std::vector<Eigen::Triplet<double>> triplets;

  //     [  4   12  -16]
  // A = [ 12   37  -43]
  //     [-16  -43   98]
  triplets.emplace_back(0, 0, 4);
  triplets.emplace_back(0, 1, 12);
  triplets.emplace_back(0, 2, -16);
  triplets.emplace_back(1, 0, 12);
  triplets.emplace_back(1, 1, 37);
  triplets.emplace_back(1, 2, -43);
  triplets.emplace_back(2, 0, -16);
  triplets.emplace_back(2, 1, -43);
  triplets.emplace_back(2, 2, 98);
  Eigen::SparseMatrix<double> A{3, 3};
  A.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::BlockedLBLT<Eigen::SparseMatrix<double>> solver{A};

  auto P = solver.permutationP();

  //     [1  0  0]
  // L = [a  1  0]
  //     [b  c  1]
  Eigen::SparseMatrix<double> L = solver.matrixL();
  for (int row = 0; row < L.rows(); ++row) {
    for (int col = 0; col < L.cols(); ++col) {
      if (row == col) {
        EXPECT_NEAR(1.0, L.coeff(row, col), 1e-9);
      } else if (row < col) {
        EXPECT_NEAR(0.0, L.coeff(row, col), 1e-9);
      }
    }
  }

  auto B = solver.matrixB();

  //     [1  a  b]
  // U = [0  1  c]
  //     [0  0  1]
  Eigen::SparseMatrix<double> U = solver.matrixU();
  for (int row = 0; row < U.rows(); ++row) {
    for (int col = 0; col < U.cols(); ++col) {
      if (row == col) {
        EXPECT_NEAR(1.0, U.coeff(row, col), 1e-9);
      } else if (row > col) {
        EXPECT_NEAR(0.0, U.coeff(row, col), 1e-9);
      }
    }
  }

  EXPECT_TRUE((P.transpose() * A * P).isApprox(L * B * U));

  Eigen::VectorXd rhs{3};
  rhs << 1, 2, 3;
  Eigen::VectorXd x = solver.solve(rhs);

  EXPECT_NEAR(0.0, (rhs - A * x).lpNorm<Eigen::Infinity>(), 1e-9);
}

TEST(BlockedLBLTTest, Matrix4x4) {
  std::vector<Eigen::Triplet<double>> triplets;

  //     [ 17  4  -2   2]
  // A = [ 4   9  -1   6]
  //     [-2  -1  14  13]
  //     [ 2   6  13  35]
  triplets.emplace_back(0, 0, 17);
  triplets.emplace_back(0, 1, 4);
  triplets.emplace_back(0, 2, -2);
  triplets.emplace_back(0, 3, 2);
  triplets.emplace_back(1, 0, 4);
  triplets.emplace_back(1, 1, 0);
  triplets.emplace_back(1, 2, -1);
  triplets.emplace_back(1, 3, 6);
  triplets.emplace_back(2, 0, -2);
  triplets.emplace_back(2, 1, -1);
  triplets.emplace_back(2, 2, 0);
  triplets.emplace_back(2, 3, 13);
  triplets.emplace_back(3, 0, 2);
  triplets.emplace_back(3, 1, 6);
  triplets.emplace_back(3, 2, 13);
  triplets.emplace_back(3, 3, 35);
  Eigen::SparseMatrix<double> A{4, 4};
  A.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::BlockedLBLT<Eigen::SparseMatrix<double>> solver{A};

  auto P = solver.permutationP();

  //     [1  0  0]
  // L = [a  1  0]
  //     [b  c  1]
  Eigen::SparseMatrix<double> L = solver.matrixL();
  for (int row = 0; row < L.rows(); ++row) {
    for (int col = 0; col < L.cols(); ++col) {
      if (row == col) {
        EXPECT_NEAR(1.0, L.coeff(row, col), 1e-9);
      } else if (row < col) {
        EXPECT_NEAR(0.0, L.coeff(row, col), 1e-9);
      }
    }
  }

  auto B = solver.matrixB();

  //     [1  a  b]
  // U = [0  1  c]
  //     [0  0  1]
  Eigen::SparseMatrix<double> U = solver.matrixU();
  for (int row = 0; row < U.rows(); ++row) {
    for (int col = 0; col < U.cols(); ++col) {
      if (row == col) {
        EXPECT_NEAR(1.0, U.coeff(row, col), 1e-9);
      } else if (row > col) {
        EXPECT_NEAR(0.0, U.coeff(row, col), 1e-9);
      }
    }
  }

  EXPECT_TRUE((P.transpose() * A * P).isApprox(L * B * U));

  Eigen::VectorXd rhs{4};
  rhs << 1, 3, 2, 7;
  Eigen::VectorXd x = solver.solve(rhs);

  EXPECT_NEAR(0.0, (rhs - A * x).lpNorm<Eigen::Infinity>(), 1e-9);
}
