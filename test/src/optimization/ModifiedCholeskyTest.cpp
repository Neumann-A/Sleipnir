// Copyright (c) Sleipnir contributors

#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/Eigenvalues>
#include <Eigen/src/OrderingMethods/Ordering.h>
#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <gtest/gtest.h>

TEST(ModifiedCholeskyTest, PositiveDefinite) {
  // [1   0   3    2]
  // [0   4   1    0]
  // [3   1   13   0]
  // [2   0   0   21]
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.emplace_back(0, 0, 1);
  triplets.emplace_back(0, 1, 0);
  triplets.emplace_back(0, 2, 3);
  triplets.emplace_back(0, 3, 2);
  triplets.emplace_back(1, 0, 0);
  triplets.emplace_back(1, 1, 4);
  triplets.emplace_back(1, 2, 1);
  triplets.emplace_back(1, 3, 0);
  triplets.emplace_back(2, 0, 3);
  triplets.emplace_back(2, 1, 1);
  triplets.emplace_back(2, 2, 13);
  triplets.emplace_back(2, 3, 3);
  triplets.emplace_back(3, 0, 2);
  triplets.emplace_back(3, 1, 0);
  triplets.emplace_back(3, 2, 3);
  triplets.emplace_back(3, 3, -2);
  Eigen::SparseMatrix<double> G{4, 4};
  G.setFromTriplets(triplets.begin(), triplets.end());

  std::cout << "G: \n" << G.toDense() << std::endl;

  // [1]
  // [1]
  // [1]
  // [1]
  Eigen::VectorXd c = Eigen::VectorXd::Ones(4);

  Eigen::EigenSolver<Eigen::MatrixXd> solver{G.toDense()};
  std::cout << "eigenvalues: \n" << solver.eigenvalues() << std::endl;

  sleipnir::OptimizationProblem problem;
  Eigen::VectorXd x = problem.ModifiedCholesky(G, c);
  
  std::cout << "x: \n" << x << std::endl;
  std::cout << "x norm: " << x.norm() << std::endl;
}
