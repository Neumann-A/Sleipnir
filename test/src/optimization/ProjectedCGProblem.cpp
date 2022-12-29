// Copyright (c) Sleipnir contributors

#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/src/OrderingMethods/Ordering.h>
#include <Eigen/src/SparseCholesky/SimplicialCholesky.h>
#include <Eigen/src/SparseCore/SparseMatrix.h>
#include <gtest/gtest.h>

double QuadraticFormula(double A, double B, double C, double sign) {
  return (-B + sign * std::sqrt(B * B - 4 * A * C)) / (2 * A);
}

void AssignSparseBlock(std::vector<Eigen::Triplet<double>>& triplets,
                       int rowOffset, int colOffset,
                       const Eigen::SparseMatrix<double>& mat,
                       bool transpose = false) {
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it{mat, k}; it; ++it) {
      if (transpose) {
        triplets.emplace_back(rowOffset + it.col(), colOffset + it.row(),
                              it.value());
      } else {
        triplets.emplace_back(rowOffset + it.row(), colOffset + it.col(),
                              it.value());
      }
    }
  }
}

Eigen::VectorXd ProjectedCG(Eigen::SparseMatrix<double> G, Eigen::VectorXd c,
                            Eigen::SparseMatrix<double> A, Eigen::VectorXd b,
                            double delta) {
  // [B  Aᵀ]
  // [A  0 ]
  std::vector<Eigen::Triplet<double>> triplets;
  // AssignSparseBlock(triplets, 0, 0, G);
  AssignSparseBlock(triplets, 0, 0,
                    Eigen::MatrixXd::Identity(G.rows(), G.cols()).sparseView());
  AssignSparseBlock(triplets, G.rows(), 0, A);
  AssignSparseBlock(triplets, 0, G.cols(), A, true);
  Eigen::SparseMatrix<double> P{G.rows() + A.rows(), G.cols() + A.rows()};
  P.setFromTriplets(triplets.begin(), triplets.end());
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver{P};

  Eigen::VectorXd x = Eigen::VectorXd::Zero(c.rows());
  Eigen::VectorXd r = (G * x + c);

  // [B  Aᵀ][g] = [r]
  // [A  0 ][v]   [0]
  Eigen::VectorXd augmentedRhs = Eigen::VectorXd::Zero(P.rows());
  augmentedRhs.topRows(r.rows()) = r;
  Eigen::VectorXd g = solver.solve(augmentedRhs).topRows(r.rows());
  Eigen::VectorXd d = -g;
  Eigen::VectorXd v;

  std::cout << "iteration \t\t objective \t\t residual" << std::endl;
  std::cout << "    " << 0 << "\t\t\t     "
            << 0.5 * x.transpose() * G * x + x.transpose() * c << "\t\t\t    "
            << (A * x).norm() << std::endl;
  for (int iteration = 0; iteration < /*2 * (x.rows() - A.rows())*/ 200;
       ++iteration) {
    double tmp1 = r.dot(g);
    double tmp2 = d.dot(G * d);

    if (tmp2 <= 0) {
      Eigen::VectorXd x1 =
          x + QuadraticFormula(d.squaredNorm(), 2 * x.dot(d),
                               x.squaredNorm() - delta * delta, -1) *
                  d;
      Eigen::VectorXd x2 =
          x + QuadraticFormula(d.squaredNorm(), 2 * x.dot(d),
                               x.squaredNorm() - delta * delta, 1) *
                  d;
      if (0.5 * x1.dot(G * x1) + x1.dot(c) < 0.5 * x2.dot(G * x2) + x2.dot(c)) {
        return x1;
      } else {
        return x2;
      }
    }

    double alpha = tmp1 / tmp2;

    if ((x + alpha * d).norm() > delta) {
      Eigen::VectorXd x1 =
          x + QuadraticFormula(d.squaredNorm(), 2 * x.dot(d),
                               x.squaredNorm() - delta * delta, -1) *
                  d;
      Eigen::VectorXd x2 =
          x + QuadraticFormula(d.squaredNorm(), 2 * x.dot(d),
                               x.squaredNorm() - delta * delta, 1) *
                  d;
      if (0.5 * x1.dot(G * x1) + x1.dot(c) < 0.5 * x2.dot(G * x2) + x2.dot(c)) {
        return x1;
      } else {
        return x2;
      }
    }

    x += alpha * d;
    r += alpha * G * d;

    // [B  Aᵀ][g⁺] = [r⁺]
    // [A  0 ][v⁺]   [0 ]
    augmentedRhs.topRows(r.rows()) = r;
    Eigen::VectorXd augmentedSol = solver.solve(augmentedRhs);
    g = augmentedSol.topRows(r.rows());
    v = augmentedSol.bottomRows(A.rows());

    // Iteratively refine step to reduce constraint violation.
    //
    // [B  Aᵀ][Δg⁺] = [p_g]
    // [A  0 ][Δv⁺]   [p_v]
    Eigen::VectorXd p_g = r - G * g - A.transpose() * v;
    Eigen::VectorXd p_v = -A * g;
    Eigen::VectorXd refinedRhs{P.rows()};
    refinedRhs.topRows(p_g.rows()) = p_g;
    refinedRhs.bottomRows(p_v.rows()) = p_v;
    Eigen::VectorXd refinedSol = solver.solve(refinedRhs);
    g += refinedSol.topRows(p_g.rows());
    v += refinedSol.bottomRows(p_v.rows());

    double beta = r.dot(g) / tmp1;
    std::cout << "    " << iteration + 1 << "\t\t\t  "
              << 0.5 * x.dot(G * x) + x.dot(c) << "\t\t    " << (A * x).norm()
              << std::endl;
    d = -g + beta * d;

    if (std::abs(r.dot(g)) < 1e-6) {
      return x;
    }
  }

  return x;
}

TEST(ProjectedCGProblemTest, 4x4) {
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
  triplets.emplace_back(2, 3, 0);
  triplets.emplace_back(3, 0, 2);
  triplets.emplace_back(3, 1, 0);
  triplets.emplace_back(3, 2, 0);
  triplets.emplace_back(3, 3, -2);
  Eigen::SparseMatrix<double> G{4, 4};
  G.setFromTriplets(triplets.begin(), triplets.end());

  // [1]
  // [1]
  // [1]
  // [1]
  Eigen::VectorXd c = Eigen::VectorXd::Ones(4);

  // [1  -1  0  0]
  triplets.clear();
  triplets.emplace_back(0, 0, 1);
  triplets.emplace_back(0, 1, -1);
  triplets.emplace_back(0, 2, 0);
  triplets.emplace_back(0, 3, 0);
  Eigen::SparseMatrix<double> A{1, 4};
  A.setFromTriplets(triplets.begin(), triplets.end());

  // [0]
  Eigen::VectorXd b = Eigen::VectorXd::Zero(1);

  //   std::cout << "G: \n" << G.toDense() << std::endl;

  //   std::cout << "A: \n" << A << std::endl;

  //   std::cout << "A eigenvalues: \n" <<
  //   Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>{G}.vectorD() <<
  //   std::endl;

  //   Eigen::VectorXd x = CG(G, c);

  //   std::cout << "x: \n" << x << std::endl;

  Eigen::VectorXd x = ProjectedCG(G, c, A, b, 1);

  std::cout << "x: \n" << x << std::endl;
  std::cout << "x norm: " << x.norm() << std::endl;
  std::cout << "objective: " << 0.5 * x.dot(G * x) + x.dot(c) << std::endl;
}
