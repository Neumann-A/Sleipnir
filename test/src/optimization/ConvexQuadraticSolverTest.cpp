// Copyright (c) Sleipnir contributors

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <Eigen/SparseCholesky>

#include <gtest/gtest.h>
#include <sleipnir/optimization/OptimizationProblem.hpp>

#include <vector>
#include <iostream>
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/SparseCholesky/SimplicialCholesky.h"
#include "Eigen/src/SparseCore/SparseMatrix.h"

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

Eigen::SparseMatrix<double> SparseDiagonal(const Eigen::VectorXd& src) {
  std::vector<Eigen::Triplet<double>> triplets;
  for (int row = 0; row < src.rows(); ++row) {
    triplets.emplace_back(row, row, src(row));
  }
  Eigen::SparseMatrix<double> dest{src.rows(), src.rows()};
  dest.setFromTriplets(triplets.begin(), triplets.end());
  return dest;
}

double FractionToTheBoundaryRule(const Eigen::Ref<const Eigen::VectorXd>& x,
                                 const Eigen::Ref<const Eigen::VectorXd>& p,
                                 double tau) {
  // αᵐᵃˣ = max(α ∈ (0, 1] : x + αp ≥ (1−τ)x)
  //      = max(α ∈ (0, 1] : αp ≥ −τx)
  double alpha = 1.0;
  for (int i = 0; i < x.rows(); ++i) {
    if (p(i) != 0.0) {
      while (alpha * p(i) < -tau * x(i)) {
        alpha *= 0.999;
      }
    }
  }

  return alpha;
}

// Solve quadratic problem:
//
//          min ½xᵀGx + xᵀc 
//   subject to Aₑx + cₑ = 0
//              Aᵢx + cᵢ ≥ 0
//
// Where G is a semi-definite matrix.
void ConvexQuadraticSolver(Eigen::SparseMatrix<double> G,
                           Eigen::VectorXd c,
                           Eigen::SparseMatrix<double> A_e,
                           Eigen::VectorXd c_e,
                           Eigen::SparseMatrix<double> A_i,
                           Eigen::VectorXd c_i) {
  std::vector<Eigen::Triplet<double>> triplets;

  Eigen::SparseMatrix<double> I = Eigen::MatrixXd::Identity(c_i.rows(), c_i.rows()).sparseView();

  Eigen::VectorXd x = Eigen::VectorXd::Zero(G.rows());
  Eigen::VectorXd s = Eigen::VectorXd::Ones(c_i.rows());
  Eigen::VectorXd y = Eigen::VectorXd::Ones(c_e.rows());
  Eigen::VectorXd z = Eigen::VectorXd::Ones(c_i.rows());

  Eigen::VectorXd e = Eigen::VectorXd::Ones(s.rows());

  double m = c_i.rows();

  std::cout << "objective \t equality constraint \t inequality constraint" << std::endl;


  for (int iterations = 0; iterations < 20; ++iterations) {
    // Complimentary measure mu µ
    double mu = s.dot(z) / m;

    Eigen::SparseMatrix<double> Z = SparseDiagonal(z);
    Eigen::SparseMatrix<double> S = SparseDiagonal(s);

    // KKT lhs
    //
    //   [G    0   -Aᵢᵀ]
    //   [Aᵢ  -I    0  ]
    //   [0    Z    S  ]
    triplets.clear();
    AssignSparseBlock(triplets, 0, 0, G);
    AssignSparseBlock(triplets, G.rows(), 0, A_i);
    AssignSparseBlock(triplets, G.rows(), G.cols(), -I);
    AssignSparseBlock(triplets, 0, G.cols() + I.cols(), -A_i.transpose());
    AssignSparseBlock(triplets, G.rows() + I.rows(), G.cols(), Z);
    AssignSparseBlock(triplets, G.rows() + I.rows(), G.cols() + I.cols(), S);
    Eigen::SparseMatrix<double> lhs{x.rows() + s.rows() + s.rows(), x.rows() + s.rows() + s.rows()};
    lhs.setFromTriplets(triplets.begin(), triplets.end());

    // std::cout << "lhs: \n" << lhs.toDense() << std::endl;

    // KKT rhs
    //
    //   [Gx - Aᵀz + c]
    //   [Aᵢx + cᵢ - s]
    //   [ SZe - σµe  ]
    Eigen::VectorXd rhs{x.rows() + s.rows() + z.rows()};
    rhs.segment(0, x.rows()) = G * x - A_i.transpose() * z + c;
    rhs.segment(x.rows(), s.rows()) = A_i * x + c_i - s;
    rhs.segment(x.rows() + s.rows(), s.rows()) = S * Z * e - mu * e;

    // std::cout << "rhs: \n" << rhs << std::endl;

    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver{lhs};
    Eigen::VectorXd step = solver.solve(-rhs);

    Eigen::VectorXd p_x = step.segment(0, x.rows());
    Eigen::VectorXd p_s = step.segment(x.rows(), s.rows());
    Eigen::VectorXd p_z = step.segment(x.rows() + s.rows(), z.rows());

    double alpha_primal = FractionToTheBoundaryRule(s, p_s, 0.995);
    double alpha_dual = FractionToTheBoundaryRule(z, p_z, 0.995);

    x += alpha_primal * p_x;
    s += alpha_primal * p_s;
    z += alpha_dual * p_z;

    if (step.norm() < 1e-9) {
      break;
    }

    double infeasibility = 0;
    for (auto element : (A_i * x + c_i)) {
      infeasibility += std::max(0.0, -element);
    }
    std::cout << " " << (0.5 * x.dot(G * x) + x.dot(c)) << "    \t\t " << c_e.norm() << "\t\t\t   " << infeasibility << std::endl;
  }

  std::cout << "x: \n" << x << std::endl;
}

TEST(ConvexQuadraticSolverTest, 3x3) {
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.emplace_back(0, 0, 1);
  triplets.emplace_back(0, 1, 0);
  triplets.emplace_back(0, 2, -3);
  triplets.emplace_back(1, 0, 0);
  triplets.emplace_back(1, 1, 5);
  triplets.emplace_back(1, 2, -5);
  triplets.emplace_back(2, 0, -3);
  triplets.emplace_back(2, 1, -5);
  triplets.emplace_back(2, 2, 15);
  Eigen::SparseMatrix<double> G{3, 3};
  G.setFromTriplets(triplets.begin(), triplets.end());
  
  std::cout << "G: \n" << G.toDense() << "\n" << std::endl;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver{G};
  std::cout << "Vector D: \n" << solver.vectorD() << std::endl;

  Eigen::VectorXd c{3};
  c(0) = 1.0;
  c(1) = 2.0;
  c(2) = -2.0;

  std::cout << "c: \n" << c << "\n" << std::endl;

  triplets.clear();
  triplets.emplace_back(0, 0, 1);
  triplets.emplace_back(0, 1, 1);
  triplets.emplace_back(0, 2, 0);
  Eigen::SparseMatrix<double> A_e{1, 3};
  A_e.setFromTriplets(triplets.begin(), triplets.end());

  std::cout << "A_e: \n" << A_e.toDense() << "\n" << std::endl;

  triplets.clear();
  triplets.emplace_back(0, 0, 0);
  triplets.emplace_back(0, 1, 0);
  triplets.emplace_back(0, 2, 1);
  Eigen::SparseMatrix<double> A_i{1, 3};
  A_i.setFromTriplets(triplets.begin(), triplets.end());

  std::cout << "A_i: \n" << A_i.toDense() << "\n" << std::endl;

  Eigen::VectorXd c_e{1};
  c_e(0) = 0.0;

  Eigen::VectorXd c_i{1};
  c_i(0) = -5.0;

  ConvexQuadraticSolver(G, c, A_e, c_e, A_i, c_i);
}