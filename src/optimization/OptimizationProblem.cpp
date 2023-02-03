// Copyright (c) Sleipnir contributors

#include "sleipnir/optimization/OptimizationProblem.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <fmt/core.h>

#include "ScopeExit.hpp"
#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"
#include "sleipnir/autodiff/Gradient.hpp"
#include "sleipnir/autodiff/Hessian.hpp"
#include "sleipnir/autodiff/Jacobian.hpp"
#include "sleipnir/autodiff/Variable.hpp"

using namespace sleipnir;

namespace {
/**
 * Filter entry consisting of objective value and constraint value.
 */
struct FilterEntry {
  /// The objective function's value
  double objective = 0.0;

  /// The constraint violation
  double constraintViolation = 0.0;

  constexpr FilterEntry() = default;

  /**
   * Constructs a FilterEntry.
   *
   * @param f The objective function.
   * @param mu The barrier parameter.
   * @param s The inequality constraint slack variables.
   * @param c_e The equality constraint values (nonzero means violation).
   * @param c_i The inequality constraint values (negative means violation).
   */
  FilterEntry(const Variable& f, double mu, Eigen::VectorXd& s,
              const Eigen::VectorXd& c_e, const Eigen::VectorXd& c_i)
      : objective{f.Value() - mu * s.array().log().sum()},
        constraintViolation{c_e.lpNorm<1>() + (c_i - s).lpNorm<1>()} {}
};

struct Filter {
  std::vector<FilterEntry> filter;

  double maxConstraintViolation;
  double minConstraintViolation;

  double gamma_constraint = 0;
  double gamma_objective = 0;

  explicit Filter(FilterEntry pair) {
    filter.push_back(pair);
    minConstraintViolation = 1e-4 * std::max(1.0, pair.constraintViolation);
    maxConstraintViolation = 1e4 * std::max(1.0, pair.constraintViolation);
  }

  void PushBack(FilterEntry pair) { filter.push_back(pair); }

  void ResetFilter(FilterEntry pair) {
    filter.clear();
    filter.push_back(pair);
  }

  bool IsStepAcceptable(const FilterEntry& pair) {
    // If current filter entry is better than all prior ones in some respect,
    // accept it
    return std::all_of(
               filter.begin(), filter.end(),
               [&](const auto& entry) {
                 return pair.objective <=
                            entry.objective -
                                gamma_objective * entry.constraintViolation ||
                        pair.constraintViolation <=
                            (1 - gamma_constraint) * entry.constraintViolation;
               }) &&
           pair.constraintViolation < maxConstraintViolation;
  }
};

/**
 * Assigns the contents of a double vector to an autodiff vector.
 *
 * @param dest The autodiff vector.
 * @param src The double vector.
 */
void SetAD(std::vector<Variable>& dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.size() == static_cast<size_t>(src.rows()));

  for (size_t row = 0; row < dest.size(); ++row) {
    dest[row] = src(row);
  }
}

/**
 * Assigns the contents of a double vector to an autodiff vector.
 *
 * @param dest The autodiff vector.
 * @param src The double vector.
 */
void SetAD(Eigen::Ref<VectorXvar> dest,
           const Eigen::Ref<const Eigen::VectorXd>& src) {
  assert(dest.rows() == src.rows());

  for (int row = 0; row < dest.rows(); ++row) {
    dest(row) = src(row);
  }
}

/**
 * Gets the contents of a autodiff vector as a double vector.
 *
 * @param src The autodiff vector.
 */
Eigen::VectorXd GetAD(std::vector<Variable> src) {
  Eigen::VectorXd dest{src.size()};
  for (int row = 0; row < dest.size(); ++row) {
    dest(row) = src[row].Value();
  }
  return dest;
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

/**
 * Applies fraction-to-the-boundary rule to a variable and its iterate, then
 * returns a fraction of the iterate step size within (0, 1].
 *
 * @param p The iterate on the variable.
 * @param tau Fraction-to-the-boundary rule scaling factor.
 * @return Fraction of the iterate step size within (0, 1].
 */
double FractionToTheBoundaryRule(const Eigen::Ref<const Eigen::VectorXd>& p,
                                 double tau) {
  // αᵐᵃˣ = max(α ∈ (0, 1] : αp ≥ −τe)
  double alpha = 1.0;
  for (int i = 0; i < p.rows(); ++i) {
    if (p(i) != 0.0) {
      while (alpha * p(i) < -tau) {
        alpha *= 0.999;
      }
    }
  }

  return alpha;
}

/**
 * Adds a sparse matrix to the list of triplets with the given row and column
 * offset.
 *
 * @param[out] triplets The triplet storage.
 * @param[in] rowOffset The row offset for each triplet.
 * @param[in] colOffset The column offset for each triplet.
 * @param[in] mat The matrix to iterate over.
 * @param[in] transpose Whether to transpose mat.
 */
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

/**
 * Converts std::chrono::duration to a number of milliseconds rounded to three
 * decimals.
 */
template <typename Rep, typename Period = std::ratio<1>>
double ToMilliseconds(const std::chrono::duration<Rep, Period>& duration) {
  using std::chrono::duration_cast;
  using std::chrono::microseconds;
  return duration_cast<microseconds>(duration).count() / 1000.0;
}

/**
 * @brief Computes the positive root of the quadratic problem.
 *
 * @param a
 * @param b
 * @param c
 * @return positive root
 */
double QuadraticFormula(double a, double b, double c, double sign) {
  return (-b + sign * std::sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
}

/**
 * Computes an approximate solution to the problem:
 *
 *        min (1/2)xᵀGx + xᵀc
 * subject to Ax = 0
 *            |x| < Δ
 *
 * @param initialX The initial step.
 * @param G
 * @param c
 * @param A
 * @param delta The trust region size.
 */
Eigen::VectorXd ProjectedCG(Eigen::VectorXd initialGuess,
                            Eigen::SparseMatrix<double> G, Eigen::VectorXd c,
                            Eigen::SparseMatrix<double> A, double delta) {
  // P = [G  Aᵀ]
  //     [A  0 ]
  std::vector<Eigen::Triplet<double>> triplets;
  AssignSparseBlock(triplets, 0, 0, G);
  AssignSparseBlock(triplets, G.rows(), 0, A);
  AssignSparseBlock(triplets, 0, G.cols(), A, true);

  // Regularize P so it can't have zero pivots, which crashes SparseLU
  for (int row = 0; row < G.rows(); ++row) {
    triplets.emplace_back(row, row, 1e-15);
  }

  Eigen::SparseMatrix<double> P{G.rows() + A.rows(), G.cols() + A.rows()};
  P.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver{P};

  Eigen::VectorXd x = initialGuess;
  Eigen::VectorXd r = G * x + c;

  // [G  Aᵀ][g] = [r]
  // [A  0 ][v]   [0]
  Eigen::VectorXd augmentedRhs = Eigen::VectorXd::Zero(P.rows());
  augmentedRhs.topRows(r.rows()) = r;
  Eigen::VectorXd g = solver.solve(augmentedRhs).topRows(r.rows());
  Eigen::VectorXd d = -g;
  Eigen::VectorXd v;

  for (int iteration = 0; iteration < 10 * (x.rows() - A.rows()); ++iteration) {
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

    if ((x + alpha * d).norm() >= delta) {
      double A = d.squaredNorm();
      double B = 2 * x.dot(d);
      double C = std::min(x.squaredNorm() - delta * delta, 0.0);
      double tau1 = QuadraticFormula(A, B, C, -1);
      double tau2 = QuadraticFormula(A, B, C, 1);
      Eigen::VectorXd x1 = x + tau1 * d;
      Eigen::VectorXd x2 = x + tau2 * d;
      if (0.5 * x1.dot(G * x1) + x1.dot(c) < 0.5 * x2.dot(G * x2) + x2.dot(c)) {
        return x1;
      } else {
        return x2;
      }
    }

    x += alpha * d;
    r += alpha * G * d;

    // [G  Aᵀ][g⁺] = [r⁺]
    // [A  0 ][v⁺]   [0 ]
    augmentedRhs.topRows(r.rows()) = r;
    Eigen::VectorXd augmentedSol = solver.solve(augmentedRhs);
    g = augmentedSol.topRows(r.rows());
    v = augmentedSol.bottomRows(A.rows());

    // Iteratively refine step to reduce constraint violation.
    //
    // [G  Aᵀ][Δg⁺] = [p_g]
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
    d = -g + beta * d;

    if (std::abs(r.dot(g)) < 1e-6) {
      return x;
    }
  }

  return x;
}
}  // namespace

OptimizationProblem::OptimizationProblem() noexcept {
  m_decisionVariables.reserve(1024);
  m_equalityConstraints.reserve(1024);
  m_inequalityConstraints.reserve(1024);
}

Variable OptimizationProblem::DecisionVariable() {
  m_decisionVariables.emplace_back(0.0);
  return m_decisionVariables.back();
}

VariableMatrix OptimizationProblem::DecisionVariable(int rows, int cols) {
  m_decisionVariables.reserve(m_decisionVariables.size() + rows * cols);

  VariableMatrix vars{rows, cols};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      m_decisionVariables.emplace_back(0.0);
      vars(row, col) = m_decisionVariables.back();
    }
  }

  return vars;
}

void OptimizationProblem::Minimize(const Variable& cost) {
  m_f = cost;
}

void OptimizationProblem::Minimize(Variable&& cost) {
  m_f = std::move(cost);
}

void OptimizationProblem::Maximize(const Variable& objective) {
  // Maximizing an objective function is the same as minimizing its negative
  m_f = -objective;
}

void OptimizationProblem::Maximize(Variable&& objective) {
  // Maximizing an objective function is the same as minimizing its negative
  m_f = -std::move(objective);
}

void OptimizationProblem::SubjectTo(EqualityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_equalityConstraints.reserve(m_equalityConstraints.size() + storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_equalityConstraints.emplace_back(std::move(storage[i]));
  }
}

void OptimizationProblem::SubjectTo(InequalityConstraints&& constraint) {
  auto& storage = constraint.constraints;

  m_inequalityConstraints.reserve(m_inequalityConstraints.size() +
                                  storage.size());

  for (size_t i = 0; i < storage.size(); ++i) {
    m_inequalityConstraints.emplace_back(std::move(storage[i]));
  }
}

SolverStatus OptimizationProblem::Solve(const SolverConfig& config) {
  m_config = config;

  // Create the initial value column vector
  Eigen::VectorXd x{m_decisionVariables.size(), 1};
  for (size_t i = 0; i < m_decisionVariables.size(); ++i) {
    x(i) = m_decisionVariables[i].Value();
  }

  SolverStatus status;

  // Get f's expression type
  if (m_f.has_value()) {
    status.costFunctionType = m_f.value().Type();
  }

  // Get the highest order equality constraint expression type
  for (const auto& constraint : m_equalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.equalityConstraintType < constraintType) {
      status.equalityConstraintType = constraintType;
    }
  }

  // Get the highest order inequality constraint expression type
  for (const auto& constraint : m_inequalityConstraints) {
    auto constraintType = constraint.Type();
    if (status.inequalityConstraintType < constraintType) {
      status.inequalityConstraintType = constraintType;
    }
  }

  if (m_config.diagnostics) {
    constexpr std::array<const char*, 5> kExprTypeToName = {
        "empty", "constant", "linear", "quadratic", "nonlinear"};

    fmt::print("The cost function is {}.\n",
               kExprTypeToName[static_cast<int>(status.costFunctionType)]);
    fmt::print(
        "The equality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.equalityConstraintType)]);
    fmt::print(
        "The inequality constraints are {}.\n",
        kExprTypeToName[static_cast<int>(status.inequalityConstraintType)]);
    fmt::print("\n");
  }

  // If the problem is empty or constant, there's nothing to do
  if (status.costFunctionType <= ExpressionType::kConstant &&
      status.equalityConstraintType <= ExpressionType::kConstant &&
      status.inequalityConstraintType <= ExpressionType::kConstant) {
    return status;
  }

  // If there's no cost function, make it zero and continue
  if (!m_f.has_value()) {
    m_f = 0.0;
  }

  // Solve the optimization problem
  Eigen::VectorXd solution = InteriorPoint(x, &status);

  if (m_config.diagnostics) {
    fmt::print("Exit condition: ");
    if (status.exitCondition == SolverExitCondition::kOk) {
      fmt::print("optimal solution found");
    } else if (status.exitCondition == SolverExitCondition::kTooFewDOFs) {
      fmt::print("problem has too few degrees of freedom");
    } else if (status.exitCondition ==
               SolverExitCondition::kLocallyInfeasible) {
      fmt::print("problem is locally infeasible");
    } else if (status.exitCondition == SolverExitCondition::kMaxIterations) {
      fmt::print("maximum iterations exceeded");
    } else if (status.exitCondition == SolverExitCondition::kTimeout) {
      fmt::print("solution returned after timeout");
    }
    fmt::print("\n");
  }

  // Assign the solution to the original Variable instances
  SetAD(m_decisionVariables, solution);

  return status;
}

Eigen::VectorXd OptimizationProblem::InteriorPoint(
    const Eigen::Ref<const Eigen::VectorXd>& initialGuess,
    SolverStatus* status) {
  // Let f(x)ₖ be the cost function, cₑ(x)ₖ be the equality constraints, and
  // cᵢ(x)ₖ be the inequality constraints. The Lagrangian of the optimization
  // problem is
  //
  //   L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
  //
  // Let H be the Hessian of the Lagrangian.
  //
  // The primal formulation takes the form [2]:
  //
  //                       m
  //          min f(x) - μ(Σ ln(sᵢ))
  //                      i=1
  //   subject to cₑ(x) = 0
  //              cᵢ(x) - s = 0
  //
  // Where m is the number of inequality constraints, and μ is the "barrier
  // parameter".
  //
  // Redefine the iterate step as
  //
  //   p = [p_x] = [  d_x ]
  //       [p_s]   [S⁻¹d_s]
  //
  // Constraint value vector
  //
  //   c = [  cₑ  ]
  //       [cᵢ - s]
  //
  // Constraint Jacobian
  //
  //   A = [Aₑ  0]
  //       [Aᵢ -S]
  //
  // Gradient of the objective function with respect to p.
  //
  //   Φ = [ ∇f]
  //       [-μe]
  //
  // Hessian of the objective function with respect to p.
  //
  //   W = [H   0]
  //       [0   S]
  //
  // Approximate the objective function and constraints as the quadratic
  // programming problem shown in equation 19.33 of [1].
  //
  //          min ½pᵀWp + pᵀΦ             (1a)
  //   subject to Ap + c = r              (1b)
  //              ||p|| < Δ               (1c)
  //              pₛ > -(τ/2)e            (1d)
  //
  // An inexact solution to the subproblem is computed in two stages.
  // The residual r is first computed from the subproblem:
  //
  //          min ||Av + c||             (2a)
  //   subject to ||v|| < 0.8Δ           (2b)
  //              vₛ > -(τ/2)e           (2c)
  //
  //   r = Av + c
  //
  // The constraints (1d) and (2c) are equivelent to the "fraction to the
  // boundary" rule, and are applied by backtracking the solution vector.
  //
  // The iterates are applied like so
  //
  //   xₖ₊₁ = xₖ + αₖpₖˣ
  //   sₖ₊₁ = sₖ + αₖpₖˢ
  //
  // [1] Nocedal, J. and Wright, S. "Numerical Optimization", 2nd. ed., Ch. 19.
  //     Springer, 2006.
  // [2] Byrd, R. and Gilbert, J. and Nocedal, J. "A Trust Region Method Based
  //     on Interior Point Techniques for Nonlinear Programming", 1998.
  //     http://users.iems.northwestern.edu/~nocedal/PDFfiles/byrd-gilb.pdf
  // [3] Wächter, A. and Biegler, L. "On the implementation of an interior-point
  //     filter line-search algorithm for large-scale nonlinear programming",
  //     2005. http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf
  // [4] Byrd, R. and Nocedal J. and Waltz R. "KNITRO: An Integrated Package for
  //     Nonlinear Optimization", 2005.
  //     https://users.iems.northwestern.edu/~nocedal/PDFfiles/integrated.pdf

  auto solveStartTime = std::chrono::system_clock::now();

  if (m_config.diagnostics) {
    fmt::print("Number of decision variables: {}\n",
               m_decisionVariables.size());
    fmt::print("Number of equality constraints: {}\n",
               m_equalityConstraints.size());
    fmt::print("Number of inequality constraints: {}\n\n",
               m_inequalityConstraints.size());
  }

  // Barrier parameter scale factor κ_μ for tolerance checks
  constexpr double kappa_epsilon = 10.0;

  // Fraction-to-the-boundary rule scale factor minimum
  constexpr double tau_min = 0.99;

  // Barrier parameter linear decrease power in "κ_μ μ". Range of (0, 1).
  constexpr double kappa_mu = 0.2;

  // Barrier parameter superlinear decrease power in "μ^(θ_μ)". Range of (1, 2).
  constexpr double theta_mu = 1.5;

  // Barrier parameter μ
  double mu = 0.1;
  double old_mu = mu;

  // Fraction-to-the-boundary rule scale factor τ
  double tau = tau_min;

  // Trust region maximum size Δ̂
  constexpr double delta_max = 1e3;

  // Trust region size Δ
  double delta = delta_max;

  // Merit function threshold η [0, 1/4). Larger values make step acceptance
  // stricter.
  constexpr double eta = 0.1;

  Eigen::VectorXd p;

  std::vector<Eigen::Triplet<double>> triplets;

  Eigen::VectorXd x = initialGuess;
  MapVectorXvar xAD(m_decisionVariables.data(), m_decisionVariables.size());

  Eigen::VectorXd s = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  VectorXvar sAD = VectorXvar::Ones(m_inequalityConstraints.size());

  Eigen::VectorXd y = Eigen::VectorXd::Zero(m_equalityConstraints.size());
  VectorXvar yAD = VectorXvar::Zero(m_equalityConstraints.size());

  Eigen::VectorXd z = Eigen::VectorXd::Ones(m_inequalityConstraints.size());
  VectorXvar zAD = VectorXvar::Ones(m_inequalityConstraints.size());

  MapVectorXvar c_eAD(m_equalityConstraints.data(),
                      m_equalityConstraints.size());
  MapVectorXvar c_iAD(m_inequalityConstraints.data(),
                      m_inequalityConstraints.size());

  const Eigen::MatrixXd e = Eigen::VectorXd::Ones(s.rows());

  // L(x, s, y, z)ₖ = f(x)ₖ − yₖᵀcₑ(x)ₖ − zₖᵀ(cᵢ(x)ₖ − sₖ)
  Variable L = m_f.value();
  if (m_equalityConstraints.size() > 0) {
    L -= yAD.transpose() * c_eAD;
  }
  if (m_inequalityConstraints.size() > 0) {
    L -= zAD.transpose() * (c_iAD - sAD);
  }
  ExpressionGraph graphL{L};

  Eigen::VectorXd step = Eigen::VectorXd::Zero(x.rows());

  SetAD(xAD, x);
  graphL.Update();

  Gradient gradientF{m_f.value(), xAD};
  Hessian hessianL{L, xAD};
  Jacobian jacobianCe{c_eAD, xAD};
  Jacobian jacobianCi{c_iAD, xAD};

  // Error estimate E_μ
  double E_mu = std::numeric_limits<double>::infinity();

  // Gradient of f ∇f
  Eigen::SparseVector<double> g = gradientF.Calculate();

  // Hessian of the Lagrangian H
  //
  // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
  Eigen::SparseMatrix<double> H = hessianL.Calculate();

  // Equality constraints cₑ
  Eigen::VectorXd c_e = GetAD(m_equalityConstraints);

  // Inequality constraints cᵢ
  Eigen::VectorXd c_i = GetAD(m_inequalityConstraints);

  Filter filter{FilterEntry(m_f.value(), mu, s, c_e, c_i)};

  // Equality constraint Jacobian Aₑ
  //
  //         [∇ᵀcₑ₁(x)ₖ]
  // Aₑ(x) = [∇ᵀcₑ₂(x)ₖ]
  //         [    ⋮    ]
  //         [∇ᵀcₑₘ(x)ₖ]
  Eigen::SparseMatrix<double> A_e = jacobianCe.Calculate();

  // Inequality constraint Jacobian Aᵢ
  //
  //         [∇ᵀcᵢ₁(x)ₖ]
  // Aᵢ(x) = [∇ᵀcᵢ₂(x)ₖ]
  //         [    ⋮    ]
  //         [∇ᵀcᵢₘ(x)ₖ]
  Eigen::SparseMatrix<double> A_i = jacobianCi.Calculate();

  triplets.clear();
  for (int row = 0; row < c_e.rows() + c_i.rows(); ++row) {
    triplets.emplace_back(row, row, 1e-15);
  }
  Eigen::SparseMatrix<double> penaltyMatrix{c_e.rows() + c_i.rows(),
                                            c_e.rows() + c_i.rows()};
  penaltyMatrix.setFromTriplets(triplets.begin(), triplets.end());

  auto iterationsStartTime = std::chrono::system_clock::now();

  if (m_config.diagnostics) {
    // Print number of nonzeros in Lagrangian Hessian and constraint Jacobians
    std::string prints;

    if (status->costFunctionType <= ExpressionType::kQuadratic &&
        status->equalityConstraintType <= ExpressionType::kQuadratic &&
        status->inequalityConstraintType <= ExpressionType::kQuadratic) {
      prints += fmt::format("Number of nonzeros in Lagrangian Hessian: {}\n",
                            H.nonZeros());
    }
    if (status->equalityConstraintType <= ExpressionType::kLinear) {
      prints += fmt::format(
          "Number of nonzeros in equality constraint Jacobian: {}\n",
          A_e.nonZeros());
    }
    if (status->inequalityConstraintType <= ExpressionType::kLinear) {
      prints += fmt::format(
          "Number of nonzeros in inequality constraint Jacobian: {}\n",
          A_i.nonZeros());
    }

    if (prints.length() > 0) {
      fmt::print("{}\n", prints);
    }

    fmt::print("Error tolerance: {}\n\n", m_config.tolerance);
  }

  // Check for overconstrained problem
  if (m_equalityConstraints.size() > m_decisionVariables.size()) {
    fmt::print("The problem has too few degrees of freedom.\n");
    fmt::print("Violated constraints (cₑ(x) = 0) in order of declaration:\n");
    for (int row = 0; row < c_e.rows(); ++row) {
      if (c_e(row) < 0.0) {
        fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
      }
    }

    status->exitCondition = SolverExitCondition::kTooFewDOFs;
    return x;
  }

  int iterations = 0;

  scope_exit exit{[&] {
    if (m_config.diagnostics) {
      auto solveEndTime = std::chrono::system_clock::now();

      fmt::print("\nSolve time: {} ms\n",
                 ToMilliseconds(solveEndTime - solveStartTime));
      fmt::print("  ↳ {} ms (IPM setup)\n",
                 ToMilliseconds(iterationsStartTime - solveStartTime));
      if (iterations > 0) {
        fmt::print(
            "  ↳ {} ms ({} IPM iterations; {} ms average)\n",
            ToMilliseconds(solveEndTime - iterationsStartTime), iterations,
            ToMilliseconds((solveEndTime - iterationsStartTime) / iterations));
      }
      fmt::print("\n");

      constexpr auto format = "{:>8}  {:>10}  {:>14}  {:>6}\n";
      fmt::print(format, "autodiff", "setup (ms)", "avg solve (ms)", "solves");
      fmt::print("{:=^44}\n", "");
      fmt::print(format, "∇f(x)", gradientF.GetProfiler().SetupDuration(),
                 gradientF.GetProfiler().AverageSolveDuration(),
                 gradientF.GetProfiler().SolveMeasurements());
      fmt::print(format, "∇²ₓₓL", hessianL.GetProfiler().SetupDuration(),
                 hessianL.GetProfiler().AverageSolveDuration(),
                 hessianL.GetProfiler().SolveMeasurements());
      fmt::print(format, "∂cₑ/∂x", jacobianCe.GetProfiler().SetupDuration(),
                 jacobianCe.GetProfiler().AverageSolveDuration(),
                 jacobianCe.GetProfiler().SolveMeasurements());
      fmt::print(format, "∂cᵢ/∂x", jacobianCi.GetProfiler().SetupDuration(),
                 jacobianCi.GetProfiler().AverageSolveDuration(),
                 jacobianCi.GetProfiler().SolveMeasurements());
      fmt::print("\n");
    }
  }};

  while (E_mu > m_config.tolerance) {
    while (true) {
      auto innerIterStartTime = std::chrono::system_clock::now();
      //     [s₁ 0 ⋯ 0 ]
      // S = [0  ⋱   ⋮ ]
      //     [⋮    ⋱ 0 ]
      //     [0  ⋯ 0 sₘ]
      Eigen::SparseMatrix<double> S = SparseDiagonal(s);

      //         [∇ᵀcₑ₁(x)ₖ]
      // Aₑ(x) = [∇ᵀcₑ₂(x)ₖ]
      //         [    ⋮    ]
      //         [∇ᵀcₑₘ(x)ₖ]
      A_e = jacobianCe.Calculate();

      //         [∇ᵀcᵢ₁(x)ₖ]
      // Aᵢ(x) = [∇ᵀcᵢ₂(x)ₖ]
      //         [    ⋮    ]
      //         [∇ᵀcᵢₘ(x)ₖ]
      A_i = jacobianCi.Calculate();

      // A = [Aₑ   0]
      //     [Aᵢ  -S]
      triplets.clear();
      AssignSparseBlock(triplets, 0, 0, A_e);
      AssignSparseBlock(triplets, A_e.rows(), 0, A_i);
      AssignSparseBlock(triplets, A_e.rows(), A_i.cols(), -S);
      Eigen::SparseMatrix<double> A{A_e.rows() + A_i.rows(),
                                    A_i.cols() + S.cols()};
      A.setFromTriplets(triplets.begin(), triplets.end());

      // Update cₑ and cᵢ
      c_e = GetAD(m_equalityConstraints);
      c_i = GetAD(m_inequalityConstraints);

      // c = [  cₑ  ]
      //     [cᵢ - s]
      Eigen::VectorXd c{c_e.rows() + c_i.rows()};
      c.topRows(c_e.rows()) = c_e;
      c.bottomRows(c_i.rows()) = c_i - s;

      // LDLᵀ factorization of AAᵀ. A penalty formulation is used to ensure the
      // existance of a solution.
      // TODO: implement exact minimum norm solver; the penalty formulation is
      // fast and works well, but could lose accuracy.
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> AATsolver{
          penaltyMatrix + A * A.transpose()};

      // Gradient of f ∇f
      g = gradientF.Calculate();

      // [ ∇f]
      // [-µe]
      Eigen::VectorXd phi{H.rows() + S.rows()};
      phi.topRows(g.rows()) = g;
      phi.bottomRows(e.rows()) = -mu * e;

      // Compute lagrange multipliers from least squares estimate.
      //
      // AAᵀ[y] = A[ ∇f]
      //    [z]    [-µe]
      Eigen::VectorXd multipliers = AATsolver.solve(A * phi);
      y = multipliers.topRows(y.rows());
      z = multipliers.bottomRows(z.rows());
      for (int i = 0; i < z.rows(); ++i) {
        if (z(i) <= 0) {
          z(i) = std::min(1e-3, mu / s(i));
        }
      }
      SetAD(yAD, y);
      SetAD(zAD, z);
      graphL.Update();

      //     [z₁ 0 ⋯ 0 ]
      // Z = [0  ⋱   ⋮ ]
      //     [⋮    ⋱ 0 ]
      //     [0  ⋯ 0 zₘ]
      triplets.clear();
      for (int k = 0; k < s.rows(); k++) {
        triplets.emplace_back(k, k, z[k]);
      }
      Eigen::SparseMatrix<double> Z{z.rows(), z.rows()};
      Z.setFromTriplets(triplets.begin(), triplets.end());

      // [H   0]
      // [0  ZS]
      triplets.clear();
      AssignSparseBlock(triplets, 0, 0, H);
      AssignSparseBlock(triplets, H.rows(), H.cols(), Z * S);
      Eigen::SparseMatrix<double> W{x.rows() + s.rows(), x.rows() + s.rows()};
      W.setFromTriplets(triplets.begin(), triplets.end());

      // Hₖ = ∇²ₓₓL(x, s, y, z)ₖ
      H = hessianL.Calculate();

      // Check for problem local infeasibility. The problem is locally
      // infeasible if
      //
      //   Aₑᵀcₑ → 0
      //   Aᵢᵀcᵢ⁺ → 0
      //   ||(cₑ, cᵢ⁺)|| > ε
      //
      // where cᵢ⁺ = min(cᵢ, 0).
      //
      // See "Infeasibility detection" in section 6 of [4].
      //
      // cᵢ⁺ is used instead of cᵢ⁻ from the paper to follow the convention that
      // feasible inequality constraints are ≥ 0.
      if (m_equalityConstraints.size() > 0 &&
          (A_e.transpose() * c_e).norm() < 1e-6 && c_e.norm() > 1e-2) {
        if (m_config.diagnostics) {
          fmt::print(
              "The problem is locally infeasible due to violated equality "
              "constraints.\n");
          fmt::print(
              "Violated constraints (cₑ(x) = 0) in order of declaration:\n");
          for (int row = 0; row < c_e.rows(); ++row) {
            if (c_e(row) < 0.0) {
              fmt::print("  {}/{}: {} = 0\n", row + 1, c_e.rows(), c_e(row));
            }
          }
        }

        status->exitCondition = SolverExitCondition::kLocallyInfeasible;
        return x;
      }
      if (m_inequalityConstraints.size() > 0) {
        Eigen::VectorXd c_i_plus = c_i.cwiseMin(0.0);
        if ((A_i.transpose() * c_i_plus).norm() < 1e-6 &&
            c_i_plus.norm() > 1e-6) {
          if (m_config.diagnostics) {
            fmt::print(
                "The problem is infeasible due to violated inequality "
                "constraints.\n");
            fmt::print(
                "Violated constraints (cᵢ(x) ≥ 0) in order of declaration:\n");
            for (int row = 0; row < c_i.rows(); ++row) {
              if (c_i(row) < 0.0) {
                fmt::print("  {}/{}: {} ≥ 0\n", row + 1, c_i.rows(), c_i(row));
              }
            }
          }

          status->exitCondition = SolverExitCondition::kLocallyInfeasible;
          return x;
        }
      }

      // s_d = max(sₘₐₓ, (||y||₁ + ||z||₁) / (m + n)) / sₘₐₓ
      constexpr double s_max = 100.0;
      double s_d = std::max(s_max, (y.lpNorm<1>() + z.lpNorm<1>()) /
                                       (m_equalityConstraints.size() +
                                        m_inequalityConstraints.size())) /
                   s_max;

      // s_c = max(sₘₐₓ, ||z||₁ / n) / sₘₐₓ
      double s_c =
          std::max(s_max, z.lpNorm<1>() / m_inequalityConstraints.size()) /
          s_max;

      // Update the error estimate using the KKT conditions from equations
      // (19.5a) through (19.5d) in [1].
      //
      //   ∇f − Aₑᵀy − Aᵢᵀz = 0
      //   Sz − μe = 0
      //   cₑ = 0
      //   cᵢ − s = 0
      //
      // The error tolerance is the max of the following infinity norms scaled
      // by s_d and s_c (see equation (5) in [3]).
      //
      //   ||∇f − Aₑᵀy − Aᵢᵀz||_∞ / s_d
      //   ||Sz − μe||_∞ / s_c
      //   ||cₑ||_∞
      //   ||cᵢ − s||_∞
      Eigen::VectorXd eq1 = g;
      if (m_equalityConstraints.size() > 0) {
        eq1 -= A_e.transpose() * y;
      }
      if (m_inequalityConstraints.size() > 0) {
        eq1 -= A_i.transpose() * z;
      }
      E_mu = std::max(eq1.lpNorm<Eigen::Infinity>() / s_d,
                      (S * z - old_mu * e).lpNorm<Eigen::Infinity>() / s_c);
      if (m_equalityConstraints.size() > 0) {
        E_mu = std::max(E_mu, c_e.lpNorm<Eigen::Infinity>());
      }
      if (m_inequalityConstraints.size() > 0) {
        E_mu = std::max(E_mu, (c_i - s).lpNorm<Eigen::Infinity>());
      }

      if (E_mu <= kappa_epsilon * old_mu) {
        break;
      }

      triplets.clear();
      for (int col = 0; col < A.cols(); ++col) {
        triplets.emplace_back(col, col, 1e-15);
      }
      Eigen::SparseMatrix<double> penaltyMatrix2{A.cols(), A.cols()};
      penaltyMatrix2.setFromTriplets(triplets.begin(), triplets.end());

      // Compute minimum norm solution of Ap_b + c = 0.
      //
      // Ap_b + c = 0
      // Ap_b = -c
      // AᵀAp_b = -Aᵀc
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ATAsolver{
          penaltyMatrix2 + A.transpose() * A};
      Eigen::VectorXd p_b = ATAsolver.solve(-A.transpose() * c);

      // Dogleg method https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method
      if (p_b.norm() <= delta) {
        p = p_b;
      } else {
        Eigen::VectorXd d_p = -A.transpose() * c;
        Eigen::VectorXd p_u =
            d_p * (d_p.squaredNorm() / (A * d_p).squaredNorm());
        if (p_u.norm() > delta) {
          p = delta * d_p / d_p.norm();
        } else {
          double dogleg_tau = QuadraticFormula(
              (p_b - p_u).squaredNorm(), 2 * p_u.dot(p_b - p_u),
              p_u.squaredNorm() - delta * delta, 1);
          p = p_u + dogleg_tau * (p_b - p_u);
        }
      }

      p *= FractionToTheBoundaryRule(p.bottomRows(s.rows()), tau / 2);

      // See algorithm 4.1 in [1]
      {
        step = ProjectedCG(p, W, phi, A, delta);

        step *= FractionToTheBoundaryRule(step.bottomRows(s.rows()), tau);

        double fOld = m_f.value().Value();
        SetAD(xAD, x + step.segment(0, x.rows()));
        m_f.value().Update();
        double fNew = m_f.value().Value();

        // Merit function for the step p
        auto m = [&](const Eigen::VectorXd& p) {
          return fOld + g.transpose() * p + 0.5 * p.transpose() * H * p;
        };

        // The merit rho
        double rho = (fOld - fNew) / (fOld - m(step.segment(0, x.rows())));

        if (rho < 0.25) {
          delta *= 0.25;
        } else {
          if (rho > 0.75 && step.segment(0, x.rows()).norm() == delta) {
            delta = std::min(2.0 * delta, delta_max);
          }
        }

        // If rho > eta, accept the step
        if (rho > eta) {
          x += step.segment(0, x.rows());
          s += S * step.segment(x.rows(), s.rows());

          SetAD(xAD, x);
          SetAD(sAD, s);
          graphL.Update();
        }
      }

      auto innerIterEndTime = std::chrono::system_clock::now();

      if (m_config.diagnostics) {
        if (iterations % 20 == 0) {
          fmt::print("{:>4}   {:>10}  {:>10}   {:>16}  {:>19}\n", "iter",
                     "time (ms)", "error", "objective", "infeasibility");
          fmt::print("{:=^70}\n", "");
        }
        fmt::print("{:>4}  {:>9}  {:>15e}  {:>16e}   {:>16e}\n", iterations,
                   ToMilliseconds(innerIterEndTime - innerIterStartTime), E_mu,
                   m_f.value().Value(),
                   std::sqrt(c_e.squaredNorm() + (c_i - s).squaredNorm()));
      }

      ++iterations;
      if (iterations >= m_config.maxIterations) {
        status->exitCondition = SolverExitCondition::kMaxIterations;
        return x;
      }

      if (innerIterEndTime - solveStartTime > m_config.timeout) {
        status->exitCondition = SolverExitCondition::kTimeout;
        return x;
      }
    }

    // Update the barrier parameter.
    //
    //   μⱼ₊₁ = max(εₜₒₗ/10, min(κ_μ μⱼ, μⱼ^θ_μ))
    //
    // See equation (7) in [3].
    old_mu = mu;
    mu = std::max(m_config.tolerance / 10.0,
                  std::min(kappa_mu * mu, std::pow(mu, theta_mu)));

    // Update the fraction-to-the-boundary rule scaling factor.
    //
    //   τⱼ = max(τₘᵢₙ, 1 − μⱼ)
    //
    // See equation (8) in [3].
    tau = std::max(tau_min, 1.0 - mu);

    // Reset the filter when the barrier parameter is updated.
    filter.ResetFilter(FilterEntry(m_f.value(), mu, s, c_e, c_i));
  }

  if (m_config.diagnostics) {
    fmt::print("{:>4}  {:>9}  {:>15e}  {:>16e}   {:>16e}\n", iterations, 0.0,
               E_mu, m_f.value().Value() - mu * s.array().log().sum(),
               c_e.lpNorm<1>() + (c_i - s).lpNorm<1>());
  }

  return x;
}
