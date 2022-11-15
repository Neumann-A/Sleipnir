// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Jacobian.hpp"

#include "sleipnir/autodiff/Gradient.hpp"

#include <iostream>

using namespace sleipnir::autodiff;

Jacobian::Jacobian(VectorXvar variables, VectorXvar wrt) noexcept
    : m_variables{std::move(variables)}, m_wrt{std::move(wrt)} {
  m_profiler.StartSetup();

  // Reserve triplet space for 99% sparsity
  m_cachedTriplets.reserve(m_variables.rows() * m_wrt.rows() * 0.01);

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  std::vector<Expression*> row;
  std::vector<Expression*> stack;
  for (int rowIndex = 0; rowIndex < m_variables.size(); ++rowIndex) {
    // BFS
    row.clear();

    stack.emplace_back(m_variables(rowIndex).expr.Get());

    // Initialize the number of instances of each node in the tree
    // (Expression::duplications)
    while (!stack.empty()) {
      auto& currentNode = stack.back();
      stack.pop_back();

      for (auto&& arg : currentNode->args) {
        // Only continue if the node is not a constant and hasn't already been
        // explored.
        if (arg != nullptr && arg->expressionType != ExpressionType::kConstant) {
          // If this is the first instance of the node encountered (it hasn't
          // been explored yet), add it to stack so it's recursed upon
          if (arg->duplications == 0) {
            stack.push_back(arg.Get());
          }
          ++arg->duplications;
        }
      }
    }

    stack.emplace_back(m_variables(rowIndex).expr.Get());

    while (!stack.empty()) {
      auto& node = stack.back();
      stack.pop_back();

      // BFS tape sorted from parent to child.
      row.emplace_back(node);
      
      for (auto&& arg : node->args) {
        // Only add node if it's not a constant and doesn't already exist in the
        // tape.
        if (arg != nullptr && arg->expressionType != ExpressionType::kConstant) {
          // Once the number of node visitations equals the number of
          // duplications (the counter hits zero), add it to the stack. Note
          // that this means the node is only enqueued once.
          --arg->duplications;
          if (arg->duplications == 0) {
            stack.emplace_back(arg.Get());
          }
        }
      }
    }

    // Cache linear paths.
    for (auto col : row) {
      col->adjoint = 0.0;
    }
    row[0]->adjoint = 1.0;
    row[0]->duplications = row[0]->operatorType == OperatorType::kLinear ? 1 : 0;

    for (auto col : row) {
      auto& lhs = col->args[0];
      auto& rhs = col->args[1];

      // If the node's number of instances in linear paths in non-zero (it exists in a linear path), push adjoints.
      if (col->duplications != 0) {
        if (lhs != nullptr) {
          // Add instance of node if it's contained in a linear path.
          lhs->duplications += lhs->operatorType == OperatorType::kLinear ? 1 : 0;
          if (rhs != nullptr) {
            rhs->duplications += rhs->operatorType == OperatorType::kLinear ? 1 : 0;
            lhs->adjoint +=
                col->gradientValueFuncs[0](lhs->value, rhs->value, col->adjoint);
            rhs->adjoint +=
                col->gradientValueFuncs[1](lhs->value, rhs->value, col->adjoint);
          } else {
            lhs->adjoint +=
                col->gradientValueFuncs[0](lhs->value, 0.0, col->adjoint);
          }
        }
      }

      if (col->row != -1) {
        m_cachedTriplets.emplace_back(rowIndex, col->row, col->adjoint);
      }
    }

    // If node is nonlinear, it's contained in nonlinear path; remove it from linear path instances.
    for (auto col : row) {
      if (col->expressionType != ExpressionType::kLinear) {
        col->duplications = 0;
      }
    }

    // Remove nodes not contained in nonlinear paths.
    for (int col = row.size() - 1; col >= 0; --col) {
      if (row[col]->duplications != 0) {
        row[col]->duplications = 0;
        row.erase(row.begin() + col);
      }
    }

    m_graph.emplace_back(std::move(row));
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_profiler.StopSetup();
}

const Eigen::SparseMatrix<double>& Jacobian::Calculate() {
  m_profiler.StartSolve();

  Update();

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = row;
  }

  // Copy the cached triplets so triplets added for the nonlinear rows are
  // thrown away at the end of the function
  auto triplets = m_cachedTriplets;

  for (int row = 0; row < m_variables.size(); ++row) {
    ComputeRow(row, triplets);
  }

  for (int row = 0; row < m_wrt.rows(); ++row) {
    m_wrt(row).expr->row = -1;
  }

  m_J.setFromTriplets(triplets.begin(), triplets.end());

  m_profiler.StopSolve();

  return m_J;
}

void Jacobian::Update(IntrusiveSharedPtr<Expression> node) { 
  // Only update node if it hasn't already been updated.
  if (node->duplications == 0) {
    for (auto arg : node->args) {
      if (arg != nullptr) {
        Update(arg);
      }
    }
    ++node->duplications;
  }
  auto& lhs = node->args[0];
  auto& rhs = node->args[1];

  if (lhs != nullptr) {
    if (rhs != nullptr) {
      node->value = node->valueFunc(lhs->value, rhs->value);
    } else {
      node->value = node->valueFunc(lhs->value, 0.0);
    }
  }  
}

void Jacobian::Update() {
  std::vector<Expression*> stack;
  for (size_t row = 0; row < m_graph.size(); ++row) {
    auto& root = m_variables(row).expr;
    Update(root);
    // Zero duplications used by Update method.
    root->duplications = 0;
    stack.emplace_back(root.Get());
    while (!stack.empty()) {
      auto& currentNode = stack.back();
      stack.pop_back();

      for (auto&& arg : currentNode->args) {
        if (arg != nullptr && arg->expressionType != ExpressionType::kConstant && arg->duplications != 0) {
          stack.push_back(arg.Get());
          arg->duplications = 0;
        }
      }
    }
  }
}

Profiler& Jacobian::GetProfiler() {
  return m_profiler;
}

void Jacobian::ComputeRow(int rowIndex,
                          std::vector<Eigen::Triplet<double>>& triplets) {
  auto& row = m_graph[rowIndex];

  if (row.size() == 0) {
    return;
  }

  for (auto col : row) {
    col->adjoint = 0.0;
  }
  row[0]->adjoint = 1.0;

  for (auto col : row) {
    auto& lhs = col->args[0];
    auto& rhs = col->args[1];

    if (lhs != nullptr) {
      if (rhs != nullptr) {
        lhs->adjoint +=
            col->gradientValueFuncs[0](lhs->value, rhs->value, col->adjoint);
        rhs->adjoint +=
            col->gradientValueFuncs[1](lhs->value, rhs->value, col->adjoint);
      } else {
        lhs->adjoint +=
            col->gradientValueFuncs[0](lhs->value, 0.0, col->adjoint);
      }
    }

    if (col->row != -1) {
      triplets.emplace_back(rowIndex, col->row, col->adjoint);
    }
  }
}
