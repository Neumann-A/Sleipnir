// Copyright (c) Joshua Nichols and Tyler Veness

#pragma once

#include <vector>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/Expression.hpp"
#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir::autodiff {

/**
 * This class is an adaptor type that performs value updates of an expression's
 * computational graph in a way that skips duplicates.
 */
class SLEIPNIR_DLLEXPORT ExpressionGraph {
 public:
  /**
   * Generates the deduplicated computational graph for the given expression.
   *
   * @param root The root node of the expression.
   */
  explicit ExpressionGraph(Variable& root);

  /**
   * Update the values of all nodes in this computational tree based on the
   * values of their dependent nodes.
   */
  void Update();

  /**
   * Returns the variable's gradient tree.
   *
   * @param wrt Variables with respect to which to compute the gradient.
   */
  VectorXvar GenerateGradientTree(Eigen::Ref<VectorXvar> wrt);

  /**
   * Updates the adjoints in the expression graph, effectively computing the
   * gradient.
   *
   * @param func A function that takes two arguments: an int for the gradient
   *   row, and a double for the adjoint (gradient value).
   */
  template <typename F>
  void ComputeAdjoints(F&& func) {
    // Zero adjoints. The root node's adjoint is 1.0 as df/df is always 1.
    std::vector<double> adjoints;
    adjoints.resize(m_adjointList.size());
    adjoints[0] = 1.0;

    // df/dx = (df/dy)(dy/dx). The adjoint of x is equal to the adjoint of y
    // multiplied by dy/dx. If there are multiple "paths" from the root node to
    // variable; the variable's adjoint is the sum of each path's adjoint
    // contribution.
    for (size_t col = 0; col < m_adjointList.size(); ++col) {
      auto& node = m_adjointList[col];
      auto& lhs = node->args[0];
      auto& rhs = node->args[1];

      int lhsIndex = m_lhsList[col];
      int rhsIndex = m_rhsList[col];
      if (lhsIndex != -1) {
        adjoints[lhsIndex] +=
            node->gradientValueFuncs[0](lhs->value, rhs->value, adjoints[col]);
        if (rhsIndex != -1) {
          adjoints[rhsIndex] += node->gradientValueFuncs[1](
              lhs->value, rhs->value, adjoints[col]);
        }
      }

      // If variable is a leaf node, assign its adjoint to the gradient.
      int row = m_rowList[col];
      if (row != -1) {
        func(row, adjoints[col]);
      }
    }
  }

 private:
  // List that maps nodes to their respective row.
  std::vector<int> m_rowList;

  // List for updating adjoints
  std::vector<Expression*> m_adjointList;

  // List for updating values
  std::vector<Expression*> m_valueList;

  // List that stores the index of the lhs child.
  std::vector<size_t> m_lhsList;

  // List that stores the index of the rhs child.
  std::vector<size_t> m_rhsList;
};

}  // namespace sleipnir::autodiff
