// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Expression.hpp"

#include "sleipnir/autodiff/Variable.hpp"

namespace sleipnir {

Expression::Expression(ZeroSingleton_t)
    : adjointExpr{nullptr},
      type{ExpressionType::kConstant},
      gradientFuncs{[](const Variable&, const Variable&, const Variable&) {
                      return Zero();
                    },
                    [](const Variable&, const Variable&, const Variable&) {
                      return Zero();
                    }},
      args{nullptr, nullptr} {}

Expression::Expression(double value, ExpressionType type)
    : value{value},
      adjointExpr{ZeroExpr()},
      type{type},
      gradientFuncs{[](const Variable&, const Variable&, const Variable&) {
                      return Zero();
                    },
                    [](const Variable&, const Variable&, const Variable&) {
                      return Zero();
                    }},
      args{ZeroExpr(), ZeroExpr()} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc, const Variable& lhs)
    : value{valueFunc(lhs.Value(), 0.0)},
      adjointExpr{ZeroExpr()},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc,
                         [](double, double, double) { return 0.0; }},
      gradientFuncs{lhsGradientFunc, [](const Variable&, const Variable&,
                                        const Variable&) { return Zero(); }},
      args{lhs.expr, ZeroExpr()} {}

Expression::Expression(ExpressionType type, BinaryFuncDouble valueFunc,
                       TrinaryFuncDouble lhsGradientValueFunc,
                       TrinaryFuncDouble rhsGradientValueFunc,
                       TrinaryFuncExpr lhsGradientFunc,
                       TrinaryFuncExpr rhsGradientFunc, const Variable& lhs,
                       const Variable& rhs)
    : value{valueFunc(lhs.Value(), rhs.Value())},
      adjointExpr{ZeroExpr()},
      type{type},
      valueFunc{valueFunc},
      gradientValueFuncs{lhsGradientValueFunc, rhsGradientValueFunc},
      gradientFuncs{lhsGradientFunc, rhsGradientFunc},
      args{lhs.expr, rhs.expr} {}

IntrusiveSharedPtr<Expression>& ZeroExpr() {
  static auto expr = AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), Expression::ZeroSingleton);
  return expr;
}

IntrusiveSharedPtr<Expression>& OneExpr() {
  static auto expr = AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), 1.0, ExpressionType::kConstant);
  return expr;
}

}  // namespace sleipnir
