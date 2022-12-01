// Copyright (c) Joshua Nichols and Tyler Veness

#include "sleipnir/autodiff/Variable.hpp"

#include <cmath>
#include <numbers>
#include <tuple>
#include <type_traits>
#include <vector>

#include <fmt/core.h>

#include "sleipnir/SymbolExports.hpp"
#include "sleipnir/autodiff/ExpressionGraph.hpp"

// https://en.cppreference.com/w/cpp/utility/to_underlying from C++23
template <class Enum>
constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

namespace sleipnir {

Variable& Zero() {
  static Variable var{ZeroExpr()};
  return var;
}

Variable& One() {
  static Variable var{OneExpr()};
  return var;
}

Variable::Variable(double value)
    : expr{AllocateIntrusiveShared<Expression>(
          GlobalPoolAllocator<Expression>(), value)} {}

Variable::Variable(int value)
    : expr{AllocateIntrusiveShared<Expression>(
          GlobalPoolAllocator<Expression>(), value)} {}

Variable::Variable(IntrusiveSharedPtr<Expression> expr)
    : expr{std::move(expr)} {}

Variable& Variable::operator=(double value) {
  if (expr == ZeroExpr()) {
    expr = AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), value);
  } else {
    if (expr->args[0] != ZeroExpr()) {
      fmt::print(stderr,
                 "WARNING: {}:{}: Modified the value of a dependent variable\n",
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

Variable& Variable::operator=(int value) {
  if (expr == ZeroExpr()) {
    expr = AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), value);
  } else {
    if (expr->args[0] != ZeroExpr()) {
      fmt::print(stderr,
                 "WARNING: {}:{}: Modified the value of a dependent variable\n",
                 __FILE__, __LINE__);
    }
    expr->value = value;
  }
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator*(double lhs, const Variable& rhs) {
  if (lhs == 0.0) {
    return Zero();
  } else if (lhs == 1.0) {
    return rhs;
  }

  return MakeConstant(lhs) * rhs;
}

SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs, double rhs) {
  if (rhs == 0.0) {
    return Zero();
  } else if (rhs == 1.0) {
    return lhs;
  }

  return lhs * MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT Variable operator*(const Variable& lhs,
                                      const Variable& rhs) {
  if (lhs.expr == ZeroExpr() || rhs.expr == ZeroExpr()) {
    return Zero();
  }

  if (lhs.Type() == ExpressionType::kConstant) {
    if (lhs.Value() == 1.0) {
      return rhs;
    } else if (lhs.Value() == 0.0) {
      return Zero();
    }
  }

  if (rhs.Type() == ExpressionType::kConstant) {
    if (rhs.Value() == 1.0) {
      return lhs;
    } else if (rhs.Value() == 0.0) {
      return Zero();
    }
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (lhs.Type() == ExpressionType::kConstant) {
    type = rhs.Type();
  } else if (rhs.Type() == ExpressionType::kConstant) {
    type = lhs.Type();
  } else if (lhs.Type() == ExpressionType::kLinear &&
             rhs.Type() == ExpressionType::kLinear) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double lhs, double rhs) { return lhs * rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * lhs;
      },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint * rhs; },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint * lhs; },
      lhs, rhs)};
}

Variable& Variable::operator*=(double rhs) {
  *this = *this * rhs;
  return *this;
}

Variable& Variable::operator*=(const Variable& rhs) {
  *this = *this * rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator/(double lhs, const Variable& rhs) {
  if (lhs == 0.0) {
    return Zero();
  }

  return MakeConstant(lhs) / rhs;
}

SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs, double rhs) {
  return lhs / MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT Variable operator/(const Variable& lhs,
                                      const Variable& rhs) {
  if (lhs.expr == ZeroExpr()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (rhs.Type() == ExpressionType::kConstant) {
    type = lhs.Type();
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double lhs, double rhs) { return lhs / rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint / rhs;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint / rhs; },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) {
        return parentAdjoint * -lhs / (rhs * rhs);
      },
      lhs, rhs)};
}

Variable& Variable::operator/=(double rhs) {
  *this = *this / rhs;
  return *this;
}

Variable& Variable::operator/=(const Variable& rhs) {
  *this = *this / rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator+(double lhs, const Variable& rhs) {
  if (lhs == 0.0) {
    return rhs;
  }

  return MakeConstant(lhs) + rhs;
}

SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs + MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs,
                                      const Variable& rhs) {
  if (lhs.IsZero()) {
    return rhs;
  } else if (rhs.IsZero()) {
    return lhs;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(),
      ExpressionType{
          std::max(to_underlying(lhs.Type()), to_underlying(rhs.Type()))},
      [](double lhs, double rhs) { return lhs + rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint; },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint; },
      lhs, rhs)};
}

Variable& Variable::operator+=(double rhs) {
  if (rhs == 0.0) {
    return *this;
  }

  return *this += MakeConstant(rhs);
}

Variable& Variable::operator+=(const Variable& rhs) {
  if (IsZero()) {
    expr = rhs.expr;
  } else if (rhs.IsZero()) {
    return *this;
  } else {
    *this = *this + rhs;
  }

  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator-(double lhs, const Variable& rhs) {
  if (lhs == 0.0) {
    return -rhs;
  }

  return MakeConstant(lhs) - rhs;
}

SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs, double rhs) {
  if (rhs == 0.0) {
    return lhs;
  }

  return lhs - MakeConstant(rhs);
}

SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs,
                                      const Variable& rhs) {
  if (lhs.IsZero()) {
    if (!rhs.IsZero()) {
      return -rhs;
    } else {
      return Zero();
    }
  } else if (rhs.IsZero()) {
    return lhs;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(),
      ExpressionType{
          std::max(to_underlying(lhs.Type()), to_underlying(rhs.Type()))},
      [](double lhs, double rhs) { return lhs - rhs; },
      [](double lhs, double rhs, double parentAdjoint) {
        return parentAdjoint;
      },
      [](double lhs, double rhs, double parentAdjoint) {
        return -parentAdjoint;
      },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint; },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return -parentAdjoint; },
      lhs, rhs)};
}

Variable& Variable::operator-=(double rhs) {
  *this = *this - rhs;
  return *this;
}

Variable& Variable::operator-=(const Variable& rhs) {
  *this = *this - rhs;
  return *this;
}

SLEIPNIR_DLLEXPORT Variable operator-(const Variable& lhs) {
  if (lhs.IsZero()) {
    return Zero();
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), lhs.Type(),
      [](double lhs, double) { return -lhs; },
      [](double lhs, double, double parentAdjoint) { return -parentAdjoint; },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return -parentAdjoint; },
      lhs)};
}

SLEIPNIR_DLLEXPORT Variable operator+(const Variable& lhs) {
  if (lhs.IsZero()) {
    return Zero();
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), lhs.Type(),
      [](double lhs, double) { return lhs; },
      [](double lhs, double, double parentAdjoint) { return parentAdjoint; },
      [](const Variable& lhs, const Variable& rhs,
         const Variable& parentAdjoint) { return parentAdjoint; },
      lhs)};
}

double Variable::Value() const {
  return expr->value;
}

ExpressionType Variable::Type() const {
  return expr->type;
}

void Variable::Update() {
  if (!IsZero()) {
    ExpressionGraph graph{*this};
    graph.Update();
  }
}

bool Variable::IsZero() const {
  return expr == ZeroExpr();
}

bool Variable::IsOne() const {
  return expr == OneExpr();
}

Variable MakeConstant(double x) {
  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), x, ExpressionType::kConstant)};
}

Variable abs(double x) {
  return sleipnir::abs(MakeConstant(x));
}

Variable abs(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::abs(x); },
      [](double x, double, double parentAdjoint) {
        if (x < 0.0) {
          return -parentAdjoint;
        } else if (x > 0.0) {
          return parentAdjoint;
        } else {
          return 0.0;
        }
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        if (x.Value() < 0.0) {
          return -parentAdjoint;
        } else if (x.Value() > 0.0) {
          return parentAdjoint;
        } else {
          return Zero();
        }
      },
      x)};
}

Variable acos(double x) {
  return sleipnir::cos(MakeConstant(x));
}

Variable acos(const Variable& x) {
  if (x.IsZero()) {
    return MakeConstant(std::numbers::pi / 2.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::acos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return -parentAdjoint / sleipnir::sqrt(1.0 - x * x);
      },
      x)};
}

Variable asin(double x) {
  return sleipnir::asin(MakeConstant(x));
}

Variable asin(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::asin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / std::sqrt(1.0 - x * x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / sleipnir::sqrt(1.0 - x * x);
      },
      x)};
}

Variable atan(double x) {
  return sleipnir::atan(MakeConstant(x));
}

Variable atan(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::atan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / (1.0 + x * x);
      },
      x)};
}

Variable atan2(double y, const Variable& x) {
  return sleipnir::atan2(MakeConstant(y), x);
}

Variable atan2(const Variable& y, double x) {
  return sleipnir::atan2(y, MakeConstant(x));
}

Variable atan2(const Variable& y, const Variable& x) {
  if (y.IsZero()) {
    return Zero();
  } else if (x.IsZero()) {
    return MakeConstant(std::numbers::pi / 2.0);
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (y.Type() == ExpressionType::kConstant &&
      x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double y, double x) { return std::atan2(y, x); },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](double y, double x, double parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      [](const Variable& y, const Variable& x, const Variable& parentAdjoint) {
        return parentAdjoint * x / (y * y + x * x);
      },
      [](const Variable& y, const Variable& x, const Variable& parentAdjoint) {
        return parentAdjoint * -y / (y * y + x * x);
      },
      y, x)};
}

Variable cos(double x) {
  return sleipnir::cos(MakeConstant(x));
}

Variable cos(const Variable& x) {
  if (x.IsZero()) {
    return One();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::cos(x); },
      [](double x, double, double parentAdjoint) {
        return -parentAdjoint * std::sin(x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint * -sleipnir::sin(x);
      },
      x)};
}

Variable cosh(double x) {
  return sleipnir::cosh(MakeConstant(x));
}

Variable cosh(const Variable& x) {
  if (x.IsZero()) {
    return One();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::cosh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::sinh(x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint * sleipnir::sinh(x);
      },
      x)};
}

Variable erf(double x) {
  return sleipnir::erf(MakeConstant(x));
}

Variable erf(const Variable& x) {
  static constexpr double sqrt_pi =
      1.7724538509055160272981674833411451872554456638435L;

  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::erf(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * 2.0 / sqrt_pi * std::exp(-x * x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint * 2.0 / sqrt_pi * sleipnir::exp(-x * x);
      },
      x)};
}

Variable exp(double x) {
  return sleipnir::exp(MakeConstant(x));
}

Variable exp(const Variable& x) {
  if (x.IsZero()) {
    return One();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::exp(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::exp(x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint * sleipnir::exp(x);
      },
      x)};
}

Variable hypot(double x, const Variable& y) {
  return sleipnir::hypot(MakeConstant(x), y);
}

Variable hypot(const Variable& x, double y) {
  return sleipnir::hypot(x, MakeConstant(y));
}

Variable hypot(const Variable& x, const Variable& y) {
  if (x.IsZero() && y.IsZero()) {
    return Zero();
  }

  if (x.IsZero() && !y.IsZero()) {
    // Evaluate the expression's type
    ExpressionType type;
    if (y.Type() == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return Variable{AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), type,
        [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const Variable& x, const Variable& y,
           const Variable& parentAdjoint) {
          return parentAdjoint * x / sleipnir::hypot(x, y);
        },
        [](const Variable& x, const Variable& y,
           const Variable& parentAdjoint) {
          return parentAdjoint * y / sleipnir::hypot(x, y);
        },
        Zero(), y)};
  } else if (!x.IsZero() && y.IsZero()) {
    // Evaluate the expression's type
    ExpressionType type;
    if (x.Type() == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return Variable{AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), type,
        [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const Variable& x, const Variable& y,
           const Variable& parentAdjoint) {
          return parentAdjoint * x / sleipnir::hypot(x, y);
        },
        [](const Variable& x, const Variable& y,
           const Variable& parentAdjoint) {
          return parentAdjoint * y / sleipnir::hypot(x, y);
        },
        x, Zero())};
  } else {
    // Evaluate the expression's type
    ExpressionType type;
    if (x.Type() == ExpressionType::kConstant &&
        y.Type() == ExpressionType::kConstant) {
      type = ExpressionType::kConstant;
    } else {
      type = ExpressionType::kNonlinear;
    }

    return Variable{AllocateIntrusiveShared<Expression>(
        GlobalPoolAllocator<Expression>(), type,
        [](double x, double y) { return std::hypot(x, y); },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * x / std::hypot(x, y);
        },
        [](double x, double y, double parentAdjoint) {
          return parentAdjoint * y / std::hypot(x, y);
        },
        [](const Variable& x, const Variable& y,
           const Variable& parentAdjoint) {
          return parentAdjoint * x / sleipnir::hypot(x, y);
        },
        [](const Variable& x, const Variable& y,
           const Variable& parentAdjoint) {
          return parentAdjoint * y / sleipnir::hypot(x, y);
        },
        x, y)};
  }
}

Variable log(double x) {
  return sleipnir::log(MakeConstant(x));
}

Variable log(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::log(x); },
      [](double x, double, double parentAdjoint) { return parentAdjoint / x; },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / x;
      },
      x)};
}

Variable log10(double x) {
  return sleipnir::log10(MakeConstant(x));
}

Variable log10(const Variable& x) {
  static constexpr double ln10 = 2.3025850929940456840179914546843L;

  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::log10(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (ln10 * x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / (ln10 * x);
      },
      x)};
}

Variable pow(double base, const Variable& power) {
  return sleipnir::pow(MakeConstant(base), power);
}

Variable pow(const Variable& base, double power) {
  return sleipnir::pow(base, MakeConstant(power));
}

Variable pow(const Variable& base, const Variable& power) {
  if (base.IsZero()) {
    return Zero();
  }
  if (power.IsZero()) {
    return One();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (base.Type() == ExpressionType::kConstant &&
      power.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else if (power.Type() == ExpressionType::kConstant &&
             power.Value() == 0.0) {
    type = ExpressionType::kConstant;
  } else if (base.Type() == ExpressionType::kLinear &&
             power.Type() == ExpressionType::kConstant &&
             power.Value() == 1.0) {
    type = ExpressionType::kLinear;
  } else if (base.Type() == ExpressionType::kLinear &&
             power.Type() == ExpressionType::kConstant &&
             power.Value() == 2.0) {
    type = ExpressionType::kQuadratic;
  } else if (base.Type() == ExpressionType::kQuadratic &&
             power.Type() == ExpressionType::kConstant &&
             power.Value() == 1.0) {
    type = ExpressionType::kQuadratic;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double base, double power) { return std::pow(base, power); },
      [](double base, double power, double parentAdjoint) {
        return parentAdjoint * std::pow(base, power - 1) * power;
      },
      [](double base, double power, double parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base == 0.0) {
          return 0.0;
        } else {
          return parentAdjoint * std::pow(base, power - 1) * base *
                 std::log(base);
        }
      },
      [](const Variable& base, const Variable& power,
         const Variable& parentAdjoint) {
        return parentAdjoint * sleipnir::pow(base, power - 1) * power;
      },
      [](const Variable& base, const Variable& power,
         const Variable& parentAdjoint) {
        // Since x * std::log(x) -> 0 as x -> 0
        if (base.Value() == 0.0) {
          return Zero();
        } else {
          return parentAdjoint * sleipnir::pow(base, power - 1) * base *
                 sleipnir::log(base);
        }
      },
      base, power)};
}

Variable sin(double x) {
  return sleipnir::sin(MakeConstant(x));
}

Variable sin(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::sin(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cos(x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint * sleipnir::cos(x);
      },
      x)};
}

Variable sinh(double x) {
  return sleipnir::sinh(MakeConstant(x));
}

Variable sinh(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::sinh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint * std::cosh(x);
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint * sleipnir::cosh(x);
      },
      x)};
}

Variable sqrt(double x) {
  return sleipnir::sqrt(MakeConstant(x));
}

Variable sqrt(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::sqrt(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (2.0 * std::sqrt(x));
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / (2.0 * sleipnir::sqrt(x));
      },
      x)};
}

Variable tan(double x) {
  return sleipnir::tan(MakeConstant(x));
}

Variable tan(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::tan(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cos(x) * std::cos(x));
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / (sleipnir::cos(x) * sleipnir::cos(x));
      },
      x)};
}

Variable tanh(double x) {
  return sleipnir::tanh(MakeConstant(x));
}

Variable tanh(const Variable& x) {
  if (x.IsZero()) {
    return Zero();
  }

  // Evaluate the expression's type
  ExpressionType type;
  if (x.Type() == ExpressionType::kConstant) {
    type = ExpressionType::kConstant;
  } else {
    type = ExpressionType::kNonlinear;
  }

  return Variable{AllocateIntrusiveShared<Expression>(
      GlobalPoolAllocator<Expression>(), type,
      [](double x, double) { return std::tanh(x); },
      [](double x, double, double parentAdjoint) {
        return parentAdjoint / (std::cosh(x) * std::cosh(x));
      },
      [](const Variable& x, const Variable&, const Variable& parentAdjoint) {
        return parentAdjoint / (sleipnir::cosh(x) * sleipnir::cosh(x));
      },
      x)};
}

}  // namespace sleipnir
