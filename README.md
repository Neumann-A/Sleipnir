# Sleipnir

![Build](https://github.com/SleipnirGroup/Sleipnir/actions/workflows/build.yml/badge.svg)
[![C++ Documentation](https://img.shields.io/badge/documentation-c%2B%2B-blue)](https://sleipnirgroup.github.io/Sleipnir/)
[![Discord](https://img.shields.io/discord/975739302933856277?color=%23738ADB&label=Join%20our%20Discord&logo=discord&logoColor=white)](https://discord.gg/ad2EEZZwsS)

> Sparsity and Linearity-Exploiting Interior-Point solver - Now Internally Readable

Named after Odin's eight-legged horse from Norse mythology, Sleipnir is a linearity-exploiting sparse nonlinear constrained optimization problem solver that uses the interior-point method.

```cpp
#include <fmt/core.h>
#include <sleipnir/OptimizationProblem.hpp>

int main() {
  // Find the x, y pair with the largest product for which x + 3y = 36
  sleipnir::OptimizationProblem problem;

  auto x = problem.DecisionVariable();
  auto y = problem.DecisionVariable();

  problem.Maximize(x * y);
  problem.SubjectTo(x + 3 * y == 36);
  problem.Solve();

  // x = 18.0, y = 6.0
  fmt::print("x = {}, y = {}\n", x.Value(), y.Value());
}
```

Sleipnir's internals are intended to be readable by those who aren't domain experts with links to explanatory material for its algorithms.

## Benchmarks

<table><tr>
  <td><img src="flywheel-scalability-results.png" alt="flywheel-results"/></td>
  <td><img src="cart-pole-scalability-results.png" alt="cart-pole-results"/></td>
</tr></table>

Generated by [tools/generate-scalability-results.sh](https://github.com/SleipnirGroup/Sleipnir/tree/main/tools/generate-scalability-results.sh) from [benchmarks/scalability](https://github.com/SleipnirGroup/Sleipnir/tree/main/benchmarks/scalability) source on a i5-8350U with 16 GB RAM.

### How we improved performance

#### Make more decisions at compile time

During problem setup, equality and inequality constraints are encoded as different types, so the appropriate setup behavior can be selected at compile time via operator overloads.

#### Reuse autodiff computation results that are still valid (aka caching)

The autodiff library automatically records the linearity of every node in the computational graph. Linear functions have constant first derivatives, and quadratic functions have constant second derivatives. The constant derivatives are computed in the initialization phase and reused for all solver iterations. Only nonlinear parts of the computational graph are recomputed during each solver iteration.

For quadratic problems, we compute the Lagrangian Hessian and constraint Jacobians once with no problem structure hints from the user.

#### Use a performant linear algebra library with fast sparse solvers

[Eigen](https://gitlab.com/libeigen/eigen/) provides these. It also has no required dependencies, which makes cross compilation much easier.

#### Use a pool allocator for autodiff expression nodes

This promotes fast allocation/deallocation and good memory locality.

We could mitigate the solver's high last-level-cache miss rate (~42% on the machine above) further by breaking apart the expression nodes into fields that are commonly iterated together. We used to use a tape, which gave computational graph updates linear access patterns, but tapes are monotonic buffers with no way to reclaim storage.

### Running the benchmarks

Benchmark projects are in the [benchmarks folder](https://github.com/SleipnirGroup/Sleipnir/tree/main/benchmarks). To compile and run the flywheel scalability benchmark, run the following in the repository root:
```bash
# Install CasADi first
cmake -B build -S .
cmake --build build
./build/FlywheelScalabilityBenchmark

# Install matplotlib, numpy, and scipy pip packages first
./tools/plot_scalability_results.py --filename flywheel-scalability-results.csv --title Flywheel
```

## Examples

See the [examples](https://github.com/SleipnirGroup/Sleipnir/tree/main/examples) and [optimization unit tests](https://github.com/SleipnirGroup/Sleipnir/tree/main/test/optimization).

## Dependencies

* C++20 compiler
  * On Linux, install GCC 11 or greater
  * On Windows, install [Visual Studio Community 2022](https://visualstudio.microsoft.com/vs/community/) and select the C++ programming language during installation
  * On macOS, install the Xcode command-line build tools via `xcode-select --install`
* [Eigen](https://gitlab.com/libeigen/eigen)
* [fmtlib](https://github.com/fmtlib/fmt) (internal only)
* [googletest](https://github.com/google/googletest) (tests only)

Library dependencies which aren't installed locally will be automatically downloaded and built by CMake.

If [CasADi](https://github.com/casadi/casadi) is installed locally, the benchmark executables will be built.

## Build instructions

Starting from the repository root, run the configure step:
```bash
cmake -B build -S .
```

This will automatically download library dependencies.

Run the build step:
```bash
cmake --build build
```

Run the tests:
```bash
cd build
ctest
```

### Supported build types

The following build types can be specified via `-DCMAKE_BUILD_TYPE`:

* Debug
  * Optimizations off
  * Debug symbols on
* Release
  * Optimizations on
  * Debug symbols off
* RelWithDebInfo (default)
  * Release build type, but with debug info
* MinSizeRel
  * Minimum size release build
* Asan
  * Enables address sanitizer
* Tsan
  * Enables thread sanitizer
* Ubsan
  * Enables undefined behavior sanitizer
* Perf
  * RelWithDebInfo build type, but with frame pointer so perf utility can use it

## Test problem solutions

Some test problems generate CSV files containing their solutions. These can be plotted with [tools/plot_test_problem_solutions.py](https://github.com/SleipnirGroup/Sleipnir/blob/main/tools/plot_test_problem_solutions.py).

## Logo

Logo: [SVG](https://github.com/SleipnirGroup/Sleipnir/tree/main/logo/sleipnir.svg), [PNG (1000px)](https://github.com/SleipnirGroup/Sleipnir/tree/main/logo/sleipnir_THcolors_1000px.png)<br>
Font: [Centaur](https://en.wikipedia.org/wiki/Centaur_(typeface))
