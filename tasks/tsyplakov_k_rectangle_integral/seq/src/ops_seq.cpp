#include "tsyplakov_k_rectangle_integral/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "tsyplakov_k_rectangle_integral/common/include/common.hpp"

namespace tsyplakov_k_rectangle_integral {

TsyplakovKRectangleIntegralSEQ::TsyplakovKRectangleIntegralSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsyplakovKRectangleIntegralSEQ::ValidationImpl() {
  if (GetInput().size() < 3) {
    return false;
  }

  if (GetInput().back() <= 0) {
    return false;
  }

  for (size_t i = 0; i + 1 < GetInput().size() - 1; i += 2) {
    if (GetInput()[i] >= GetInput()[i + 1]) {
      return false;
    }
  }

  return true;
}

bool TsyplakovKRectangleIntegralSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsyplakovKRectangleIntegralSEQ::RunImpl() {
  const auto &input = GetInput();
  const int steps = static_cast<int>(input.back());
  const size_t dim = (input.size() - 1) / 2;

  std::vector<double> a(dim);
  std::vector<double> b(dim);
  std::vector<double> h(dim);

  for (size_t i = 0; i < dim; ++i) {
    a[i] = input[static_cast<size_t>(2) * i];
    b[i] = input[(static_cast<size_t>(2) * i) + 1];
    h[i] = (b[i] - a[i]) / steps;
  }

  const int total_points = static_cast<int>(std::pow(steps, dim));

  for (int idx = 0; idx < total_points; ++idx) {
    int tmp = idx;
    double f_value = 0.0;

    for (size_t dd = 0; dd < dim; ++dd) {
      int coord = tmp % steps;
      tmp /= steps;

      double x = a[dd] + (coord * h[dd]);
      f_value += x;
    }

    GetOutput() += f_value;
  }

  double volume = 1.0;
  for (size_t dd = 0; dd < dim; ++dd) {
    volume *= h[dd];
  }

  GetOutput() *= volume;
  return true;
}

bool TsyplakovKRectangleIntegralSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_rectangle_integral
