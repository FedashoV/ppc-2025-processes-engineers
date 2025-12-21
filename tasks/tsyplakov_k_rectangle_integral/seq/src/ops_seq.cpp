#include "tsyplakov_k_rectangle_integral/seq/include/ops_seq.hpp"

#include <cmath>
#include <vector>

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
  const int dim = (input.size() - 1) / 2;

  std::vector<double> a(dim), b(dim), h(dim);

  for (int i = 0; i < dim; ++i) {
    a[i] = input[2 * i];
    b[i] = input[2 * i + 1];
    h[i] = (b[i] - a[i]) / steps;
  }

  const int total_points = static_cast<int>(std::pow(steps, dim));

  for (int idx = 0; idx < total_points; ++idx) {
    int tmp = idx;
    double f_value = 0.0;

    for (int d = 0; d < dim; ++d) {
      int coord = tmp % steps;
      tmp /= steps;

      double x = a[d] + coord * h[d];
      f_value += x;
    }

    GetOutput() += f_value;
  }

  double volume = 1.0;
  for (int d = 0; d < dim; ++d) {
    volume *= h[d];
  }

  GetOutput() *= volume;
  return true;
}

bool TsyplakovKRectangleIntegralSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_rectangle_integral
