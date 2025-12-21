#include "tsyplakov_k_rectangle_integral/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "tsyplakov_k_rectangle_integral/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tsyplakov_k_rectangle_integral {

TsyplakovKRectangleIntegralMPI::TsyplakovKRectangleIntegralMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool TsyplakovKRectangleIntegralMPI::ValidationImpl() {
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

bool TsyplakovKRectangleIntegralMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool TsyplakovKRectangleIntegralMPI::RunImpl() {
  int rank = 0;
  int size = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  const int steps = static_cast<int>(input.back());
  const size_t dim = (input.size() - 1) / 2;

  std::vector<double> a(dim);
  std::vector<double> b(dim);
  std::vector<double> h(dim);

  for (size_t i = 0; i < dim; ++i) {
    a[i] = input[static_cast<size_t>(2) * i];
    b[i] = input[static_cast<size_t>(2) * i + 1];
    h[i] = (b[i] - a[i]) / steps;
  }

  const int total_points = static_cast<int>(std::pow(steps, dim));

  const int points_per_proc = total_points / size;
  const int remainder = total_points % size;

  int start = rank * points_per_proc + std::min(rank, remainder);
  int end = start + points_per_proc + (rank < remainder ? 1 : 0);

  double local_sum = 0.0;

  for (int idx = start; idx < end; ++idx) {
    int tmp = idx;
    double f_value = 0.0;

    for (size_t d = 0; d < dim; ++d) {
      int coord = tmp % steps;
      tmp /= steps;

      double x = a[d] + coord * h[d];
      f_value += x;
    }

    local_sum += f_value;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  double result = 0.0;

  if (rank == 0) {
    double volume = 1.0;
    for (size_t d = 0; d < dim; ++d) {
      volume *= h[d];
    }
    result = global_sum * volume;
  }

  MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = result;

  return true;
}

bool TsyplakovKRectangleIntegralMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_rectangle_integral
