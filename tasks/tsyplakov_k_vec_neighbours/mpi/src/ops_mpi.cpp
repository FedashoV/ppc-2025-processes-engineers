#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"

namespace tsyplakov_k_vec_neighbours {

TsyplakovKVecNeighboursMPI::TsyplakovKVecNeighboursMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool TsyplakovKVecNeighboursMPI::ValidationImpl() {
  return true;
}

bool TsyplakovKVecNeighboursMPI::PreProcessingImpl() {
  return true;
}

bool TsyplakovKVecNeighboursMPI::RunImpl() {
  int my_rank = 0;
  int world_size = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const auto &arr = GetInput();
  const int n = static_cast<int>(arr.size());

  if (n < 2) {
    GetOutput() = std::make_tuple(-1, -1);
    return true;
  }

  int best_local_gap = std::numeric_limits<int>::max();
  int best_local_pos = -1;

  for (int idx = my_rank; idx < n - 1; idx += world_size) {
    int diff = std::abs(arr[idx] - arr[idx + 1]);

    if (diff < best_local_gap) {
      best_local_gap = diff;
      best_local_pos = idx;
    }
  }

  int best_global_gap = 0;
  MPI_Allreduce(&best_local_gap, &best_global_gap, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int candidate_index = std::numeric_limits<int>::max();
  if (best_local_gap == best_global_gap) {
    candidate_index = best_local_pos;
  }

  int final_index = 0;
  MPI_Allreduce(&candidate_index, &final_index, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = std::make_tuple(final_index, final_index + 1);
  return true;
}

bool TsyplakovKVecNeighboursMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_vec_neighbours
