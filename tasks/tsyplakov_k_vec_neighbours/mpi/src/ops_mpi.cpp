#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"

namespace tsyplakov_k_vec_neighbours {

struct Result {
  int delta;
  int index;
};

TsyplakovKVecNeighboursMPI::TsyplakovKVecNeighboursMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool TsyplakovKVecNeighboursMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool valid = true;
  if (rank == 0) {
    valid = GetInput().size() >= 2;
  }
  MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return valid;
}

namespace {

Result FindLocalMinimum(const std::vector<int> &vec, int local_start, int local_end) {
  Result local_res{.delta = std::numeric_limits<int>::max(), .index = -1};

  for (int i = local_start; i + 1 < local_end; ++i) {
    int64_t diff = std::abs(static_cast<int64_t>(vec[i + 1]) - static_cast<int64_t>(vec[i]));
    if (diff < local_res.delta || (diff == local_res.delta && i < local_res.index)) {
      local_res.delta = static_cast<int>(diff);
      local_res.index = i;
    }
  }
  return local_res;
}

void ExchangeBoundaryValues(int rank, int comm_size, int left_value, int right_value, int &recv_left, int &recv_right) {
  std::vector<MPI_Request> reqs(4, MPI_REQUEST_NULL);
  int rc = 0;

  if (rank > 0) {
    MPI_Irecv(&recv_left, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[rc]);
    rc++;
    MPI_Isend(&left_value, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &reqs[rc]);
    rc++;
  }
  if (rank + 1 < comm_size) {
    MPI_Irecv(&recv_right, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &reqs[rc]);
    rc++;
    MPI_Isend(&right_value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[rc]);
    rc++;
  }

  if (rc > 0) {
    MPI_Waitall(rc, reqs.data(), MPI_STATUSES_IGNORE);
  }
}

void CheckBoundaryPairs(int rank, int comm_size, int left_value, int right_value, int recv_left, int recv_right,
                        int local_start, int local_end, Result &local_res) {
  if (rank > 0) {
    int64_t diff = std::abs(static_cast<int64_t>(left_value) - static_cast<int64_t>(recv_left));
    int idx = local_start - 1;
    if (diff < local_res.delta || (diff == local_res.delta && idx < local_res.index)) {
      local_res.delta = static_cast<int>(diff);
      local_res.index = idx;
    }
  }
  if (rank + 1 < comm_size) {
    int64_t diff = std::abs(static_cast<int64_t>(right_value) - static_cast<int64_t>(recv_right));
    int idx = local_end - 1;
    if (diff < local_res.delta || (diff == local_res.delta && idx < local_res.index)) {
      local_res.delta = static_cast<int>(diff);
      local_res.index = idx;
    }
  }
}

}  // namespace

bool TsyplakovKVecNeighboursMPI::RunImpl() {
  const auto &vec = GetInput();
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int global_size = static_cast<int>(vec.size());
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base = global_size / comm_size;
  const int rem = global_size % comm_size;
  const int local_start = (rank * base) + std::min(rank, rem);
  const int local_end = local_start + base + ((rank < rem) ? 1 : 0);

  Result local_res = FindLocalMinimum(vec, local_start, local_end);

  int left_value = (local_start > 0) ? vec[local_start] : 0;
  int right_value = (local_end < global_size) ? vec[local_end - 1] : 0;
  int recv_left = 0;
  int recv_right = 0;

  ExchangeBoundaryValues(rank, comm_size, left_value, right_value, recv_left, recv_right);
  CheckBoundaryPairs(rank, comm_size, left_value, right_value, recv_left, recv_right, local_start, local_end,
                     local_res);

  struct MpiResult {
    int delta;
    int index;
  };

  MpiResult send_res{.delta = local_res.delta, .index = local_res.index};
  MpiResult recv_res{};

  MPI_Allreduce(&send_res, &recv_res, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

  if (recv_res.index >= 0) {
    GetOutput() = std::make_tuple(recv_res.index, recv_res.index + 1);
    return true;
  }
  return false;
}

bool TsyplakovKVecNeighboursMPI::PreProcessingImpl() {
  return true;
}

bool TsyplakovKVecNeighboursMPI::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_vec_neighbours
