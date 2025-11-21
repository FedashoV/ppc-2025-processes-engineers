#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"
#include "util/include/util.hpp"

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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool valid = true;
  if (rank == 0) {
    valid = GetInput().size() >= 2;
  }
  MPI_Bcast(&valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return valid;
}

bool TsyplakovKVecNeighboursMPI::RunImpl() {
  const auto &vec = GetInput();
  int rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int global_size = vec.size();
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int base = global_size / comm_size;
  int rem = global_size % comm_size;
  int local_start = rank * base + std::min(rank, rem);
  int local_end = local_start + base + (rank < rem ? 1 : 0);

  Result local_res{std::numeric_limits<int>::max(), -1};

  for (int i = local_start; i + 1 < local_end; ++i) {
    int diff = std::abs(vec[i + 1] - vec[i]);
    if (diff < local_res.delta || (diff == local_res.delta && i < local_res.index)) {
      local_res.delta = diff;
      local_res.index = i;
    }
  }

  int left_value = (local_start > 0) ? vec[local_start] : 0;
  int right_value = (local_end < global_size) ? vec[local_end - 1] : 0;
  int recv_left = 0, recv_right = 0;

  MPI_Request reqs[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  int rc = 0;

  if (rank > 0) {
    MPI_Irecv(&recv_left, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &reqs[rc++]);
    MPI_Isend(&left_value, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &reqs[rc++]);
  }
  if (rank + 1 < comm_size) {
    MPI_Irecv(&recv_right, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &reqs[rc++]);
    MPI_Isend(&right_value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &reqs[rc++]);
  }
  if (rc > 0) {
    MPI_Waitall(rc, reqs, MPI_STATUSES_IGNORE);
  }

  if (rank > 0) {
    int diff = std::abs(left_value - recv_left);
    int idx = local_start - 1;
    if (diff < local_res.delta || (diff == local_res.delta && idx < local_res.index)) {
      local_res.delta = diff;
      local_res.index = idx;
    }
  }
  if (rank + 1 < comm_size) {
    int diff = std::abs(right_value - recv_right);
    int idx = local_end - 1;
    if (diff < local_res.delta || (diff == local_res.delta && idx < local_res.index)) {
      local_res.delta = diff;
      local_res.index = idx;
    }
  }

  struct {
    int delta;
    int index;
  } send_res{local_res.delta, local_res.index}, recv_res;

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
