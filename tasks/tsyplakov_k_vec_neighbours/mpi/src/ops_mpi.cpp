#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <cstdlib>
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

  int size = static_cast<int>(GetInput().size());

  MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (size < 2) {
    if (rank == 0) {
      GetOutput() = std::make_tuple(-1, -1);
    }
    return true;
  }

  return true;
}

namespace {

Result FindLocalMinimum(const std::vector<int> &local_vec, int local_size) {
  Result local_res{.delta = std::numeric_limits<int>::max(), .index = -1};

  for (int i = 0; i + 1 < local_size; ++i) {
    int64_t diff = std::llabs(static_cast<int64_t>(local_vec[i + 1]) - static_cast<int64_t>(local_vec[i]));
    if (diff < local_res.delta) {
      local_res.delta = static_cast<int>(diff);
      local_res.index = i;
    } else if (diff == local_res.delta && i < local_res.index) {
      local_res.index = i;
    }
  }
  return local_res;
}

void ExchangeBoundaryValues(int rank, int comm_size, int left_value, int right_value, int &recv_left, int &recv_right) {
  std::vector<MPI_Request> reqs(4, MPI_REQUEST_NULL);
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
    MPI_Waitall(rc, reqs.data(), MPI_STATUSES_IGNORE);
  }
}

}  // namespace

bool TsyplakovKVecNeighboursMPI::PreProcessingImpl() {
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int global_size = static_cast<int>(GetInput().size());
  MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (global_size < 2) {
    return true;
  }

  const int base = global_size / comm_size;
  const int rem = global_size % comm_size;
  const int local_size = base + ((rank < rem) ? 1 : 0);

  local_vec_.resize(local_size);

  std::vector<int> recvcounts(comm_size);
  std::vector<int> displs(comm_size);

  for (int i = 0; i < comm_size; ++i) {
    recvcounts[i] = base + ((i < rem) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
  }

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, recvcounts.data(), displs.data(), MPI_INT, local_vec_.data(),
               local_size, MPI_INT, 0, MPI_COMM_WORLD);

  return true;
}

bool TsyplakovKVecNeighboursMPI::RunImpl() {
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int global_size = static_cast<int>(GetInput().size());

  if (global_size < 2) {
    return true;
  }

  const int base = global_size / comm_size;
  const int rem = global_size % comm_size;
  const int displ = (rank < rem) ? (rank * (base + 1)) : (rem * (base + 1) + (rank - rem) * base);

  Result local_res = FindLocalMinimum(local_vec_, static_cast<int>(local_vec_.size()));

  if (local_res.index >= 0) {
    local_res.index += displ;
  }

  int left_value = (!local_vec_.empty()) ? local_vec_.front() : 0;
  int right_value = (!local_vec_.empty()) ? local_vec_.back() : 0;
  int recv_left = 0, recv_right = 0;

  ExchangeBoundaryValues(rank, comm_size, left_value, right_value, recv_left, recv_right);

  if (rank > 0 && !local_vec_.empty()) {
    int64_t diff = std::llabs(static_cast<int64_t>(left_value) - static_cast<int64_t>(recv_left));
    int global_idx = displ - 1;

    if (diff < local_res.delta || (diff == local_res.delta && global_idx < local_res.index)) {
      local_res.delta = static_cast<int>(diff);
      local_res.index = global_idx;
    }
  }

  if (rank + 1 < comm_size && !local_vec_.empty()) {
    int64_t diff = std::llabs(static_cast<int64_t>(recv_right) - static_cast<int64_t>(right_value));
    int global_idx = displ + local_vec_.size() - 1;
    if (diff < local_res.delta || (diff == local_res.delta && global_idx < local_res.index)) {
      local_res.delta = static_cast<int>(diff);
      local_res.index = global_idx;
    }
  }

  struct MpiResult {
    int delta;
    int index;
  };

  MpiResult send_res{.delta = local_res.delta, .index = local_res.index};
  MpiResult recv_res{.delta = std::numeric_limits<int>::max(), .index = -1};

  MPI_Datatype mpi_result_type;
  int blocklengths[2] = {1, 1};
  MPI_Aint displacements[2] = {0, sizeof(int)};
  MPI_Datatype types[2] = {MPI_INT, MPI_INT};
  MPI_Type_create_struct(2, blocklengths, displacements, types, &mpi_result_type);
  MPI_Type_commit(&mpi_result_type);

  MPI_Op minloc_op;
  MPI_Op_create([](void *invec, void *inoutvec, int *len, MPI_Datatype * /*datatype*/) {
    MpiResult *in = static_cast<MpiResult *>(invec);
    MpiResult *inout = static_cast<MpiResult *>(inoutvec);

    for (int i = 0; i < *len; ++i) {
      if (in[i].delta < inout[i].delta || (in[i].delta == inout[i].delta && in[i].index < inout[i].index)) {
        inout[i] = in[i];
      }
    }
  }, 1, &minloc_op);

  MPI_Allreduce(&send_res, &recv_res, 1, mpi_result_type, minloc_op, MPI_COMM_WORLD);

  MPI_Op_free(&minloc_op);
  MPI_Type_free(&mpi_result_type);

  if (recv_res.index >= 0) {
    GetOutput() = std::make_tuple(recv_res.index, recv_res.index + 1);
  } else {
    GetOutput() = std::make_tuple(-1, -1);
  }

  return true;
}

bool TsyplakovKVecNeighboursMPI::PostProcessingImpl() {
  local_vec_.clear();
  return true;
}

}  // namespace tsyplakov_k_vec_neighbours
