#include "tsyplakov_k_from_all_to_one/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdlib>
#include <cstring>

namespace tsyplakov_k_from_all_to_one {

template <typename T>
TsyplakovKFromAllToOneMPI<T>::TsyplakovKFromAllToOneMPI(const InTypeT<T> &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::ValidationImpl() {
  const auto &[data, root] = this->GetInput();
  return !data.empty() && root >= 0;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::PreProcessingImpl() {
  gathered_.clear();
  return true;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::RunImpl() {
#ifdef USE_MPI
  int rank = 0 int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[local_vec, root] = this->GetInput();
  const int sendcount = static_cast<int>(local_vec.size());

  MPI_Datatype mpi_type;
  if constexpr (std::is_same_v<T, int>) {
    mpi_type = MPI_INT;
  } else if constexpr (std::is_same_v<T, float>) {
    mpi_type = MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mpi_type = MPI_DOUBLE;
  } else {
    static_assert(!sizeof(T *), "Unsupported type for MPI_Gather");
  }

  std::vector<T> recvbuf;
  if (rank == root) {
    recvbuf.resize(sendcount * size);
  }

  MyMpiGather(local_vec.data(), sendcount, mpi_type, rank == root ? recvbuf.data() : nullptr, sendcount, mpi_type, root,
              MPI_COMM_WORLD);

  if (rank == root) {
    gathered_ = std::move(recvbuf);
    this->GetOutput() = gathered_;
  }
#else
  this->GetOutput() = std::get<0>(this->GetInput());
#endif
  return true;
}

template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::PostProcessingImpl() {
  return true;
}

int MyMpiGather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                MPI_Datatype recvtype, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (sendcount != recvcount) {
    return MPI_ERR_COUNT;
  }

  if (sendtype != recvtype) {
    return MPI_ERR_TYPE;
  }

  int type_size = 0;
  MPI_Type_size(sendtype, &type_size);
  const int block_bytes = sendcount * type_size;

  int blocks = 1;

  std::vector<int> ranks(1, rank);
  std::vector<std::byte> data(static_cast<std::size_t>(block_bytes));

  std::memcpy(data.data(), sendbuf, static_cast<std::size_t>(block_bytes));

  const int rel_rank = (rank - root + size) % size;
  int step = 1;

  while (step < size) {
    if (rel_rank % (2 * step) == 0) {
      const int src = rel_rank + step;
      if (src < size) {
        const int real_src = (src + root) % size;

        int recv_blocks = 0;
        MPI_Recv(&recv_blocks, 1, MPI_INT, real_src, 0, comm, MPI_STATUS_IGNORE);

        const int old_blocks = blocks;
        blocks += recv_blocks;

        ranks.resize(static_cast<std::size_t>(blocks));
        data.resize(static_cast<std::size_t>(blocks) * static_cast<std::size_t>(block_bytes));

        for (int i = 0; i < recv_blocks; ++i) {
          MPI_Recv(&ranks[old_blocks + i], 1, MPI_INT, real_src, 0, comm, MPI_STATUS_IGNORE);

          const auto offset = static_cast<std::ptrdiff_t>(old_blocks + i) * static_cast<std::ptrdiff_t>(block_bytes);

          MPI_Recv(data.data() + offset, block_bytes, MPI_BYTE, real_src, 0, comm, MPI_STATUS_IGNORE);
        }
      }
    } else {
      const int dest = rel_rank - step;
      const int real_dest = (dest + root) % size;

      MPI_Send(&blocks, 1, MPI_INT, real_dest, 0, comm);

      for (int i = 0; i < blocks; ++i) {
        MPI_Send(&ranks[i], 1, MPI_INT, real_dest, 0, comm);

        const auto offset = static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(block_bytes);

        MPI_Send(data.data() + offset, block_bytes, MPI_BYTE, real_dest, 0, comm);
      }
      break;
    }
    step <<= 1;
  }

  if (rank == root) {
    for (int i = 0; i < blocks; ++i) {
      const auto src_offset = static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(block_bytes);

      const auto dst_offset = static_cast<std::ptrdiff_t>(ranks[i]) * static_cast<std::ptrdiff_t>(block_bytes);

      std::memcpy(static_cast<std::byte *>(recvbuf) + dst_offset, data.data() + src_offset,
                  static_cast<std::size_t>(block_bytes));
    }
  }

  return MPI_SUCCESS;
}

template class TsyplakovKFromAllToOneMPI<int>;
template class TsyplakovKFromAllToOneMPI<float>;
template class TsyplakovKFromAllToOneMPI<double>;

}  // namespace tsyplakov_k_from_all_to_one
