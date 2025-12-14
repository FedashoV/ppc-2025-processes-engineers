#include "tsyplakov_k_from_all_to_one/mpi/include/ops_mpi.hpp"

#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace tsyplakov_k_from_all_to_one {

// =======================================================
// Конструктор
// =======================================================
template <typename T>
TsyplakovKFromAllToOneMPI<T>::TsyplakovKFromAllToOneMPI(const InTypeT<T> &in) {
  this->SetTypeOfTask(GetStaticTypeOfTask());
  this->GetInput() = in;
}

// =======================================================
// Validation
// =======================================================
template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::ValidationImpl() {
  const auto &[data, root] = this->GetInput();
  return !data.empty() && root >= 0;
}

// =======================================================
// PreProcessing
// =======================================================
template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::PreProcessingImpl() {
  gathered_.clear();
  return true;
}

// =======================================================
// RunImpl
// =======================================================
template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::RunImpl() {
#ifdef USE_MPI
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &[local_vec, root] = this->GetInput();
  const int sendcount = static_cast<int>(local_vec.size());

  // Определяем MPI_Datatype по T
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

  My_MPI_Gather(local_vec.data(), sendcount, mpi_type, rank == root ? recvbuf.data() : nullptr, sendcount, mpi_type,
                root, MPI_COMM_WORLD);

  if (rank == root) {
    gathered_ = std::move(recvbuf);
    this->GetOutput() = gathered_;
  }
#else
  this->GetOutput() = std::get<0>(this->GetInput());
#endif
  return true;
}

// =======================================================
// PostProcessing
// =======================================================
template <typename T>
bool TsyplakovKFromAllToOneMPI<T>::PostProcessingImpl() {
  return true;
}

// =======================================================
// Реализация My_MPI_Gather (универсальная)
// =======================================================
int My_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount,
                  MPI_Datatype recvtype, int root, MPI_Comm comm) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (sendcount != recvcount || sendtype != recvtype) {
    return MPI_ERR_COUNT;
  }

  int type_size = 0;
  MPI_Type_size(sendtype, &type_size);
  const int block_bytes = sendcount * type_size;

  int blocks = 1;
  int *ranks = static_cast<int *>(std::malloc(sizeof(int)));
  void *data = std::malloc(block_bytes);

  ranks[0] = rank;
  std::memcpy(data, sendbuf, block_bytes);

  const int rel_rank = (rank - root + size) % size;
  int step = 1;

  while (step < size) {
    if (rel_rank % (2 * step) == 0) {
      int src = rel_rank + step;
      if (src < size) {
        const int real_src = (src + root) % size;

        int recv_blocks = 0;
        MPI_Recv(&recv_blocks, 1, MPI_INT, real_src, 0, comm, MPI_STATUS_IGNORE);

        ranks = static_cast<int *>(std::realloc(ranks, (blocks + recv_blocks) * sizeof(int)));
        data = std::realloc(data, (blocks + recv_blocks) * block_bytes);

        for (int i = 0; i < recv_blocks; ++i) {
          MPI_Recv(&ranks[blocks + i], 1, MPI_INT, real_src, 0, comm, MPI_STATUS_IGNORE);
          MPI_Recv(static_cast<char *>(data) + (blocks + i) * block_bytes, block_bytes, MPI_BYTE, real_src, 0, comm,
                   MPI_STATUS_IGNORE);
        }

        blocks += recv_blocks;
      }
    } else {
      int dest = rel_rank - step;
      const int real_dest = (dest + root) % size;

      MPI_Send(&blocks, 1, MPI_INT, real_dest, 0, comm);

      for (int i = 0; i < blocks; ++i) {
        MPI_Send(&ranks[i], 1, MPI_INT, real_dest, 0, comm);
        MPI_Send(static_cast<char *>(data) + i * block_bytes, block_bytes, MPI_BYTE, real_dest, 0, comm);
      }
      break;
    }
    step <<= 1;
  }

  if (rank == root) {
    for (int i = 0; i < blocks; ++i) {
      std::memcpy(static_cast<char *>(recvbuf) + ranks[i] * block_bytes, static_cast<char *>(data) + i * block_bytes,
                  block_bytes);
    }
  }

  std::free(ranks);
  std::free(data);
  return MPI_SUCCESS;
}

// =======================================================
// Явные инстанцирования шаблона
// =======================================================
template class TsyplakovKFromAllToOneMPI<int>;
template class TsyplakovKFromAllToOneMPI<float>;
template class TsyplakovKFromAllToOneMPI<double>;

}  // namespace tsyplakov_k_from_all_to_one
