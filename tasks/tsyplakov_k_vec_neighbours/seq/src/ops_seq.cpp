#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>

namespace tsyplakov_k_vec_neighbours {

TsyplakovKVecNeighboursSEQ::TsyplakovKVecNeighboursSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::make_tuple(-1, -1);
}

bool TsyplakovKVecNeighboursSEQ::ValidationImpl() {
  return GetInput().size() >= 2;
}

bool TsyplakovKVecNeighboursSEQ::PreProcessingImpl() {
  vector_data_ = GetInput();
  return true;
}

bool TsyplakovKVecNeighboursSEQ::RunImpl() {
  int min_diff = std::numeric_limits<int>::max();
  int min_index = -1;

  for (size_t i = 0; i + 1 < vector_data_.size(); i++) {
    int diff = std::abs(vector_data_[i] - vector_data_[i + 1]);
    if (diff < min_diff) {
      min_diff = diff;
      min_index = static_cast<int>(i);
    }
  }

  if (min_index >= 0) {
    GetOutput() = std::make_tuple(min_index, min_index + 1);
  } else {
    GetOutput() = std::make_tuple(-1, -1);
  }

  return true;
}

bool TsyplakovKVecNeighboursSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace tsyplakov_k_vec_neighbours
