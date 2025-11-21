#include <gtest/gtest.h>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"
#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

#ifdef USE_MPI
#  include <mpi.h>
#endif

namespace tsyplakov_k_vec_neighbours {

class TsyplakovKVecNeighboursPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 75000000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kCount_);
    for (int i = 0; i < kCount_; i++) {
      input_data_[i] = i;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
#ifdef USE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }
#endif

    if (input_data_.size() < 2) {
      return output_data == std::make_tuple(-1, -1);
    }

    int best = std::numeric_limits<int>::max();
    int best_i = -1;

    for (size_t i = 0; i + 1 < input_data_.size(); i++) {
      int diff = std::abs(input_data_[i] - input_data_[i + 1]);
      if (diff < best) {
        best = diff;
        best_i = static_cast<int>(i);
      }
    }

    OutType expected = (best_i >= 0) ? std::make_tuple(best_i, best_i + 1) : std::make_tuple(-1, -1);

    return output_data == expected;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(TsyplakovKVecNeighboursPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TsyplakovKVecNeighboursMPI, TsyplakovKVecNeighboursSEQ>(
    PPC_SETTINGS_tsyplakov_k_vec_neighbours);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsyplakovKVecNeighboursPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsyplakovKVecNeighboursPerfTest, kGtestValues, kPerfTestName);

}  // namespace tsyplakov_k_vec_neighbours
