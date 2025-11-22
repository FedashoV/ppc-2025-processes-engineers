#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <tuple>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"
#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

#ifdef USE_MPI
#  include <mpi.h>
#endif

namespace tsyplakov_k_vec_neighbours {

class TsyplakovKVecNeighboursPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  static constexpr int kCount = 75000000;
  InType input_data;

  void SetUp() override {
    input_data.resize(kCount);
    for (int i = 0; i < kCount; ++i) {
      input_data[i] = i;
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

    if (input_data.size() < 2) {
      return output_data == std::make_tuple(-1, -1);
    }

    int best = std::numeric_limits<int>::max();
    int best_i = -1;

    const std::size_t n = input_data.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
      const int64_t diff = std::llabs(static_cast<int64_t>(input_data[i + 1]) - static_cast<int64_t>(input_data[i]));
      if (diff < best || (diff == best && std::cmp_less(i, static_cast<std::size_t>(best_i)))) {
        best = static_cast<int>(diff);
        best_i = static_cast<int>(i);
      }
    }

    const OutType expected = (best_i >= 0) ? std::make_tuple(best_i, best_i + 1) : std::make_tuple(-1, -1);
    return output_data == expected;
  }

  InType GetTestInputData() final {
    return input_data;
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
