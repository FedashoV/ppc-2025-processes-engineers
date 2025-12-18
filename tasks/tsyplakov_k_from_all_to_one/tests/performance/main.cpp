#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "tsyplakov_k_from_all_to_one/common/include/common.hpp"
#include "tsyplakov_k_from_all_to_one/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_from_all_to_one/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsyplakov_k_from_all_to_one {

template <typename T>
class TsyplakovKRunPerfTestFromAllToOne : public ppc::util::BaseRunPerfTests<InTypeT<T>, OutTypeT<T>> {
 protected:
  static constexpr unsigned int kLocalCount = 7000000;
  InTypeT<T> input_data;

  void SetUp() override {
#ifdef USE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<T> local_vec(kLocalCount, static_cast<T>(rank));
    input_data = std::make_tuple(local_vec, 0);
#else
    std::vector<T> vec(kLocalCount);
    for (unsigned int i = 0; i < kLocalCount; ++i) {
      vec[i] = static_cast<T>(i);
    }
    input_data = std::make_tuple(vec, 0);
#endif
  }

  bool CheckTestOutputData(OutTypeT<T> &output_data) final {
#ifdef USE_MPI
    int rank = 0 int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int root = std::get<1>(input_data);
    if (rank != root) {
      return true;
    }

    if (output_data.size() != size * kLocalCount) {
      return false;
    }

    for (int r = 0; r < size; ++r) {
      for (size_t i = 0; i < kLocalCount; ++i) {
        if (output_data[r * kLocalCount + i] != static_cast<T>(r)) {
          return false;
        }
      }
    }
    return true;
#else
    return output_data == std::get<0>(input_data);
#endif
  }

  InTypeT<T> GetTestInputData() final {
    return input_data;
  }
};

using PerfTestInt = TsyplakovKRunPerfTestFromAllToOne<int>;
using PerfTestFloat = TsyplakovKRunPerfTestFromAllToOne<float>;
using PerfTestDouble = TsyplakovKRunPerfTestFromAllToOne<double>;

TEST_P(PerfTestInt, RunPerfModes) {
  ExecuteTest(GetParam());
}
TEST_P(PerfTestFloat, RunPerfModes) {
  ExecuteTest(GetParam());
}
TEST_P(PerfTestDouble, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasksInt =
    ppc::util::MakeAllPerfTasks<InTypeT<int>, TsyplakovKFromAllToOneMPI<int>, TsyplakovKFromAllToOneSEQ>(
        PPC_SETTINGS_tsyplakov_k_from_all_to_one);

const auto kAllPerfTasksFloat = ppc::util::MakeAllPerfTasks<InTypeT<float>, TsyplakovKFromAllToOneMPI<float>>(
    PPC_SETTINGS_tsyplakov_k_from_all_to_one);

const auto kAllPerfTasksDouble = ppc::util::MakeAllPerfTasks<InTypeT<double>, TsyplakovKFromAllToOneMPI<double>>(
    PPC_SETTINGS_tsyplakov_k_from_all_to_one);

INSTANTIATE_TEST_SUITE_P(IntPerf, PerfTestInt, ppc::util::TupleToGTestValues(kAllPerfTasksInt),
                         PerfTestInt::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(FloatPerf, PerfTestFloat, ppc::util::TupleToGTestValues(kAllPerfTasksFloat),
                         PerfTestFloat::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(DoublePerf, PerfTestDouble, ppc::util::TupleToGTestValues(kAllPerfTasksDouble),
                         PerfTestDouble::CustomPerfTestName);

}  // namespace
}  // namespace tsyplakov_k_from_all_to_one
