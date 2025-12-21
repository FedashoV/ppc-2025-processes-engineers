#include <gtest/gtest.h>

#include <cmath>

#include "tsyplakov_k_rectangle_integral/common/include/common.hpp"
#include "tsyplakov_k_rectangle_integral/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_rectangle_integral/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsyplakov_k_rectangle_integral {

class TsyplakovKRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 200};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double expected = 1.5;
    const double eps = 5e-2;
    return std::abs(output_data - expected) < eps;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(TsyplakovKRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, TsyplakovKRectangleIntegralMPI, TsyplakovKRectangleIntegralSEQ>(
        PPC_SETTINGS_tsyplakov_k_rectangle_integral);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsyplakovKRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsyplakovKRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace tsyplakov_k_rectangle_integral
