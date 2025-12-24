#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "tsyplakov_k_rectangle_integral/common/include/common.hpp"
#include "tsyplakov_k_rectangle_integral/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_rectangle_integral/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsyplakov_k_rectangle_integral {

class TsyplakovKRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    int steps = static_cast<int>(std::get<0>(param).back());
    return "steps_" + std::to_string(steps);
  }

 protected:
  void SetUp() override {
    const auto &params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
    expected_ = std::get<1>(params);
  }

  InType GetTestInputData() override {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    constexpr double kEps = 3e-2;
    return std::abs(output_data - expected_) < kEps;
  }

 private:
  InType input_data_;
  OutType expected_{};
};

namespace {

const std::array<TestType, 3> kTestParams = {
    TestType{{0.0, 1.0, 0.0, 1.0, 50}, 1.0},
    TestType{{0.0, 1.0, 0.0, 1.0, 100}, 1.0},
    TestType{{0.0, 1.0, 0.0, 1.0, 200}, 1.0},
};

TEST_P(TsyplakovKRunFuncTests, RectangleIntegral2D) {
  ExecuteTest(GetParam());
}

const auto kTestTasks = std::tuple_cat(ppc::util::AddFuncTask<TsyplakovKRectangleIntegralMPI, InType>(
                                           kTestParams, PPC_SETTINGS_tsyplakov_k_rectangle_integral),
                                       ppc::util::AddFuncTask<TsyplakovKRectangleIntegralSEQ, InType>(
                                           kTestParams, PPC_SETTINGS_tsyplakov_k_rectangle_integral));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasks);

const auto kTestName = TsyplakovKRunFuncTests::PrintFuncTestName<TsyplakovKRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(RectangleIntegralTests, TsyplakovKRunFuncTests, kGtestValues, kTestName);

}  // namespace
}  // namespace tsyplakov_k_rectangle_integral
