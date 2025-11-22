#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tsyplakov_k_vec_neighbours/common/include/common.hpp"
#include "tsyplakov_k_vec_neighbours/mpi/include/ops_mpi.hpp"
#include "tsyplakov_k_vec_neighbours/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace tsyplakov_k_vec_neighbours {

class TsyplakovKVecNeighboursFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(
      const testing::TestParamInfo<std::tuple<std::function<std::shared_ptr<BaseTask>(InType)>, std::string, TestType>>
          &info) {
    const TestType &p = std::get<2>(info.param);
    const std::string &task_type = std::get<1>(info.param);
    return task_type + "_" + std::to_string(std::get<0>(p)) + "_" + std::get<1>(p);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const int vec_size = std::get<0>(params);
    const std::string &case_type = std::get<1>(params);

    input_data_.resize(vec_size);

    if (case_type == "normal") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = (i * 13 + 7) % 50;
      }
    } else if (case_type == "zeros" || case_type == "all_same") {
      std::ranges::fill(input_data_, (case_type == "zeros") ? 0 : 42);
    } else if (case_type == "negatives") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = -i;
      }
    } else if (case_type == "ascending") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = i;
      }
    } else if (case_type == "descending") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = vec_size - i;
      }
    } else if (case_type == "big") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = i % 1000;
      }
    } else {
      throw std::runtime_error("Unknown test case type");
    }

    expected_output_ = ComputeReference(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static OutType ComputeReference(const std::vector<int> &v) {
    if (v.size() < 2) {
      return std::make_tuple(-1, -1);
    }

    int best = std::numeric_limits<int>::max();
    int best_i = -1;

    const std::size_t n = v.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
      int64_t diff = std::abs(static_cast<int64_t>(v[i + 1]) - static_cast<int64_t>(v[i]));

      if (diff < best || (diff == best && static_cast<int>(i) < best_i)) {
        best = static_cast<int>(diff);
        best_i = static_cast<int>(i);
      }
    }

    if (best_i >= 0) {
      return std::make_tuple(best_i, best_i + 1);
    }
    return std::make_tuple(-1, -1);
  }

  InType input_data_{};
  OutType expected_output_ = std::make_tuple(-1, -1);
};

namespace {

TEST_P(TsyplakovKVecNeighboursFuncTests, VecMinNeighbourDiff) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 7> kTestParam = {
    std::make_tuple(10, "normal"),    std::make_tuple(10, "zeros"),     std::make_tuple(10, "all_same"),
    std::make_tuple(10, "negatives"), std::make_tuple(10, "ascending"), std::make_tuple(10, "descending"),
    std::make_tuple(1000000, "big"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TsyplakovKVecNeighboursMPI, InType>(kTestParam, PPC_SETTINGS_tsyplakov_k_vec_neighbours),
    ppc::util::AddFuncTask<TsyplakovKVecNeighboursSEQ, InType>(kTestParam, PPC_SETTINGS_tsyplakov_k_vec_neighbours));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsyplakovKVecNeighboursFuncTests::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(VectorFuncTests, TsyplakovKVecNeighboursFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace tsyplakov_k_vec_neighbours
