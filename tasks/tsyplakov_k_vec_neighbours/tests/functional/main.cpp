#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
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

  static OutType ComputeReference(const std::vector<int> &v) {
    if (v.size() < 2) {
      return std::make_tuple(-1, -1);
    }

    int64_t best = std::numeric_limits<int64_t>::max();
    int best_i = -1;

    const std::size_t n = v.size();
    for (std::size_t i = 0; i + 1 < n; ++i) {
      const int64_t val1 = static_cast<int64_t>(v[i]);
      const int64_t val2 = static_cast<int64_t>(v[i + 1]);
      const int64_t diff = std::llabs(val2 - val1);

      if (diff < best) {
        best = diff;
        best_i = static_cast<int>(i);
      } else if (diff == best && std::cmp_less(i, static_cast<std::size_t>(best_i))) {
        best_i = static_cast<int>(i);
      }
    }

    if (best_i >= 0) {
      return std::make_tuple(best_i, best_i + 1);
    }
    return std::make_tuple(-1, -1);
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
    } else if (case_type == "zeros") {
      std::ranges::fill(input_data_, 0);
    } else if (case_type == "all_same") {
      std::ranges::fill(input_data_, 42);
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
    } else if (case_type == "empty") {
    } else if (case_type == "single_element") {
      if (vec_size > 0) {
        input_data_[0] = 42;
      }
    } else if (case_type == "minimal") {
      if (vec_size >= 2) {
        input_data_[0] = 10;
        input_data_[1] = 20;
      }
    } else if (case_type == "small") {
      if (vec_size >= 3) {
        input_data_[0] = 5;
        input_data_[1] = 15;
        input_data_[2] = 10;
      }
    } else if (case_type == "medium") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = (i * 7) % 100;
      }
    } else if (case_type == "first_pair_best") {
      if (vec_size >= 2) {
        input_data_[0] = 10;
        input_data_[1] = 11;
        for (int i = 2; i < vec_size; ++i) {
          input_data_[i] = 20 + i;
        }
      }
    } else if (case_type == "last_pair_best") {
      if (vec_size >= 2) {
        for (int i = 0; i < vec_size - 2; ++i) {
          input_data_[i] = 100 + i;
        }
        input_data_[vec_size - 2] = 10;
        input_data_[vec_size - 1] = 11;
      }
    } else if (case_type == "multiple_same") {
      if (vec_size >= 4) {
        input_data_[0] = 10;
        input_data_[1] = 12;
        input_data_[2] = 20;
        input_data_[3] = 22;
        for (int i = 4; i < vec_size; ++i) {
          input_data_[i] = 30 + i;
        }
      }
    } else if (case_type == "large_values") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = std::numeric_limits<int>::max() - (i % 100) * 1000000;
      }
    } else if (case_type == "overflow_risk") {
      if (vec_size >= 2) {
        input_data_[0] = 0;
        input_data_[1] = 1;
        for (int i = 2; i < vec_size; ++i) {
          input_data_[i] = 100 + i;
        }
      }
    } else if (case_type == "mixed_signs") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = (i % 2 == 0) ? i : -i;
      }
    } else if (case_type == "alternating") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = (i % 2 == 0) ? 100 : 101;
      }
    } else if (case_type == "single_process") {
      if (vec_size > 0) {
        input_data_[0] = 10;
        if (vec_size > 1) {
          input_data_[1] = 11;
        }
      }
    } else if (case_type == "two_processes") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = (i % 2 == 0) ? i * 10 : i * 10 + 1;
      }
    } else if (case_type == "three_processes") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = i * 5 + (i % 3);
      }
    } else if (case_type == "small_even") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = i * 2;
      }
    } else if (case_type == "small_odd") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = i * 2 + 1;
      }
    } else if (case_type == "medium_distributed") {
      for (int i = 0; i < vec_size; ++i) {
        input_data_[i] = (i * 7) % 50 + (i % 10);
      }
    } else {
      throw std::runtime_error("Unknown test case type: " + case_type);
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
  InType input_data_;
  OutType expected_output_ = std::make_tuple(-1, -1);
};

namespace {

TEST_P(TsyplakovKVecNeighboursFuncTests, VecMinNeighbourDiff) {
  ExecuteTest(GetParam());
}

class TsyplakovKVecNeighboursUnitTests : public testing::Test {};

TEST_F(TsyplakovKVecNeighboursUnitTests, ComputeReferenceEdgeCases) {
  EXPECT_EQ(std::make_tuple(-1, -1), TsyplakovKVecNeighboursFuncTests::ComputeReference({}));

  EXPECT_EQ(std::make_tuple(-1, -1), TsyplakovKVecNeighboursFuncTests::ComputeReference({42}));

  EXPECT_EQ(std::make_tuple(0, 1), TsyplakovKVecNeighboursFuncTests::ComputeReference({10, 20}));

  EXPECT_EQ(std::make_tuple(0, 1), TsyplakovKVecNeighboursFuncTests::ComputeReference({10, 12, 20, 22}));
}

TEST_F(TsyplakovKVecNeighboursUnitTests, LargeValuesHandling) {
  std::vector<int> large_values = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max() - 1};
  auto result = TsyplakovKVecNeighboursFuncTests::ComputeReference(large_values);
  EXPECT_EQ(std::make_tuple(0, 1), result);
}

TEST_F(TsyplakovKVecNeighboursUnitTests, OverflowProtection) {
  std::vector<int> overflow_values = {std::numeric_limits<int>::min(), std::numeric_limits<int>::min() + 1};
  auto result = TsyplakovKVecNeighboursFuncTests::ComputeReference(overflow_values);
  EXPECT_EQ(std::make_tuple(0, 1), result);
}

TEST_F(TsyplakovKVecNeighboursUnitTests, MixedSigns) {
  std::vector<int> mixed = {-5, 3, -1, 0, 2};
  auto result = TsyplakovKVecNeighboursFuncTests::ComputeReference(mixed);
  EXPECT_EQ(std::make_tuple(2, 3), result);
}

TEST_F(TsyplakovKVecNeighboursUnitTests, ExtremeOverflow) {
  std::vector<int> extreme = {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
  auto result = TsyplakovKVecNeighboursFuncTests::ComputeReference(extreme);
  EXPECT_EQ(std::make_tuple(0, 1), result);
}

TEST_F(TsyplakovKVecNeighboursUnitTests, SingleProcessCase) {
  std::vector<int> single = {1, 2, 3};
  auto result = TsyplakovKVecNeighboursFuncTests::ComputeReference(single);
  EXPECT_EQ(std::make_tuple(0, 1), result);
}

TEST_F(TsyplakovKVecNeighboursUnitTests, BoundaryElementsMPI) {
  std::vector<int> boundary = {10, 20, 5, 15, 25};
  auto result = TsyplakovKVecNeighboursFuncTests::ComputeReference(boundary);
  EXPECT_EQ(std::make_tuple(0, 1), result);
}

const std::array<TestType, 28> kTestParam = {std::make_tuple(10, "normal"),
                                             std::make_tuple(10, "zeros"),
                                             std::make_tuple(10, "all_same"),
                                             std::make_tuple(10, "negatives"),
                                             std::make_tuple(10, "ascending"),
                                             std::make_tuple(10, "descending"),
                                             std::make_tuple(1000000, "big"),
                                             std::make_tuple(0, "empty"),
                                             std::make_tuple(1, "single_element"),
                                             std::make_tuple(2, "minimal"),
                                             std::make_tuple(3, "small"),
                                             std::make_tuple(100, "medium"),
                                             std::make_tuple(10, "first_pair_best"),
                                             std::make_tuple(10, "last_pair_best"),
                                             std::make_tuple(10, "multiple_same"),
                                             std::make_tuple(10, "large_values"),
                                             std::make_tuple(10, "overflow_risk"),
                                             std::make_tuple(10, "mixed_signs"),
                                             std::make_tuple(10, "alternating"),
                                             std::make_tuple(500, "normal"),
                                             std::make_tuple(1000, "ascending"),
                                             std::make_tuple(1000, "descending"),
                                             std::make_tuple(1, "single_process"),
                                             std::make_tuple(2, "two_processes"),
                                             std::make_tuple(3, "three_processes"),
                                             std::make_tuple(4, "small_even"),
                                             std::make_tuple(5, "small_odd"),
                                             std::make_tuple(100, "medium_distributed")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<TsyplakovKVecNeighboursMPI, InType>(kTestParam, PPC_SETTINGS_tsyplakov_k_vec_neighbours),
    ppc::util::AddFuncTask<TsyplakovKVecNeighboursSEQ, InType>(kTestParam, PPC_SETTINGS_tsyplakov_k_vec_neighbours));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = TsyplakovKVecNeighboursFuncTests::PrintTestParam;

INSTANTIATE_TEST_SUITE_P(VectorFuncTests, TsyplakovKVecNeighboursFuncTests, kGtestValues, kPerfTestName);

}  // namespace
}  // namespace tsyplakov_k_vec_neighbours
