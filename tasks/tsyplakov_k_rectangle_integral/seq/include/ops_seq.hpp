#pragma once

#include "tsyplakov_k_rectangle_integral/common/include/common.hpp"

namespace tsyplakov_k_rectangle_integral {

class TsyplakovKRectangleIntegralSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit TsyplakovKRectangleIntegralSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsyplakov_k_rectangle_integral
